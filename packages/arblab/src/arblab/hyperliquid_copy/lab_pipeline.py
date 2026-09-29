"""One effective strategy, historical membership evidence and independent benchmark."""

from dataclasses import asdict, replace
from datetime import timedelta
from math import ceil

from .contracts import utc
from .data_quality import validate_fills
from .lab_config import day
from .lab_ranking import rank_universe, cohort_snapshot, validate_ranking_bound
from .pipeline import RollingScale, RunResult
from .positions import PositionReplay
from .signals import aggregate_with_weights, conviction_value
from .simulator import SimulationConfig, simulate


def is_decision(at, start, frequency):
    return at == start or (
        at.hour == at.minute == at.second == 0
        and (
            frequency == "daily"
            or frequency == "weekly"
            and at.weekday() == 0
            or frequency == "monthly"
            and at.day == 1
        )
    )


def validate_inputs(fills, market, config, metadata):
    validate_ranking_bound(len(fills), config)
    start, end = day(config.start), day(config.end)
    warmup = start - timedelta(
        days=max(
            config.lookback_days,
            config.scale_lookback_days
            if config.aggregation == "conviction_trimmed"
            else 1,
        )
    )
    if day(metadata["coverage_start"]) > warmup or day(metadata["coverage_end"]) < end:
        raise ValueError("Dataset does not cover full lookback/warmup and simulation")
    minutes = int((end - start).total_seconds() / 60)
    if (
        minutes * len(config.coins) > 250000
        or (minutes // config.update_minutes + 1)
        * len(config.coins)
        * config.max_cohort
        > 1000000
    ):
        raise ValueError(
            "Run exceeds bounded replay/output ceiling; no wallet sampling performed"
        )
    validate_fills(fills, day(metadata["coverage_start"]), end).assert_accepted()
    coins = sorted(set(config.coins) | {"BTC"})
    for coin in coins:
        at = warmup
        while at <= end:
            market.mark(coin, at)
            if (
                start < at <= end
                and at.minute == 0
                and (coin, at) not in market.funding
            ):
                raise ValueError("Missing hourly funding; no zero-rate imputation")
            at += timedelta(minutes=1)
    requests, missing = 0, 0
    at = start
    while at < end:
        for coin in coins:
            requests += 1
            missing += (
                market.execution_book(
                    coin, at + timedelta(seconds=config.latency_seconds)
                )
                is None
            )
        at += timedelta(minutes=config.update_minutes)
    if missing / requests > 0.05:
        raise ValueError("Insufficient executable books at configured cadence/delay")
    return start, end, warmup


def run_configured(fills, market, config, metadata):
    end = day(config.end)
    fills = sorted(
        [f for f in fills if f.exchange_time < end], key=lambda f: f.order_key
    )
    start, end, warmup = validate_inputs(fills, market, config, metadata)
    semantics = metadata["fee_semantics"]
    replay = PositionReplay(fills)
    selected, previous, scales = {}, {}, {}
    scores, cohorts, signals, contributions = [], [], [], []
    at = start
    while at < end:
        if is_decision(at, start, config.reselection):
            for scope in config.coins if config.scope == "per_asset" else [None]:
                rows = rank_universe(fills, at, config, scope, semantics, smoke=True)
                scores.extend(rows)
                snapshot = cohort_snapshot(rows, at, scope, previous.get(scope))
                snapshot["requested_count"] = (
                    config.top_n
                    if config.selection == "n"
                    else ceil(config.top_fraction * snapshot["eligible_count"])
                )
                cohorts.append(snapshot)
                previous[scope] = snapshot["members"]
                for coin in config.coins if scope is None else [scope]:
                    selected[coin] = [r for r in rows if r["selected"]]
            if config.aggregation == "conviction_trimmed":
                required = {
                    (r["user"], c) for c, rows in selected.items() for r in rows
                }
                scales = {
                    key: value for key, value in scales.items() if key in required
                }
                for key in sorted(required - set(scales)):
                    scale = RollingScale(
                        config.scale_lookback_days, config.scale_quantile
                    )
                    moment = at - timedelta(days=config.scale_lookback_days)
                    while moment < at:
                        qty = replay.position(key, moment, inclusive=False)
                        if qty is not None:
                            scale.add(moment, abs(qty) * market.mark(key[1], moment))
                        moment += timedelta(minutes=1)
                    scales[key] = scale
        update = int((at - start).total_seconds() / 60) % config.update_minutes == 0
        for coin in config.coins:
            mid = market.mark(coin, at)
            positions = {
                r["user"]: replay.position((r["user"], coin), at, inclusive=False)
                for r in selected[coin]
            }
            if update:
                values = (
                    {
                        u: conviction_value(
                            position_qty=q, mid=mid, scale=scales[u, coin].value(at)
                        )
                        if q is not None
                        else None
                        for u, q in positions.items()
                    }
                    if config.aggregation == "conviction_trimmed"
                    else positions
                )
                score = {r["user"]: r["score"] for r in selected[coin]}
                signal, effective = aggregate_with_weights(
                    values,
                    score,
                    config.aggregation,
                    min_known=config.min_known,
                    min_weight=config.min_known_weight,
                    trim=config.trim,
                )
                budget = config.asset_weights[coin] * config.gross_cap
                cap_factor = (
                    min(1.0, config.asset_cap / abs(signal.value * budget))
                    if signal.value
                    else 1.0
                )
                target = signal.value * budget * cap_factor
                signals.append(
                    dict(
                        time=at,
                        coin=coin,
                        signal_name=config.aggregation,
                        **asdict(signal),
                    )
                    | {"value": target, "aggregate_signal": signal.value}
                )
                for row in selected[coin]:
                    user = row["user"]
                    value = values[user]
                    input_value = (
                        value
                        if config.aggregation == "conviction_trimmed" or value is None
                        else (1.0 if value > 0 else -1.0 if value < 0 else 0.0)
                    )
                    contributions.append(
                        dict(
                            time=at,
                            coin=coin,
                            user=user,
                            decision_time=row["decision_time"],
                            position_qty=positions[user],
                            signal_input=input_value,
                            known=value is not None,
                            nominal_weight=row["weight"],
                            effective_weight=effective[user],
                            target_contribution=(input_value or 0)
                            * effective[user]
                            * budget
                            * cap_factor,
                            aggregate_signal=signal.value,
                            portfolio_target=target,
                            reason=signal.reason,
                        )
                    )
            for user, qty in positions.items():
                if config.aggregation == "conviction_trimmed" and qty is not None:
                    scales[user, coin].add(at, abs(qty) * mid)
        at += timedelta(minutes=1)
    simulation = SimulationConfig(
        initial_equity=config.initial_equity,
        asset_cap=1,
        gross_cap=config.gross_cap,
        deadband=config.deadband,
        min_trade_usd=config.min_trade_usd,
        latency_seconds=config.latency_seconds,
        fee_bps=config.fee_bps,
    )
    strategy = simulate(signals, market, start, end, config.coins, simulation)
    benchmark = simulate(
        [dict(time=start, coin="BTC", value=1.0)],
        market,
        start,
        end,
        ["BTC"],
        replace(simulation, gross_cap=1, deadband=0, min_trade_usd=0),
    )
    cash = simulate([], market, start, end, config.coins, simulation)
    if any(s.insolvent for s in (strategy, benchmark, cash)):
        raise ValueError("Insolvent scenario; liquidation modelling is not available")
    warnings = [
        "DEVELOPMENT ONLY: new runner is not research-qualified",
        "funding_uses_settlement_mid_approximation",
    ]
    if metadata.get("synthetic"):
        warnings.insert(
            0,
            "SYNTHETIC DEMO: fabricated wallets, prices and funding; not market performance",
        )
    if semantics == "unknown":
        warnings.append("unresolved_fee_semantics")
    return RunResult(
        scores,
        cohorts,
        signals,
        {(config.aggregation, config.latency_seconds): strategy},
        {("btc_buy_hold", config.latency_seconds): benchmark, ("cash", None): cash},
        warnings,
    ), contributions
