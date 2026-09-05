"""Two-stage market/trader replay with lifetime-aware valuation and exit targets."""

from dataclasses import dataclass, asdict, replace
from datetime import timedelta
from .lab_config import day
from .lab_config_v2 import ExplicitUniverse
from .lab_schedule import SelectionState
from .lab_ranking import validate_ranking_bound
from .data_quality import validate_fills
from .pipeline import RollingScale, RunResult
from .positions import PositionReplay
from .signals import aggregate_with_weights, conviction_value
from .simulator import simulate, SimulationConfig


@dataclass
class MarketRun:
    result: RunResult
    contributions: list
    market_rankings: list
    market_cohorts: list


def participating_ids(dataset, config):
    u = config.market_universe
    if isinstance(u, ExplicitUniverse):
        return u.instrument_ids
    return sorted(
        {
            r.instrument_id
            for r in dataset.catalogue.records
            if r.known_at < day(config.end)
            and r.effective_from < day(config.end)
            and (u.general or r.asset_class in u.classes)
        }
    )


def validate_v2(dataset, config, ids):
    start, end = day(config.start), day(config.end)
    f = config.follower
    effective = config.effective(ids, {})
    validate_ranking_bound(len(dataset.fills), effective)
    minutes = int((end - start).total_seconds() / 60)
    evidence_ids = (
        ids
        if isinstance(config.market_universe, ExplicitUniverse)
        else dataset.catalogue.known_ids(end)
    )
    if len(evidence_ids) * (end - start).days > 1000000:
        raise ValueError("Market history exceeds bounded output ceiling")
    if (
        minutes * len(ids) > 250000
        or (minutes // f.update_minutes + 1) * len(ids) * config.trader.max_cohort
        > 1000000
    ):
        raise ValueError("Run exceeds bounded replay/output ceiling")
    warm_days = max(
        config.trader.lookback_days,
        f.scale_lookback_days if f.aggregation == "conviction_trimmed" else 1,
        getattr(config.market_universe, "lookback_days", 0)
        + getattr(config.market_universe, "publication_lag_days", 0),
    )
    warmup = start - timedelta(days=warm_days)
    if (
        day(dataset.manifest["coverage_start"]) > warmup
        or day(dataset.manifest["coverage_end"]) < end
    ):
        raise ValueError("Missing required market/trader warmup")
    validate_fills(
        [r for r in dataset.fills if r.exchange_time < end],
        day(dataset.manifest["coverage_start"]),
        end,
    ).assert_accepted()
    required = sorted(set(ids) | {"BTC"})
    lifetimes = dataset.catalogue.lifetimes(required)
    for coin in required:
        life = lifetimes[coin]
        if not life.supported:
            continue  # selector excludes unsupported contracts
        if life.delisted_at is not None and start <= life.delisted_at <= end:
            raise ValueError("Delisting settlement is not supported")
        at = max(warmup, life.listed_at)
        # Validation must use the same UTC minute grid as portfolio replay,
        # regardless of the exchange's subminute listing timestamp.
        if at.second or at.microsecond:
            at = at.replace(second=0, microsecond=0) + timedelta(minutes=1)
        while at <= end:
            dataset.market.mark(coin, at)
            if (
                start < at <= end
                and at.minute == at.second == 0
                and (coin, at) not in dataset.market.funding
            ):
                raise ValueError("Missing hourly funding")
            at += timedelta(minutes=1)
    for fill in dataset.fills:
        if fill.coin in lifetimes and not lifetimes[fill.coin].active(
            fill.exchange_time
        ):
            raise ValueError("Leader fill outside instrument lifetime")
    missing, total = 0, 0
    at = start
    while at < end:
        for coin in required:
            if lifetimes[coin].supported and lifetimes[coin].active(at):
                total += 1
                missing += (
                    dataset.market.execution_book(
                        coin, at + timedelta(seconds=f.latency_seconds)
                    )
                    is None
                )
        at += timedelta(minutes=f.update_minutes)
    if total and missing / total > 0.05:
        raise ValueError("Insufficient executable books")
    if not lifetimes["BTC"].supported or not lifetimes["BTC"].active(start):
        raise ValueError("BTC benchmark unavailable")
    return start, end, lifetimes


def run_configured_v2(dataset, config):
    ids = participating_ids(dataset, config)
    start, end, lifetimes = validate_v2(dataset, config, ids)
    f = config.follower
    state = SelectionState(dataset, config)
    replay = PositionReplay(dataset.fills)
    scales = {}
    signals = []
    contributions = []
    at = start
    while at < end:
        reranked = state.advance(at)
        if f.aggregation == "conviction_trimmed" and reranked:
            required = {
                (r["user"], coin)
                for coin in state.active
                for r in state.selected.get(coin, [])
            }
            scales = {k: v for k, v in scales.items() if k in required}
            for key in sorted(required - set(scales)):
                scale = RollingScale(f.scale_lookback_days, f.scale_quantile)
                moment = at - timedelta(days=f.scale_lookback_days)
                while moment < at:
                    qty = replay.position(key, moment, inclusive=False)
                    if qty is not None:
                        scale.add(
                            moment, abs(qty) * dataset.market.mark(key[1], moment)
                        )
                    moment += timedelta(minutes=1)
                scales[key] = scale
        update = int((at - start).total_seconds() / 60) % f.update_minutes == 0
        for coin in sorted(state.ever):
            if not lifetimes[coin].active(at):
                continue
            rows = state.selected.get(coin, []) if coin in state.active else []
            mid = dataset.market.mark(coin, at)
            positions = {
                r["user"]: replay.position((r["user"], coin), at, inclusive=False)
                for r in rows
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
                    if f.aggregation == "conviction_trimmed"
                    else positions
                )
                signal, weights = aggregate_with_weights(
                    values,
                    {r["user"]: r["score"] for r in rows},
                    f.aggregation,
                    min_known=f.min_known,
                    min_weight=f.min_known_weight,
                    trim=f.trim,
                )
                budget = state.budgets.get(coin, 0) * f.gross_cap
                cap = (
                    min(1.0, f.asset_cap / abs(signal.value * budget))
                    if signal.value * budget
                    else 1.0
                )
                target = signal.value * budget * cap
                signals.append(
                    dict(
                        time=at, coin=coin, signal_name=f.aggregation, **asdict(signal)
                    )
                    | {"value": target, "aggregate_signal": signal.value}
                )
                for row in rows:
                    user = row["user"]
                    value = values[user]
                    input_value = (
                        value
                        if f.aggregation == "conviction_trimmed" or value is None
                        else (1.0 if value > 0 else -1.0 if value < 0 else 0.0)
                    )
                    contributions.append(
                        dict(
                            time=at,
                            coin=coin,
                            user=user,
                            decision_time=row["decision_time"],
                            market_decision_time=state.market_time,
                            position_qty=positions[user],
                            signal_input=input_value,
                            known=value is not None,
                            nominal_weight=row["weight"],
                            effective_weight=weights[user],
                            target_contribution=(input_value or 0)
                            * weights[user]
                            * budget
                            * cap,
                            aggregate_signal=signal.value,
                            portfolio_target=target,
                            reason=signal.reason,
                        )
                    )
            if f.aggregation == "conviction_trimmed":
                for user, qty in positions.items():
                    if qty is not None:
                        scales[user, coin].add(at, abs(qty) * mid)
        at += timedelta(minutes=1)
    simulation = SimulationConfig(
        initial_equity=f.initial_equity,
        asset_cap=1,
        gross_cap=f.gross_cap,
        deadband=f.deadband,
        min_trade_usd=f.min_trade_usd,
        latency_seconds=f.latency_seconds,
        fee_bps=f.fee_bps,
    )
    tradable_ids = [c for c in ids if lifetimes[c].supported]
    strategy = simulate(
        signals,
        dataset.market,
        start,
        end,
        tradable_ids,
        simulation,
        lifetimes=lifetimes,
    )
    benchmark = simulate(
        [dict(time=start, coin="BTC", value=1.0)],
        dataset.market,
        start,
        end,
        ["BTC"],
        replace(simulation, gross_cap=1, deadband=0, min_trade_usd=0),
        lifetimes=lifetimes,
    )
    cash = simulate([], dataset.market, start, end, [], simulation, lifetimes={})
    if any(s.insolvent for s in (strategy, benchmark, cash)):
        raise ValueError("Insolvent scenario; liquidation model unavailable")
    warnings = [
        "DEVELOPMENT ONLY: cross-class runner is not research-qualified",
        "funding_uses_settlement_mid_approximation",
        "market_volume_uses_one_day_publication_lag",
    ]
    if dataset.manifest["synthetic"]:
        warnings.insert(
            0, "SYNTHETIC DEMO: fabricated markets and wallets, not market performance"
        )
    if dataset.manifest["fee_semantics"] == "unknown":
        warnings.append("unresolved_fee_semantics")
    result = RunResult(
        state.rankings,
        state.trader_cohorts,
        signals,
        {(f.aggregation, f.latency_seconds): strategy},
        {("btc_buy_hold", f.latency_seconds): benchmark, ("cash", None): cash},
        warnings,
    )
    return MarketRun(result, contributions, state.market_rankings, state.market_cohorts)
