"""Observed-market trader selection feeding the separate hourly proxy engine."""

from dataclasses import asdict, replace
from datetime import timedelta

from .lab_config import day
from .lab_config_v2 import ExplicitUniverse
from .lab_pipeline_v2 import MarketRun
from .lab_schedule import SelectionState
from .pipeline import RunResult
from .proxy_selection import select_proxy_markets
from .proxy_conviction import ProxyConviction
from .proxy_sessions import hourly_windows
from .proxy_schedule import decision_times
from .proxy_simulator import ProxySimulationConfig, simulate_proxy
from .signals import aggregate_with_weights
from .lab_config_proxy import LabConfigProxyScheduled
from .ranking_artifact import RankingSink
from .scheduled_activity import history_days
from .follower_positions import selected_positions


class ProxySelectionState(SelectionState):
    def _prepare(self, at):
        prepare = getattr(self.dataset.activity, "prepare", None)
        if prepare is not None:
            prepare(at)

    def select_markets(self, at):
        self._prepare(at)
        return select_proxy_markets(
            self.dataset,
            self.config.market_universe,
            at,
            None if not self.market_cohorts else self.active,
            history_days=history_days(self.config),
        )

    def rank_traders(self, at, effective, scope):
        self._prepare(at)
        return self.dataset.activity.rank(
            at, effective, scope, self.dataset.manifest["fee_semantics"], smoke=True
        )


def validate_proxy_run(dataset, config, *, annual=False):
    start, end = day(config.start), day(config.end)
    if getattr(dataset, "native_starts", {}).get("BTC", start) > start:
        raise ValueError("BTC benchmark requires full-window native history")
    u = config.market_universe
    warmup_days = history_days(config)
    if (
        dataset.coverage_start > start - timedelta(days=warmup_days)
        or dataset.coverage_end < end
    ):
        raise ValueError("Missing required proxy trader/market warmup")
    maximum = 366 if annual and isinstance(config, LabConfigProxyScheduled) else 93
    if not 0 < (end - start).days <= maximum:
        raise ValueError(
            f"Proxy range must be at most {maximum} days; annual runs require disk ranking output"
        )
    ids = (
        set(u.instrument_ids)
        if isinstance(u, ExplicitUniverse)
        else {
            m.instrument_id
            for m in dataset.mappings.records
            if day(m.valid_from) < end
            and day(m.valid_to) > start
            and (
                u.general
                or m.asset_class
                in {
                    {
                        "commodity": "commodities",
                        "equity": "equities",
                        "index": "indices",
                        "crypto": "crypto",
                    }[c]
                    for c in u.classes
                }
            )
        }
    )
    ids.add("BTC")
    if len(ids) * int((end - start).total_seconds() / 3600) > 250_000:
        raise ValueError("Proxy asset-hour output ceiling exceeded")
    ticks = sum(
        1 for _ in decision_times(start, end, getattr(config, "rebalance", "hourly"))
    )
    if len(ids) > 50 or len(ids) * ticks * config.trader.max_cohort > 1_000_000:
        raise ValueError("Proxy replay output ceiling exceeded")
    actual = {(b.instrument_id, b.start, b.end) for b in dataset.bars}
    for coin in sorted(ids):
        mappings = [
            m
            for m in dataset.mappings.records
            if m.instrument_id == coin
            and day(m.valid_from) < end
            and day(m.valid_to) > start
        ]
        if not mappings and coin != "BTC":
            continue  # Explicit missing mappings remain visible selector exclusions.
        if len(mappings) != 1:
            raise ValueError(f"Missing or changing proxy mapping: {coin}")
        mapping = mappings[0]
        if not mapping.active(end - timedelta(microseconds=1)) or (
            coin == "BTC" and not mapping.active(start)
        ):
            raise ValueError(f"Missing full-window proxy mapping: {coin}")
        active_from = max(
            start,
            day(mapping.valid_from),
            getattr(dataset, "native_starts", {}).get(coin, start),
        )
        native_start = getattr(dataset, "native_starts", {}).get(coin)
        needs_seed = False
        if coin != "BTC" and native_start is not None:
            # The selector cannot admit this market before its native warmup
            # completes. Require every session from that conservative lower
            # bound onward, not merely sessions where the realized run trades.
            # BTC remains a full-period benchmark, independent of selection.
            warmed_from = native_start + timedelta(days=warmup_days)
            needs_seed = mapping.calendar != "24/7" and warmed_from > active_from
            active_from = max(active_from, warmed_from)
        if active_from >= end:
            continue  # Qualified future markets remain visible but ineligible.
        required = hourly_windows(mapping.calendar, active_from, end)
        if needs_seed and required:
            first_open = min(required)
            first_tick = first_open.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
            if not any(
                bar.instrument_id == coin
                and bar.end <= first_open
                and (first_tick - bar.end).total_seconds()
                <= config.proxy.max_mark_age_seconds
                for bar in dataset.bars
            ):
                raise ValueError(f"Missing fresh completed-close seed: {coin}")
        for begin, finish in required.items():
            if (coin, begin, finish) not in actual:
                raise ValueError(f"Missing scheduled proxy bar: {coin} at {begin}")
    return start, end


def _signals_at(dataset, state, config, at, conviction):
    f = config.follower
    signals, contributions = [], []
    for coin in sorted(state.ever):
        rows = state.selected.get(coin, []) if coin in state.active else []
        positions = selected_positions(
            dataset.activity, [r["user"] for r in rows], coin, at
        )
        values = (
            {user: conviction.value((user, coin), at) for user in positions}
            if conviction
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
        uncapped = signal.value * budget
        cap = min(1.0, f.asset_cap / abs(uncapped)) if uncapped else 1.0
        target = uncapped * cap
        signals.append(
            dict(time=at, coin=coin, signal_name=f.aggregation, **asdict(signal))
            | dict(value=target, aggregate_signal=signal.value)
        )
        for row in rows:
            user = row["user"]
            qty = positions[user]
            value = (
                values[user]
                if conviction
                else None
                if qty is None
                else (1.0 if qty > 0 else -1.0 if qty < 0 else 0.0)
            )
            contributions.append(
                dict(
                    time=at,
                    coin=coin,
                    user=user,
                    decision_time=row["decision_time"],
                    market_decision_time=state.market_time,
                    position_qty=qty,
                    signal_input=value,
                    known=value is not None,
                    nominal_weight=row["weight"],
                    effective_weight=weights[user],
                    target_contribution=(value or 0) * weights[user] * budget * cap,
                    aggregate_signal=signal.value,
                    portfolio_target=target,
                    reason=signal.reason,
                )
            )
    return signals, contributions


def run_configured_proxy(dataset, config, *, rankings_path=None):
    if rankings_path is None:
        return _run_configured_proxy(dataset, config)
    with RankingSink(rankings_path) as sink:
        output = _run_configured_proxy(dataset, config, sink)
    output.result.scores = sink.artifact
    return output


def _run_configured_proxy(dataset, config, sink=None):
    annual = sink is not None and isinstance(config, LabConfigProxyScheduled)
    start, end = validate_proxy_run(dataset, config, annual=annual)
    state = ProxySelectionState(dataset, config)
    if sink is not None:
        state.rankings = sink
    conviction = (
        ProxyConviction(dataset.activity, config, end)
        if config.follower.aggregation == "conviction_trimmed"
        else None
    )
    signals, contributions = [], []
    cadence = getattr(config, "rebalance", "hourly")
    for at in decision_times(start, end, cadence):
        state.advance(at)
        if conviction:
            conviction.advance(
                {
                    (r["user"], coin)
                    for coin in state.active
                    for r in state.selected.get(coin, [])
                },
                at,
            )
        new_signals, new_contributions = _signals_at(
            dataset, state, config, at, conviction
        )
        signals.extend(new_signals)
        contributions.extend(new_contributions)
        if (
            max(
                len(contributions),
                len(state.rankings) if sink is None else 0,
                len(state.market_rankings),
            )
            > 1_000_000
        ):
            raise ValueError("Proxy history output ceiling exceeded; no truncation")
    f = config.follower
    simulation = ProxySimulationConfig(
        initial_equity=f.initial_equity,
        gross_cap=f.gross_cap,
        asset_cap=f.asset_cap,
        fee_bps=f.fee_bps,
        latency_seconds=f.latency_seconds,
        min_trade_usd=f.min_trade_usd,
        deadband=f.deadband,
        **asdict(config.proxy),
    )
    strategy = simulate_proxy(
        signals,
        dataset.bars,
        dataset.funding,
        start,
        end,
        sorted(state.ever),
        simulation,
        max_days=366 if annual else 93,
        funding_starts={
            coin: at
            for coin, at in getattr(
                dataset,
                "funding_starts",
                getattr(dataset, "native_starts", {}),
            ).items()
            if coin in state.ever
        },
    )
    benchmark = simulate_proxy(
        [dict(time=start, coin="BTC", value=1)],
        dataset.bars,
        dataset.funding,
        start,
        end,
        ["BTC"],
        replace(simulation, gross_cap=1, asset_cap=1, deadband=0, min_trade_usd=0),
        max_days=366 if annual else 93,
        funding_starts={
            "BTC": getattr(
                dataset,
                "funding_starts",
                getattr(dataset, "native_starts", {}),
            )["BTC"]
        }
        if "BTC"
        in getattr(
            dataset,
            "funding_starts",
            getattr(dataset, "native_starts", {}),
        )
        else None,
    )
    cash = simulate_proxy(
        [], dataset.bars, [], start, end, [], simulation, max_days=366 if annual else 93
    )
    warnings = [
        "approximate_proxy_priced",
        "observed_mapped_market_universe_only",
        f"{cadence}_decisions_strict_next_open_execution",
        "funding_uses_proxy_completed_close_notional",
        "market_volume_uses_one_day_publication_lag",
        "native_leader_metrics_are_not_account_returns",
    ]
    if dataset.manifest.get("synthetic"):
        warnings.append("SYNTHETIC DEMO: fabricated activity, not market performance")
    if dataset.manifest["fee_semantics"] == "unknown":
        warnings.append("unresolved_fee_semantics")
    if conviction:
        warnings.append("conviction_uses_last_native_trade_price_not_exchange_mark")
    result = RunResult(
        state.rankings,
        state.trader_cohorts,
        signals,
        {(f.aggregation, f.latency_seconds): strategy},
        {("btc_buy_hold", f.latency_seconds): benchmark, ("cash", None): cash},
        warnings,
    )
    return MarketRun(result, contributions, state.market_rankings, state.market_cohorts)
