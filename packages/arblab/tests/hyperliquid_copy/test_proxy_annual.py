from dataclasses import replace
from datetime import timedelta

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from arblab.hyperliquid_copy.proxy_bars import ProxyBar
from arblab.hyperliquid_copy.proxy_funding import FundingEvent
from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
from .test_proxy_weekly import multiweek, weekly_config
from .test_proxy_activity import fills, partition


def annual_dataset(tmp_path):
    data = multiweek(tmp_path)
    start = day("2026-08-03")
    warmup = start - timedelta(days=90)
    end = start + timedelta(days=365)
    data.coverage_start, data.coverage_end = warmup, end
    data.mappings = ProxyMappings(
        [
            replace(
                data.mappings.records[0],
                valid_from=str(warmup.date()),
                valid_to=str((end + timedelta(days=1)).date()),
            )
        ]
    )
    data.bars = [
        ProxyBar(
            "BTC",
            start + timedelta(hours=h),
            start + timedelta(hours=h + 1),
            100 + h * 0.001,
            101 + h * 0.001,
            100 + h * 0.001,
            101 + h * 0.001,
        )
        for h in range(-1, 8760)
    ]
    data.funding = [
        FundingEvent(
            "BTC", start + timedelta(hours=h), start + timedelta(hours=h), 0.000001, 0
        )
        for h in range(8760)
    ]
    initial = replace(
        fills()[0],
        exchange_time=warmup,
        tid=10000,
        px=100,
        sz=1,
        start_position=0,
        post_position=1,
        side="B",
        closed_pnl=0,
        fee=0,
    )
    rows = [initial]
    for i in range(1, 455):
        at = warmup + timedelta(days=i)
        rows.extend(
            [
                replace(
                    initial,
                    exchange_time=at,
                    tid=10000 + i * 2,
                    event_id=f"close{i}",
                    side="A",
                    start_position=1,
                    post_position=0,
                    closed_pnl=1,
                ),
                replace(
                    initial,
                    exchange_time=at + timedelta(hours=1),
                    tid=10001 + i * 2,
                    event_id=f"open{i}",
                ),
            ]
        )
    data.activity.close()
    data.activity = ProxyActivity(
        [partition(tmp_path, rows, "annual")], temp_root=tmp_path
    )
    return data


def annual_config():
    c = weekly_config(end="2027-08-03")
    return replace(
        c,
        trader=replace(c.trader, lookback_days=90),
        follower=replace(c.follower, fee_bps=1),
        proxy=replace(c.proxy, slippage_bps=5),
    )


def test_annual_scheduled_disk_history_and_daily_comparison(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
    from arblab.hyperliquid_copy.report import write_report

    data = annual_dataset(tmp_path)
    c = annual_config()
    try:
        with pytest.raises(ValueError, match="93|disk"):
            run_configured_proxy(data, c)
        weekly = run_configured_proxy(
            data, c, rankings_path=tmp_path / "weekly.parquet"
        )
        daily_config = replace(
            c,
            rebalance="daily",
            trader=replace(c.trader, reselection="daily"),
            market_universe=replace(c.market_universe, reselection="daily"),
        )
        daily = run_configured_proxy(
            data, daily_config, rankings_path=tmp_path / "daily.parquet"
        )
    finally:
        data.activity.close()
    for output, config, decisions, name in [
        (weekly, c, 53, "weekly"),
        (daily, daily_config, 365, "daily"),
    ]:
        result = output.result
        assert len(result.signals) == decisions
        assert len(result.scores) == decisions
        assert pq.read_table(result.scores.path).num_rows == decisions
        strategy = next(iter(result.strategies.values()))
        assert len(strategy.equity) == 8761
        assert len(strategy.funding) == 8760
        assert strategy.fills and any(r["cash_delta"] for r in strategy.funding)
        final = strategy.equity[-1]
        assert final["equity"] == pytest.approx(strategy.cash + final["unrealized_pnl"])
        report = tmp_path / ("report_" + name)
        write_report(
            report,
            result,
            config.to_dict() | {"research_eligible": False},
            data.manifest,
            {},
        )
        assert (
            report / "trader_scores.parquet"
        ).read_bytes() == result.scores.path.read_bytes()
        assert pq.read_table(report / "control_funding_ledger.parquet").num_rows == 8760


def test_streamed_short_run_matches_original_and_preserves_null_metrics(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
    from arblab.hyperliquid_copy.ranking_artifact import RANKING_SCHEMA
    import pyarrow as pa

    data = multiweek(tmp_path)
    try:
        old = run_configured_proxy(data, weekly_config())
        streamed = run_configured_proxy(
            data, weekly_config(), rankings_path=tmp_path / "scores.parquet"
        )
    finally:
        data.activity.close()
    normalized = [
        {k: None if v == {} else v for k, v in row.items()} for row in old.result.scores
    ]
    assert pq.read_table(streamed.result.scores.path).equals(
        pa.Table.from_pylist(normalized, schema=RANKING_SCHEMA)
    )
    assert old.result.signals == streamed.result.signals
    assert old.result.strategies == streamed.result.strategies


def test_late_year_preview_discards_old_rankings_but_keeps_cohort_history(tmp_path):
    from arblab.hyperliquid_copy.lab_schedule import preview_selection
    from arblab.hyperliquid_copy.lab_pipeline_proxy import ProxySelectionState

    data = annual_dataset(tmp_path)
    c = annual_config()
    c = replace(
        c,
        rebalance="daily",
        trader=replace(c.trader, reselection="daily"),
        market_universe=replace(c.market_universe, reselection="daily"),
    )
    try:
        rows, state, hypothetical = preview_selection(
            data,
            c,
            day(c.end) - timedelta(days=1),
            "BTC",
            state_type=ProxySelectionState,
        )
    finally:
        data.activity.close()
    assert not hypothetical and len(rows) == 1
    assert len(state.rankings) == 1
    assert len(state.trader_cohorts) == 365


@pytest.mark.parametrize(
    "maximum,days,coins,match",
    [
        (93, 94, 0, "93 days"),
        (366, 367, 0, "366 days"),
        (400, 365, 0, "day limit"),
        (366, 365, 29, "asset-hour"),
    ],
)
def test_simulator_annual_opt_in_and_resource_bounds(maximum, days, coins, match):
    from arblab.hyperliquid_copy.proxy_simulator import (
        simulate_proxy,
        ProxySimulationConfig,
    )

    start = day("2026-08-03")
    with pytest.raises(ValueError, match=match):
        simulate_proxy(
            [],
            [],
            [],
            start,
            start + timedelta(days=days),
            [f"ASSET{i}" for i in range(coins)],
            ProxySimulationConfig(),
            max_days=maximum,
        )
