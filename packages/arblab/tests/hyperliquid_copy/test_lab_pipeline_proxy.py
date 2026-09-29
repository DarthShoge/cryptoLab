from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxy
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from arblab.hyperliquid_copy.proxy_bars import ProxyBar
from arblab.hyperliquid_copy.proxy_funding import FundingEvent
from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
from .test_proxy_activity import fills, partition
from .test_proxy_selection import mapping


def config(**changes):
    return LabConfigProxy(
        **(
            dict(
                start="2026-08-03",
                end="2026-08-04",
                market_universe=dict(mode="explicit", instrument_ids=["BTC"]),
                trader=dict(
                    lookback_days=1,
                    min_active_days=1,
                    min_episodes=1,
                    min_notional=0,
                    min_minutes=0,
                    min_cohort=1,
                    selection="n",
                    top_n=2,
                    top_fraction=None,
                    metric_weights={"pnl_efficiency": 1},
                ),
                follower=dict(
                    update_minutes=60,
                    min_known=1,
                    fee_bps=0,
                    deadband=0,
                    min_trade_usd=0,
                ),
                proxy=dict(slippage_bps=0),
            )
            | changes
        )
    )


def dataset(tmp_path):
    start = day("2026-08-03")
    original = fills()
    earliest = min(f.exchange_time for f in original)
    rows = [
        replace(
            f, exchange_time=start - timedelta(hours=12) + (f.exchange_time - earliest)
        )
        for f in original
    ]
    rows += [
        replace(
            f,
            tid=100 + i,
            event_id="reopen" + str(i),
            exchange_time=start - timedelta(hours=1),
        )
        for i, f in enumerate(original[::2])
    ]
    bars = [
        ProxyBar(
            "BTC",
            start + timedelta(hours=h),
            start + timedelta(hours=h + 1),
            100 + h,
            101 + h,
            100 + h,
            101 + h,
        )
        for h in range(-1, 24)
    ]
    rates = [
        FundingEvent(
            "BTC", start + timedelta(hours=h), start + timedelta(hours=h), 0, 0
        )
        for h in range(24)
    ]
    return SimpleNamespace(
        activity=ProxyActivity([partition(tmp_path, rows)], temp_root=tmp_path),
        bars=bars,
        funding=rates,
        mappings=ProxyMappings([mapping()]),
        coverage_start=start - timedelta(days=2),
        coverage_end=start + timedelta(days=1),
        manifest={"synthetic": True, "fee_semantics": "gross_excludes_fee"},
    )


def test_copy_pipeline_retains_market_trader_history_contributions_and_benchmarks(
    tmp_path,
):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    data = dataset(tmp_path)
    try:
        run = run_configured_proxy(data, config())
    finally:
        data.activity.close()
    assert run.market_cohorts[0]["members"] == ["BTC"]
    assert run.result.cohorts[0]["selected_count"] == 2
    assert len(run.contributions) == 48
    assert all(r["known"] for r in run.contributions)
    strategy = next(iter(run.result.strategies.values()))
    assert strategy.fills
    assert strategy.equity[-1]["equity"] > 10000
    assert len(strategy.equity) == 25
    assert set(k[0] for k in run.result.controls) == {"btc_buy_hold", "cash"}
    assert "approximate_proxy_priced" in run.result.warnings
    assert run.result.signals[0]["value"] == 0.5


def test_runner_rejects_missing_warmup_and_scheduled_price_holes(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    data = dataset(tmp_path)
    try:
        data.coverage_start = day(config().start)
        with pytest.raises(ValueError, match="warmup"):
            run_configured_proxy(data, config())
        data.coverage_start -= timedelta(days=2)
        data.bars.pop(5)
        with pytest.raises(ValueError, match="scheduled proxy bar"):
            run_configured_proxy(data, config())
    finally:
        data.activity.close()


def test_conviction_uses_native_hourly_exposure_and_preserves_contributions(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    data = dataset(tmp_path)
    c = config(
        follower=dict(
            update_minutes=60,
            aggregation="conviction_trimmed",
            min_known=1,
            scale_lookback_days=1,
            trim=0,
            fee_bps=0,
            deadband=0,
            min_trade_usd=0,
        )
    )
    try:
        run = run_configured_proxy(data, c)
    finally:
        data.activity.close()
    assert run.result.signals[0]["value"] == 0.5
    assert (
        "conviction_uses_last_native_trade_price_not_exchange_mark"
        in run.result.warnings
    )
    for signal in run.result.signals:
        rows = [r for r in run.contributions if r["time"] == signal["time"]]
        assert sum(r["target_contribution"] for r in rows) == pytest.approx(
            signal["value"]
        )


def test_future_position_and_price_do_not_change_earlier_conviction(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    data = dataset(tmp_path)
    c = config(
        follower=dict(
            update_minutes=60,
            aggregation="conviction_trimmed",
            min_known=1,
            scale_lookback_days=1,
            trim=0,
            fee_bps=0,
            deadband=0,
            min_trade_usd=0,
        )
    )
    try:
        baseline = run_configured_proxy(data, c)
    finally:
        data.activity.close()
    future = replace(
        fills()[-2],
        event_id="future-reversal",
        tid=9999,
        exchange_time=day(c.start) + timedelta(hours=2),
        px=10000,
        side="A",
        start_position=1,
        sz=2,
        post_position=-1,
    )
    path = partition(tmp_path, [future], "future")
    data.activity = ProxyActivity(
        [tmp_path / "fills.parquet", path], temp_root=tmp_path
    )
    try:
        changed = run_configured_proxy(data, c)
    finally:
        data.activity.close()
    assert baseline.result.signals[:3] == changed.result.signals[:3]
    assert baseline.result.signals[3]["value"] != changed.result.signals[3]["value"]


def test_pooled_fraction_strategy_uses_same_hourly_pipeline(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    data = dataset(tmp_path)
    trader = config().to_dict()["trader"] | dict(
        scope="pooled", selection="fraction", top_n=None, top_fraction=0.5
    )
    try:
        run = run_configured_proxy(data, config(trader=trader))
    finally:
        data.activity.close()
    assert run.result.cohorts[0]["coin"] is None
    assert run.result.cohorts[0]["selected_count"] == 3


def test_later_market_mapping_cannot_bypass_scheduled_price_coverage(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import validate_proxy_run

    data = dataset(tmp_path)
    start = day("2026-08-03")
    data.coverage_end = start + timedelta(days=2)
    data.bars += [
        ProxyBar(
            "BTC",
            start + timedelta(hours=h),
            start + timedelta(hours=h + 1),
            100,
            100,
            100,
            100,
        )
        for h in range(24, 48)
    ]
    data.mappings = ProxyMappings(
        [
            mapping(),
            mapping(instrument_id="ETH", ticker="ETHUSDT", valid_from="2026-08-04"),
        ]
    )
    c = config(
        end="2026-08-05",
        market_universe=dict(mode="liquidity", top_n=1, lookback_days=1),
    )
    try:
        with pytest.raises(ValueError, match="Missing scheduled proxy bar: ETH"):
            validate_proxy_run(data, c)
    finally:
        data.activity.close()


def test_proxy_report_writes_accounting_and_pending_request_evidence(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
    from arblab.hyperliquid_copy.report import write_report
    import pyarrow.parquet as pq
    data = dataset(tmp_path)
    c = config()
    try:
        result = run_configured_proxy(data, c).result
    finally:
        data.activity.close()
    output = tmp_path / "proxy-report"
    summary = write_report(output, result, c.to_dict() | {"research_eligible": False}, data.manifest, {})
    strategy = next(r for r in summary["scenarios"] if r["scenario_type"] == "strategy")
    assert strategy["fill_ratio"] is None  # No order-book/depth fill-rate claim.
    assert strategy["execution_model"] == "hourly_proxy_open"
    assert strategy["final_equity"] == pytest.approx(strategy["final_collateral"]+strategy["final_unrealized_pnl"])
    assert strategy["unexecuted_requests"] >= 1
    requests = pq.read_table(output / "proxy_requests.parquet").to_pylist()
    assert requests
    controls = [r for r in requests if r["scenario_type"] == "control"]
    assert controls and all(r["control_name"] == "btc_buy_hold" and r["signal_name"] is None for r in controls)
    assert pq.read_table(output / "control_funding_ledger.parquet").num_rows == 24
    prose = (output / "report.md").read_text()
    assert "proxy units" in prose
    assert "All four latencies" not in prose
