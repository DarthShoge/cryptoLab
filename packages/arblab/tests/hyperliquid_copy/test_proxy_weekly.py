from dataclasses import replace
from datetime import timedelta

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.proxy_bars import ProxyBar
from arblab.hyperliquid_copy.proxy_funding import FundingEvent
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from .test_lab_pipeline_proxy import dataset, config
from .test_proxy_activity import fills, partition


def test_scheduled_report_retains_proxy_requests_and_benchmark_funding(tmp_path):
    import pyarrow.parquet as pq
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
    from arblab.hyperliquid_copy.report import write_report

    data = multiweek(tmp_path)
    c = weekly_config()
    try:
        result = run_configured_proxy(data, c).result
    finally:
        data.activity.close()
    output = tmp_path / "report"
    write_report(
        output, result, c.to_dict() | {"research_eligible": False}, data.manifest, {}
    )
    assert (output / "proxy_requests.parquet").is_file()
    assert pq.read_table(output / "control_funding_ledger.parquet").num_rows == 360
    prose = (output / "report.md").read_text()
    assert "quantities in proxy units" in prose
    assert "every weekly membership" in prose
    assert "All four latencies" not in prose


def weekly_config(**changes):
    from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxyScheduled

    data = config().to_dict()
    data.pop("schema_version")
    data.update(end="2026-08-18")
    data["trader"].update(
        reselection="weekly",
        lookback_days=14,
        metric_weights={"gross_volume": 1},
        metric_directions={},
    )
    data["market_universe"]["reselection"] = "weekly"
    return LabConfigProxyScheduled(**(data | changes))


def multiweek(tmp_path, reverse=False):
    data = dataset(tmp_path)
    start = day("2026-08-03")
    data.coverage_start = start - timedelta(days=14)
    data.coverage_end = start + timedelta(days=15)
    data.bars = [
        ProxyBar(
            "BTC",
            start + timedelta(hours=h),
            start + timedelta(hours=h + 1),
            100 + h * 0.01,
            101 + h * 0.01,
            100 + h * 0.01,
            101 + h * 0.01,
        )
        for h in range(-1, 360)
    ]
    data.funding = [
        FundingEvent(
            "BTC", start + timedelta(hours=h), start + timedelta(hours=h), 0.00001, 0
        )
        for h in range(360)
    ]
    if reverse:
        data.activity.close()
        event = replace(
            fills()[-2],
            event_id="midweek",
            tid=99999,
            exchange_time=start + timedelta(days=2),
            side="A",
            start_position=1,
            sz=2,
            post_position=-1,
        )
        path = partition(tmp_path, [event], "reversal")
        data.activity = ProxyActivity(
            [tmp_path / "fills.parquet", path], temp_root=tmp_path
        )
    return data


def test_weekly_targets_ignore_midweek_leaders_but_value_and_fund_hourly(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    roots = [tmp_path / "baseline", tmp_path / "reversed"]
    runs = []
    for root, reverse in zip(roots, [False, True]):
        root.mkdir()
        data = multiweek(root, reverse)
        try:
            runs.append(run_configured_proxy(data, weekly_config()))
        finally:
            data.activity.close()
    first, changed = [r.result for r in runs]
    assert [s["time"] for s in changed.signals] == [
        day(d) for d in ["2026-08-03", "2026-08-10", "2026-08-17"]
    ]
    assert first.signals[0] == changed.signals[0]
    assert first.signals[1]["value"] != changed.signals[1]["value"]
    strategy = next(iter(changed.strategies.values()))
    assert len(strategy.equity) == 361
    assert len(strategy.funding) == 360
    assert any(r["cash_delta"] != 0 for r in strategy.funding)
    held = [
        r["qty"]
        for r in strategy.funding
        if day("2026-08-03") + timedelta(hours=2) <= r["time"] <= day("2026-08-10")
    ]
    assert held and held[0] != 0 and len(set(held)) == 1
    assert all(f["signal_time"].weekday() == 0 for f in strategy.fills)
    assert "weekly_decisions_strict_next_open_execution" in changed.warnings


def test_non_monday_start_stays_cash_until_first_weekly_decision(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    data = multiweek(tmp_path)
    try:
        run = run_configured_proxy(data, weekly_config(start="2026-08-04"))
    finally:
        data.activity.close()
    assert run.result.signals[0]["time"] == day("2026-08-10")
    strategy = next(iter(run.result.strategies.values()))
    assert all(
        r["gross_exposure"] == 0
        for r in strategy.equity
        if r["time"] <= day("2026-08-10")
    )
    assert run.market_cohorts[0]["decision_time"] == day("2026-08-10")


def test_weekly_preview_before_first_monday_has_no_selected_history(tmp_path):
    from arblab.hyperliquid_copy.lab_schedule import preview_selection
    from arblab.hyperliquid_copy.lab_pipeline_proxy import ProxySelectionState

    data = multiweek(tmp_path)
    try:
        rows, state, hypothetical = preview_selection(
            data,
            weekly_config(start="2026-08-04"),
            day("2026-08-05"),
            "BTC",
            state_type=ProxySelectionState,
        )
    finally:
        data.activity.close()
    assert rows == [] and state.trader_cohorts == [] and hypothetical


def test_weekly_schedule_delays_to_real_open_and_expires_or_supersedes():
    from arblab.hyperliquid_copy.proxy_schedule import decision_times
    from arblab.hyperliquid_copy.proxy_simulator import (
        ProxySimulationConfig,
        simulate_proxy,
    )

    start = day("2026-08-03")
    end = start + timedelta(days=15)
    decisions = list(decision_times(start, end, "weekly"))
    signals = [
        dict(time=t, coin="BTC", value=1 if i == 0 else -1)
        for i, t in enumerate(decisions)
    ]
    # Artificial long closure: first actual open is Tuesday of the second week.
    opening = start + timedelta(days=8)
    bars = [
        ProxyBar("BTC", t, t + timedelta(hours=1), 100, 100, 100, 100)
        for t in [start - timedelta(hours=1), opening]
    ]
    funding = [
        FundingEvent(
            "BTC", start + timedelta(hours=h), start + timedelta(hours=h), 0, 0
        )
        for h in range(360)
    ]
    for wait, reason in [(4 * 86400, "no_open_before_end"), (10 * 86400, "superseded")]:
        result = simulate_proxy(
            signals,
            bars,
            funding,
            start,
            end,
            ["BTC"],
            ProxySimulationConfig(
                max_wait_seconds=wait, max_mark_age_seconds=20 * 86400
            ),
        )
        assert result.requests[0]["reason"] == reason
        assert len(result.fills) == 1
        assert result.fills[0]["time"] == opening
        assert result.fills[0]["signal_time"] == decisions[1]
        assert result.fills[0]["filled_qty"] < 0
