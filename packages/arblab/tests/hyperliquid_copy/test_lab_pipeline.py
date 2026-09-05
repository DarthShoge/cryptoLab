from dataclasses import replace
from datetime import timedelta

import pytest


def test_schedule_and_cohort_deltas():
    from arblab.hyperliquid_copy.lab_config import day
    from arblab.hyperliquid_copy.lab_pipeline import is_decision
    from arblab.hyperliquid_copy.lab_ranking import cohort_snapshot

    start = day("2026-01-03")
    assert is_decision(start, start, "weekly")
    assert not is_decision(start + timedelta(days=1), start, "weekly")
    assert is_decision(day("2026-01-05"), start, "weekly")
    assert is_decision(day("2026-02-01"), start, "monthly")
    rows = [dict(user=u, selected=True, eligible=True) for u in ("a", "c")]
    snap = cohort_snapshot(rows, start, "SOL", ["a", "b"])
    assert snap["entries"] == ["c"] and snap["exits"] == ["b"]
    assert snap["membership_turnover"] == 0.5 and snap["retention"] == 0.5


def test_configured_backtest_records_universe_and_reconciles_contributions():
    from arblab.hyperliquid_copy.lab_fixture import fixture
    from arblab.hyperliquid_copy.lab_pipeline import run_configured

    fills, market, config, metadata = fixture()
    result, contributions = run_configured(fills, market, config, metadata)
    assert set(result.strategies) == {(config.aggregation, config.latency_seconds)}
    assert set(result.controls) == {
        ("btc_buy_hold", config.latency_seconds),
        ("cash", None),
    }
    assert {r["coin"] for r in result.signals} == {"ETH", "SOL"}
    assert len(result.cohorts) == 4
    assert any(r["exits"] for r in result.cohorts)
    assert (
        result.strategies[config.aggregation, config.latency_seconds].equity[-1][
            "equity"
        ]
        > 0
    )
    for signal in result.signals[::100]:
        rows = [
            r
            for r in contributions
            if (r["time"], r["coin"]) == (signal["time"], signal["coin"])
        ]
        assert sum(r["target_contribution"] for r in rows) == pytest.approx(
            signal["value"]
        )
        assert abs(signal["value"]) <= config.asset_cap


def test_append_future_fills_does_not_change_historical_output():
    from arblab.hyperliquid_copy.lab_fixture import fixture
    from arblab.hyperliquid_copy.lab_pipeline import run_configured
    from arblab.hyperliquid_copy.lab_config import day

    fills, market, config, metadata = fixture()
    result, _ = run_configured(fills, market, config, metadata)
    future = [
        replace(
            f,
            exchange_time=day(config.end) + timedelta(days=2),
            event_id=f.event_id + "future",
        )
        for f in fills[:2]
    ]
    other, _ = run_configured(fills + future, market, config, metadata)
    assert result.cohorts == other.cohorts
    assert result.signals == other.signals
