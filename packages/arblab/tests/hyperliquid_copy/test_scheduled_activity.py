from dataclasses import replace
from datetime import timedelta

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.activity_checkpoints import ActivityCheckpoints
from arblab.hyperliquid_copy.contracts import FillEvent
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_pipeline_proxy import (
    run_configured_proxy,
    ProxySelectionState,
)
from arblab.hyperliquid_copy.lab_schedule import preview_selection
from .test_activity_checkpoints import catalog_for
from .test_proxy_weekly import multiweek, weekly_config


def rolling(tmp_path, config):
    from arblab.hyperliquid_copy.scheduled_activity import (
        ScheduledActivity,
        history_days,
    )

    rows = [
        FillEvent(**row)
        for row in pq.read_table(tmp_path / "fills.parquet").to_pylist()
    ]
    catalog, ids, _ = catalog_for(tmp_path, [rows])
    store = ActivityCheckpoints(tmp_path / "checkpoints")
    identity = store.build(
        catalog,
        ids,
        day(config.start) - timedelta(days=history_days(config)),
        temp_root=tmp_path,
    )
    return ScheduledActivity(store, identity, config, temp_root=tmp_path)


@pytest.mark.parametrize("cadence", ["weekly", "daily"])
@pytest.mark.parametrize(
    "aggregation", ["direction_equal", "direction_score_weighted", "conviction_trimmed"]
)
def test_checkpoint_pipeline_matches_full_reader(tmp_path, aggregation, cadence):
    data = multiweek(tmp_path)
    c = weekly_config()
    c = replace(
        c,
        rebalance=cadence,
        trader=replace(c.trader, reselection=cadence),
        market_universe=replace(c.market_universe, reselection=cadence),
    )
    c = replace(
        c, follower=replace(c.follower, aggregation=aggregation, scale_lookback_days=2)
    )
    try:
        expected = run_configured_proxy(data, c)
    finally:
        data.activity.close()
    data.activity = rolling(tmp_path, c)
    try:
        actual = run_configured_proxy(data, c)
        assert actual == expected
        assert data.activity.reader.query_window == (
            day("2026-08-03"),
            day("2026-08-17") + timedelta(hours=1),
        )
    finally:
        data.activity.close()
    assert not list(tmp_path.glob("proxy_activity_*"))


def test_checkpoint_preview_prepares_hypothetical_midweek_window(tmp_path):
    data = multiweek(tmp_path)
    c = weekly_config()
    at = day("2026-08-12")
    try:
        expected = preview_selection(data, c, at, "BTC", state_type=ProxySelectionState)
    finally:
        data.activity.close()
    data.activity = rolling(tmp_path, c)
    try:
        actual = preview_selection(data, c, at, "BTC", state_type=ProxySelectionState)
        assert actual[0] == expected[0]
        assert actual[2] == expected[2]
        assert actual[1].trader_cohorts == expected[1].trader_cohorts
    finally:
        data.activity.close()


def test_reader_lifecycle_and_monotonic_decisions(tmp_path):
    data = multiweek(tmp_path)
    data.activity.close()
    c = weekly_config()
    activity = rolling(tmp_path, c)
    at = day(c.start)
    with activity:
        with pytest.raises(ValueError, match="prepare"):
            activity.observed(at)
        activity.prepare(at)
        reader = activity.reader
        activity.prepare(at)
        assert activity.reader is reader
        assert activity.position("0x" + "f" * 40, "BTC", at) is None
        for invalid in [at - timedelta(hours=1), at + timedelta(minutes=1), day(c.end)]:
            with pytest.raises(ValueError, match="decision"):
                activity.prepare(invalid)
    with pytest.raises(ValueError, match="closed"):
        activity.prepare(at)


def test_failed_open_does_not_leave_stale_query_reader(tmp_path):
    data = multiweek(tmp_path)
    data.activity.close()
    c = weekly_config()
    with rolling(tmp_path, c) as activity:
        activity.prepare(day(c.start))
        old_reader = activity.reader
        # Corrupt an actual frozen partition, not a mocked query implementation.
        from pathlib import Path

        entry = activity.store.metadata(activity.identity)["partitions"][0]
        Path(entry["path"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="identity"):
            activity.prepare(day("2026-08-10"))
        assert activity.reader is None
        assert old_reader.db is None
        with pytest.raises(ValueError, match="prepared"):
            activity.observed(day("2026-08-10"))


def test_warmup_includes_lagged_market_volume_and_conviction():
    from arblab.hyperliquid_copy.scheduled_activity import history_days

    c = weekly_config(
        market_universe=dict(
            mode="liquidity",
            top_n=1,
            lookback_days=30,
            publication_lag_days=1,
            reselection="weekly",
        )
    )
    assert history_days(c) == 31
    c = replace(
        c,
        follower=replace(
            c.follower, aggregation="conviction_trimmed", scale_lookback_days=60
        ),
    )
    assert history_days(c) == 60


def test_failed_new_window_can_retry_previous_successful_decision(
    tmp_path, monkeypatch
):
    data = multiweek(tmp_path)
    data.activity.close()
    c = weekly_config()
    at, next_at = day(c.start), day("2026-08-10")
    with rolling(tmp_path, c) as activity:
        activity.prepare(at)
        identity, cutoff = activity.identity, activity.cutoff
        original_open = activity.store.open

        def failing_open(identity, start, end, **kwargs):
            if end == next_at + timedelta(hours=1):
                raise OSError("injected new-window open failure")
            return original_open(identity, start, end, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(activity.store, "open", failing_open)
            with pytest.raises(OSError, match="injected"):
                activity.prepare(next_at)
        assert activity.identity == identity
        assert activity.cutoff == cutoff
        assert activity.last_decision == at
        assert activity.reader is None
        activity.prepare(at)
        assert activity.observed(at) == ["BTC"]
        activity.prepare(next_at)
        assert activity.last_decision == next_at
