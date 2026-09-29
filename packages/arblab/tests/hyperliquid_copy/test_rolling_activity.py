from dataclasses import replace
from datetime import timedelta
import json

import pytest

from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from .test_proxy_activity import fills, partition
from .test_lab_ranking import settings


def fixture(tmp_path):
    old = fills()
    start = max(f.exchange_time for f in old) + timedelta(days=3)
    recent = [
        replace(
            f,
            tid=f.tid + 100,
            event_id=f.event_id + "recent",
            exchange_time=f.exchange_time + timedelta(days=4),
        )
        for f in old[:2]
    ]
    future = replace(
        old[0],
        tid=999,
        event_id="future",
        user="0x" + "f" * 40,
        exchange_time=start + timedelta(days=5),
    )
    rows = [*old, *recent, future]
    return partition(tmp_path, rows), rows, start, start + timedelta(days=2)


def test_window_retains_dormant_wallets_and_matches_reference(tmp_path):
    path, rows, start, end = fixture(tmp_path)
    with (
        ProxyActivity([path], temp_root=tmp_path) as full,
        ProxyActivity([path], temp_root=tmp_path, query_window=(start, end)) as rolling,
    ):
        assert rolling.seed_count == 6
        assert rolling.count == 8 < full.count
        for at in [start, start + timedelta(days=1), end]:
            assert rolling.observed(at) == full.observed(at)
            for user in {f.user for f in rows}:
                assert rolling.position(user, "BTC", at) == full.position(
                    user, "BTC", at
                )
            if at > start:
                assert rolling.volume("BTC", start, at) == full.volume("BTC", start, at)
            if at >= start + timedelta(days=1):
                assert rolling.rank(
                    at, settings(lookback_days=1), "BTC", "gross_excludes_fee"
                ) == full.rank(
                    at, settings(lookback_days=1), "BTC", "gross_excludes_fee"
                )
        args = (rows[0].user, "BTC", start, end)
        assert rolling.hourly_exposure(
            *args, max_price_age_seconds=864000
        ) == full.hourly_exposure(*args, max_price_age_seconds=864000)


def test_evicted_history_queries_reject_and_wider_reopen_works(tmp_path):
    path, rows, start, end = fixture(tmp_path)
    with ProxyActivity([path], temp_root=tmp_path, query_window=(start, end)) as a:
        calls = [
            lambda: a.position(rows[0].user, "BTC", start - timedelta(seconds=1)),
            lambda: a.observed(end + timedelta(seconds=1)),
            lambda: a.volume("BTC", start - timedelta(days=1), end),
            lambda: a.rank(end, settings(lookback_days=5), "BTC", "gross_excludes_fee"),
            lambda: a.hourly_exposure(
                rows[0].user,
                "BTC",
                start - timedelta(hours=1),
                end,
                max_price_age_seconds=3600,
            ),
            lambda: a.validate_registered_scope(["BTC"], start, end),
        ]
        for call in calls:
            with pytest.raises(ValueError, match="[Ww]indow"):
                call()
    with ProxyActivity(
        [path], temp_root=tmp_path, query_window=(start - timedelta(days=5), end)
    ) as b:
        assert b.rank(end, settings(lookback_days=5), "BTC", "gross_excludes_fee")


def test_old_conflicting_duplicates_still_reject(tmp_path):
    old = fills()[0]
    path = partition(tmp_path, [old, replace(old, px=999)])
    start = old.exchange_time + timedelta(days=90)
    with pytest.raises(ValueError, match="conflicting"):
        ProxyActivity(
            [path], temp_root=tmp_path, query_window=(start, start + timedelta(days=7))
        )


def test_seed_order_and_exact_lower_boundary(tmp_path):
    first = fills()[0]
    last = replace(
        first,
        tid=999,
        event_id="last",
        source_line=first.source_line + 1,
        start_position=1,
        post_position=2,
    )
    start = first.exchange_time + timedelta(days=2)
    boundary = replace(
        first,
        tid=1000,
        event_id="boundary",
        exchange_time=start,
        start_position=2,
        post_position=3,
    )
    path = partition(tmp_path, [boundary, last, first])
    with ProxyActivity(
        [path], temp_root=tmp_path, query_window=(start, start + timedelta(days=1))
    ) as a:
        assert a.seed_count == 1 and a.count == 2
        assert a.position(first.user, "BTC", start) == 2
        assert a.position(first.user, "BTC", start + timedelta(microseconds=1)) == 3


def test_catalog_adapter_reads_prefix_and_rejects_corruption(tmp_path):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog
    from arblab.hyperliquid_copy.rolling_activity import open_rolling_activity
    from .test_compact_catalog import compact

    manifest, rows = compact(tmp_path)
    catalog = CompactCatalog(tmp_path / "catalog.sqlite3")
    identity = catalog.register(manifest)
    start = max(f.exchange_time for f in rows) + timedelta(days=2)
    end = start + timedelta(days=1)
    assert catalog.partitions([identity], start, end) == []
    with open_rolling_activity(
        catalog, [identity], start, end, temp_root=tmp_path
    ) as a:
        assert a.seed_count == 6
        assert a.position(rows[0].user, "BTC", start) == 0
    path = manifest.parent / json.loads(manifest.read_text())["files"][0]["name"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        open_rolling_activity(catalog, [identity], start, end, temp_root=tmp_path)


def test_reopening_advancing_windows_keeps_original_state_and_staleness(tmp_path):
    path, rows, start, end = fixture(tmp_path)
    with ProxyActivity([path], temp_root=tmp_path) as full:
        for days in [0, 3, 8]:
            low, high = start + timedelta(days=days), end + timedelta(days=days)
            with ProxyActivity(
                [path], temp_root=tmp_path, query_window=(low, high)
            ) as a:
                for user in {f.user for f in rows}:
                    assert a.position(user, "BTC", high) == full.position(
                        user, "BTC", high
                    )
                assert a.rank(
                    high, settings(lookback_days=1), "BTC", "gross_excludes_fee"
                ) == full.rank(
                    high, settings(lookback_days=1), "BTC", "gross_excludes_fee"
                )
                args = (rows[0].user, "BTC", low, high)
                assert a.hourly_exposure(
                    *args, max_price_age_seconds=3600
                ) == full.hourly_exposure(*args, max_price_age_seconds=3600)


@pytest.mark.parametrize("fault", ["missing", "symlink", "byte_limit"])
def test_adapter_rejects_unsafe_or_oversized_inputs(tmp_path, fault):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog
    from arblab.hyperliquid_copy.rolling_activity import open_rolling_activity
    from .test_compact_catalog import compact

    manifest, rows = compact(tmp_path)
    catalog = CompactCatalog(tmp_path / "catalog.sqlite3")
    identity = catalog.register(manifest)
    path = manifest.parent / json.loads(manifest.read_text())["files"][0]["name"]
    if fault in {"missing", "symlink"}:
        moved = path.with_suffix(".retained")
        path.rename(moved)
        if fault == "symlink":
            path.symlink_to(moved)
    start = rows[0].exchange_time
    limits = {"max_input_bytes": 1} if fault == "byte_limit" else {}
    with pytest.raises(ValueError, match="identity|byte"):
        open_rolling_activity(catalog, [identity], start,
            start + timedelta(days=3), temp_root=tmp_path, **limits)
    assert not list(tmp_path.glob("proxy_activity_*"))
