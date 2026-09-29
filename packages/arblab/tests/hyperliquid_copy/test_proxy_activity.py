from dataclasses import asdict, replace
from datetime import timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from .test_lab_ranking import settings
from .test_ranking import history


def fills():
    return [replace(f, tid=i) for i, f in enumerate(history())]


def partition(tmp_path, rows, name="fills"):
    path = tmp_path / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist([asdict(f) for f in rows]), path)
    return path


def test_positions_and_observation_are_strictly_past(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    rows = fills()
    opening, closing = rows[:2]
    path = partition(tmp_path, rows)
    with ProxyActivity([path], temp_root=tmp_path) as activity:
        assert activity.position(opening.user, "BTC", opening.exchange_time) is None
        assert activity.observed(opening.exchange_time) == []
        assert activity.position(opening.user, "BTC", closing.exchange_time) == 1
        after = closing.exchange_time + timedelta(seconds=1)
        assert activity.position(opening.user, "BTC", after) == 0
        assert activity.position(opening.user, "ETH", after) is None
        assert activity.observed(after) == ["BTC"]


@pytest.mark.parametrize("days_after", [0, 2])
def test_disk_ranking_matches_existing_including_dormant_wallets(tmp_path, days_after):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from arblab.hyperliquid_copy.lab_ranking import rank_universe

    rows = fills()
    at = max(f.exchange_time for f in rows) + timedelta(days=days_after, seconds=1)
    config = settings(lookback_days=1)
    expected = rank_universe(rows, at, config, "BTC", "gross_excludes_fee")
    future = [
        replace(
            f,
            tid=f.tid + 100,
            event_id=f.event_id + "future",
            exchange_time=at + timedelta(days=1),
        )
        for f in rows
    ]
    path = partition(tmp_path, list(reversed(rows + future)))
    with ProxyActivity([path], temp_root=tmp_path) as activity:
        assert activity.rank(at, config, "BTC", "gross_excludes_fee") == expected


def test_market_volume_counts_each_trade_once_not_both_wallets(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    opening = fills()[0]
    counterparty = replace(
        opening,
        user="0x" + "f" * 40,
        event_id="counterparty",
        side="A",
        post_position=-1,
        oid=999,
    )
    duplicate = replace(opening, event_id="other-partition", source_key="later")
    path = partition(tmp_path, [opening, counterparty, duplicate])
    with ProxyActivity([path], temp_root=tmp_path) as activity:
        at = opening.exchange_time
        assert activity.volume("BTC", at, at + timedelta(hours=1)) == opening.px
        assert activity.volume("BTC", at - timedelta(hours=1), at) == 0
        assert activity.count == 2


def test_conflicting_economic_duplicates_are_rejected(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    opening = fills()[0]
    path = partition(tmp_path, [opening, replace(opening, px=999)])
    with pytest.raises(ValueError, match="conflicting"):
        with ProxyActivity([path], temp_root=tmp_path):
            pass


def test_inconsistent_counterparty_trade_volume_is_rejected(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    opening = fills()[0]
    opposite = replace(
        opening,
        user="0x" + "f" * 40,
        event_id="opposite",
        oid=999,
        side="A",
        px=opening.px + 1,
    )
    path = partition(tmp_path, [opening, opposite])
    with pytest.raises(ValueError, match="market trade"):
        with ProxyActivity([path], temp_root=tmp_path):
            pass


def test_old_optional_schema_and_new_partitions_can_be_ranked_together(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from arblab.hyperliquid_copy.lab_ranking import rank_universe

    rows = fills()
    old = tmp_path / "old.parquet"
    old_rows = [
        {k: v for k, v in asdict(f).items() if k != "raw_details_json"}
        for f in rows[:6]
    ]
    pq.write_table(pa.Table.from_pylist(old_rows), old)
    new = partition(tmp_path, rows[6:])
    at = max(f.exchange_time for f in rows) + timedelta(seconds=1)
    with ProxyActivity([old, new], temp_root=tmp_path) as activity:
        assert activity.rank(
            at, settings(), "BTC", "gross_excludes_fee"
        ) == rank_universe(rows, at, settings(), "BTC", "gross_excludes_fee")


def test_same_timestamp_position_uses_complete_archive_order(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    first = fills()[0]
    last = replace(
        first,
        tid=999,
        event_id="last",
        source_line=first.source_line + 1,
        start_position=1,
        post_position=2,
    )
    path = partition(tmp_path, [last, first])
    with ProxyActivity([path], temp_root=tmp_path) as activity:
        assert activity.position(first.user, "BTC", first.exchange_time) is None
        assert (
            activity.position(
                first.user, "BTC", first.exchange_time + timedelta(microseconds=1)
            )
            == 2
        )


def test_bounds_reject_without_sampling(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    rows = fills()
    path = partition(tmp_path, rows)
    with pytest.raises(ValueError, match="input byte"):
        with ProxyActivity([path], temp_root=tmp_path, max_input_bytes=1):
            pass
    at = max(f.exchange_time for f in rows) + timedelta(seconds=1)
    with ProxyActivity([path], temp_root=tmp_path, max_wallet_fills=1) as activity:
        with pytest.raises(ValueError, match="wallet history"):
            activity.rank(at, settings(), "BTC", "gross_excludes_fee")
    with ProxyActivity([path], temp_root=tmp_path, max_candidates=1) as activity:
        with pytest.raises(ValueError, match="candidate"):
            activity.rank(at, settings(), "BTC", "gross_excludes_fee")


def test_hourly_native_exposure_is_causal_and_rejects_stale_prices(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    first = fills()[0]
    start = first.exchange_time
    second = replace(
        first,
        event_id="second",
        tid=1000,
        exchange_time=start + timedelta(minutes=30),
        start_position=1,
        post_position=2,
        px=110,
    )
    path = partition(tmp_path, [first, second])
    with ProxyActivity([path], temp_root=tmp_path) as activity:
        rows = activity.hourly_exposure(
            first.user,
            "BTC",
            start,
            start + timedelta(hours=3),
            max_price_age_seconds=3600,
        )
        assert [value for _, value in rows] == [None, 220, None]
        with pytest.raises(ValueError, match="hourly exposure"):
            activity.hourly_exposure(
                first.user,
                "BTC",
                start,
                start + timedelta(days=5000),
                max_price_age_seconds=3600,
            )
