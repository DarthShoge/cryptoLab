from dataclasses import replace
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from .test_proxy_activity import fills, partition


@pytest.mark.parametrize("same_orders", [False, True])
def test_time_disambiguates_market_and_wallet_identity(tmp_path, same_orders):
    first = replace(fills()[0], tid=840478295334001, px=103308.0, sz=0.14631)
    second = replace(
        first,
        event_id="second",
        exchange_time=first.exchange_time + timedelta(days=18),
        px=86751.0,
        sz=0.024,
        oid=first.oid if same_orders else first.oid + 1,
    )
    rows = [first, second]
    rows += [
        replace(
            f,
            user="0x" + "f" * 40,
            side="A",
            oid=f.oid + 100,
            event_id=f.event_id + "opposite",
        )
        for f in rows
    ]
    path = partition(tmp_path, rows)
    with ProxyActivity([path], temp_root=tmp_path) as activity:
        assert activity.count == 4
        assert activity.volume(
            "BTC", first.exchange_time, second.exchange_time + timedelta(milliseconds=1)
        ) == pytest.approx(103308 * 0.14631 + 86751 * 0.024)
        assert activity.volume(
            "BTC", first.exchange_time, second.exchange_time
        ) == pytest.approx(103308 * 0.14631)


@pytest.mark.parametrize(
    "case", ["null", "sub_ms", "missing_envelope", "mismatched_envelope"]
)
def test_invalid_trade_timestamp_is_rejected_before_deduplication(tmp_path, case):
    first = fills()[0]
    if case == "null":
        first = replace(first, exchange_time=None)
    elif case == "sub_ms":
        first = replace(
            first, exchange_time=first.exchange_time + timedelta(microseconds=1)
        )
    else:
        first = replace(
            first,
            source_key="node_fills_by_block/hourly/20251123/14.lz4",
            block_time=None
            if case == "missing_envelope"
            else first.exchange_time + timedelta(milliseconds=1),
        )
    path = partition(tmp_path, [first])
    with pytest.raises(ValueError, match="timestamp"):
        ProxyActivity([path], temp_root=tmp_path)


def test_legacy_and_submillisecond_envelope_duplicate_collapse(tmp_path):
    first = fills()[0]
    second = replace(
        first,
        source_key="node_fills_by_block/hourly/20251123/14.lz4",
        block_number=805984548,
        block_time=first.exchange_time + timedelta(microseconds=656),
        event_id="block-copy",
    )
    path = partition(tmp_path, [second, first])
    with ProxyActivity([path], temp_root=tmp_path) as activity:
        assert activity.count == 1
        assert (
            activity.db.execute("SELECT event_id FROM fills").fetchone()[0]
            == first.event_id
        )


@pytest.mark.parametrize(
    "field,value", [("px", 999), ("sz", 5), ("fee", 3), ("post_position", 99)]
)
def test_same_time_wallet_economic_conflict_still_fails(tmp_path, field, value):
    first = fills()[0]
    path = partition(tmp_path, [first, replace(first, **{field: value})])
    with pytest.raises(ValueError, match="conflicting"):
        ProxyActivity([path], temp_root=tmp_path)


def test_seed_and_current_fill_with_same_ids_remain_distinct(tmp_path):
    first = fills()[0]
    second = replace(
        first,
        exchange_time=first.exchange_time + timedelta(days=18),
        event_id="second",
        start_position=1,
        post_position=2,
        px=101,
    )
    seed = partition(tmp_path, [first], "seed")
    current = partition(tmp_path, [second], "current")
    cutoff = first.exchange_time + timedelta(days=1)
    end = second.exchange_time + timedelta(seconds=1)
    with ProxyActivity(
        [current],
        temp_root=tmp_path,
        seed_path=seed,
        seed_cutoff=cutoff,
        query_window=(cutoff, end),
    ) as activity:
        assert activity.count == 2
        assert activity.seed_count == 1
        assert activity.position(first.user, "BTC", end) == 2
        assert activity.volume("BTC", cutoff, end) == 101


def test_day_projection_preserves_cross_time_wallet_keys(tmp_path):
    from types import SimpleNamespace
    import pyarrow.parquet as pq
    from arblab.hyperliquid_copy.day_projection import ProjectionQuery

    first = fills()[0]
    start = first.exchange_time.replace(hour=0, minute=0, second=0, microsecond=0)
    first = replace(first, exchange_time=start + timedelta(hours=1))
    second = replace(first, exchange_time=start + timedelta(hours=2), event_id="second")
    path = partition(tmp_path, [first, second])
    entry = SimpleNamespace(path=path)
    day = SimpleNamespace(
        start=start, end=start + timedelta(days=1), entries=(entry,), witness=entry
    )
    with ProjectionQuery(day, tmp_path / "spill") as query:
        output = tmp_path / "projected.parquet"
        assert query.write(day.start, day.end, output) == 2
    assert pq.read_table(output)["exchange_time"].to_pylist() == [
        first.exchange_time,
        second.exchange_time,
    ]


def test_sharded_qualification_preserves_cross_time_identity(tmp_path):
    import pyarrow.parquet as pq
    from arblab.hyperliquid_copy.download import file_hash
    from arblab.hyperliquid_copy.sharded_validation import qualified_seed

    first = fills()[0]
    second = replace(
        first,
        exchange_time=first.exchange_time + timedelta(days=18),
        event_id="second",
        start_position=1,
        post_position=2,
        px=101,
    )
    path = partition(tmp_path, [first, second])
    entries = [dict(path=str(path), bytes=path.stat().st_size, sha256=file_hash(path))]
    end = second.exchange_time + timedelta(seconds=1)
    with qualified_seed(
        entries, ["BTC"], first.exchange_time, end, end, temp_root=tmp_path, buckets=2
    ) as seed:
        assert pq.read_table(seed)["exchange_time"].to_pylist() == [
            second.exchange_time
        ]
