from datetime import timedelta

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from .test_qualified_day import qualified


def test_projection_deduplicates_all_spill_and_preserves_values(qualified, tmp_path):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay
    from arblab.hyperliquid_copy.day_projection import ProjectionQuery, COLUMNS

    day = QualifiedDay(qualified, "2026-08-01")
    with ProxyActivity([e.path for e in day.entries], temp_root=tmp_path) as reference:
        expected = reference.db.execute(
            f"SELECT {','.join(COLUMNS)} FROM fills ORDER BY user,exchange_time"
        ).fetchall()
    with ProjectionQuery(day, tmp_path / "spill") as query:
        intervals = query.intervals(250000)
        assert intervals == ((day.start, day.end),)
        path = tmp_path / "projection.parquet"
        assert query.write(*intervals[0], path, 1024**2) == 48
    table = pq.read_table(path)
    actual = [tuple(row[col] for col in COLUMNS) for row in table.to_pylist()]
    assert actual == expected
    assert table.column("native_order").to_pylist() == list(range(1, 49))


def test_split_intervals_preserve_boundary_and_global_wallet_order(qualified, tmp_path):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay
    from arblab.hyperliquid_copy.day_projection import ProjectionQuery

    day = QualifiedDay(qualified, "2026-08-02")
    rows = []
    with ProjectionQuery(day, tmp_path / "spill") as query:
        intervals = query.intervals(2)
        assert len(intervals) > 1
        assert intervals[0][0] == day.start and intervals[-1][1] == day.end
        assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
        for i, (start, end) in enumerate(intervals):
            path = tmp_path / f"part-{i}.parquet"
            query.write(start, end, path, 1024**2)
            batch = pq.read_table(path).to_pylist()
            assert len(batch) <= 2
            assert all(start <= row["exchange_time"] < end for row in batch)
            rows.extend(batch)
    assert len(rows) == 48
    ordered = sorted(
        rows, key=lambda r: (r["user"], r["exchange_time"], r["native_order"])
    )
    assert (
        len({(r["user"], r["coin"], r["tid"], r["oid"], r["side"]) for r in ordered})
        == 48
    )


def test_indivisible_timestamp_and_byte_limit_fail_closed(qualified, tmp_path):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay
    from arblab.hyperliquid_copy.day_projection import ProjectionQuery

    day = QualifiedDay(qualified, "2026-08-02")
    with ProjectionQuery(day, tmp_path / "spill") as query:
        with pytest.raises(ValueError, match="indivisible"):
            query.intervals(1)
        with pytest.raises(ValueError, match="byte limit"):
            query.write(day.start, day.end, tmp_path / "small.parquet", 10)


def test_empty_day_has_typed_empty_output(qualified, tmp_path):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay
    from arblab.hyperliquid_copy.day_projection import ProjectionQuery, COLUMNS

    day = QualifiedDay(qualified, "2026-08-03")
    with ProjectionQuery(day, tmp_path / "spill") as query:
        assert query.intervals(2) == ((day.start, day.end),)
        path = tmp_path / "empty.parquet"
        assert query.write(day.start, day.end, path, 1024**2) == 0
    assert pq.read_schema(path).names == [*COLUMNS, "native_order"]
    assert pq.read_metadata(path).num_rows == 0


def test_equal_time_multi_market_order_and_integer_values(tmp_path):
    from dataclasses import asdict, replace
    from types import SimpleNamespace
    import pyarrow as pa
    from arblab.hyperliquid_copy.day_projection import ProjectionQuery, COLUMNS
    from arblab.hyperliquid_copy.proxy_activity import ORDER
    from .test_streaming_wallet_metrics import mixed_history

    original = mixed_history()
    start = original[0].exchange_time.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = [
        replace(
            row,
            exchange_time=start + timedelta(seconds=i // 3),
            source_key=f"source-{2 - i % 3}",
            px=2**53 + 1,
        )
        for i, row in enumerate(original)
    ]
    rows.append(replace(rows[0], source_key="zz-duplicate", event_id="duplicate"))
    path = tmp_path / "canonical.parquet"
    pq.write_table(pa.Table.from_pylist([asdict(row) for row in rows]), path)
    # Query-unit fixture; qualification/source-pin admission is tested separately.
    entry = SimpleNamespace(path=path)
    day = SimpleNamespace(
        start=start, end=start + timedelta(days=1), entries=(entry,), witness=entry
    )
    with ProxyActivity([path], temp_root=tmp_path) as reference:
        expected = reference.db.execute(
            f"SELECT {','.join(COLUMNS)} FROM fills ORDER BY user,{ORDER}"
        ).fetchall()
    with ProjectionQuery(day, tmp_path / "spill") as query:
        output = tmp_path / "projected.parquet"
        query.write(day.start, day.end, output, 1024**2)
    table = pq.read_table(output)
    assert table.schema.field("px").type == pa.int64()
    assert [tuple(row[col] for col in COLUMNS) for row in table.to_pylist()] == expected
    assert len(expected) == 9
