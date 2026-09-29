from dataclasses import replace
from datetime import timedelta

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity, REVERSE_ORDER
from .test_proxy_activity import fills, partition


def entry(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_hash(path))


def bounds(rows):
    return min(r.exchange_time for r in rows) - timedelta(days=1), max(
        r.exchange_time for r in rows
    ) + timedelta(days=1)


@pytest.mark.parametrize("cutoff_kind", ["start", "middle", "end"])
def test_sharded_seed_matches_full_reader_with_future_and_dormant_positions(
    tmp_path, cutoff_kind
):
    from arblab.hyperliquid_copy.sharded_validation import qualified_seed

    rows = fills()
    rows.append(replace(rows[0], event_id="duplicate", source_key="zzz"))
    paths = [partition(tmp_path, rows[::2], "a"), partition(tmp_path, rows[1::2], "b")]
    start, end = bounds(rows)
    cutoff = {"start": start, "middle": rows[1].exchange_time, "end": end}[cutoff_kind]
    with ProxyActivity(paths, temp_root=tmp_path) as full:
        expected = full.db.execute(
            f"SELECT event_id FROM fills WHERE exchange_time < ? QUALIFY row_number() OVER (PARTITION BY user,coin ORDER BY {REVERSE_ORDER})=1 ORDER BY event_id",
            [cutoff],
        ).fetchall()
    with qualified_seed(
        [entry(p) for p in paths],
        {"BTC"},
        start,
        end,
        cutoff,
        temp_root=tmp_path,
        buckets=4,
    ) as seed:
        got = sorted((r["event_id"],) for r in pq.read_table(seed).to_pylist())
        assert got == expected
        assert all(r["exchange_time"] < cutoff for r in pq.read_table(seed).to_pylist())
    assert not seed.exists()
    assert all(p.exists() for p in paths)


@pytest.mark.parametrize("counterparty", [False, True])
def test_conflicts_across_source_files_never_yield_qualification(
    tmp_path, counterparty
):
    from arblab.hyperliquid_copy.sharded_validation import qualified_seed

    first = fills()[0]
    other = (
        replace(first, user="0x" + "f" * 40, oid=999, px=first.px + 1)
        if counterparty
        else replace(first, fee=first.fee + 1)
    )
    paths = [partition(tmp_path, [first], "old"), partition(tmp_path, [other], "later")]
    start, end = bounds([first])
    with pytest.raises(ValueError, match="conflicting|counterparties"):
        with qualified_seed(
            [entry(p) for p in paths],
            {"BTC"},
            start,
            end,
            start,
            temp_root=tmp_path,
            buckets=4,
        ):
            pytest.fail("Conflicting history was qualified")


def test_invalid_future_scope_is_checked_even_when_seed_is_empty(tmp_path):
    from arblab.hyperliquid_copy.sharded_validation import qualified_seed

    row = fills()[0]
    path = partition(tmp_path, [replace(row, post_position=999)])
    start, end = bounds([row])
    with pytest.raises(ValueError, match="economic"):
        with qualified_seed(
            [entry(path)], {"BTC"}, start, end, start, temp_root=tmp_path, buckets=2
        ):
            pytest.fail("Future malformed fill was ignored")


def test_changed_source_identity_rejected_before_validation(tmp_path):
    from arblab.hyperliquid_copy.sharded_validation import qualified_seed

    rows = fills()
    path = partition(tmp_path, rows)
    bad = entry(path) | {"sha256": "0" * 64}
    start, end = bounds(rows)
    with pytest.raises(ValueError, match="identity"):
        with qualified_seed([bad], {"BTC"}, start, end, start, temp_root=tmp_path):
            pytest.fail("Changed source was qualified")


@pytest.mark.parametrize(
    "limit", ["MAX_CORPUS_BYTES", "MAX_BUCKET_BYTES", "MAX_SEED_BYTES", "MAX_SEED_ROWS"]
)
def test_resource_caps_never_yield_partial_qualification(tmp_path, monkeypatch, limit):
    from arblab.hyperliquid_copy import sharded_validation

    rows = fills()
    path = partition(tmp_path, rows)
    start, end = bounds(rows)
    before = file_hash(path)
    monkeypatch.setattr(sharded_validation, limit, 0 if limit == "MAX_SEED_ROWS" else 1)
    with pytest.raises(ValueError, match="limit"):
        with sharded_validation.qualified_seed(
            [entry(path)], {"BTC"}, start, end, end, temp_root=tmp_path, buckets=2
        ):
            pytest.fail("Resource failure was qualified")
    assert file_hash(path) == before
    assert not list(tmp_path.glob("qualified_seed_*"))


def test_source_mutation_during_work_prevents_final_yield(tmp_path):
    from arblab.hyperliquid_copy.sharded_validation import qualified_seed

    rows = fills()
    path = partition(tmp_path, rows)
    start, end = bounds(rows)

    def mutate(item):
        if item["completed_buckets"] == item["buckets"]:
            path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="identity"):
        with qualified_seed(
            [entry(path)],
            {"BTC"},
            start,
            end,
            end,
            temp_root=tmp_path,
            buckets=2,
            progress=mutate,
        ):
            pytest.fail("Mutated source was qualified")


def test_global_union_numeric_types_and_signed_zero_match_full_reader(tmp_path):
    from arblab.hyperliquid_copy.sharded_validation import qualified_seed

    first = replace(fills()[0], px=100, fee=0.0)
    other = replace(first, event_id="other", source_key="zzz", px=100.0, fee=-0.0)
    paths = [partition(tmp_path, [first], "a"), partition(tmp_path, [other], "b")]
    start, end = bounds([first])
    with ProxyActivity(paths, temp_root=tmp_path) as full:
        assert full.count == 1
        expected = full.db.execute("SELECT event_id FROM fills").fetchone()[0]
    with qualified_seed(
        [entry(p) for p in paths],
        {"BTC"},
        start,
        end,
        end,
        temp_root=tmp_path,
        buckets=2,
    ) as seed:
        assert [r["event_id"] for r in pq.read_table(seed).to_pylist()] == [expected]


def test_native_start_contradictions_are_not_ignored(tmp_path):
    from arblab.hyperliquid_copy.sharded_validation import qualified_seed

    rows = fills()
    path = partition(tmp_path, rows)
    start, end = bounds(rows)
    with pytest.raises(ValueError, match="native"):
        with qualified_seed(
            [entry(path)],
            {"BTC"},
            start,
            end,
            start,
            temp_root=tmp_path,
            buckets=2,
            native_starts={"BTC": end},
        ):
            pytest.fail("Contradictory history was qualified")
