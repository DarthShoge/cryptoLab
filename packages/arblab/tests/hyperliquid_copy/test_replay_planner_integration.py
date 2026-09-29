from contextlib import contextmanager
from dataclasses import astuple
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.wallet_partition_plan import counted_plan
from arblab.hyperliquid_copy.wallet_replay_groups import coalesce_plan
from .test_ordered_wallet_partitions import window
from .test_qualified_day import qualified


def test_reader_aggregates_source_once_and_coalesces(qualified, tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    original = reader._query
    with original() as db:
        raw = counted_plan(db, max_rows=100)
    expected = coalesce_plan(raw, max_rows=100)
    assert len(expected) < len(raw)
    queries = []

    class Trace:
        def __init__(self, db):
            self.db = db

        def execute(self, sql, *args):
            queries.append(sql)
            return self.db.execute(sql, *args)

    @contextmanager
    def traced():
        with original() as db:
            yield Trace(db)

    monkeypatch.setattr(reader, "_query", traced)
    actual = reader.plan(max_rows=100)
    assert tuple(astuple(part) for part in actual) == expected
    assert sum("FROM source" in sql for sql in queries) == 1
    assert any("CREATE TEMP TABLE wallet_partition_counts" in sql for sql in queries)
    before = list(queries)
    assert reader.plan(max_rows=100) is actual and queries == before
    with pytest.raises(ValueError):
        reader.plan(max_rows=99)


@pytest.mark.parametrize(
    "helper", ["wallet_partition_plan.py", "wallet_replay_groups.py"]
)
def test_helper_hash_binds_reader_and_producer(
    qualified, tmp_path, monkeypatch, helper
):
    from arblab.hyperliquid_copy import ordered_wallet_partitions as ordered
    from arblab.hyperliquid_copy import candidate_metric_producer as producer

    scratch = tmp_path / "ordered"
    scratch.mkdir()
    reader = ordered.OrderedWalletPartitions(window(qualified), ["BTC"], scratch)
    reader.plan()
    engine = producer._engine()
    assert helper in engine["code"]
    original = ordered.file_hash

    def changed(path):
        return "f" * 64 if Path(path).name == helper else original(path)

    monkeypatch.setattr(ordered, "file_hash", changed)
    monkeypatch.setattr(producer, "file_hash", changed)
    with pytest.raises(ValueError, match="engine|context"):
        reader.plan()
    assert producer._engine() != engine
