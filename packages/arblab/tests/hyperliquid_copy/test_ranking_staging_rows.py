import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.lab_config import METRICS
from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner
from .test_candidate_day import resources
from .test_disk_metric_rows import row


def test_writer_rejects_valid_same_inode_change_during_final_verification(
    resources, monkeypatch
):
    import pyarrow as pa
    from arblab.hyperliquid_copy.ranking_staging_rows import write_pending_metrics
    from arblab.hyperliquid_copy.disk_metric_rows import METRIC_ROWS_SCHEMA

    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    original, calls = RankingStagingOwner.verify, 0

    def mutate(self):
        nonlocal calls
        original(self)
        calls += 1
        if calls == 2:
            path = self.path("metrics")
            inode = path.stat().st_ino
            pq.write_table(
                pa.Table.from_pylist([row(value=99.0)], schema=METRIC_ROWS_SCHEMA), path
            )
            assert path.stat().st_ino == inode

    monkeypatch.setattr(RankingStagingOwner, "verify", mutate)
    try:
        with pytest.raises(ValueError):
            write_pending_metrics(owner, [row()], METRICS)
    finally:
        owner.close()


@pytest.mark.parametrize("count", [0, 1, 4097])
def test_pending_writer_keeps_full_rows_without_publication(resources, count):
    from arblab.hyperliquid_copy.ranking_staging_rows import write_pending_metrics
    from arblab.hyperliquid_copy.disk_metric_rows import METRIC_ROWS_SCHEMA

    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    before = resources.audit()
    try:
        result = write_pending_metrics(owner, (row(i) for i in range(count)), METRICS)
        assert result.path == owner.path("metrics")
        result.verify()
        assert pq.read_schema(result.path) == METRIC_ROWS_SCHEMA
        assert pq.read_table(result.path).to_pylist() == [row(i) for i in range(count)]
        metadata = pq.read_metadata(result.path)
        assert all(
            metadata.row_group(i).num_rows <= 4096
            for i in range(metadata.num_row_groups)
        )
        owner.verify()
        assert resources.audit() == before
        with resources._connect() as db:
            assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0
            assert db.execute("SELECT DISTINCT state FROM allocations").fetchall() == [
                ("pending",)
            ]
    finally:
        owner.close()


@pytest.mark.parametrize(
    "fault",
    ["nonfinite", "missing", "row_limit", "byte_limit", "interrupted", "context"],
)
def test_failed_pending_write_preserves_obligations(resources, fault):
    from arblab.hyperliquid_copy.ranking_staging_rows import write_pending_metrics

    context = {"query": "fixture"}
    owner = RankingStagingOwner.create(resources, context)
    before = resources.audit()

    def records():
        yield row()
        if fault == "interrupted":
            raise RuntimeError("iterator interrupted")
        if fault == "context":
            context["query"] = "changed"
        yield row(
            1,
            value=float("nan")
            if fault == "nonfinite"
            else None
            if fault == "missing"
            else 1.0,
        )

    options = (
        {"max_rows": 1}
        if fault == "row_limit"
        else {"max_bytes": 16}
        if fault == "byte_limit"
        else {}
    )
    try:
        with pytest.raises((ValueError, RuntimeError)):
            write_pending_metrics(owner, records(), METRICS, **options)
        assert resources.audit() == before
    finally:
        owner.close()


def test_writer_captures_inode_before_iterator_callback(resources):
    from arblab.hyperliquid_copy.ranking_staging_rows import write_pending_metrics

    owner = RankingStagingOwner.create(resources, {"query": "fixture"})

    def records():
        path = owner.path("metrics")
        path.unlink()
        with path.open("xb") as stream:
            stream.write(b"replacement")
        yield row()

    try:
        with pytest.raises(ValueError, match="identity"):
            write_pending_metrics(owner, records(), METRICS)
        assert owner.path("metrics").read_bytes() == b"replacement"
    finally:
        owner.close()
