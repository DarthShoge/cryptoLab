import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from .test_qualified_day import qualified


@pytest.fixture
def resources(tmp_path):
    root = tmp_path / "candidate_cache"
    root.mkdir()
    with CacheLease(root) as lease:
        yield CacheResources.create(lease, "candidate-day-test")


def test_first_observation_matches_all_source_rows(qualified, resources):
    from arblab.hyperliquid_copy.candidate_day import build_candidate_day

    result = build_candidate_day(resources, qualified, "2026-08-01")
    report = json.loads(Path(qualified["path"]).read_text())
    expected = {}
    for entry in report["files"]:
        for row in pq.read_table(entry["path"]).to_pylist():
            at = row["exchange_time"]
            if at.date().isoformat() == "2026-08-01":
                key = row["user"], row["coin"]
                expected[key] = min(expected.get(key, at), at)
    actual = pq.read_table(resources.root / result.artifacts[0].path).to_pylist()
    assert actual == [
        dict(user=u, coin=c, first_observed=t) for (u, c), t in sorted(expected.items())
    ]
    assert len(actual) == 2
    assert resources.audit()["reserved_bytes"] == 0
    assert list((resources.root / "scratch").iterdir()) == []


def test_candidate_index_does_not_retain_event_level_copies(qualified, resources):
    from arblab.hyperliquid_copy.candidate_day import build_candidate_day

    result = build_candidate_day(resources, qualified, "2026-08-01")
    with resources._connect() as db:
        descriptors = [
            json.loads(r[0]) for r in db.execute("SELECT descriptor FROM publications")
        ]
    assert [d["kind"] for d in descriptors] == ["candidate_day"]
    assert (
        len(list((resources.root / "artifacts").iterdir()))
        == len(result.artifacts)
        == 1
    )


def test_empty_day_has_typed_publication_and_reuses(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import candidate_day as module

    first = module.build_candidate_day(resources, qualified, "2026-08-03")
    table = pq.read_table(resources.root / first.artifacts[0].path)
    assert table.num_rows == 0
    assert table.schema == module.SCHEMA
    before = resources.audit()

    def unexpected(*args, **kwargs):
        raise AssertionError("candidate cache hit reran query")

    monkeypatch.setattr(module, "_write", unexpected)
    assert module.build_candidate_day(resources, qualified, "2026-08-03") == first
    assert resources.audit() == before


def test_footer_overflow_retains_charge_without_candidate_publication(
    qualified, resources
):
    from arblab.hyperliquid_copy.candidate_day import build_candidate_day

    with pytest.raises(ValueError, match="byte limit"):
        build_candidate_day(resources, qualified, "2026-08-01", max_bytes=10)
    assert resources.audit()["reserved_bytes"] == 10
    with resources._connect() as db:
        descriptors = [
            json.loads(r[0]) for r in db.execute("SELECT descriptor FROM publications")
        ]
    assert all(d["kind"] != "candidate_day" for d in descriptors)


@pytest.mark.parametrize("target", ["source", "result"])
def test_cache_hit_rejects_changed_payload(qualified, resources, target):
    from arblab.hyperliquid_copy.candidate_day import build_candidate_day

    result = build_candidate_day(resources, qualified, "2026-08-01")
    path = resources.root / result.artifacts[0].path
    if target == "source":
        path = Path(json.loads(Path(qualified["path"]).read_text())["files"][0]["path"])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        build_candidate_day(resources, qualified, "2026-08-01")


@pytest.mark.parametrize("fault", ["source", "engine", "interrupt"])
def test_changed_inputs_during_write_cannot_publish(
    qualified, resources, monkeypatch, fault
):
    from arblab.hyperliquid_copy import candidate_day as module

    original = module._write
    engine = module._engine()

    def changed(paths, *args):
        original(paths, *args)
        if fault == "engine":
            monkeypatch.setattr(module, "_engine", lambda: dict(engine, schema=999))
        elif fault == "interrupt":
            raise RuntimeError("simulated writer interruption")
        else:
            path = Path(
                json.loads(Path(qualified["path"]).read_text())["files"][0]["path"]
            )
            with path.open("ab") as stream:
                stream.write(b"changed")

    monkeypatch.setattr(module, "_write", changed)
    with pytest.raises((ValueError, RuntimeError)):
        module.build_candidate_day(resources, qualified, "2026-08-01")
    with resources._connect() as db:
        descriptors = [
            json.loads(r[0]) for r in db.execute("SELECT descriptor FROM publications")
        ]
    assert all(d["kind"] != "candidate_day" for d in descriptors)
    if fault == "interrupt":
        assert resources.audit()["reserved_bytes"] == module.MAX_BYTES
    assert list((resources.root / "scratch").iterdir()) == []


def test_row_limit_fails_without_truncation(qualified, resources):
    from arblab.hyperliquid_copy.candidate_day import build_candidate_day

    with pytest.raises(ValueError, match="row/metadata limit"):
        build_candidate_day(resources, qualified, "2026-08-01", max_rows=1)


def test_shared_budget_is_reserved_before_query(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import candidate_day as module
    from arblab.hyperliquid_copy.derived_day_builder import build_projection_day

    build_projection_day(resources, qualified, "2026-08-01")
    resources.reserve("staging/" + "a" * 32, 6 * 1024**3, "payload")
    before = resources.audit()

    def unexpected(*args, **kwargs):
        raise AssertionError("unbudgeted candidate query")

    monkeypatch.setattr(module, "_write", unexpected)
    with pytest.raises(ValueError, match="budget|limit"):
        module.build_candidate_day(resources, qualified, "2026-08-01")
    assert resources.audit() == before


def test_closed_lease_rejected(qualified, resources):
    from arblab.hyperliquid_copy.candidate_day import build_candidate_day

    resources.lease.__exit__(None, None, None)
    with pytest.raises(ValueError):
        build_candidate_day(resources, qualified, "2026-08-01")


def test_cross_class_and_late_first_observations(tmp_path, resources):
    import lz4.frame
    from arblab.hyperliquid_copy.candidate_day import build_candidate_day
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    _, source, _, _ = inputs(tmp_path, days=1)
    late_users = ["0x" + "ef" * 20, "0x" + "fe" * 20]
    for key, body in list(source.bodies.items()):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        data = json.loads(lz4.frame.decompress(body))
        for index, event in enumerate(data["events"][:2]):
            event[1]["coin"] = "xyz:GOLD" if hour >= 12 else "BTC"
            if hour == 23:
                event[0] = late_users[index]
        source.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    raw = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "downloaded")
    normalized = import_archive(
        raw, ["BTC", "xyz:GOLD"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    report = qualify([compact], tmp_path)
    result = build_candidate_day(resources, report, "2026-08-01")
    rows = pq.read_table(resources.root / result.artifacts[0].path).to_pylist()
    assert len(rows) == 6
    for row in rows:
        expected_hour = (
            23 if row["user"] in late_users else 12 if row["coin"] == "xyz:GOLD" else 0
        )
        assert row["first_observed"].hour == expected_hour
