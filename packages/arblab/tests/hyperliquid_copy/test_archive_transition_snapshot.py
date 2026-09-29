import json
import sqlite3
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.archive_job import ArchiveJob
from arblab.hyperliquid_copy.download import file_hash
from .test_archive_job import inputs


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import archive_job_qualification as qualification

    inventory, source, total, daily = inputs(tmp_path)
    job = ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        max_download_bytes=total,
        max_batch_bytes=daily,
    )
    job.run_next(source=source)

    def fail(*args, **kwargs):
        raise RuntimeError("stopped before qualification")

    with monkeypatch.context() as patch:
        patch.setattr(qualification, "complete", fail)
        with pytest.raises(RuntimeError, match="stopped"):
            job.run_next(source=source)
    return job, source


def test_preparation_preserves_state_and_archives_engine_without_network(stopped):
    from arblab.hyperliquid_copy.archive_transition_snapshot import (
        prepare_transition,
        read_snapshot,
    )

    job, source = stopped
    original = {p.name: file_hash(p) for p in (job.store.path, job.budget.path)}
    requests = (list(source.heads), list(source.gets))
    result = prepare_transition(job.store.root)
    snapshot = read_snapshot(job.store.root, result["sha256"])
    assert snapshot["boundary"] == 1
    assert snapshot["old_engine"] == job.metadata["engine"]
    assert snapshot["budget"]["reserved_bytes"] == job.budget.reserved_bytes
    assert len(snapshot["records"]) == 7
    assert snapshot["sources"]
    assert snapshot["cleanup"]
    assert {p.name: file_hash(p) for p in (job.store.path, job.budget.path)} == original
    assert (source.heads, source.gets) == requests
    assert not (Path(result["path"]).parent / "activation.json").exists()
    assert prepare_transition(job.store.root) == result


@pytest.mark.parametrize(
    "target", ["raw", "compact", "source", "manifest", "budget", "cleanup"]
)
def test_snapshot_rejects_changed_evidence(stopped, target):
    from arblab.hyperliquid_copy.archive_transition_snapshot import (
        prepare_transition,
        read_snapshot,
    )

    job, _ = stopped
    result = prepare_transition(job.store.root)
    data = json.loads(Path(result["path"]).read_text())
    if target in ("raw", "compact"):
        path = job.store.records()[1, target]["path"]
        artifact = json.loads(path.read_text())
        entry = artifact["objects" if target == "raw" else "files"][0]
        (path.parent / entry["file" if target == "raw" else "name"]).write_bytes(b"bad")
    elif target == "source":
        (
            Path(result["path"]).parent / "source" / data["sources"][0]["name"]
        ).write_bytes(b"bad")
    elif target == "manifest":
        Path(result["path"]).write_text("{}")
    elif target == "budget":
        with sqlite3.connect(job.budget.path) as db:
            db.execute(
                "DELETE FROM reservations WHERE key=(SELECT min(key) FROM reservations)"
            )
    else:
        with sqlite3.connect(job.store.path) as db:
            db.execute("UPDATE cleanup SET qualification_sha256=?", ("0" * 64,))
    with pytest.raises(ValueError):
        read_snapshot(job.store.root, result["sha256"])


def test_preparation_rejects_missing_ledger_without_recreating_it(stopped):
    from arblab.hyperliquid_copy.archive_transition_snapshot import prepare_transition

    job, _ = stopped
    job.budget.path.unlink()
    with pytest.raises(ValueError, match="budget"):
        prepare_transition(job.store.root)
    assert not job.budget.path.exists()


def test_preparation_rejects_changed_engine_and_lock(stopped, monkeypatch):
    from arblab.hyperliquid_copy import archive_job
    from arblab.hyperliquid_copy.archive_transition_snapshot import prepare_transition

    job, _ = stopped
    with job.store.locked(), pytest.raises(ValueError, match="running"):
        prepare_transition(job.store.root)
    monkeypatch.setattr(archive_job, "_engine", lambda: {"changed": True})
    with pytest.raises(ValueError, match="engine"):
        prepare_transition(job.store.root)


def test_preparation_rejects_corrupt_pending_raw(stopped):
    from arblab.hyperliquid_copy.archive_transition_snapshot import prepare_transition

    job, _ = stopped
    raw = job.store.records()[1, "raw"]["path"]
    data = json.loads(raw.read_text())
    (raw.parent / data["objects"][0]["file"]).write_bytes(b"bad")
    with pytest.raises(ValueError):
        prepare_transition(job.store.root)
    assert not (job.store.root / "engine_transition_v2").exists()


def test_preparation_requires_qualified_prefix_and_pending_compact(tmp_path):
    from arblab.hyperliquid_copy.archive_transition_snapshot import prepare_transition

    inventory, _, total, daily = inputs(tmp_path)
    job = ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        max_download_bytes=total,
        max_batch_bytes=daily,
    )
    with pytest.raises(ValueError, match="sequence"):
        prepare_transition(job.store.root)


@pytest.mark.parametrize("change", ["boundary", "retained"])
def test_existing_untrusted_preparation_cannot_omit_required_evidence(stopped, change):
    from arblab.hyperliquid_copy.archive_transition_snapshot import prepare_transition

    job, _ = stopped
    result = prepare_transition(job.store.root)
    path = Path(result["path"])
    data = json.loads(path.read_text())
    if change == "boundary":
        data[change] = 99
    else:
        data[change].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        prepare_transition(job.store.root)


def test_preparation_rejects_unrecorded_later_stage(stopped):
    from arblab.hyperliquid_copy.archive_transition_snapshot import prepare_transition

    job, _ = stopped
    (job.store.root / "batches/0002/raw/orphan").mkdir(parents=True)
    with pytest.raises(ValueError, match="stage|sequence"):
        prepare_transition(job.store.root)


def test_snapshot_rejects_symlink_source(stopped, tmp_path):
    from arblab.hyperliquid_copy.archive_transition_snapshot import (
        prepare_transition,
        read_snapshot,
    )

    job, _ = stopped
    result = prepare_transition(job.store.root)
    data = json.loads(Path(result["path"]).read_text())
    path = Path(result["path"]).parent / "source" / data["sources"][0]["name"]
    moved = tmp_path / "source.py"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError):
        read_snapshot(job.store.root, result["sha256"])
