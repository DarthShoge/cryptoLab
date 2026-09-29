import json
from pathlib import Path

import pytest

from arblab.hyperliquid_copy import archive_job_qualification as qualification
from arblab.hyperliquid_copy import prefix_qualification as prefix
from arblab.hyperliquid_copy.archive_job import ArchiveJob
from arblab.hyperliquid_copy.archive_transition_snapshot import (
    prepare_transition,
    _state,
)
from .test_archive_job import inputs
from .test_archive_transition_snapshot import stopped


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    inventory, source, total, daily = inputs(tmp_path, days=3)
    # Simulate an old qualification generation without editing on-disk code or
    # rewriting job metadata. Actual time-identity behavior has separate tests.
    old = prefix._engine() | {"version": 1}
    with monkeypatch.context() as patch:
        patch.setattr(prefix, "_engine", lambda: old)
        patch.setattr(qualification, "_engine", lambda: old)
        job = ArchiveJob.create(
            inventory,
            tmp_path / "job",
            ["BTC"],
            max_download_bytes=total,
            max_batch_bytes=daily,
        )
        job.run_next(source=source)
        with monkeypatch.context() as stop:

            def fail(*args, **kwargs):
                raise RuntimeError("qualification stopped")

            stop.setattr(qualification, "complete", fail)
            with pytest.raises(RuntimeError, match="stopped"):
                job.run_next(source=source)
        pin = prepare_transition(job.store.root)
    return job, source, pin


def test_activation_requires_new_baseline_and_preserves_old_state(prepared):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, source, preparation = prepared
    before = _state(job.store)
    requests = (list(source.heads), list(source.gets))
    with pytest.raises(ValueError, match="engine"):
        ArchiveJob(job.store.root)
    activated = activate_transition(
        job.store.root, snapshot_sha256=preparation["sha256"]
    )
    assert _state(job.store) == before
    assert (source.heads, source.gets) == requests
    data = json.loads(Path(activated["path"]).read_text())
    baseline = prefix._previous(data["baseline"], prefix._engine())
    assert baseline["previous"] is None
    assert len(baseline["manifests"]) == 1
    with pytest.raises(ValueError, match="engine"):
        prefix._previous(
            qualification.pin(job.store.records()[0, "qualified"]), prefix._engine()
        )
    assert ArchiveJob(job.store.root).metadata == job.metadata
    assert (
        activate_transition(job.store.root, snapshot_sha256=preparation["sha256"])
        == activated
    )


def test_offline_recovery_and_later_download_keep_original_ledger(prepared):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, source, preparation = prepared
    before = _state(job.store)
    activated = activate_transition(
        job.store.root, snapshot_sha256=preparation["sha256"]
    )
    baseline = json.loads(Path(activated["path"]).read_text())["baseline"]
    requests = (list(source.heads), list(source.gets))
    result = ArchiveJob(job.store.root).run_next()
    assert result["completed_batches"] == 2
    assert result["reserved_bytes"] == before["budget"]["reserved_bytes"]
    assert (source.heads, source.gets) == requests
    report = prefix._previous(result["qualification_report"], prefix._engine())
    assert report["previous"] == baseline
    after = _state(job.store)
    assert after["metadata_sha256"] == before["metadata_sha256"]
    assert all(row in after["cleanup"] for row in before["cleanup"])
    assert all(row in after["records"] for row in before["records"])
    assert after["budget"] == before["budget"]
    final = ArchiveJob(job.store.root).run_next(source=source)
    assert final["completed_batches"] == 3
    assert len(source.gets) == len(requests[1]) + 24
    latest = _state(job.store)
    assert latest["budget"]["path"] == before["budget"]["path"]
    assert all(
        row in latest["budget"]["reservations"]
        for row in before["budget"]["reservations"]
    )
    assert ArchiveJob(job.store.root).run_next()["phase"] == "all_batches_qualified"


def test_changed_snapshot_or_missing_ledger_cannot_activate(prepared):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, _, preparation = prepared
    with pytest.raises(ValueError):
        activate_transition(job.store.root, snapshot_sha256="0" * 64)
    job.budget.path.unlink()
    with pytest.raises(ValueError, match="budget"):
        activate_transition(job.store.root, snapshot_sha256=preparation["sha256"])
    assert not job.budget.path.exists()
    assert not (Path(preparation["path"]).parent / "activation.json").exists()


def test_activation_honors_existing_job_lock(prepared):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, _, preparation = prepared
    with job.store.locked(), pytest.raises(ValueError, match="running"):
        activate_transition(job.store.root, snapshot_sha256=preparation["sha256"])


def test_baseline_mutation_before_activation_never_publishes(prepared):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, _, preparation = prepared

    def mutate(event):
        if event.get("transition") == "baseline_verified":
            Path(event["baseline"]["path"]).write_text("{}")

    with pytest.raises(ValueError):
        activate_transition(
            job.store.root, snapshot_sha256=preparation["sha256"], progress=mutate
        )
    assert not (Path(preparation["path"]).parent / "activation.json").exists()


def test_crashed_baseline_is_fully_requalified_before_activation(prepared, monkeypatch):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, _, preparation = prepared

    def stop(event):
        if event.get("transition") == "baseline_verified":
            raise RuntimeError("crash before activation")

    with pytest.raises(RuntimeError, match="crash"):
        activate_transition(
            job.store.root, snapshot_sha256=preparation["sha256"], progress=stop
        )
    calls = []
    original = prefix.qualify_prefix

    def traced(*args, **kwargs):
        calls.append(kwargs["previous"])
        return original(*args, **kwargs)

    monkeypatch.setattr(prefix, "qualify_prefix", traced)
    activate_transition(job.store.root, snapshot_sha256=preparation["sha256"])
    assert calls == [None]


@pytest.mark.parametrize(
    "target", ["activation", "baseline", "budget", "reservation", "engine"]
)
def test_activated_job_rejects_damaged_evidence_before_requests(
    prepared, monkeypatch, target
):
    import sqlite3
    from arblab.hyperliquid_copy import archive_job
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, source, preparation = prepared
    activated = activate_transition(
        job.store.root, snapshot_sha256=preparation["sha256"]
    )
    data = json.loads(Path(activated["path"]).read_text())
    requests = (list(source.heads), list(source.gets))
    if target == "activation":
        Path(activated["path"]).write_text("{}")
    elif target == "baseline":
        Path(data["baseline"]["path"]).write_text("{}")
    elif target == "budget":
        job.budget.path.unlink()
    elif target == "reservation":
        with sqlite3.connect(job.budget.path) as db:
            db.execute(
                "DELETE FROM reservations WHERE key=(SELECT min(key) FROM reservations)"
            )
    else:
        original = archive_job._engine
        monkeypatch.setattr(archive_job, "_engine", lambda: original() | {"schema": 99})
    with pytest.raises(ValueError):
        ArchiveJob(job.store.root).run_next(source=source)
    assert (source.heads, source.gets) == requests


def test_registration_and_status_expose_transition_provenance(prepared):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition
    from arblab.hyperliquid_copy.annual_registration import completed_job_inputs

    job, source, preparation = prepared
    activated = activate_transition(
        job.store.root, snapshot_sha256=preparation["sha256"]
    )
    ArchiveJob(job.store.root).run_next()
    result = ArchiveJob(job.store.root).run_next(source=source)
    evidence = completed_job_inputs(
        job.store.root, start="2026-08-02", end="2026-08-03"
    )
    assert evidence["archive_engine"] == result["archive_engine"]
    assert evidence["archive_engine"]["transition"] == activated
    assert (
        evidence["archive_engine"]["original"]
        != evidence["archive_engine"]["effective"]
    )


def test_cleanup_cannot_dispose_pending_payload_before_qualification(prepared):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition
    from arblab.hyperliquid_copy.archive_job_cleanup import dispose_prefix

    job, _, preparation = prepared
    activate_transition(job.store.root, snapshot_sha256=preparation["sha256"])
    raw = job.store.records()[1, "raw"]["path"]
    data = json.loads(raw.read_text())
    with job.store.locked(), pytest.raises((ValueError, KeyError)):
        dispose_prefix(job.store, 2)
    assert all((raw.parent / e["file"]).is_file() for e in data["objects"])


def test_only_legacy_qualification_generation_can_transition(stopped, monkeypatch):
    from arblab.hyperliquid_copy import archive_job
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, _ = stopped
    preparation = prepare_transition(job.store.root)
    original = archive_job._engine
    monkeypatch.setattr(archive_job, "_engine", lambda: original() | {"schema": 99})
    with pytest.raises(ValueError, match="successor|generation"):
        activate_transition(job.store.root, snapshot_sha256=preparation["sha256"])
    assert not (Path(preparation["path"]).parent / "baseline").exists()
    folder = Path(preparation["path"]).parent
    (folder / "activation.json").write_text(
        json.dumps(
            dict(
                schema="hyperliquid_engine_activation_v1",
                snapshot=preparation,
                baseline=dict(
                    path=str(folder / "baseline/fake/manifest.json"), sha256="0" * 64
                ),
                boundary=1,
                new_engine=archive_job._engine(),
            )
        )
    )
    with pytest.raises(ValueError, match="generation"):
        ArchiveJob(job.store.root)


@pytest.mark.parametrize("stage", ["intent_committed", "payload_unlinked"])
def test_transition_cleanup_crash_recovers_without_spend(prepared, stage):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, source, preparation = prepared
    activate_transition(job.store.root, snapshot_sha256=preparation["sha256"])
    spent = job.budget.reserved_bytes
    gets = list(source.gets)

    def stop(event):
        if event.get("cleanup") == stage:
            raise RuntimeError("interrupted cleanup")

    with pytest.raises(RuntimeError, match="cleanup"):
        ArchiveJob(job.store.root).run_next(progress=stop)
    # Recovery must finish local cleanup before a subsequent offline cache miss.
    with pytest.raises(ValueError, match="not cached"):
        ArchiveJob(job.store.root).run_next()
    assert job.budget.reserved_bytes == spent
    assert source.gets == gets
    assert all(row[-1] == "deleted" for row in _state(job.store)["cleanup"])


def test_transition_orphan_batch_report_is_revalidated(prepared, monkeypatch):
    from arblab.hyperliquid_copy.archive_engine_transition import activate_transition

    job, _, preparation = prepared
    activated = activate_transition(
        job.store.root, snapshot_sha256=preparation["sha256"]
    )
    baseline = json.loads(Path(activated["path"]).read_text())["baseline"]

    def stop(event):
        if event.get("published") == "qualified":
            raise RuntimeError("before report commit")

    with pytest.raises(RuntimeError, match="report commit"):
        ArchiveJob(job.store.root).run_next(progress=stop)
    calls = []
    original = qualification.qualify_prefix

    def traced(*args, **kwargs):
        calls.append(kwargs["previous"])
        return original(*args, **kwargs)

    monkeypatch.setattr(qualification, "qualify_prefix", traced)
    assert ArchiveJob(job.store.root).run_next()["completed_batches"] == 2
    assert calls == [baseline]


def test_interrupted_activation_requires_pending_file_fsync_on_retry(prepared, monkeypatch):
    from arblab.hyperliquid_copy import archive_engine_transition as transition

    job, _, preparation = prepared
    original = transition._sync
    calls = []

    def fail_pending(path):
        if path.name == "activation.pending":
            calls.append(path)
            raise OSError("pending fsync failed")
        return original(path)

    monkeypatch.setattr(transition, "_sync", fail_pending)
    for attempt in range(2):
        with pytest.raises(OSError, match="pending fsync"):
            transition.activate_transition(job.store.root, snapshot_sha256=preparation["sha256"])
        assert len(calls) == attempt + 1
        assert not (Path(preparation["path"]).parent / "activation.json").exists()
