import os
import sqlite3
from contextlib import contextmanager

import pytest

from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from arblab.hyperliquid_copy.lab_config import METRICS, day
from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner
from arblab.hyperliquid_copy.ranking_staging_rows import write_pending_metrics
from arblab.hyperliquid_copy.ranking_staging_score import score_pending_metrics
from .test_candidate_day import resources
from .test_disk_metric_rows import row
from .test_lab_ranking import settings


def prepared(resources):
    query = {"fixture_query": "full-result"}
    owner = RankingStagingOwner.create(resources, {"query": query})
    metrics = write_pending_metrics(owner, [row()], METRICS)
    result = score_pending_metrics(
        owner,
        metrics,
        settings(),
        day("2026-08-03"),
        "BTC",
        "gross_excludes_fee",
        verify_source=lambda: {"source": "fixture"},
    )
    resources.settle(owner._allocations["ranking"]["token"])
    inputs = {"query": query, "ranking": "a" * 64}
    receipt = PublishedArtifacts(resources).publish(
        "saved_feature_ranking", inputs, [owner._allocations["ranking"]["token"]]
    )

    def verify():
        result.ranking.verify()
        return PublishedArtifacts(resources).lookup("saved_feature_ranking", inputs)

    return owner, result, receipt, verify


def test_successful_cleanup_preserves_full_saved_ranking(resources):
    from arblab.hyperliquid_copy.ranking_staging_cleanup import finish_staging

    owner, result, receipt, verify = prepared(resources)
    paths = {role: owner.path(role) for role in owner._allocations}
    finish_staging(owner, result, receipt, verify_receipt=verify)
    assert verify() == receipt
    assert paths["ranking"].exists()
    assert all(not path.exists() for role, path in paths.items() if role != "ranking")
    audit = resources.audit()
    assert audit["reserved_bytes"] == 0
    assert audit["retained_bytes"] == result.ranking.bytes


def test_failed_receipt_verification_prevents_any_cleanup(resources):
    from arblab.hyperliquid_copy.ranking_staging_cleanup import finish_staging

    owner, result, receipt, _ = prepared(resources)
    paths = [owner.path(role) for role in owner._allocations]
    before = resources.audit()

    def invalid():
        raise ValueError("source changed")

    with pytest.raises(ValueError, match="source changed"):
        finish_staging(owner, result, receipt, verify_receipt=invalid)
    assert all(path.exists() for path in paths)
    assert resources.audit() == before


def test_interrupted_unlink_keeps_missing_file_charged(resources, monkeypatch):
    from arblab.hyperliquid_copy import ranking_staging_cleanup as cleanup

    owner, result, receipt, verify = prepared(resources)
    before = resources.audit()
    original = cleanup._unlink_owned
    removed = []

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        removed.append(True)
        raise RuntimeError("unlink interrupted")

    monkeypatch.setattr(cleanup, "_unlink_owned", interrupted)
    with pytest.raises(RuntimeError, match="unlink interrupted"):
        cleanup.finish_staging(owner, result, receipt, verify_receipt=verify)
    assert removed == [True]
    assert resources.audit() == before
    assert not owner.path("metrics").exists()
    assert verify() == receipt
    with pytest.raises(ValueError):
        cleanup.finish_staging(owner, result, receipt, verify_receipt=verify)


@pytest.mark.parametrize("after", [1, 2, 3])
def test_each_file_unlink_failure_rolls_back_all_refunds(resources, monkeypatch, after):
    from arblab.hyperliquid_copy import ranking_staging_cleanup as cleanup

    owner, result, receipt, verify = prepared(resources)
    before = resources.audit()
    original, count = cleanup._unlink_owned, 0

    def interrupted(*args, **kwargs):
        nonlocal count
        original(*args, **kwargs)
        count += 1
        if count == after:
            raise RuntimeError("unlink boundary")

    monkeypatch.setattr(cleanup, "_unlink_owned", interrupted)
    with pytest.raises(RuntimeError, match="unlink boundary"):
        cleanup.finish_staging(owner, result, receipt, verify_receipt=verify)
    assert resources.audit() == before
    assert verify() == receipt
    assert owner._closed
    with pytest.raises(ValueError, match="pending"):
        RankingStagingOwner.create(resources, {"query": "new"})


@pytest.mark.parametrize(
    "change", ["bytes", "hardlink", "symlink", "missing", "scratch", "query", "ledger"]
)
def test_all_targets_are_authenticated_before_first_removal(
    resources, tmp_path, change
):
    from arblab.hyperliquid_copy.ranking_staging_cleanup import finish_staging

    owner, result, receipt, verify = prepared(resources)
    path = owner.path("scores")
    if change == "bytes":
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif change == "hardlink":
        os.link(path, tmp_path / "linked")
    elif change == "symlink":
        moved = tmp_path / "moved"
        path.rename(moved)
        path.symlink_to(moved)
    elif change == "missing":
        path.unlink()
    elif change == "scratch":
        (owner.path("scratch") / "unexpected").touch()
    elif change == "query":
        owner._context["query"] = {"different": "query"}
    else:
        with resources._connect() as db, db:
            db.execute(
                "UPDATE allocations SET bytes=1 WHERE token=?",
                (owner._allocations["scores"]["token"],),
            )
    with pytest.raises((ValueError, FileNotFoundError)):
        finish_staging(owner, result, receipt, verify_receipt=verify)
    assert owner.path("metrics").exists()
    assert owner.path("manifest").exists()
    assert verify() == receipt
    assert owner._closed


def test_late_namespace_swap_prevents_first_unlink(resources, monkeypatch):
    from arblab.hyperliquid_copy import ranking_staging_cleanup as cleanup
    from arblab.hyperliquid_copy.ranking_staging_artifact import StagingArtifact

    owner, result, receipt, verify = prepared(resources)
    original, count = StagingArtifact.verify, 0

    def swap(pin):
        nonlocal count
        original(pin)
        if pin.path == owner.path("manifest"):
            count += 1
            if count == 3:  # Initial guard, loop guard, final pre-unlink guard.
                parent = resources.root / "scratch"
                old = resources.root / "scratch-old"
                parent.rename(old)
                parent.mkdir()
                (old / owner.path("scratch").name).rename(owner.path("scratch"))

    monkeypatch.setattr(StagingArtifact, "verify", swap)
    with pytest.raises(ValueError):
        cleanup.finish_staging(owner, result, receipt, verify_receipt=verify)
    assert count == 3
    assert owner.path("metrics").exists()


def test_late_source_change_prevents_first_unlink(resources, monkeypatch):
    from arblab.hyperliquid_copy import ranking_staging_cleanup as cleanup
    from arblab.hyperliquid_copy.ranking_staging_artifact import StagingArtifact

    owner, result, receipt, verify = prepared(resources)
    original, count, changed = StagingArtifact.verify, 0, False

    def mutate(pin):
        nonlocal count, changed
        original(pin)
        if pin.path == owner.path("manifest"):
            count += 1
            if count == 3:
                changed = True

    def source_verify():
        if changed:
            raise ValueError("source changed")
        return verify()

    monkeypatch.setattr(StagingArtifact, "verify", mutate)
    with pytest.raises(ValueError, match="source changed"):
        cleanup.finish_staging(owner, result, receipt, verify_receipt=source_verify)
    assert owner.path("metrics").exists()


@pytest.mark.parametrize("after", range(1, 9))
def test_fsync_failure_never_refunds_partial_cleanup(resources, monkeypatch, after):
    from arblab.hyperliquid_copy.ranking_staging_cleanup import finish_staging

    owner, result, receipt, verify = prepared(resources)
    before = resources.audit()
    original, count = os.fsync, 0

    def interrupted(fd):
        nonlocal count
        original(fd)
        count += 1
        if count == after:
            raise OSError("sync boundary")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", interrupted)
        with pytest.raises(OSError, match="sync boundary"):
            finish_staging(owner, result, receipt, verify_receipt=verify)
    assert resources.audit() == before
    assert verify() == receipt
    assert owner._closed


@pytest.mark.parametrize("boundary", ["delete", "commit"])
def test_database_failure_rolls_back_refunds(resources, monkeypatch, boundary):
    from arblab.hyperliquid_copy.ranking_staging_cleanup import finish_staging

    owner, result, receipt, verify = prepared(resources)
    before = resources.audit()
    original = type(resources)._connect

    @contextmanager
    def connect(self):
        def authorize(action, first, second, database, trigger):
            if (
                boundary == "delete"
                and action == sqlite3.SQLITE_DELETE
                and first == "allocations"
            ):
                return sqlite3.SQLITE_DENY
            if (
                boundary == "commit"
                and action == sqlite3.SQLITE_TRANSACTION
                and first == "COMMIT"
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        with original(self) as db:
            db.set_authorizer(authorize)
            yield db

    with monkeypatch.context() as patch:
        patch.setattr(type(resources), "_connect", connect)
        with pytest.raises(ValueError, match="Invalid existing cache catalog") as error:
            finish_staging(owner, result, receipt, verify_receipt=verify)
        assert isinstance(error.value.__cause__, sqlite3.DatabaseError)
    assert resources.audit() == before
    assert verify() == receipt
    assert owner._closed
