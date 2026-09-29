import json
import os
import sqlite3
import subprocess
import sys
import uuid

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts


def fixture_cache(root, lease):
    resources = CacheResources.create(lease, "expansion-test")
    relative = f"artifacts/{uuid.uuid4().hex}"
    token = resources.reserve(relative, 100, "payload")
    (root / relative).write_bytes(b"preserve this published history")
    resources.settle(token)
    publication = PublishedArtifacts(resources).publish("history", {"day": 1}, [token])
    return resources, publication


def rows(resources):
    with resources._connect() as db:
        return (
            db.execute("SELECT * FROM allocations ORDER BY token").fetchall(),
            db.execute("SELECT * FROM publications ORDER BY key").fetchall(),
        )


def prepare(resources):
    from arblab.hyperliquid_copy.derived_cache_expansion import prepare_expansion

    return prepare_expansion(
        resources,
        approved_from=8 * 1024**3,
        approved_to=16 * 1024**3,
        approval="explicit-user-approval-2026-09-15",
    )


def test_expansion_preserves_publications_and_reopens(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_expansion import apply_expansion
    from arblab.hyperliquid_copy.derived_cache_policy import open_expanded_cache

    with CacheLease(tmp_path) as lease:
        old, publication = fixture_cache(tmp_path, lease)
        old_rows = rows(old)
        inputs = prepare(old)
        expanded = apply_expansion(lease, "expansion-test", inputs)
        assert expanded.limit == 16 * 1024**3
        assert expanded.audit()["reserved_bytes"] == 0
        new_rows = rows(expanded)
        assert set(old_rows[0]) <= set(new_rows[0])
        assert set(old_rows[1]) <= set(new_rows[1])
        assert PublishedArtifacts(expanded).lookup("history", {"day": 1}) == publication
        with pytest.raises(ValueError):
            CacheResources(lease, "expansion-test")
    with CacheLease(tmp_path) as lease:
        reopened = open_expanded_cache(lease, "expansion-test", inputs)
        assert PublishedArtifacts(reopened).lookup("history", {"day": 1}) == publication
        reopened.reserve(f"scratch/{uuid.uuid4().hex}", 123, "scratch")
        assert reopened.audit()["reserved_bytes"] == 123


def test_expanded_marker_without_receipt_is_not_authority(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_policy import open_expanded_cache

    with CacheLease(tmp_path) as lease:
        resources, _ = fixture_cache(tmp_path, lease)
        marker = json.loads(resources.marker.read_text())
        marker["limit_bytes"] = 16 * 1024**3
        with resources._connect() as db, db:
            db.execute(
                "UPDATE metadata SET value=?", (json.dumps(marker, sort_keys=True),)
            )
        resources.marker.write_text(json.dumps(marker, sort_keys=True))
        with pytest.raises(ValueError):
            open_expanded_cache(lease, "expansion-test", {})


@pytest.mark.parametrize(
    "point", ["before_replace", "after_replace", "before_release", "after_release"]
)
def test_interruption_retains_obligations_and_recovers(tmp_path, monkeypatch, point):
    from arblab.hyperliquid_copy import derived_cache_expansion as migration

    with CacheLease(tmp_path) as lease:
        old, publication = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
        replace, release = os.replace, CacheResources.release_missing

        def interrupted_replace(*args, **kwargs):
            if point == "before_replace":
                raise RuntimeError("interrupted")
            replace(*args, **kwargs)
            if point == "after_replace":
                raise RuntimeError("interrupted")

        def interrupted_release(*args, **kwargs):
            if point == "before_release":
                raise RuntimeError("interrupted")
            release(*args, **kwargs)
            if point == "after_release":
                raise RuntimeError("interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", interrupted_replace)
            patch.setattr(CacheResources, "release_missing", interrupted_release)
            with pytest.raises(RuntimeError, match="interrupted"):
                migration.apply_expansion(lease, "expansion-test", inputs)
        with sqlite3.connect(tmp_path / "resources.sqlite3") as db:
            assert db.execute(
                "SELECT sum(maximum) FROM allocations WHERE state='pending'"
            ).fetchone()[0] == (None if point == "after_release" else 4096)
    with CacheLease(tmp_path) as lease:
        resources = migration.apply_expansion(lease, "expansion-test", inputs)
        assert resources.audit()["reserved_bytes"] == 0
        assert (
            PublishedArtifacts(resources).lookup("history", {"day": 1}) == publication
        )


def test_recovery_fsyncs_replaced_root_before_refund(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import derived_cache_expansion as migration

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
        replace = os.replace

        def crash(*args, **kwargs):
            replace(*args, **kwargs)
            raise RuntimeError("power loss before root fsync")

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", crash)
            with pytest.raises(RuntimeError):
                migration.apply_expansion(lease, "expansion-test", inputs)
    events = []
    fsync, release = os.fsync, CacheResources.release_missing

    def track_sync(fd):
        if os.fstat(fd).st_ino == tmp_path.stat().st_ino:
            events.append("root_sync")
        return fsync(fd)

    def track_release(*args, **kwargs):
        events.append("release")
        return release(*args, **kwargs)

    with CacheLease(tmp_path) as lease, monkeypatch.context() as patch:
        patch.setattr(os, "fsync", track_sync)
        patch.setattr(CacheResources, "release_missing", track_release)
        migration.apply_expansion(lease, "expansion-test", inputs)
    assert "root_sync" in events
    assert events.index("root_sync") < events.index("release")


@pytest.mark.parametrize(
    "from_limit,to_limit",
    [
        (True, 16 * 1024**3),
        (8 * 1024**3, True),
        (8 * 1024**3, 32 * 1024**3),
        (4 * 1024**3, 16 * 1024**3),
    ],
)
def test_unapproved_limits_rejected_without_allocating(tmp_path, from_limit, to_limit):
    from arblab.hyperliquid_copy.derived_cache_expansion import prepare_expansion

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        before = rows(old)
        with pytest.raises(ValueError):
            prepare_expansion(
                old, approved_from=from_limit, approved_to=to_limit, approval="yes"
            )
        assert rows(old) == before


def test_pending_work_blocks_preparation_without_refund(tmp_path):
    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        old.reserve(f"scratch/{uuid.uuid4().hex}", 77, "scratch")
        before = rows(old)
        with pytest.raises(ValueError, match="pending"):
            prepare(old)
        assert rows(old) == before


@pytest.mark.parametrize(
    "fault",
    [
        "extra_allocation",
        "extra_publication",
        "stage_bytes",
        "stage_symlink",
        "stage_hardlink",
        "marker_nonce",
        "database_nonce",
        "metadata_schema",
        "receipt_bytes",
    ],
)
def test_apply_rejects_changed_evidence_without_migrating(tmp_path, fault):
    from arblab.hyperliquid_copy.derived_cache_expansion import apply_expansion
    from arblab.hyperliquid_copy.derived_cache_policy import _receipt

    with CacheLease(tmp_path) as lease:
        old, publication = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
        receipt_pub, receipt = _receipt(old, inputs)
        stage = tmp_path / receipt["stage"]["path"]
        if fault == "extra_allocation":
            old.reserve(f"scratch/{uuid.uuid4().hex}", 1, "scratch")
        elif fault == "extra_publication":
            PublishedArtifacts(old).publish(
                "extra", {}, [publication.artifacts[0].token]
            )
        elif fault == "stage_bytes":
            raw = stage.read_bytes()
            stage.write_bytes(raw.replace(b'"version": 2', b'"version": 3'))
        elif fault == "stage_symlink":
            kept = tmp_path.parent / (tmp_path.name + "-kept")
            stage.rename(kept)
            stage.symlink_to(kept)
        elif fault == "stage_hardlink":
            os.link(stage, tmp_path.parent / (tmp_path.name + "-linked"))
        elif fault == "marker_nonce":
            value = json.loads(old.marker.read_text())
            value["nonce"] = "f" * 32
            old.marker.write_text(json.dumps(value, sort_keys=True))
        elif fault == "database_nonce":
            value = dict(inputs["old_marker"], nonce="f" * 32)
            with old._connect() as db, db:
                db.execute(
                    "UPDATE metadata SET value=?", (json.dumps(value, sort_keys=True),)
                )
        elif fault == "metadata_schema":
            with old._connect() as db, db:
                db.execute("PRAGMA user_version=3")
        else:
            path = tmp_path / receipt_pub.artifacts[0].path
            path.write_bytes(b"x" * path.stat().st_size)
        with pytest.raises((ValueError, OSError)):
            apply_expansion(lease, "expansion-test", inputs)
        assert json.loads(old.marker.read_text())["limit_bytes"] == 8 * 1024**3
        assert (
            tmp_path / publication.artifacts[0].path
        ).read_bytes() == b"preserve this published history"


def test_changed_caller_receipt_during_state_check_rejects_before_commit(
    tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import derived_cache_expansion as migration

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
        original = migration._state

        def changing(*args, **kwargs):
            result = original(*args, **kwargs)
            inputs["approval"] = "different approval"
            return result

        monkeypatch.setattr(migration, "_state", changing)
        with pytest.raises(ValueError):
            migration.apply_expansion(lease, "expansion-test", inputs)
        assert json.loads(old.marker.read_text())["limit_bytes"] == 8 * 1024**3


def test_preparation_rejects_directory_substitution(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import derived_cache_expansion as migration

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        original = migration._write

        def swapped(path, raw):
            original(path, raw)
            if path.parent.name == "staging":
                previous = tmp_path.parent / (tmp_path.name + "-old-staging")
                path.parent.rename(previous)
                path.parent.mkdir()
                (previous / path.name).rename(path)

        monkeypatch.setattr(migration, "_write", swapped)
        with pytest.raises(ValueError, match="directory"):
            prepare(old)
        assert json.loads(old.marker.read_text())["limit_bytes"] == 8 * 1024**3


@pytest.mark.parametrize("write_number", [1, 2])
def test_interrupted_preparation_preserves_charges(tmp_path, monkeypatch, write_number):
    from arblab.hyperliquid_copy import derived_cache_expansion as migration

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        original = migration._write
        calls = 0

        def interrupted(path, raw):
            nonlocal calls
            calls += 1
            original(path, raw)
            if calls == write_number:
                raise RuntimeError("interrupted preparation")

        monkeypatch.setattr(migration, "_write", interrupted)
        with pytest.raises(RuntimeError):
            prepare(old)
        assert old.audit()["reserved_bytes"] == 4096 + 64 * 1024
        with pytest.raises(ValueError, match="pending"):
            prepare(old)


def test_policy_change_during_preparation_publication_is_reported(
    tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import derived_cache_policy as policy

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        original, engine = PublishedArtifacts.publish, policy._engine

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            monkeypatch.setattr(
                policy,
                "_engine",
                lambda: dict(engine(), derived_cache_policy="changed"),
            )
            return result

        monkeypatch.setattr(PublishedArtifacts, "publish", changed)
        with pytest.raises(ValueError, match="policy"):
            prepare(old)
        assert json.loads(old.marker.read_text())["limit_bytes"] == 8 * 1024**3


def test_hot_rollback_journal_recovers_before_read_only_inspection(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_expansion import apply_expansion

    with CacheLease(tmp_path) as lease:
        old, publication = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
    script = """
import json, os, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute('PRAGMA cache_size=1')
db.execute('BEGIN IMMEDIATE')
db.execute('UPDATE metadata SET value=?', (sys.argv[2],))
db.execute('CREATE TABLE crash_only (value BLOB)')
db.executemany('INSERT INTO crash_only VALUES (?)', [(b'x'*4096,)]*40)
os._exit(0)
"""
    new_marker = dict(inputs["old_marker"], limit_bytes=16 * 1024**3)
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "resources.sqlite3"),
            json.dumps(new_marker, sort_keys=True),
        ],
        check=True,
    )
    journal = tmp_path / "resources.sqlite3-journal"
    assert journal.stat().st_size > 0
    with sqlite3.connect(
        (tmp_path / "resources.sqlite3").as_uri() + "?mode=ro", uri=True
    ) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("SELECT value FROM metadata").fetchall()
    with CacheLease(tmp_path) as lease:
        resources = apply_expansion(lease, "expansion-test", inputs)
        assert resources.audit()["reserved_bytes"] == 0
        assert (
            PublishedArtifacts(resources).lookup("history", {"day": 1}) == publication
        )
        with resources._connect() as db:
            assert (
                db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='crash_only'"
                ).fetchall()
                == []
            )
