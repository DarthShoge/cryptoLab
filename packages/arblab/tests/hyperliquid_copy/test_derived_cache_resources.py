import os
import multiprocessing
import sqlite3
import uuid

import pytest


def name(namespace="staging"):
    return f"{namespace}/{uuid.uuid4().hex}"


def create(root, lease):
    from arblab.hyperliquid_copy.derived_cache_resources import (
        CacheResources,
        METADATA_BYTES,
    )

    return CacheResources.create(lease, "test-v1", limit_bytes=METADATA_BYTES + 100)


def test_pending_and_retained_share_budget(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import METADATA_BYTES

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        relative = name()
        token = ledger.reserve(relative, 80, "payload")
        assert not (tmp_path / relative).exists()
        assert ledger.audit()["total_bytes"] == METADATA_BYTES + 80
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve(name(), 21, "payload")
        (tmp_path / relative).write_bytes(b"a" * 30)
        ledger.settle(token)
        ledger.reserve(name("scratch"), 70, "scratch")
        assert ledger.audit() == dict(
            reserved_bytes=70,
            retained_bytes=30,
            metadata_bytes=METADATA_BYTES,
            total_bytes=METADATA_BYTES + 100,
        )


def test_reopen_retains_unwritten_obligation_and_explicit_release(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import (
        CacheResources,
        METADATA_BYTES,
    )

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        token = ledger.reserve(name(), 100, "payload")
    with CacheLease(tmp_path) as lease:
        ledger = CacheResources(lease, "test-v1", limit_bytes=METADATA_BYTES + 100)
        assert ledger.audit()["reserved_bytes"] == 100
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve(name(), 1, "payload")
        ledger.release_missing(token)
        assert ledger.audit()["reserved_bytes"] == 0


@pytest.mark.parametrize("fault", ["deleted", "truncated", "identity", "limit"])
def test_catalog_cannot_silently_reset(tmp_path, fault):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import (
        CacheResources,
        METADATA_BYTES,
    )

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        ledger.reserve(name(), 100, "payload")
        if fault == "deleted":
            ledger.path.unlink()
        elif fault == "truncated":
            ledger.path.write_bytes(b"")
        with pytest.raises(ValueError):
            CacheResources(
                lease,
                "wrong" if fault == "identity" else "test-v1",
                limit_bytes=METADATA_BYTES + (101 if fault == "limit" else 100),
            )
        with pytest.raises(ValueError):
            create(tmp_path, lease)


@pytest.mark.parametrize(
    "fault", ["oversized", "changed", "unknown", "symlink", "hardlink"]
)
def test_audit_rejects_unsafe_or_changed_outputs_without_deleting(tmp_path, fault):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        relative = name()
        token = ledger.reserve(relative, 50, "payload")
        path = tmp_path / relative
        path.write_bytes(b"keep")
        if fault == "oversized":
            path.write_bytes(b"a" * 51)
        elif fault == "changed":
            ledger.settle(token)
            path.write_bytes(b"edit")
        elif fault == "unknown":
            (tmp_path / name()).write_bytes(b"unknown")
        elif fault == "symlink":
            path.rename(tmp_path / "keep")
            path.symlink_to(tmp_path / "keep")
        else:
            os.link(path, tmp_path / "keep")
        with pytest.raises(ValueError):
            ledger.audit()
        assert path.exists()


def test_existing_scratch_and_settled_payload_cannot_be_refunded(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        relative = name("scratch")
        token = ledger.reserve(relative, 50, "scratch")
        (tmp_path / relative).mkdir()
        with pytest.raises(ValueError):
            ledger.release_missing(token)
        with pytest.raises(ValueError):
            ledger.settle(token)
        (tmp_path / relative).rmdir()
        ledger.release_missing(token)
        relative = name()
        token = ledger.reserve(relative, 50, "payload")
        (tmp_path / relative).write_bytes(b"data")
        ledger.settle(token)
        with pytest.raises(ValueError):
            ledger.release_missing(token)


@pytest.mark.parametrize(
    "relative",
    ["../escape", "/tmp/escape", "staging/../escape", "other/a", "staging/a/b"],
)
def test_invalid_allocation_paths(tmp_path, relative):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        with pytest.raises(ValueError):
            ledger.reserve(relative, 1, "payload")


def test_closed_lease_and_low_space_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy import derived_cache_resources as module

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        token = ledger.reserve(name(), 50, "payload")
        monkeypatch.setattr(
            module.shutil, "disk_usage", lambda _: SimpleNamespace(free=64 * 1024**2)
        )
        with pytest.raises(ValueError, match="space"):
            ledger.reserve(name(), 1, "payload")
    for operation in [
        ledger.audit,
        lambda: ledger.reserve(name(), 1, "payload"),
        lambda: ledger.settle(token),
        lambda: ledger.release_missing(token),
    ]:
        with pytest.raises(ValueError, match="held"):
            operation()


def crash_after_reservation(root, relative):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    with CacheLease(root) as lease:
        ledger = create(root, lease)
        ledger.reserve(relative, 100, "payload")
        (root / relative).write_bytes(b"partial")
        os._exit(17)


def contender(root, output):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease, CacheBusyError

    try:
        with CacheLease(root):
            output.send("acquired")
    except CacheBusyError:
        output.send("busy")
    finally:
        output.close()


def test_crash_keeps_partial_output_fully_charged(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import (
        CacheResources,
        METADATA_BYTES,
    )

    relative = name()
    child = multiprocessing.get_context("spawn").Process(
        target=crash_after_reservation, args=(tmp_path, relative)
    )
    child.start()
    child.join(10)
    assert child.exitcode == 17
    with CacheLease(tmp_path) as lease:
        ledger = CacheResources(lease, "test-v1", limit_bytes=METADATA_BYTES + 100)
        assert ledger.audit()["reserved_bytes"] == 100
        assert (tmp_path / relative).read_bytes() == b"partial"
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve(name(), 1, "payload")


def test_competing_process_cannot_start_independent_budget(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import (
        CacheResources,
        METADATA_BYTES,
    )

    ctx = multiprocessing.get_context("spawn")
    output, sender = ctx.Pipe(duplex=False)
    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        ledger.reserve(name(), 100, "payload")
        child = ctx.Process(target=contender, args=(tmp_path, sender))
        child.start()
        try:
            assert output.poll(10)
            assert output.recv() == "busy"
        finally:
            child.join(10)
            if child.is_alive():
                child.terminate()
                child.join(10)
            output.close()
            sender.close()
        assert child.exitcode == 0
    with CacheLease(tmp_path) as lease:
        ledger = CacheResources(lease, "test-v1", limit_bytes=METADATA_BYTES + 100)
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve(name(), 1, "payload")


def test_deleted_retained_payload_is_a_validation_error(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        relative = name()
        token = ledger.reserve(relative, 10, "payload")
        (tmp_path / relative).write_bytes(b"data")
        ledger.settle(token)
        (tmp_path / relative).unlink()
        with pytest.raises(ValueError):
            ledger.audit()


def test_record_limit_and_schema_change_rejected(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy import derived_cache_resources as module

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        monkeypatch.setattr(module, "MAX_RECORDS", 1)
        ledger.reserve(name(), 1, "payload")
        with pytest.raises(ValueError, match="record limit"):
            ledger.reserve(name(), 1, "payload")
        with sqlite3.connect(ledger.path) as db:
            db.execute("PRAGMA user_version=999")
        with pytest.raises(ValueError, match="schema"):
            ledger.audit()


@pytest.mark.parametrize("fault", ["page_size", "journal"])
def test_database_storage_settings_cannot_expand_metadata_budget(tmp_path, fault):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        with sqlite3.connect(ledger.path) as db:
            if fault == "page_size":
                db.execute("PRAGMA page_size=8192")
                db.execute("VACUUM")
            else:
                db.execute("PRAGMA journal_mode=WAL")
        db.close()
        with pytest.raises(ValueError):
            ledger.audit()


def test_refund_syncs_removed_output_directory_before_commit(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy import derived_cache_resources as module

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        relative = name()
        token = ledger.reserve(relative, 10, "payload")
        seen = []
        original = module._sync

        def sync(path):
            db = sqlite3.connect(ledger.path)
            try:
                assert db.execute("SELECT token FROM allocations").fetchall() == [
                    (token,)
                ]
            finally:
                db.close()
            seen.append(path)
            original(path)

        monkeypatch.setattr(module, "_sync", sync)
        ledger.release_missing(token)
        assert seen == [tmp_path / "staging"]


def test_failed_directory_sync_preserves_reservation(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy import derived_cache_resources as module

    with CacheLease(tmp_path) as lease:
        ledger = create(tmp_path, lease)
        token = ledger.reserve(name(), 10, "payload")

        def fail(_):
            raise OSError("sync failed")

        monkeypatch.setattr(module, "_sync", fail)
        with pytest.raises(OSError, match="sync failed"):
            ledger.release_missing(token)
        assert ledger.audit()["reserved_bytes"] == 10


def test_interrupted_initialization_cannot_be_recreated(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy import derived_cache_resources as module

    with CacheLease(tmp_path) as lease:
        with monkeypatch.context() as scoped:

            def fail(_):
                raise OSError("interrupted initialization")

            scoped.setattr(module, "_sync", fail)
            with pytest.raises(OSError, match="interrupted"):
                create(tmp_path, lease)
        assert (tmp_path / ".initialized").exists()
        with pytest.raises(ValueError):
            create(tmp_path, lease)
        with pytest.raises(ValueError):
            module.CacheResources(
                lease, "test-v1", limit_bytes=module.METADATA_BYTES + 100
            )
