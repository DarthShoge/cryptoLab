import multiprocessing
import os

import pytest


def hold(root, ready, stop):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    with CacheLease(root):
        ready.set()
        stop.wait(20)


def test_context_busy_and_exception_release(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheBusyError, CacheLease

    with pytest.raises(RuntimeError):
        with CacheLease(tmp_path) as lease:
            lease.check()
            with pytest.raises(CacheBusyError):
                with CacheLease(tmp_path):
                    pass
            raise RuntimeError("caller")
    with pytest.raises(ValueError, match="held"):
        lease.check()
    inode = (tmp_path / ".resource.lock").stat().st_ino
    with CacheLease(tmp_path) as replacement:
        replacement.check()
        assert (tmp_path / ".resource.lock").stat().st_ino == inode


def test_process_death_releases_same_lock(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheBusyError, CacheLease

    ctx = multiprocessing.get_context("spawn")
    ready, stop = ctx.Event(), ctx.Event()
    child = ctx.Process(target=hold, args=(tmp_path, ready, stop))
    child.start()
    try:
        assert ready.wait(10)
        inode = (tmp_path / ".resource.lock").stat().st_ino
        with pytest.raises(CacheBusyError):
            with CacheLease(tmp_path):
                pass
        child.terminate()
        child.join(10)
        assert not child.is_alive()
        with CacheLease(tmp_path) as lease:
            lease.check()
            assert (tmp_path / ".resource.lock").stat().st_ino == inode
    finally:
        # A terminated waiter can leave Event's condition semaphore poisoned.
        # Never notify that event after deliberately killing its waiter.
        if child.is_alive():
            child.terminate()
        child.join(10)


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "directory", "nonempty"])
def test_unsafe_lock_rejected(tmp_path, fault):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    path = tmp_path / ".resource.lock"
    target = tmp_path / "other"
    target.write_text("keep")
    if fault == "symlink":
        path.symlink_to(target)
    elif fault == "hardlink":
        os.link(target, path)
    elif fault == "directory":
        path.mkdir()
    else:
        path.write_text("unexpected lock payload")
    with pytest.raises(ValueError):
        with CacheLease(tmp_path):
            pass
    assert target.read_text() == "keep"


@pytest.mark.parametrize("fault", ["root", "lock", "pid"])
def test_changed_ownership_rejected(tmp_path, monkeypatch, fault):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    root = tmp_path / "cache"
    root.mkdir()
    with CacheLease(root) as lease:
        if fault == "root":
            root.rename(tmp_path / "old")
            root.mkdir()
        elif fault == "lock":
            (root / ".resource.lock").rename(root / "old.lock")
            (root / ".resource.lock").touch()
        else:
            original = os.getpid()
            monkeypatch.setattr(os, "getpid", lambda: original + 1)
        with pytest.raises(ValueError):
            lease.check()


def test_symlink_root_rejected(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    target = tmp_path / "real"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        with CacheLease(alias):
            pass


def fork_ownership(root):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease, CacheBusyError

    with CacheLease(root) as lease:
        child = os.fork()
        if child == 0:
            try:
                lease.check()
            except ValueError:
                lease.__exit__(None)
                os._exit(0)
            os._exit(1)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        lease.check()
        with pytest.raises(CacheBusyError):
            with CacheLease(root):
                pass


def test_fork_child_cannot_use_or_unlock_parent_lease(tmp_path):
    # Fork in a fresh single-threaded process, not pytest's Arrow-loaded process.
    child = multiprocessing.get_context("spawn").Process(
        target=fork_ownership, args=(tmp_path,)
    )
    child.start()
    child.join(10)
    try:
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)
