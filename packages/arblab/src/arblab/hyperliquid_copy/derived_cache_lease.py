"""Exclusive Linux cache-work ownership; lock files are never removed."""

import fcntl
import os
from pathlib import Path
import stat


class CacheBusyError(RuntimeError):
    """Another operation owns this cache's resource envelope."""


def _identity(value):
    return value.st_dev, value.st_ino


def _safe_root(root):
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Unsafe cache root")
    return _identity(root.stat())


def _regular(value):
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or value.st_size != 0:
        raise ValueError("Expected regular unaliased cache lock")
    return _identity(value)


class CacheLease:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.path = self.root / ".resource.lock"
        self._fd = None
        self._used = False

    def __enter__(self):
        if self._used:
            raise ValueError("Cache lease cannot be reopened")
        self._used = True
        self._root_identity = _safe_root(self.root)
        try:
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            try:
                fd = os.open(self.path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                fd = os.open(self.path, flags)
            self._fd = fd
            self._lock_identity = _regular(os.fstat(fd))
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CacheBusyError("Cache resources are busy") from exc
            self._pid = os.getpid()
            self.check()
            return self
        except OSError as exc:
            self._close()
            raise ValueError("Cannot safely open cache lock") from exc
        except BaseException:
            self._close()
            raise

    def check(self):
        if self._fd is None or os.getpid() != self._pid:
            raise ValueError("Cache lease is not held by this process")
        if _safe_root(self.root) != self._root_identity:
            raise ValueError("Cache root identity changed")
        try:
            if (
                _regular(self.path.lstat()) != self._lock_identity
                or _regular(os.fstat(self._fd)) != self._lock_identity
            ):
                raise ValueError("Cache lock identity changed")
        except OSError as exc:
            raise ValueError("Cache lock identity unavailable") from exc

    def _close(self):
        if self._fd is not None:
            # Closing, not LOCK_UN: an inherited child must not unlock its parent.
            os.close(self._fd)
            self._fd = None

    def __exit__(self, *_):
        self._close()
