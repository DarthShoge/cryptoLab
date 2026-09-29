"""Bounded checksum reuse for an explicitly opted-in worker verification session.

Reuse a SHA-256 only within this process while file identity and timestamps stay
unchanged. This assumes normal local filesystem change tracking, as the existing
QualifiedSourceSession does. It is not detection of metadata-invisible bit rot.
"""

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import stat
import time


def signature(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class FileHashSession:
    def __init__(self, max_entries=4096):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("Positive cache size required")
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.pid = os.getpid()
        self.hits = self.misses = self.bytes_read = 0

    def _digest(self, handle):
        result = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            self.bytes_read += len(chunk)
            result.update(chunk)
        return result.hexdigest()

    def __call__(self, path):
        if os.getpid() != self.pid:
            raise ValueError("Hash session cannot cross process boundaries")
        path = Path(path).absolute()
        with os.fdopen(
            os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
        ) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Expected a regular hash input")
            identity = signature(before)
            # Linux inode timestamps can share a clock tick. Do not reuse a
            # digest for a newly written file until that ambiguity has passed.
            stable = (
                time.time_ns() - max(before.st_ctime_ns, before.st_mtime_ns)
                >= 1_000_000_000
            )
            cached = self.entries.get(path)
            if stable and cached is not None and cached[0] == identity:
                result = cached[1]
                self.hits += 1
            else:
                result = self._digest(handle)
                self.misses += 1
                if not stable:
                    handle.seek(0)
                    if self._digest(handle) != result:
                        self.entries.pop(path, None)
                        raise ValueError("Hash input changed during verification")
            if (
                signature(os.fstat(handle.fileno())) != identity
                or signature(path.lstat()) != identity
            ):
                self.entries.pop(path, None)
                raise ValueError("Hash input changed during verification")
        if not stable:
            self.entries.pop(path, None)
            return result
        self.entries[path] = identity, result
        self.entries.move_to_end(path)
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return result
