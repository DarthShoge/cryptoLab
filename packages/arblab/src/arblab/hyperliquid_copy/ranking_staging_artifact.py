"""Content pins for completed pending outputs; not causal provenance proofs."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

from .disk_metric_rows import MAX_BYTES
from .download import file_hash


def _identity(info):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Unsafe staging artifact identity")
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True)
class StagingArtifact:
    path: Path
    bytes: int
    sha256: str
    identity: tuple

    @classmethod
    def capture(cls, path, fd, *, maximum=MAX_BYTES):
        path = Path(path)
        info = os.fstat(fd)
        identity = _identity(info)
        if (
            type(maximum) is not int
            or not 0 < maximum <= MAX_BYTES
            or not 0 < info.st_size <= maximum
        ):
            raise ValueError("Invalid staging artifact byte bound")
        if _identity(path.lstat()) != identity:
            raise ValueError("Staging artifact path/FD identity mismatch")
        digest, offset = hashlib.sha256(), 0
        while offset < info.st_size:
            block = os.pread(fd, min(1024**2, info.st_size - offset), offset)
            if not block:
                raise ValueError("Incomplete staging artifact content")
            digest.update(block)
            offset += len(block)
        if _identity(os.fstat(fd)) != identity or _identity(path.lstat()) != identity:
            raise ValueError("Staging artifact changed during capture")
        return cls(path, info.st_size, digest.hexdigest(), identity)

    def verify(self):
        if (
            _identity(self.path.lstat()) != self.identity
            or file_hash(self.path) != self.sha256
            or _identity(self.path.lstat()) != self.identity
        ):
            raise ValueError("Staging artifact content/identity changed")
