"""Explicit receipt-backed 16-GiB policy; legacy derivation code stays unchanged."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .derived_cache_resources import CacheResources, MAX_BYTES
from .derived_publication import PublishedArtifacts, _encode, _inputs
from .download import file_hash

EXPANDED_BYTES = 16 * 1024**3
KIND = "approved_cache_expansion"
RECEIPT_BYTES = 64 * 1024


def _engine():
    root = Path(__file__).parent
    return {
        name: file_hash(root / (name + ".py"))
        for name in (
            "derived_cache_policy",
            "derived_cache_expansion",
            "derived_cache_resources",
            "derived_publication",
        )
    }


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


class _PinnedFile:
    """Hold a bounded regular file across validation and an explicit final check."""

    def __init__(self, path, maximum):
        self.path = Path(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(self.fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or not 0 < info.st_size <= maximum
            ):
                raise ValueError("Unsafe/oversized expansion file")
            self.identity = _identity(info)
            self.check()
        except BaseException:
            os.close(self.fd)
            raise

    def check(self):
        if (
            _identity(os.fstat(self.fd)) != self.identity
            or _identity(self.path.lstat()) != self.identity
        ):
            raise ValueError("Changed expansion file identity")

    def read(self):
        self.check()
        raw = os.pread(self.fd, self.identity[2] + 1, 0)
        self.check()
        if len(raw) != self.identity[2]:
            raise ValueError("Changed expansion file length")
        return raw

    def __enter__(self):
        return self

    def __exit__(self, *_):
        os.close(self.fd)


def _validate(lease, identity, inputs):
    lease.check()
    frozen = _inputs(KIND, inputs)
    if set(frozen) != {"schema", "old_marker", "new_limit", "approval", "engine"}:
        raise ValueError("Invalid expansion receipt inputs")
    marker = frozen["old_marker"]
    if (
        type(marker) is not dict
        or set(marker) != {"version", "identity", "limit_bytes", "nonce", "root"}
        or type(marker["version"]) is not int
        or marker["version"] != 2
        or type(marker["limit_bytes"]) is not int
        or marker["limit_bytes"] != MAX_BYTES
        or marker["identity"] != identity
        or type(marker["nonce"]) is not str
        or not re.fullmatch("[a-f0-9]{32}", marker["nonce"])
        or type(marker["root"]) is not list
        or len(marker["root"]) != 2
        or any(type(v) is not int for v in marker["root"])
        or marker["root"] != [lease.root.stat().st_dev, lease.root.stat().st_ino]
        or type(frozen["schema"]) is not int
        or frozen["schema"] != 1
        or type(frozen["new_limit"]) is not int
        or frozen["new_limit"] != EXPANDED_BYTES
        or type(frozen["approval"]) is not str
        or not 1 <= len(frozen["approval"].strip()) <= 1024
        or _encode(frozen["engine"]) != _encode(_engine())
    ):
        raise ValueError("Expansion policy/identity/approval mismatch")
    lease.check()
    if _encode(inputs) != _encode(frozen):
        raise ValueError("Changed expansion inputs")
    return frozen


def _new_marker(inputs):
    return dict(inputs["old_marker"], limit_bytes=EXPANDED_BYTES)


def _receipt(resources, inputs):
    publication = PublishedArtifacts(resources).lookup(KIND, inputs)
    if publication is None or len(publication.artifacts) != 1:
        raise ValueError("Missing expansion receipt publication")
    artifact = publication.artifacts[0]
    with _PinnedFile(resources.root / artifact.path, RECEIPT_BYTES) as pin:
        raw = pin.read()
        if (
            len(raw) != artifact.bytes
            or hashlib.sha256(raw).hexdigest() != artifact.sha256
        ):
            raise ValueError("Changed expansion receipt payload")
        data = json.loads(raw)
        if (
            type(data) is not dict
            or set(data) != {"inputs", "stage", "baseline"}
            or _encode(data["inputs"]) != _encode(inputs)
        ):
            raise ValueError("Invalid expansion receipt")
        stage = data["stage"]
        if (
            type(stage) is not dict
            or set(stage) != {"path", "token", "bytes", "sha256"}
            or type(stage["path"]) is not str
            or not re.fullmatch(r"staging/[a-f0-9]{32}", stage["path"])
            or type(stage["token"]) is not str
            or not re.fullmatch("[a-f0-9]{32}", stage["token"])
            or type(stage["bytes"]) is not int
            or not 0 < stage["bytes"] <= 4096
            or type(stage["sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", stage["sha256"])
        ):
            raise ValueError("Invalid expansion staging receipt")
        baseline = data["baseline"]
        if type(baseline) is not dict or set(baseline) != {
            "allocations",
            "publications",
        }:
            raise ValueError("Invalid expansion baseline")
        for value in baseline.values():
            if (
                type(value) is not list
                or len(value) != 2
                or type(value[0]) is not int
                or not 0 <= value[0] <= 100_000
                or type(value[1]) is not str
                or not re.fullmatch("[a-f0-9]{64}", value[1])
            ):
                raise ValueError("Invalid expansion baseline digest")
        pin.check()
    return publication, data


class _ExpandedResources(CacheResources):
    @staticmethod
    def _arguments(lease, identity, limit_bytes):
        CacheResources._arguments(lease, identity, MAX_BYTES)
        if type(limit_bytes) is not int or limit_bytes != EXPANDED_BYTES:
            raise ValueError("Expected explicit 16-GiB cache policy")


def open_expanded_cache(lease, identity, receipt_inputs):
    """Open an already completed migration; never repair or infer authorization."""
    inputs = _validate(lease, identity, receipt_inputs)
    resources = _ExpandedResources(lease, identity, limit_bytes=EXPANDED_BYTES)
    with _PinnedFile(resources.marker, 4096) as marker:
        if _encode(json.loads(marker.read())) != _encode(_new_marker(inputs)):
            raise ValueError("Expansion marker does not match receipt")
        publication, data = _receipt(resources, inputs)
        with resources._connect() as db:
            if db.execute(
                "SELECT 1 FROM allocations WHERE token=?", (data["stage"]["token"],)
            ).fetchone():
                raise ValueError("Expansion staging obligation still pending")
        stage = resources._path(data["stage"]["path"])
        if stage.exists() or stage.is_symlink():
            raise ValueError("Expansion staging output still present")
        if _encode(_validate(lease, identity, receipt_inputs)) != _encode(inputs):
            raise ValueError("Changed caller expansion inputs")
        if _receipt(resources, inputs)[0] != publication:
            raise ValueError("Changed expansion receipt")
        if _encode(receipt_inputs) != _encode(inputs):
            raise ValueError("Changed caller expansion inputs")
        marker.check()
        lease.check()
    return resources
