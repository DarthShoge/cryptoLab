"""Receipt-chained policy for the explicitly approved 16-to-64-GiB cache."""

import hashlib
import json
from pathlib import Path
import re

from .derived_cache_policy import (
    EXPANDED_BYTES,
    RECEIPT_BYTES,
    _ExpandedResources,
    _PinnedFile,
    _new_marker as predecessor_marker,
    _receipt as predecessor_receipt,
    _validate as validate_predecessor,
)
from .derived_cache_resources import CacheResources, MAX_BYTES
from .derived_publication import PublishedArtifacts, _encode, _inputs
from .download import file_hash

EXPANDED_64_BYTES = 64 * 1024**3
KIND_64 = "approved_cache_expansion_64gib"
RECEIPT_64_BYTES = RECEIPT_BYTES


def _engine_64():
    root = Path(__file__).parent
    return {
        name: file_hash(root / (name + ".py"))
        for name in (
            "derived_cache_policy_64",
            "derived_cache_expansion_64",
            "derived_cache_policy",
            "derived_cache_expansion",
            "derived_cache_resources",
            "derived_publication",
        )
    }


def _validate_64(lease, identity, inputs):
    lease.check()
    frozen = _inputs(KIND_64, inputs)
    if set(frozen) != {
        "schema",
        "predecessor_receipt",
        "old_marker",
        "new_limit",
        "approval",
        "engine",
    }:
        raise ValueError("Invalid 64-GiB expansion receipt inputs")
    predecessor = validate_predecessor(lease, identity, frozen["predecessor_receipt"])
    if (
        type(frozen["schema"]) is not int
        or frozen["schema"] != 1
        or _encode(frozen["old_marker"]) != _encode(predecessor_marker(predecessor))
        or type(frozen["new_limit"]) is not int
        or frozen["new_limit"] != EXPANDED_64_BYTES
        or type(frozen["approval"]) is not str
        or not 1 <= len(frozen["approval"].strip()) <= 1024
        or _encode(frozen["engine"]) != _encode(_engine_64())
    ):
        raise ValueError("64-GiB expansion policy/identity/approval mismatch")
    lease.check()
    if _encode(inputs) != _encode(frozen):
        raise ValueError("Changed 64-GiB expansion inputs")
    return frozen


def _new_marker_64(inputs):
    return dict(inputs["old_marker"], limit_bytes=EXPANDED_64_BYTES)


def _receipt_64(resources, inputs):
    publication = PublishedArtifacts(resources).lookup(KIND_64, inputs)
    if publication is None or len(publication.artifacts) != 1:
        raise ValueError("Missing 64-GiB expansion receipt publication")
    artifact = publication.artifacts[0]
    with _PinnedFile(resources.root / artifact.path, RECEIPT_64_BYTES) as pin:
        raw = pin.read()
        if len(raw) != artifact.bytes or hashlib.sha256(raw).hexdigest() != artifact.sha256:
            raise ValueError("Changed 64-GiB expansion receipt payload")
        data = json.loads(raw)
        if (
            type(data) is not dict
            or set(data) != {"inputs", "stage", "baseline"}
            or _encode(data["inputs"]) != _encode(inputs)
        ):
            raise ValueError("Invalid 64-GiB expansion receipt")
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
            raise ValueError("Invalid 64-GiB expansion staging receipt")
        baseline = data["baseline"]
        if type(baseline) is not dict or set(baseline) != {"allocations", "publications"}:
            raise ValueError("Invalid 64-GiB expansion baseline")
        for value in baseline.values():
            if (
                type(value) is not list
                or len(value) != 2
                or type(value[0]) is not int
                or not 0 <= value[0] <= 100_000
                or type(value[1]) is not str
                or not re.fullmatch("[a-f0-9]{64}", value[1])
            ):
                raise ValueError("Invalid 64-GiB expansion baseline digest")
        pin.check()
    return publication, data


class _Expanded64Resources(CacheResources):
    @staticmethod
    def _arguments(lease, identity, limit_bytes):
        CacheResources._arguments(lease, identity, MAX_BYTES)
        if type(limit_bytes) is not int or limit_bytes != EXPANDED_64_BYTES:
            raise ValueError("Expected explicit 64-GiB cache policy")


def _verify_predecessor(resources, identity, inputs):
    predecessor = validate_predecessor(resources.lease, identity, inputs)
    publication, data = predecessor_receipt(resources, predecessor)
    with resources._connect() as db:
        if db.execute(
            "SELECT 1 FROM allocations WHERE token=?", (data["stage"]["token"],)
        ).fetchone():
            raise ValueError("Predecessor expansion staging obligation still pending")
    stage = resources._path(data["stage"]["path"])
    if stage.exists() or stage.is_symlink():
        raise ValueError("Predecessor expansion staging output still present")
    return publication


def open_expanded_64_cache(lease, identity, predecessor_inputs, receipt_inputs):
    """Open a completed 64-GiB transition; never infer or repair authority."""
    inputs = _validate_64(lease, identity, receipt_inputs)
    if _encode(inputs["predecessor_receipt"]) != _encode(predecessor_inputs):
        raise ValueError("64-GiB predecessor receipt mismatch")
    resources = _Expanded64Resources(lease, identity, limit_bytes=EXPANDED_64_BYTES)
    with _PinnedFile(resources.marker, 4096) as marker:
        if _encode(json.loads(marker.read())) != _encode(_new_marker_64(inputs)):
            raise ValueError("64-GiB expansion marker does not match receipt")
        predecessor = _verify_predecessor(resources, identity, predecessor_inputs)
        publication, data = _receipt_64(resources, inputs)
        with resources._connect() as db:
            if db.execute(
                "SELECT 1 FROM allocations WHERE token=?", (data["stage"]["token"],)
            ).fetchone():
                raise ValueError("64-GiB expansion staging obligation still pending")
        stage = resources._path(data["stage"]["path"])
        if stage.exists() or stage.is_symlink():
            raise ValueError("64-GiB expansion staging output still present")
        if _verify_predecessor(resources, identity, predecessor_inputs) != predecessor:
            raise ValueError("Changed predecessor expansion receipt")
        if _receipt_64(resources, inputs)[0] != publication:
            raise ValueError("Changed 64-GiB expansion receipt")
        if _encode(receipt_inputs) != _encode(inputs):
            raise ValueError("Changed caller 64-GiB expansion inputs")
        marker.check()
        lease.check()
    return resources
