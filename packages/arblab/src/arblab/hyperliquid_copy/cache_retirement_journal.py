"""Bounded immutable retirement intent; never unlinks a cache payload."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import uuid

from .cache_retirement_inventory import (
    PROTECTED_KINDS,
    RetirementInventory,
    _descriptor,
    _digest,
    _identity,
    _namespace,
    _rows,
    retirement_inventory,
    _engine as inventory_engine,
)
from .derived_cache_policy import _PinnedFile, _identity as fd_identity
from .derived_cache_resources import _sync
from .derived_publication import (
    MAX_ARTIFACTS,
    MAX_DESCRIPTOR_BYTES,
    PublishedArtifacts,
    _encode,
    _inputs,
    publication_key,
)
from .download import file_hash

INTENT = "cache_retirement_intent"
COMMITTED = "cache_retirement_committed"
MAX_BYTES = 4 * 1024**2


def engine():
    root = Path(__file__).parent
    return dict(
        inventory=inventory_engine(),
        code={
            name: file_hash(root / name)
            for name in (
                "cache_retirement_journal.py",
                "cache_retirement.py",
            )
        },
    )


def context(resources, inputs):
    resources.lease.check()
    frozen = _inputs(INTENT, inputs)
    if (
        set(frozen)
        != {"schema", "target", "journal_sha256", "engine"}
        | ({"owner"} if "owner" in frozen else set())
        | ({"max_bytes"} if "max_bytes" in frozen else set())
        | ({"max_descriptor_bytes"} if "max_descriptor_bytes" in frozen else set())
        or type(frozen["schema"]) is not int
        or frozen["schema"] != 1
        or not _digest(frozen["target"])
        or not _digest(frozen["journal_sha256"])
        or "owner" in frozen
        and not _digest(frozen["owner"])
        or "max_bytes" in frozen
        and (
            type(frozen["max_bytes"]) is not int
            or not 0 < frozen["max_bytes"] <= MAX_BYTES
        )
        or "max_descriptor_bytes" in frozen
        and (
            type(frozen["max_descriptor_bytes"]) is not int
            or not 0 < frozen["max_descriptor_bytes"] <= MAX_DESCRIPTOR_BYTES
        )
        or frozen["engine"] != engine()
    ):
        raise ValueError("Invalid retirement receipt context")
    resources.lease.check()
    if _encode(inputs) != _encode(frozen):
        raise ValueError("Retirement caller context changed")
    return frozen


def allocations(db, target):
    return [
        list(
            db.execute("SELECT * FROM allocations WHERE token=?", (p.token,)).fetchone()
        )
        for p in target.artifacts
    ]


def catalogue_digest(resources, db, *, omit=()):
    digest = hashlib.sha256()
    for row in _rows(db):
        _descriptor(resources, db, *row)
        if row[0] not in omit:
            digest.update(_encode(row))
            digest.update(b"\n")
    return digest.hexdigest()


def validate_body(resources, body, inputs):
    if (
        type(body) is not dict
        or set(body)
        != {"schema", "inventory", "allocations", "protected", "reason"}
        | ({"owner"} if "owner" in inputs else set())
        or type(body["schema"]) is not int
        or body["schema"] != 1
    ):
        raise ValueError("Invalid retirement journal structure")
    if "owner" in inputs and (
        not _digest(body["owner"]) or body["owner"] != inputs["owner"]
    ):
        raise ValueError("Retirement journal owner mismatch")
    inv = body["inventory"]
    if (
        type(inv) is not dict
        or set(inv) != set(RetirementInventory.__dataclass_fields__)
        or type(inv["descriptor"]) is not str
        or len(inv["descriptor"].encode()) > MAX_DESCRIPTOR_BYTES
        or inv["marker"] != resources._metadata()
        or inv["namespace"] != [list(p) for p in _namespace(resources)]
        or inv["engine"] != _encode(inventory_engine()).decode()
        or not _digest(inv["catalogue_digest"])
        or type(body["reason"]) is not str
        or len(body["reason"]) > 1024
        or not body["reason"].strip()
    ):
        raise ValueError("Retirement journal cache/context mismatch")
    protected = _inputs(INTENT, dict(protected=body["protected"]))["protected"]
    if type(protected) is not list or any(not _digest(k) for k in protected):
        raise ValueError("Invalid retirement protected keys")
    data = json.loads(inv["descriptor"])
    if (
        type(data) is not dict
        or set(data) != {"schema", "kind", "inputs", "artifacts"}
        or type(data["schema"]) is not int
        or data["schema"] != 1
        or _encode(data).decode() != inv["descriptor"]
        or publication_key(data["kind"], data["inputs"]) != inputs["target"]
        or data["kind"] in PROTECTED_KINDS
        or data["kind"].startswith("cache_retirement")
        or inputs["target"] in protected
        or inv["target"] != dict(key=inputs["target"], artifacts=data["artifacts"])
        or type(data["artifacts"]) is not list
        or not 1 <= len(data["artifacts"]) <= MAX_ARTIFACTS
    ):
        raise ValueError("Invalid journal retirement target")
    pins, records, identities = data["artifacts"], body["allocations"], inv["files"]
    if (
        type(records) is not list
        or type(identities) is not list
        or len(records) != len(pins)
        or len(identities) != len(pins)
    ):
        raise ValueError("Invalid journal ownership records")
    tokens = []
    for pin, row, identity in zip(pins, records, identities, strict=True):
        if (
            type(pin) is not dict
            or set(pin) != {"token", "path", "bytes", "sha256"}
            or type(pin["token"]) is not str
            or len(pin["token"]) != 32
            or not _digest(pin["token"] * 2)
            or not _digest(pin["sha256"])
            or type(pin["bytes"]) is not int
            or not 0 <= pin["bytes"] <= resources.limit
            or type(row) is not list
            or len(row) != 7
            or row[:2] != [pin["token"], pin["path"]]
            or type(row[2]) is not int
            or not max(1, pin["bytes"]) <= row[2] <= resources.limit
            or row[3:] != ["payload", "retained", pin["bytes"], pin["sha256"]]
            or type(identity) is not list
            or len(identity) != 2
            or identity[0] != pin["token"]
            or type(identity[1]) is not list
            or len(identity[1]) != 6
            or any(type(v) is not int for v in identity[1])
            or identity[1][2] != pin["bytes"]
            or identity[1][5] != 1
        ):
            raise ValueError("Invalid journal artifact/physical ownership")
        resources._path(pin["path"])
        if not pin["path"].startswith("artifacts/"):
            raise ValueError("Journal path is outside artifacts")
        tokens.append(pin["token"])
    exclusive, shared = inv["exclusive"], inv["shared"]
    if (
        type(exclusive) is not list
        or type(shared) is not list
        or any(type(t) is not str for t in exclusive + shared)
        or len(set(tokens)) != len(tokens)
        or len(exclusive) + len(shared) != len(tokens)
        or set(exclusive) & set(shared)
        or set(exclusive + shared) != set(tokens)
    ):
        raise ValueError("Invalid journal ownership partition")
    return body


def load_journal(resources, inputs):
    frozen = context(resources, inputs)
    with resources._connect() as db:
        publications = PublishedArtifacts(resources)
        publication = publications._read(
            db, publication_key(INTENT, frozen), INTENT, frozen
        )
        if publication is None:
            publication = publications._read(
                db, publication_key(COMMITTED, frozen), COMMITTED, frozen
            )
        if publication is None:
            raise ValueError("Retirement journal receipt is missing")
    if len(publication.artifacts) != 1 or not 0 < publication.artifacts[
        0
    ].bytes <= inputs.get("max_bytes", MAX_BYTES):
        raise ValueError("Invalid retirement journal size")
    pin = publication.artifacts[0]
    if pin.sha256 != frozen["journal_sha256"]:
        raise ValueError("Retirement journal pin mismatch")
    with _PinnedFile(resources.root / pin.path, MAX_BYTES) as held:
        raw = held.read()
        if len(raw) != pin.bytes or hashlib.sha256(raw).hexdigest() != pin.sha256:
            raise ValueError("Retirement journal changed")
        try:
            body = json.loads(raw)
            if _encode(body) != raw:
                raise ValueError("Noncanonical retirement journal")
            validate_body(resources, body, frozen)
        except (TypeError, KeyError, RecursionError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed retirement journal") from exc
        if context(resources, inputs) != frozen:
            raise ValueError("Changed retirement receipt context")
        held.check()
    return body, publication


def prepare_retirement(
    resources,
    kind,
    inputs,
    *,
    protected_keys=(),
    reason,
    owner=None,
    max_bytes=MAX_BYTES,
    max_descriptor_bytes=MAX_DESCRIPTOR_BYTES,
):
    resources.lease.check()
    if (
        owner is not None
        and not _digest(owner)
        or type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_BYTES
        or type(max_descriptor_bytes) is not int
        or not 0 < max_descriptor_bytes <= MAX_DESCRIPTOR_BYTES
    ):
        raise ValueError("Invalid retirement owner")
    normalized = _inputs(kind, inputs)
    protected = _inputs(INTENT, dict(protected=protected_keys))["protected"]
    if type(reason) is not str or len(reason) > 1024 or not reason.strip():
        raise ValueError("Explicit retirement reason required")
    initial_engine = engine()
    inventory = retirement_inventory(
        resources, kind, inputs, protected_keys=protected_keys
    )
    with resources._connect() as db:
        records = allocations(db, inventory.target)
    body = dict(
        schema=1,
        inventory=asdict(inventory),
        allocations=records,
        protected=protected,
        reason=reason,
    )
    if owner is not None:
        body["owner"] = owner
    raw = _encode(body)
    if len(raw) > max_bytes:
        raise ValueError("Retirement journal byte limit exceeded")
    receipt_inputs = dict(
        schema=1,
        target=inventory.target.key,
        journal_sha256=hashlib.sha256(raw).hexdigest(),
        engine=initial_engine,
    )
    if owner is not None:
        receipt_inputs["owner"] = owner
    if max_bytes != MAX_BYTES:
        receipt_inputs["max_bytes"] = max_bytes
    if max_descriptor_bytes != MAX_DESCRIPTOR_BYTES:
        receipt_inputs["max_descriptor_bytes"] = max_descriptor_bytes
    relative = "artifacts/" + uuid.uuid4().hex
    token = resources.reserve(relative, max_bytes, "payload")
    path = resources.root / relative
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
        physical = fd_identity(os.fstat(output.fileno()))
        if _identity(path) != physical:
            raise ValueError("Retirement journal output inode changed")
        resources.settle(token)
        again = retirement_inventory(
            resources, kind, inputs, protected_keys=protected_keys
        )
        with resources._connect() as db:
            if allocations(db, inventory.target) != records:
                raise ValueError("Retirement allocations changed during preparation")
        if (
            again != inventory
            or engine() != initial_engine
            or _encode(inputs) != _encode(normalized)
            or _encode(protected_keys) != _encode(protected)
            or _identity(path) != physical
            or fd_identity(os.fstat(output.fileno())) != physical
            or file_hash(path) != receipt_inputs["journal_sha256"]
        ):
            raise ValueError("Retirement preparation changed")
        resources.lease.check()
        with resources._connect() as db:
            artifacts = PublishedArtifacts(resources)._records(db, [token])
        for kind in (INTENT, COMMITTED):
            descriptor = _encode(
                dict(
                    schema=1,
                    kind=kind,
                    inputs=receipt_inputs,
                    artifacts=[asdict(pin) for pin in artifacts],
                )
            )
            if len(descriptor) > max_descriptor_bytes:
                raise ValueError("Retirement publication descriptor limit exceeded")
        PublishedArtifacts(resources).publish(INTENT, receipt_inputs, [token])
        _sync(path.parent)
        load_journal(resources, receipt_inputs)
        if (
            _identity(path) != physical
            or fd_identity(os.fstat(output.fileno())) != physical
            or _encode(inputs) != _encode(normalized)
            or _encode(protected_keys) != _encode(protected)
        ):
            raise ValueError("Retirement final preparation context changed")
    return receipt_inputs
