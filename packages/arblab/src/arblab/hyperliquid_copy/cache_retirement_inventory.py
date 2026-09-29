"""Read-only publication ownership evidence; not a semantic expiration policy."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat

from .derived_cache_resources import _regular, METADATA_BYTES
from .derived_cache_policy import _PinnedFile
from .derived_publication import (
    ArtifactPin,
    Publication,
    MAX_ARTIFACTS,
    MAX_DESCRIPTOR_BYTES,
    MAX_PUBLICATIONS,
    _encode,
    _inputs,
    publication_key,
)
from .download import file_hash
from .query_directory_pin import pin_directory

PROTECTED_KINDS = frozenset(
    {
        "approved_cache_expansion",
        "saved_feature_ranking",
        "candidate_capacity",
        "qualified_feature_anchor",
    }
)


def _engine():
    root = Path(__file__).parent
    return {
        name: file_hash(root / name)
        for name in (
            "cache_retirement_inventory.py",
            "derived_cache_resources.py",
            "derived_publication.py",
            "derived_cache_lease.py",
            "query_directory_pin.py",
            "download.py",
            "derived_cache_policy.py",
        )
    }


def _digest(value):
    return type(value) is str and re.fullmatch("[a-f0-9]{64}", value) is not None


def _identity(path):
    info = _regular(path)
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _namespace(resources):
    result = []
    for path in (resources.root, resources.root / "artifacts"):
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("Unsafe retirement namespace")
        result.append((info.st_dev, info.st_ino))
    return tuple(result)


@dataclass(frozen=True)
class RetirementInventory:
    target: Publication
    descriptor: str
    exclusive: tuple[str, ...]
    shared: tuple[str, ...]
    catalogue_digest: str
    marker: str
    namespace: tuple
    files: tuple
    engine: str


def _descriptor(resources, db, key, raw, digest):
    if (
        type(raw) is not str
        or not 0 < len(raw.encode()) <= MAX_DESCRIPTOR_BYTES
        or not _digest(key)
        or not _digest(digest)
        or hashlib.sha256(raw.encode()).hexdigest() != digest
    ):
        raise ValueError("Invalid retirement catalogue descriptor/hash")
    try:
        data = json.loads(raw)
        if (
            type(data) is not dict
            or set(data) != {"schema", "kind", "inputs", "artifacts"}
            or type(data["schema"]) is not int
            or data["schema"] != 1
            or _encode(data).decode() != raw
            or publication_key(data["kind"], data["inputs"]) != key
            or type(data["artifacts"]) is not list
            or not 1 <= len(data["artifacts"]) <= MAX_ARTIFACTS
        ):
            raise ValueError("Invalid retirement publication metadata")
        pins, seen = [], set()
        for row in data["artifacts"]:
            if (
                type(row) is not dict
                or set(row) != {"token", "path", "bytes", "sha256"}
                or type(row["token"]) is not str
                or not re.fullmatch("[a-f0-9]{32}", row["token"])
                or row["token"] in seen
                or not _digest(row["sha256"])
                or type(row["bytes"]) is not int
                or not 0 <= row["bytes"] <= resources.limit
            ):
                raise ValueError("Invalid retirement artifact pin")
            resources._path(row["path"])
            if not row["path"].startswith("artifacts/"):
                raise ValueError("Non-artifact retirement reference")
            record = db.execute(
                "SELECT path,bytes,sha256,purpose,state,maximum FROM allocations WHERE token=?",
                (row["token"],),
            ).fetchone()
            if (
                record is None
                or record[:5]
                != (row["path"], row["bytes"], row["sha256"], "payload", "retained")
                or type(record[5]) is not int
                or not max(1, row["bytes"]) <= record[5] <= resources.limit
            ):
                raise ValueError("Retirement reference/allocation mismatch")
            seen.add(row["token"])
            pins.append(ArtifactPin(**row))
        return Publication(key, tuple(pins))
    except (TypeError, KeyError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed retirement descriptor") from exc


def _rows(db):
    if db.execute("SELECT count(*) FROM publications").fetchone()[0] > MAX_PUBLICATIONS:
        raise ValueError("Retirement catalogue record limit exceeded")
    return db.execute(
        "SELECT key, CASE WHEN length(CAST(descriptor AS BLOB)) BETWEEN 1 AND ? "
        "THEN descriptor END, sha256 FROM publications ORDER BY key",
        (MAX_DESCRIPTOR_BYTES,),
    )


def retirement_inventory(resources, kind, inputs, *, protected_keys=()):
    resources.lease.check()
    normalized = _inputs(kind, inputs)
    if (
        type(protected_keys) not in (list, tuple)
        or len(protected_keys) > MAX_PUBLICATIONS
        or any(not _digest(k) for k in protected_keys)
    ):
        raise ValueError("Expected bounded protected publication keys")
    protected = tuple(protected_keys)
    _inputs("retirement_inventory", dict(protected=protected))
    key = publication_key(kind, normalized)
    if (
        key in protected
        or kind in PROTECTED_KINDS
        or kind.startswith("cache_retirement")
    ):
        raise ValueError("Protected publication cannot be retired")
    frozen_engine = _encode(_engine()).decode()
    marker, namespace = resources._metadata(), _namespace(resources)
    shared, digest, target, raw_target = set(), hashlib.sha256(), None, None
    with (
        pin_directory(resources.root, namespace[0]),
        pin_directory(resources.root / "artifacts", namespace[1]),
        _PinnedFile(resources.path, METADATA_BYTES // 2) as database_pin,
        _PinnedFile(resources.marker, 4096) as marker_pin,
    ):
        with resources._connect() as db:
            version = db.execute("PRAGMA data_version").fetchone()
            row = db.execute(
                "SELECT key, CASE WHEN length(CAST(descriptor AS BLOB)) BETWEEN 1 AND ? "
                "THEN descriptor END, sha256 FROM publications WHERE key=?",
                (MAX_DESCRIPTOR_BYTES, key),
            ).fetchone()
            if row is None:
                raise ValueError("Retirement target publication is missing")
            target = _descriptor(resources, db, *row)
            raw_target = row[1]
            tokens = {pin.token for pin in target.artifacts}
            for record in _rows(db):
                publication = _descriptor(resources, db, *record)
                digest.update(_encode(record))
                digest.update(b"\n")
                if publication.key != key:
                    shared.update(
                        pin.token
                        for pin in publication.artifacts
                        if pin.token in tokens
                    )
            files = []
            for pin in target.artifacts:
                path = resources.root / pin.path
                identity = _identity(path)
                if (
                    identity[2] != pin.bytes
                    or file_hash(path) != pin.sha256
                    or _identity(path) != identity
                ):
                    raise ValueError("Retirement target payload changed")
                files.append((pin.token, identity))
            if (
                _encode(_engine()).decode() != frozen_engine
                or _encode(inputs) != _encode(normalized)
                or tuple(protected_keys) != protected
                or resources._metadata() != marker
                or _namespace(resources) != namespace
            ):
                raise ValueError("Retirement inventory context changed")
            if tuple(
                (p.token, _identity(resources.root / p.path)) for p in target.artifacts
            ) != tuple(files):
                raise ValueError("Retirement payload identity changed")
            resources.lease.check()
            if db.execute("PRAGMA data_version").fetchone() != version:
                raise ValueError("Retirement catalogue changed during inventory")
            database_pin.check()
            marker_pin.check()
    return RetirementInventory(
        target,
        raw_target,
        tuple(p.token for p in target.artifacts if p.token not in shared),
        tuple(p.token for p in target.artifacts if p.token in shared),
        digest.hexdigest(),
        marker,
        namespace,
        tuple(files),
        frozen_engine,
    )
