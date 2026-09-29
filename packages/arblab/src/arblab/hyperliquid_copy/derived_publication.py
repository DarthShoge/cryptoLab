"""Bounded content identities and atomic visibility for immutable cache files."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from .derived_cache_resources import _regular, _sync
from .download import file_hash

MAX_INPUT_BYTES = 64 * 1024
MAX_NODES = 8192
MAX_DESCRIPTOR_BYTES = 1024**2
MAX_PUBLICATIONS = 10_000
MAX_ARTIFACTS = 5000


def _encode(value):
    # Preserve JSON scalar types: the source semantic hash intentionally turns
    # floats into decimal strings, which is unsuitable for typed cache inputs.
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _inputs(kind, inputs):
    if (
        not isinstance(kind, str)
        or not re.fullmatch(r"[a-z][a-z0-9_-]{0,79}", kind)
        or type(inputs) is not dict
    ):
        raise ValueError("Invalid publication kind/inputs")
    stack, count, text_bytes = [(inputs, 0)], 0, 0
    while stack:
        value, depth = stack.pop()
        count += 1
        if count > MAX_NODES or depth > 16:
            raise ValueError("Publication input structure limit exceeded")
        if type(value) in (dict, list, tuple):
            if len(value) + count + len(stack) > MAX_NODES:
                raise ValueError("Publication input node limit exceeded")
            if isinstance(value, dict):
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise ValueError("Expected string input keys")
                    stack.extend(((key, depth + 1), (child, depth + 1)))
            else:
                stack.extend((v, depth + 1) for v in value)
        elif type(value) is str:
            if len(value) > MAX_INPUT_BYTES:
                raise ValueError("Publication input byte limit exceeded")
            text_bytes += len(value.encode())
            if text_bytes > MAX_INPUT_BYTES:
                raise ValueError("Publication input byte limit exceeded")
        elif type(value) is int:
            if value.bit_length() > 256:
                raise ValueError("Publication integer limit exceeded")
        elif value is not None and type(value) not in (bool, float):
            raise ValueError("Unsupported publication input type")
    encoded = _encode(inputs)
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValueError("Publication input byte limit exceeded")
    return json.loads(encoded)


def publication_key(kind, inputs):
    normalized = _inputs(kind, inputs)
    return hashlib.sha256(
        _encode(dict(schema=1, kind=kind, inputs=normalized))
    ).hexdigest()


@dataclass(frozen=True)
class ArtifactPin:
    token: str
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Publication:
    key: str
    artifacts: tuple[ArtifactPin, ...]


def _tokens(tokens):
    if (
        type(tokens) not in (tuple, list)
        or not 1 <= len(tokens) <= MAX_ARTIFACTS
        or any(
            type(t) is not str or not re.fullmatch("[a-f0-9]{32}", t) for t in tokens
        )
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("Expected bounded distinct artifact tokens")
    return tuple(tokens)


class PublishedArtifacts:
    def __init__(self, resources):
        resources.lease.check()
        self.resources = resources

    def _records(self, db, tokens, *, sync=False):
        records = []
        for token in _tokens(tokens):
            row = db.execute(
                "SELECT path,bytes,sha256,purpose,state FROM allocations WHERE token=?",
                (token,),
            ).fetchone()
            if (
                row is None
                or row[3:] != ("payload", "retained")
                or not row[0].startswith("artifacts/")
                or type(row[1]) is not int
                or row[1] < 0
                or not isinstance(row[2], str)
                or not re.fullmatch("[a-f0-9]{64}", row[2])
            ):
                raise ValueError("Publication requires finalized artifact payloads")
            path = self.resources._path(row[0])
            if _regular(path).st_size != row[1]:
                raise ValueError("Changed published artifact size")
            if sync:
                _sync(path)
                _sync(path.parent)
            if file_hash(path) != row[2] or _regular(path).st_size != row[1]:
                raise ValueError("Changed published artifact identity")
            records.append(ArtifactPin(token, row[0], row[1], row[2]))
        return tuple(records)

    def _read(self, db, key, kind, normalized):
        size = db.execute(
            "SELECT length(CAST(descriptor AS BLOB)) FROM publications WHERE key=?",
            (key,),
        ).fetchone()
        if size is None:
            return None
        if type(size[0]) is not int or not 0 < size[0] <= MAX_DESCRIPTOR_BYTES:
            raise ValueError("Publication descriptor byte limit exceeded")
        raw, digest = db.execute(
            "SELECT descriptor,sha256 FROM publications WHERE key=?", (key,)
        ).fetchone()
        if (
            not isinstance(raw, str)
            or hashlib.sha256(raw.encode()).hexdigest() != digest
        ):
            raise ValueError("Changed publication descriptor")
        try:
            data = json.loads(raw)
            if (
                type(data) is not dict
                or set(data) != {"schema", "kind", "inputs", "artifacts"}
                or data["schema"] != 1
                or data["kind"] != kind
                or data["inputs"] != normalized
                or type(data["artifacts"]) is not list
                or not 1 <= len(data["artifacts"]) <= MAX_ARTIFACTS
            ):
                raise ValueError("Invalid publication descriptor")
            records = self._records(db, [item["token"] for item in data["artifacts"]])
            expected = dict(
                schema=1,
                kind=kind,
                inputs=normalized,
                artifacts=[asdict(record) for record in records],
            )
            if _encode(data) != _encode(expected):
                raise ValueError("Publication artifact references changed")
        except (TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid publication descriptor") from exc
        return Publication(key, records)

    def lookup(self, kind, inputs):
        self.resources.lease.check()
        normalized = _inputs(kind, inputs)
        key = publication_key(kind, normalized)
        with self.resources._connect() as db:
            return self._read(db, key, kind, normalized)

    def publish(self, kind, inputs, tokens):
        self.resources.lease.check()
        normalized, tokens = _inputs(kind, inputs), _tokens(tokens)
        key = publication_key(kind, normalized)
        with self.resources._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            self.resources._audit(db)
            records = self._records(db, tokens, sync=True)
            data = dict(
                schema=1,
                kind=kind,
                inputs=normalized,
                artifacts=[asdict(record) for record in records],
            )
            raw = _encode(data)
            if len(raw) > MAX_DESCRIPTOR_BYTES:
                raise ValueError("Publication descriptor byte limit exceeded")
            existing = self._read(db, key, kind, normalized)
            result = Publication(key, records)
            if existing is not None:
                if existing != result:
                    raise ValueError("Publication key already has different artifacts")
                return existing
            if (
                db.execute("SELECT count(*) FROM publications").fetchone()[0]
                >= MAX_PUBLICATIONS
            ):
                raise ValueError("Publication row limit exceeded")
            db.execute(
                "INSERT INTO publications VALUES (?,?,?)",
                (key, raw.decode(), hashlib.sha256(raw).hexdigest()),
            )
            return result
