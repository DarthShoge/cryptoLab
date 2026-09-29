"""Retain verified capacity evidence without a live cache or raw-query dependency."""

from dataclasses import asdict, dataclass
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import re

from .archive_cache import _safe
from .candidate_capacity import (
    CandidateCapacity,
    build_candidate_capacity,
    KIND,
    MAX_BYTES,
    MAX_ROWS,
    _engine as capacity_engine,
)
from .candidate_capacity_records import read_counts
from .contracts import utc
from .derived_publication import _encode, publication_key
from .download import file_hash
from .lab_config import day
from .proxy_compact import _sync
from .qualified_source_session import _identity, _engine as source_engine
from .registration_provenance import verify_registration_provenance, NAMES

DESCRIPTOR = "candidate_capacity.json"
PAYLOAD = "candidate_capacity.parquet"
MAX_DESCRIPTOR = 1024**2


def _reader_engine():
    return dict(
        reader=file_hash(Path(__file__)),
        provenance=file_hash(Path(__file__).with_name("registration_provenance.py")),
        capacity=capacity_engine(),
        source=source_engine(),
        schedule={
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "capacity_schedule.py",
                "lab_pipeline.py",
                "proxy_schedule.py",
            )
        },
    )


def export_registration_capacity(resources, source, stage):
    engine = _reader_engine()
    stage = Path(stage)
    _safe(stage)
    if not stage.is_dir():
        raise ValueError("Existing unpublished capacity staging required")
    inputs = build_candidate_capacity(resources, source)
    counts = CandidateCapacity(resources, source, inputs)
    pin = counts.publication.artifacts[0]
    target = stage / PAYLOAD
    with counts.path.open("rb") as reader, target.open("xb") as writer:
        remaining = pin.bytes
        while remaining:
            block = reader.read(min(remaining, 1024**2))
            if not block:
                raise ValueError("Capacity copy source truncated")
            writer.write(block)
            remaining -= len(block)
        if reader.read(1):
            raise ValueError("Capacity copy source grew")
    counts._verify_artifact()
    identity = _identity(target)
    if file_hash(target) != pin.sha256:
        raise ValueError("Capacity copy checksum mismatch")
    read_counts(target, source, max_bytes=MAX_BYTES, max_rows=MAX_ROWS)
    descriptor = dict(
        schema=1,
        inputs=inputs,
        publication=asdict(counts.publication),
        payload=dict(name=PAYLOAD, bytes=pin.bytes, sha256=pin.sha256),
    )
    raw = _encode(descriptor)
    if len(raw) > MAX_DESCRIPTOR:
        raise ValueError("Capacity descriptor byte limit")
    path = stage / DESCRIPTOR
    with path.open("xb") as writer:
        writer.write(raw)
    descriptor_identity = _identity(path)
    _sync(target)
    _sync(path)
    _sync(stage)
    counts._verify_artifact()
    digest = hashlib.sha256(raw).hexdigest()
    if _reader_engine() != engine or file_hash(path) != digest:
        raise ValueError("Capacity export descriptor/engine changed")
    if _identity(target) != identity or _identity(path) != descriptor_identity:
        raise ValueError("Capacity copy changed during export")
    source.verify_identity()
    resources.lease.check()
    return dict(name=DESCRIPTOR, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _json(path, maximum):
    _identity(path)
    with path.open("rb") as reader:
        raw = reader.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise ValueError("Registered capacity metadata byte limit")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid registered capacity metadata") from exc


@dataclass(frozen=True)
class _Source:
    origin: object
    finish: object
    coins: tuple


class RegisteredCapacity:
    def __init__(self, directory, metadata):
        self._engine = _reader_engine()
        self.directory, self.metadata = Path(directory), metadata
        self._provenance_identities = self._provenance_snapshot()
        self._frozen = _encode(metadata)
        entry = metadata.get("candidate_capacity")
        if (
            type(entry) is not dict
            or set(entry) != {"name", "bytes", "sha256"}
            or entry["name"] != DESCRIPTOR
            or type(entry["bytes"]) is not int
            or not 0 < entry["bytes"] <= MAX_DESCRIPTOR
            or type(entry["sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", entry["sha256"])
        ):
            raise ValueError("Invalid registered capacity reference")
        verify_registration_provenance(self.directory, metadata)
        self.path = self.directory / DESCRIPTOR
        descriptor_identity = _identity(self.path)
        value = _json(self.path, MAX_DESCRIPTOR)
        if (
            self.path.stat().st_size != entry["bytes"]
            or file_hash(self.path) != entry["sha256"]
            or type(value) is not dict
            or set(value) != {"schema", "inputs", "publication", "payload"}
            or type(value["schema"]) is not int
            or value["schema"] != 1
        ):
            raise ValueError("Registered capacity descriptor changed")
        report = _json(self.directory / "activity_qualification.json", 16 * 1024**2)
        original = _json(self.directory / "activity_source.json", 16 * 1024**2)
        self.source = _Source(
            day(report["source_start"]),
            day(report["source_end"]),
            tuple(report["coins"]),
        )
        expected_source = dict(
            schema=1,
            pin=original["qualification"],
            origin=report["source_start"],
            finish=report["source_end"],
            coins=report["coins"],
            files=len(report["files"]),
            membership_sha256=hashlib.sha256(_encode(report["files"])).hexdigest(),
            engine=source_engine(),
        )
        inputs = dict(
            source=expected_source,
            engine=capacity_engine(),
            max_bytes=MAX_BYTES,
            max_rows=MAX_ROWS,
        )
        payload, publication = value["payload"], value["publication"]
        if (
            _encode(value["inputs"]) != _encode(inputs)
            or not 0 < (self.source.finish - self.source.origin).days <= 732
            or list(self.source.coins) != metadata["coins"]
            or expected_source["pin"]["sha256"]
            != metadata["activity_provenance"]["source_manifest_hash"]
            or type(payload) is not dict
            or set(payload) != {"name", "bytes", "sha256"}
            or payload["name"] != PAYLOAD
            or type(payload["bytes"]) is not int
            or not 0 < payload["bytes"] <= MAX_BYTES
            or type(publication) is not dict
            or set(publication) != {"key", "artifacts"}
            or publication["key"] != publication_key(KIND, inputs)
            or type(publication["artifacts"]) is not list
            or len(publication["artifacts"]) != 1
        ):
            raise ValueError("Registered capacity source/publication binding mismatch")
        pin = publication["artifacts"][0]
        if (
            type(pin) is not dict
            or set(pin) != {"token", "path", "bytes", "sha256"}
            or type(pin["token"]) is not str
            or not re.fullmatch("[a-f0-9]{32}", pin["token"])
            or type(pin["path"]) is not str
            or not re.fullmatch(r"artifacts/[a-f0-9]{32}\.parquet", pin["path"])
            or type(pin["sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", pin["sha256"])
            or (pin["bytes"], pin["sha256"]) != (payload["bytes"], payload["sha256"])
        ):
            raise ValueError("Invalid original capacity artifact pin")
        self.payload, self._payload_pin = self.directory / PAYLOAD, payload
        payload_identity = _identity(self.payload)
        if (
            self.payload.stat().st_size != payload["bytes"]
            or file_hash(self.payload) != payload["sha256"]
        ):
            raise ValueError("Registered capacity payload changed")
        self._rows = read_counts(
            self.payload, self.source, max_bytes=MAX_BYTES, max_rows=MAX_ROWS
        )
        self._identities = descriptor_identity, payload_identity
        self._verify()

    def _verify(self):
        verify_registration_provenance(self.directory, self.metadata)
        if _reader_engine() != self._engine or _encode(self.metadata) != self._frozen:
            raise ValueError("Registered capacity context changed")
        if (
            (_identity(self.path), _identity(self.payload)) != self._identities
            or self._provenance_snapshot() != self._provenance_identities
        ):
            raise ValueError("Registered capacity file changed")

    def _provenance_snapshot(self):
        return tuple(_identity(self.directory / name) for name in sorted(NAMES))

    def upper_bound(self, decision, coins, scope):
        self._verify()
        decision = utc(decision)
        if (
            not self.source.origin <= decision <= self.source.finish
            or any(
                (decision.hour, decision.minute, decision.second, decision.microsecond)
            )
            or type(coins) not in (list, tuple)
            or any(type(c) is not str for c in coins)
            or len(set(coins)) != len(coins)
            or not set(coins) <= set(self.source.coins)
            or scope is not None
            and scope not in coins
        ):
            raise ValueError("Invalid registered capacity query")
        totals = {coin: 0 for coin in (*self.source.coins, None)}
        for date, coin, entrants in self._rows:
            if date < decision.date():
                totals[coin] += entrants
        return (
            totals[scope]
            if scope is not None
            else min(sum(totals[c] for c in coins), totals[None])
        )

    def ranking_rows(self, config, coins):
        """Bound a whole run with two integrity checks and a bounded prefix index."""
        from .capacity_schedule import ranking_upper_bound

        frozen = _encode(config.to_dict())
        if type(coins) not in (list, tuple):
            raise ValueError("Invalid registered capacity markets")
        selected = tuple(coins)
        self._verify()
        if _encode(config.to_dict()) != frozen or tuple(coins) != selected:
            raise ValueError("Registered capacity run context changed")
        if (
            type(coins) not in (list, tuple)
            or any(type(c) is not str for c in coins)
            or len(set(coins)) != len(coins)
            or not set(coins) <= set(self.source.coins)
            or not self.source.origin
            <= day(config.start)
            < day(config.end)
            <= self.source.finish
        ):
            raise ValueError("Invalid registered capacity run")
        index = {coin: ([], [0]) for coin in (*self.source.coins, None)}
        for date, coin, entrants in self._rows:
            dates, totals = index[coin]
            dates.append(date)
            totals.append(totals[-1] + entrants)

        def count(at, markets, scope):
            def cumulative(coin):
                dates, totals = index[coin]
                return totals[bisect_left(dates, at.date())]

            return (
                cumulative(scope)
                if scope is not None
                else min(sum(cumulative(c) for c in markets), cumulative(None))
            )

        result = ranking_upper_bound(config, selected, count)
        self._verify()
        if _encode(config.to_dict()) != frozen or tuple(coins) != selected:
            raise ValueError("Registered capacity run context changed")
        return result


def load_registration_capacity(directory, metadata):
    if "candidate_capacity" not in metadata:
        return None
    return RegisteredCapacity(directory, metadata)
