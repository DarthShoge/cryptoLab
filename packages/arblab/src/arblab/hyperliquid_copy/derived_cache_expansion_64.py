"""Journaled, explicitly approved 16-to-64-GiB cache transition."""

from contextlib import closing, contextmanager, ExitStack
import hashlib
import json
import os
import sqlite3
import uuid

from .derived_cache_expansion import _baseline, _database_marker, _directories, _write
from .derived_cache_policy import EXPANDED_BYTES, _PinnedFile, open_expanded_cache
from .derived_cache_policy_64 import (
    EXPANDED_64_BYTES,
    KIND_64,
    RECEIPT_64_BYTES,
    _Expanded64Resources,
    _engine_64,
    _new_marker_64,
    _receipt_64,
    _validate_64,
    _verify_predecessor,
    open_expanded_64_cache,
)
from .derived_cache_resources import CacheResources, METADATA_BYTES, NAMESPACES
from .derived_publication import PublishedArtifacts, _encode


def prepare_expansion_64(
    resources, predecessor_receipt, *, approved_from, approved_to, approval
):
    """Publish a 64-GiB intent under the verified 16-GiB predecessor."""
    if (
        type(approved_from) is not int
        or approved_from != EXPANDED_BYTES
        or type(approved_to) is not int
        or approved_to != EXPANDED_64_BYTES
        or resources.limit != EXPANDED_BYTES
    ):
        raise ValueError("Expected explicitly approved 16-to-64-GiB expansion")
    verified = open_expanded_cache(
        resources.lease, resources.identity, predecessor_receipt
    )
    with (
        _directories(resources.lease) as (_, check_dirs),
        _PinnedFile(resources.marker, 4096) as marker,
    ):
        before = verified.audit()  # Includes streaming hashes of every retained payload.
        if before["reserved_bytes"]:
            raise ValueError("Resolve pending obligations before 64-GiB expansion")
        check_dirs()
        inputs = _validate_64(
            resources.lease,
            resources.identity,
            dict(
                schema=1,
                predecessor_receipt=predecessor_receipt,
                old_marker=json.loads(marker.read()),
                new_limit=EXPANDED_64_BYTES,
                approval=approval,
                engine=_engine_64(),
            ),
        )
        with resources._connect() as db:
            for (raw,) in db.execute("SELECT descriptor FROM publications"):
                if json.loads(raw).get("kind") == KIND_64:
                    raise ValueError("64-GiB expansion already prepared")
            baseline = _baseline(db)
        stage_path = f"staging/{uuid.uuid4().hex}"
        stage_token = resources.reserve(stage_path, 4096, "payload")
        receipt_path = f"artifacts/{uuid.uuid4().hex}"
        receipt_token = resources.reserve(receipt_path, RECEIPT_64_BYTES, "payload")
        check_dirs()
        raw_marker = json.dumps(_new_marker_64(inputs), sort_keys=True).encode()
        if len(raw_marker) > 4096:
            raise ValueError("64-GiB expansion marker too large")
        _write(resources.root / stage_path, raw_marker)
        check_dirs()
        stage = dict(
            path=stage_path,
            token=stage_token,
            bytes=len(raw_marker),
            sha256=hashlib.sha256(raw_marker).hexdigest(),
        )
        raw_receipt = _encode(dict(inputs=inputs, stage=stage, baseline=baseline))
        if len(raw_receipt) > RECEIPT_64_BYTES:
            raise ValueError("64-GiB expansion receipt too large")
        _write(resources.root / receipt_path, raw_receipt)
        check_dirs()
        resources.settle(receipt_token)
        marker.check()
        _validate_64(resources.lease, resources.identity, inputs)
        with resources._connect() as db:
            if _baseline(db, tokens=(stage_token, receipt_token)) != baseline:
                raise ValueError("Changed cache during 64-GiB expansion preparation")
        PublishedArtifacts(resources).publish(KIND_64, inputs, [receipt_token])
        _validate_64(resources.lease, resources.identity, inputs)
        check_dirs()
        marker.check()
        resources.lease.check()
        return inputs


class _Transition64View(CacheResources):
    def __init__(self, lease, identity, inputs, marker, database_marker):
        self._expected_marker = json.dumps(marker, sort_keys=True)
        self._database_marker = json.dumps(database_marker, sort_keys=True)
        self._allowed = {
            json.dumps(inputs["old_marker"], sort_keys=True),
            json.dumps(_new_marker_64(inputs), sort_keys=True),
        }
        super().__init__(lease, identity, limit_bytes=marker["limit_bytes"])

    @staticmethod
    def _arguments(lease, identity, limit_bytes):
        CacheResources._arguments(lease, identity, 8 * 1024**3)
        if type(limit_bytes) is not int or limit_bytes not in (
            EXPANDED_BYTES,
            EXPANDED_64_BYTES,
        ):
            raise ValueError("Invalid 64-GiB transition budget")

    def _metadata(self):
        actual = super()._metadata()
        if actual != self._expected_marker or actual not in self._allowed:
            raise ValueError("Changed 64-GiB transition marker")
        return self._database_marker

    @contextmanager
    def _connect(self):
        with super()._connect() as db:
            db.execute("PRAGMA query_only=ON")
            yield db


def _state_64(lease, identity, inputs):
    lease.check()
    allowed = {
        ".resource.lock",
        ".initialized",
        "resources.sqlite3",
        "resources.sqlite3-journal",
        *NAMESPACES,
    }
    if any(path.name not in allowed for path in lease.root.iterdir()):
        raise ValueError("Unknown 64-GiB expansion root file")
    with _PinnedFile(lease.root / ".initialized", 4096) as pin:
        marker = json.loads(pin.read())
        old, new = inputs["old_marker"], _new_marker_64(inputs)
        if _encode(marker) not in (_encode(old), _encode(new)):
            raise ValueError("Unrecognized 64-GiB expansion marker")
        database = _database_marker(lease.root / "resources.sqlite3")
        pair = (_encode(database), _encode(marker))
        if pair not in (
            (_encode(old), _encode(old)),
            (_encode(new), _encode(old)),
            (_encode(new), _encode(new)),
        ):
            raise ValueError("Unrecognized 64-GiB expansion metadata state")
        view = _Transition64View(lease, identity, inputs, marker, database)
        _verify_predecessor(view, identity, inputs["predecessor_receipt"])
        publication, receipt = _receipt_64(view, inputs)
        stage = receipt["stage"]
        path = view._path(stage["path"])
        stage_present = path.exists() or path.is_symlink()
        if stage_present != (marker == old):
            raise ValueError("Unrecognized 64-GiB expansion staging state")
        with view._connect() as db:
            row = db.execute(
                "SELECT path,maximum,purpose,state,bytes,sha256 FROM allocations WHERE token=?",
                (stage["token"],),
            ).fetchone()
            expected = (stage["path"], 4096, "payload", "pending", None, None)
            if row != expected and not (row is None and marker == new):
                raise ValueError("Changed 64-GiB expansion staging obligation")
            actual = _baseline(
                db,
                tokens=(stage["token"], publication.artifacts[0].token),
                publication_key=publication.key,
            )
            if actual != receipt["baseline"]:
                raise ValueError("Changed 64-GiB expansion baseline")
        pin.check()
        lease.check()
        return marker, database, receipt, publication, row is not None


def apply_expansion_64(lease, identity, predecessor_receipt, receipt_inputs):
    """Complete only the exact receipt-chained transition, including recovery."""
    inputs = _validate_64(lease, identity, receipt_inputs)
    if _encode(inputs["predecessor_receipt"]) != _encode(predecessor_receipt):
        raise ValueError("64-GiB predecessor receipt mismatch")
    old, new = inputs["old_marker"], _new_marker_64(inputs)
    with _directories(lease) as (directories, check_dirs), ExitStack() as stack:
        marker, database, receipt, publication, pending = _state_64(
            lease, identity, inputs
        )
        marker_pin = stack.enter_context(_PinnedFile(lease.root / ".initialized", 4096))
        stage_pin = None
        if marker == old:
            stage_pin = stack.enter_context(
                _PinnedFile(lease.root / receipt["stage"]["path"], 4096)
            )
            raw = stage_pin.read()
            if (
                len(raw) != receipt["stage"]["bytes"]
                or hashlib.sha256(raw).hexdigest() != receipt["stage"]["sha256"]
                or _encode(json.loads(raw)) != _encode(new)
            ):
                raise ValueError("Changed 64-GiB expansion staging marker")
        if _encode(_validate_64(lease, identity, receipt_inputs)) != _encode(inputs):
            raise ValueError("Changed caller 64-GiB expansion inputs")
        if _state_64(lease, identity, inputs) != (
            marker,
            database,
            receipt,
            publication,
            pending,
        ):
            raise ValueError("Changed cache before 64-GiB expansion commit")
        if (
            _encode(receipt_inputs) != _encode(inputs)
            or _encode(predecessor_receipt) != _encode(inputs["predecessor_receipt"])
        ):
            raise ValueError("Changed caller 64-GiB expansion inputs")
        check_dirs()
        marker_pin.check()
        if stage_pin:
            stage_pin.check()
        if database == old:
            path = lease.root / "resources.sqlite3"
            with _PinnedFile(path, METADATA_BYTES // 2) as db_pin:
                with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as db, db:
                    db.execute("PRAGMA synchronous=FULL")
                    db.execute("PRAGMA max_page_count=4000")
                    db.execute("BEGIN IMMEDIATE")
                    db_pin.check()
                    if db.execute("SELECT value FROM metadata").fetchall() != [
                        (json.dumps(old, sort_keys=True),)
                    ]:
                        raise ValueError("Changed database before 64-GiB expansion commit")
                    db.execute(
                        "UPDATE metadata SET value=?", (json.dumps(new, sort_keys=True),)
                    )
                    check_dirs()
                    marker_pin.check()
                    stage_pin.check()
        if stage_pin:
            os.replace(
                receipt["stage"]["path"].split("/")[1],
                ".initialized",
                src_dir_fd=directories[lease.root / "staging"],
                dst_dir_fd=directories[lease.root],
            )
        os.fsync(directories[lease.root])
        os.fsync(directories[lease.root / "staging"])
        check_dirs()
        _, _, _, _, pending = _state_64(lease, identity, inputs)
        if pending:
            resources = _Expanded64Resources(
                lease, identity, limit_bytes=EXPANDED_64_BYTES
            )
            resources.release_missing(receipt["stage"]["token"])
        check_dirs()
    return open_expanded_64_cache(
        lease, identity, predecessor_receipt, receipt_inputs
    )
