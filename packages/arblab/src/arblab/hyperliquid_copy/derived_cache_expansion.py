"""Journaled, explicitly approved 8-to-16-GiB cache metadata transition."""

from contextlib import closing, contextmanager, ExitStack
import hashlib
import json
import os
import sqlite3
import uuid

from .derived_cache_policy import (
    EXPANDED_BYTES,
    KIND,
    RECEIPT_BYTES,
    _PinnedFile,
    _engine,
    _new_marker,
    _receipt,
    _validate,
    open_expanded_cache,
)
from .derived_cache_resources import (
    CacheResources,
    MAX_BYTES,
    METADATA_BYTES,
    NAMESPACES,
    _regular,
    _sync,
)
from .derived_publication import PublishedArtifacts, _encode


def _baseline(db, *, tokens=(), publication_key=None):
    result = {}
    for table, order in (("allocations", "token"), ("publications", "key")):
        count, digest = 0, hashlib.sha256()
        for row in db.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            if row[0] in tokens or (
                table == "publications" and row[0] == publication_key
            ):
                continue
            count += 1
            if count > 100_000:
                raise ValueError("Expansion baseline row limit exceeded")
            raw = _encode(row)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        result[table] = [count, digest.hexdigest()]
    return result


def _write(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _sync(path.parent)


@contextmanager
def _directories(lease):
    with ExitStack() as stack:
        directories = {}
        for path in (lease.root, lease.root / "staging", lease.root / "artifacts"):
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
            stack.callback(os.close, fd)
            directories[path] = fd

        def check():
            lease.check()
            for path, fd in directories.items():
                current, held = path.lstat(), os.fstat(fd)
                if (current.st_dev, current.st_ino) != (held.st_dev, held.st_ino):
                    raise ValueError("Changed expansion directory")

        check()
        yield directories, check
        check()


def prepare_expansion(resources, *, approved_from, approved_to, approval):
    """Publish migration intent under the old cap; unfinished work stays charged."""
    if (
        type(approved_from) is not int
        or approved_from != MAX_BYTES
        or type(approved_to) is not int
        or approved_to != EXPANDED_BYTES
        or resources.limit != MAX_BYTES
    ):
        raise ValueError("Expected explicitly approved 8-to-16-GiB expansion")
    resources.lease.check()
    with (
        _directories(resources.lease) as (_, check_dirs),
        _PinnedFile(resources.marker, 4096) as marker,
    ):
        before = resources.audit()
        if before["reserved_bytes"]:
            raise ValueError("Resolve pending obligations before expansion")
        check_dirs()
        inputs = dict(
            schema=1,
            old_marker=json.loads(marker.read()),
            new_limit=EXPANDED_BYTES,
            approval=approval,
            engine=_engine(),
        )
        inputs = _validate(resources.lease, resources.identity, inputs)
        with resources._connect() as db:
            for (raw,) in db.execute("SELECT descriptor FROM publications"):
                if json.loads(raw).get("kind") == KIND:
                    raise ValueError("Expansion already prepared")
            baseline = _baseline(db)
        stage_path = f"staging/{uuid.uuid4().hex}"
        stage_token = resources.reserve(stage_path, 4096, "payload")
        receipt_path = f"artifacts/{uuid.uuid4().hex}"
        receipt_token = resources.reserve(receipt_path, RECEIPT_BYTES, "payload")
        check_dirs()
        raw_marker = json.dumps(_new_marker(inputs), sort_keys=True).encode()
        if len(raw_marker) > 4096:
            raise ValueError("Expansion marker too large")
        _write(resources.root / stage_path, raw_marker)
        check_dirs()
        stage = dict(
            path=stage_path,
            token=stage_token,
            bytes=len(raw_marker),
            sha256=hashlib.sha256(raw_marker).hexdigest(),
        )
        raw_receipt = _encode(dict(inputs=inputs, stage=stage, baseline=baseline))
        if len(raw_receipt) > RECEIPT_BYTES:
            raise ValueError("Expansion receipt too large")
        _write(resources.root / receipt_path, raw_receipt)
        check_dirs()
        resources.settle(receipt_token)
        marker.check()
        _validate(resources.lease, resources.identity, inputs)
        with resources._connect() as db:
            if _baseline(db, tokens=(stage_token, receipt_token)) != baseline:
                raise ValueError("Changed cache during expansion preparation")
        PublishedArtifacts(resources).publish(KIND, inputs, [receipt_token])
        _validate(resources.lease, resources.identity, inputs)
        check_dirs()
        marker.check()
        resources.lease.check()
        return inputs


class _TransitionView(CacheResources):
    """Private read-only audit view for one receipt-pinned DB/marker split."""

    def __init__(self, lease, identity, inputs, marker, database_marker):
        self._expected_marker = json.dumps(marker, sort_keys=True)
        self._database_marker = json.dumps(database_marker, sort_keys=True)
        self._allowed = {
            json.dumps(inputs["old_marker"], sort_keys=True),
            json.dumps(_new_marker(inputs), sort_keys=True),
        }
        super().__init__(lease, identity, limit_bytes=marker["limit_bytes"])

    @staticmethod
    def _arguments(lease, identity, limit_bytes):
        CacheResources._arguments(lease, identity, MAX_BYTES)
        if type(limit_bytes) is not int or limit_bytes not in (
            MAX_BYTES,
            EXPANDED_BYTES,
        ):
            raise ValueError("Invalid transition budget")

    def _metadata(self):
        actual = super()._metadata()
        if actual != self._expected_marker or actual not in self._allowed:
            raise ValueError("Changed transition marker")
        return self._database_marker

    @contextmanager
    def _connect(self):
        with super()._connect() as db:
            db.execute("PRAGMA query_only=ON")
            yield db


def _database_marker(path):
    with _PinnedFile(path, METADATA_BYTES // 2) as pin:
        # SQLite must be able to roll back its own hot journal after process
        # death. Bound/pin both metadata files before allowing native recovery;
        # execute no application writes through this bootstrap connection.
        journal = path.with_name(path.name + "-journal")
        if journal.exists() or journal.is_symlink():
            if _regular(journal).st_size > METADATA_BYTES // 2:
                raise ValueError("Oversized expansion rollback journal")
        with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as db:
            values = db.execute("SELECT value FROM metadata LIMIT 2").fetchall()
            if (
                len(values) != 1
                or type(values[0][0]) is not str
                or len(values[0][0]) > 4096
            ):
                raise ValueError("Invalid expansion database metadata")
        # Native rollback legitimately changes size/timestamps, never identity.
        current, held = _regular(path), os.fstat(pin.fd)
        if (
            (current.st_dev, current.st_ino) != pin.identity[:2]
            or (held.st_dev, held.st_ino) != pin.identity[:2]
            or held.st_nlink != 1
            or not 0 < current.st_size <= METADATA_BYTES // 2
        ):
            raise ValueError("Changed expansion database identity")
        return json.loads(values[0][0])


def _state(lease, identity, inputs):
    lease.check()
    allowed = {
        ".resource.lock",
        ".initialized",
        "resources.sqlite3",
        "resources.sqlite3-journal",
        *NAMESPACES,
    }
    if any(p.name not in allowed for p in lease.root.iterdir()):
        raise ValueError("Unknown expansion root file")
    with _PinnedFile(lease.root / ".initialized", 4096) as pin:
        marker = json.loads(pin.read())
        old, new = inputs["old_marker"], _new_marker(inputs)
        if _encode(marker) not in (_encode(old), _encode(new)):
            raise ValueError("Unrecognized expansion marker")
        database = _database_marker(lease.root / "resources.sqlite3")
        pair = (_encode(database), _encode(marker))
        if pair not in (
            (_encode(old), _encode(old)),
            (_encode(new), _encode(old)),
            (_encode(new), _encode(new)),
        ):
            raise ValueError("Unrecognized expansion metadata state")
        view = _TransitionView(lease, identity, inputs, marker, database)
        publication, receipt = _receipt(view, inputs)
        stage = receipt["stage"]
        path = view._path(stage["path"])
        stage_present = path.exists() or path.is_symlink()
        if stage_present != (marker == old):
            raise ValueError("Unrecognized expansion staging state")
        with view._connect() as db:
            row = db.execute(
                "SELECT path,maximum,purpose,state,bytes,sha256 FROM allocations WHERE token=?",
                (stage["token"],),
            ).fetchone()
            expected = (stage["path"], 4096, "payload", "pending", None, None)
            if row != expected and not (row is None and marker == new):
                raise ValueError("Changed expansion staging obligation")
            actual = _baseline(
                db,
                tokens=(stage["token"], publication.artifacts[0].token),
                publication_key=publication.key,
            )
            if actual != receipt["baseline"]:
                raise ValueError("Changed expansion baseline")
        pin.check()
        lease.check()
        return marker, database, receipt, publication, row is not None


def apply_expansion(lease, identity, receipt_inputs):
    """Complete only the exact receipt-pinned migration, including crash recovery."""
    inputs = _validate(lease, identity, receipt_inputs)
    old, new = inputs["old_marker"], _new_marker(inputs)
    with _directories(lease) as (directories, check_dirs), ExitStack() as stack:
        marker, database, receipt, publication, pending = _state(
            lease, identity, inputs
        )
        check_dirs()
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
                raise ValueError("Changed expansion staging marker")
        if _encode(_validate(lease, identity, receipt_inputs)) != _encode(inputs):
            raise ValueError("Changed caller expansion inputs")
        if _state(lease, identity, inputs) != (
            marker,
            database,
            receipt,
            publication,
            pending,
        ):
            raise ValueError("Changed expansion before commit")
        if _encode(receipt_inputs) != _encode(inputs):
            raise ValueError("Changed caller expansion inputs")
        check_dirs()
        marker_pin.check()
        if stage_pin:
            stage_pin.check()
        if database == old:
            path = lease.root / "resources.sqlite3"
            with _PinnedFile(path, METADATA_BYTES // 2) as db_pin:
                with (
                    closing(
                        sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
                    ) as db,
                    db,
                ):
                    db.execute("PRAGMA synchronous=FULL")
                    db.execute("PRAGMA max_page_count=4000")
                    db.execute("BEGIN IMMEDIATE")
                    db_pin.check()
                    if db.execute("SELECT value FROM metadata").fetchall() != [
                        (json.dumps(old, sort_keys=True),)
                    ]:
                        raise ValueError("Changed database before expansion commit")
                    db.execute(
                        "UPDATE metadata SET value=?",
                        (json.dumps(new, sort_keys=True),),
                    )
                    check_dirs()
                    marker_pin.check()
                    stage_pin.check()
        if stage_pin:
            check_dirs()
            marker_pin.check()
            stage_pin.check()
            os.replace(
                receipt["stage"]["path"].split("/")[1],
                ".initialized",
                src_dir_fd=directories[lease.root / "staging"],
                dst_dir_fd=directories[lease.root],
            )
        # Recovery may begin after replace but before either directory fsync.
        # Repeat both durability barriers even when the staged path is gone.
        os.fsync(directories[lease.root])
        os.fsync(directories[lease.root / "staging"])
        check_dirs()
        # Reconcile every pre-existing obligation and receipt before releasing
        # the one now-missing staged marker. The private view is read-only.
        _, _, _, _, pending = _state(lease, identity, inputs)
        if pending:
            from .derived_cache_policy import _ExpandedResources

            resources = _ExpandedResources(lease, identity, limit_bytes=EXPANDED_BYTES)
            resources.release_missing(receipt["stage"]["token"])
        check_dirs()
    return open_expanded_cache(lease, identity, receipt_inputs)
