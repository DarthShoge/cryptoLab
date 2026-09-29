"""Explicit journaled cache disposal; no automatic expiration or raw-source access."""

from dataclasses import asdict
import hashlib
import os
import stat

from .cache_retirement_inventory import _descriptor, _identity, _namespace, _rows
from .cache_retirement_journal import (
    INTENT,
    COMMITTED,
    MAX_BYTES,
    allocations,  # noqa: F401 - compatibility re-export
    catalogue_digest,
    context,
    load_journal,
    prepare_retirement,  # noqa: F401 - compatibility re-export
)
from .derived_cache_policy import _PinnedFile
from .derived_cache_resources import METADATA_BYTES
from .derived_publication import (
    MAX_DESCRIPTOR_BYTES,
    MAX_PUBLICATIONS,
    PublishedArtifacts,
    _encode,
    publication_key,
)
from .download import file_hash
from .query_directory_pin import pin_directory


def _database_inode(held):
    """Allow our transaction's bytes to change, never its underlying file."""
    fd = os.fstat(held.fd)
    path = held.path.lstat()
    for info in (fd, path):
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != held.identity[:2]
            or not 0 < info.st_size <= METADATA_BYTES // 2
        ):
            raise ValueError("Retirement database identity changed")


def _metadata_guard(resources, namespace, database, marker, journal, directory_fd=None):
    """Cheap held-identity checks; safe after the final semantic validation."""
    resources.lease.check()
    marker.check()
    journal.check()
    _database_inode(database)
    if _namespace(resources) != namespace:
        raise ValueError("Retirement namespace changed")
    if directory_fd is not None:
        info = os.fstat(directory_fd)
        if (info.st_dev, info.st_ino) != namespace[1]:
            raise ValueError("Retirement namespace changed")


def _files(resources, body):
    inv = body["inventory"]
    for pin, (_, identity) in zip(
        inv["target"]["artifacts"], inv["files"], strict=True
    ):
        path = resources.root / pin["path"]
        if (
            _identity(path) != tuple(identity)
            or file_hash(path) != pin["sha256"]
            or _identity(path) != tuple(identity)
        ):
            raise ValueError("Retirement target file changed")


def _insert_committed(db, inputs, publication):
    raw = _encode(
        dict(
            schema=1,
            kind=COMMITTED,
            inputs=inputs,
            artifacts=[asdict(p) for p in publication.artifacts],
        )
    )
    if (
        len(raw) > MAX_DESCRIPTOR_BYTES
        or db.execute("SELECT count(*) FROM publications").fetchone()[0]
        >= MAX_PUBLICATIONS
    ):
        raise ValueError("Retirement commit metadata capacity exceeded")
    db.execute(
        "INSERT INTO publications VALUES (?,?,?)",
        (
            publication_key(COMMITTED, inputs),
            raw.decode(),
            hashlib.sha256(raw).hexdigest(),
        ),
    )


def _intent(resources, db, inputs, publication):
    expected = {
        publication_key(INTENT, inputs): INTENT,
        publication_key(COMMITTED, inputs): COMMITTED,
    }
    kind = expected.get(publication.key)
    if (
        kind is None
        or PublishedArtifacts(resources)._read(db, publication.key, kind, inputs)
        != publication
    ):
        raise ValueError("Retirement journal receipt changed or disappeared")


def _ownership(resources, db, body, target_key):
    inv = body["inventory"]
    tokens = [p["token"] for p in inv["target"]["artifacts"]]
    candidates, shared = set(tokens), set()
    for row in _rows(db):
        publication = _descriptor(resources, db, *row)
        if publication.key != target_key:
            shared.update(
                p.token for p in publication.artifacts if p.token in candidates
            )
    if inv["exclusive"] != [t for t in tokens if t not in shared] or inv["shared"] != [
        t for t in tokens if t in shared
    ]:
        raise ValueError("Retirement ownership classification mismatch")


def begin_retirement(resources, intent_inputs, *, validation=None):
    frozen = context(resources, intent_inputs)
    body, publication = load_journal(resources, intent_inputs)
    inv = body["inventory"]
    namespace = tuple(tuple(v) for v in inv["namespace"])
    with (
        pin_directory(resources.root, namespace[0]),
        pin_directory(resources.root / "artifacts", namespace[1]),
        _PinnedFile(resources.path, METADATA_BYTES // 2) as database,
        _PinnedFile(resources.marker, 4096) as marker,
        _PinnedFile(
            resources.root / publication.artifacts[0].path, MAX_BYTES
        ) as journal,
    ):
        with resources._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            resources._audit(db)
            _intent(resources, db, frozen, publication)
            # Exact baseline excludes only the already-verified intent receipt.
            if (
                catalogue_digest(resources, db, omit=(publication.key,))
                != inv["catalogue_digest"]
            ):
                raise ValueError("Retirement catalogue changed since preparation")
            target = db.execute(
                "SELECT descriptor FROM publications WHERE key=?", (frozen["target"],)
            ).fetchone()
            if target != (inv["descriptor"],):
                raise ValueError("Retirement target changed or is already detached")
            _ownership(resources, db, body, frozen["target"])
            for row in body["allocations"]:
                if db.execute(
                    "SELECT * FROM allocations WHERE token=?", (row[0],)
                ).fetchone() != tuple(row):
                    raise ValueError("Retirement allocation changed")
            _files(resources, body)
            if (
                context(resources, intent_inputs) != frozen
                or _namespace(resources) != namespace
            ):
                raise ValueError("Retirement detachment context changed")
            marker.check()
            journal.check()
            database.check()
            if validation is not None:
                validation()
            if _encode(intent_inputs) != _encode(frozen):
                raise ValueError("Retirement validation changed inputs")
            _metadata_guard(resources, namespace, database, marker, journal)
            database.check()
            db.execute("DELETE FROM publications WHERE key=?", (frozen["target"],))
            exclusive = set(inv["exclusive"])
            for row in body["allocations"]:
                if row[0] in exclusive:
                    db.execute(
                        "UPDATE allocations SET maximum=?,state='pending',bytes=NULL,sha256=NULL WHERE token=?",
                        (max(1, row[5]), row[0]),
                    )
            _insert_committed(db, frozen, publication)
            resources._audit(db)
            if context(resources, intent_inputs) != frozen:
                raise ValueError("Retirement inputs changed during detachment")
            _files(resources, body)
            if context(resources, intent_inputs) != frozen:
                raise ValueError("Retirement final detachment context changed")
            marker.check()
            journal.check()
            _database_inode(database)
            resources.lease.check()
            if validation is not None:
                validation()
            if _encode(intent_inputs) != _encode(frozen):
                raise ValueError("Retirement validation changed inputs")
            _metadata_guard(resources, namespace, database, marker, journal)
        _database_inode(database)
    return dict(committed=publication_key(COMMITTED, frozen), target=frozen["target"])


def _detached(resources, db, body, inputs, publication):
    _intent(resources, db, inputs, publication)
    committed = PublishedArtifacts(resources)._read(
        db, publication_key(COMMITTED, inputs), COMMITTED, inputs
    )
    if committed is None or committed.artifacts != publication.artifacts:
        raise ValueError("Matching committed retirement receipt required")
    if db.execute(
        "SELECT 1 FROM publications WHERE key=?", (inputs["target"],)
    ).fetchone():
        raise ValueError("Retirement target was republished")
    exclusive = set(body["inventory"]["exclusive"])
    for row in _rows(db):
        current = _descriptor(resources, db, *row)
        if any(p.token in exclusive for p in current.artifacts):
            raise ValueError("Retirement token still has surviving references")
    targets = []
    for row, (token, identity) in zip(
        body["allocations"], body["inventory"]["files"], strict=True
    ):
        if token not in exclusive:
            continue
        expected = (token, row[1], max(1, row[5]), "payload", "pending", None, None)
        current = db.execute(
            "SELECT * FROM allocations WHERE token=?", (token,)
        ).fetchone()
        path = resources._path(row[1])
        if current is None:
            if path.exists() or path.is_symlink():
                raise ValueError("Released retirement path has reappeared")
        elif current != expected:
            raise ValueError("Retirement pending obligation changed")
        targets.append((row, tuple(identity), current is not None))
    return targets


def _unlink_owned(resources, directory_fd, row, identity, guard):
    path = resources._path(row[1])
    if not path.exists() and not path.is_symlink():
        return
    fd = os.open(
        path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
    )
    try:
        info = os.fstat(fd)
        actual = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
        )
        if (
            not stat.S_ISREG(info.st_mode)
            or actual != identity
            or _identity(path) != identity
        ):
            raise ValueError("Retirement file ownership changed")
        if file_hash(path) != row[6]:
            raise ValueError("Retirement file bytes changed")
        guard()
        info = os.fstat(fd)
        final = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
        )
        if final != identity or _identity(path) != identity:
            raise ValueError("Retirement file changed before unlink")
        os.unlink(path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(fd)


def finish_retirement(resources, intent_inputs, *, validation=None):
    frozen = context(resources, intent_inputs)
    body, publication = load_journal(resources, intent_inputs)
    namespace = tuple(tuple(v) for v in body["inventory"]["namespace"])
    with (
        pin_directory(resources.root, namespace[0]),
        pin_directory(resources.root / "artifacts", namespace[1]),
        _PinnedFile(resources.path, METADATA_BYTES // 2) as database,
        _PinnedFile(resources.marker, 4096) as marker,
        _PinnedFile(
            resources.root / publication.artifacts[0].path, MAX_BYTES
        ) as journal,
    ):
        directory_fd = os.open(
            resources.root / "artifacts", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            with resources._connect() as db, db:
                db.execute("BEGIN IMMEDIATE")
                targets = _detached(resources, db, body, frozen, publication)

                def guard():
                    if context(resources, intent_inputs) != frozen:
                        raise ValueError("Retirement disposal context changed")
                    _metadata_guard(
                        resources, namespace, database, marker, journal, directory_fd
                    )
                    # All context hashing precedes the caller's final protected
                    # file checks. Do not hash again between those checks/unlink.
                    if validation is not None:
                        validation()
                    if _encode(intent_inputs) != _encode(frozen):
                        raise ValueError("Retirement validation changed inputs")
                    _metadata_guard(
                        resources, namespace, database, marker, journal, directory_fd
                    )

                database.check()
                for row, identity, pending in targets:
                    guard()
                    if pending:
                        _unlink_owned(resources, directory_fd, row, identity, guard)
                        path = resources._path(row[1])
                        if path.exists() or path.is_symlink():
                            raise ValueError("Retirement output is still present")
                        os.fsync(directory_fd)
                        db.execute("DELETE FROM allocations WHERE token=?", (row[0],))
                if context(resources, intent_inputs) != frozen:
                    raise ValueError("Retirement completion context changed")
                guard()
                # Missing paths remain charged until this transaction commits.
                resources._audit(db)
                guard()
                # The committed receipt owns the same journal payload and is
                # sufficient after disposal. Retain only one durable row.
                db.execute(
                    "DELETE FROM publications WHERE key=?",
                    (publication_key(INTENT, frozen),),
                )
                resources._audit(db)
            _database_inode(database)
        finally:
            os.close(directory_fd)
    return dict(
        committed=publication_key(COMMITTED, frozen),
        target=frozen["target"],
        retired_files=len(targets),
        retired_bytes=sum(row[5] for row, _, _ in targets),
    )
