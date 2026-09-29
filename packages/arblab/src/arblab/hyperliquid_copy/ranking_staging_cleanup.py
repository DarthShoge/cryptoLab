"""Successful live-invocation disposal only; failed work remains charged."""

import json
import os
import stat

from .cache_retirement import _database_inode, _unlink_owned
from .cache_retirement_inventory import _descriptor, _digest, _rows
from .derived_cache_policy import _PinnedFile
from .derived_cache_resources import METADATA_BYTES
from .derived_publication import ArtifactPin, Publication, _encode
from .ranking_staging_artifact import StagingArtifact, _identity
from .ranking_staging_manifest import MAX_BYTES, ROLES, encode_manifest
from .ranking_staging_owner import RankingStagingOwner
from .ranking_staging_score import PendingScores


def _catalogue(resources, db, owner, result, receipt):
    allocation = owner._allocations["ranking"]
    expected = ArtifactPin(
        allocation["token"],
        allocation["path"],
        result.ranking.bytes,
        result.ranking.sha256,
    )
    if receipt.artifacts != (expected,):
        raise ValueError("Saved receipt does not bind the owned full ranking")
    temporary = {
        a["token"] for role, a in owner._allocations.items() if role != "ranking"
    }
    found = False
    for row in _rows(db):
        publication = _descriptor(resources, db, *row)
        if any(pin.token in temporary for pin in publication.artifacts):
            raise ValueError("Temporary staging token has surviving references")
        if publication.key == receipt.key:
            data = json.loads(row[1])
            inputs = data["inputs"]
            if (
                publication != receipt
                or data["kind"] != "saved_feature_ranking"
                or set(inputs) != {"query", "ranking"}
                or _encode(inputs["query"]) != _encode(owner._context.get("query"))
                or not _digest(inputs["ranking"])
            ):
                raise ValueError("Saved receipt query/binding changed")
            found = True
    if not found:
        raise ValueError("Saved ranking receipt is missing")


def finish_staging(owner, result, receipt, *, verify_receipt):
    """Release successful temporary obligations atomically, never the ranking.

    Unlinks cannot roll back. Any failure rolls back ledger releases and closes
    the live owner, leaving even missing files charged without implicit retry.
    The verifier must be read-only; it executes within the ledger transaction.
    """
    if (
        type(owner) is not RankingStagingOwner
        or type(result) is not PendingScores
        or type(receipt) is not Publication
        or not callable(verify_receipt)
    ):
        raise ValueError("Expected live staging result and saved receipt verifier")
    try:
        owner._check_context()
        if set(owner._fds) != set(ROLES):
            raise ValueError("Incomplete staging ownership")
        resources = owner._resources
        pins = {
            role: getattr(result, role) for role in ("metrics", "scores", "ranking")
        }
        pins["manifest"] = StagingArtifact.capture(
            owner.path("manifest"),
            owner._fds["manifest"],
            maximum=MAX_BYTES,
        )
        removed = set()
        with (
            _PinnedFile(resources.path, METADATA_BYTES // 2) as database,
            _PinnedFile(resources.marker, 4096) as marker,
            resources._connect() as db,
            db,
        ):
            db.execute("BEGIN IMMEDIATE")

            def physical():
                resources.lease.check()
                marker.check()
                _database_inode(database)
                for path, fd in owner._directories.items():
                    actual, held = path.lstat(), os.fstat(fd)
                    if not stat.S_ISDIR(actual.st_mode) or (
                        actual.st_dev,
                        actual.st_ino,
                    ) != (held.st_dev, held.st_ino):
                        raise ValueError("Staging cleanup namespace changed")
                if (
                    encode_manifest(
                        dict(
                            schema=1,
                            context=owner._context,
                            allocations=owner._allocations,
                        )
                    )
                    != owner._encoded
                ):
                    raise ValueError("Staging manifest context changed")
                if os.pread(owner._fds["manifest"], MAX_BYTES + 1, 0) != owner._encoded:
                    raise ValueError("Staging manifest bytes changed")
                for role in ROLES:
                    if role in removed:
                        if owner.path(role).exists() or owner.path(role).is_symlink():
                            raise ValueError("Removed staging path reappeared")
                        continue
                    owner._check_fd(role, owner._fds[role])
                    if role == "scratch":
                        with os.scandir(owner._fds[role]) as entries:
                            if next(entries, None) is not None:
                                raise ValueError("Staging scratch is not empty")
                    elif (
                        pins[role].path != owner.path(role)
                        or _identity(owner.path(role).lstat()) != pins[role].identity
                    ):
                        raise ValueError("Staging artifact pin changed")

            def guard():
                owner._check_context()
                for role, pin in pins.items():
                    if role not in removed:
                        pin.verify()
                # The supplied verifier is the final source/receipt operation;
                # only bounded metadata and physical checks follow it.
                if verify_receipt() != receipt:
                    raise ValueError("Independent saved receipt verification failed")
                _catalogue(resources, db, owner, result, receipt)
                for role, allocation in owner._allocations.items():
                    row = db.execute(
                        "SELECT * FROM allocations WHERE token=?",
                        (allocation["token"],),
                    ).fetchone()
                    expected = (
                        allocation["token"],
                        allocation["path"],
                        allocation["maximum"],
                        allocation["purpose"],
                        "pending",
                        None,
                        None,
                    )
                    if role in removed:
                        expected = None
                    elif role == "ranking":
                        expected = (
                            *expected[:4],
                            "retained",
                            result.ranking.bytes,
                            result.ranking.sha256,
                        )
                    if row != expected:
                        raise ValueError("Staging cleanup allocation changed")
                # All callback/hash work precedes final physical/manifest checks.
                physical()

            guard()  # Authenticate every target before the first unlink.
            for role in ("metrics", "scores", "scratch", "manifest"):
                guard()
                allocation = owner._allocations[role]
                path = owner.path(role)
                directory_fd = owner._directories[path.parent]
                if role == "scratch":
                    os.rmdir(path.name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                else:
                    pin = pins[role]
                    dev, ino, _, links, size, mtime, ctime = pin.identity
                    row = (
                        allocation["token"],
                        allocation["path"],
                        allocation["maximum"],
                        "payload",
                        "pending",
                        None,
                        pin.sha256,
                    )
                    _unlink_owned(
                        resources,
                        directory_fd,
                        row,
                        (dev, ino, size, mtime, ctime, links),
                        guard,
                    )
                if path.exists() or path.is_symlink():
                    raise ValueError("Staging output is still present")
                os.fsync(directory_fd)
                resources._pending(db, allocation["token"])
                db.execute(
                    "DELETE FROM allocations WHERE token=?", (allocation["token"],)
                )
                removed.add(role)
            resources._audit(db)
            guard()
    finally:
        owner.close()  # Handles only; no failure-path deletion or refund.
