"""Explicit recovery of manifest-bound ranking work left by a dead worker.

The manifest is retained as the durable recovery receipt.  Missing partial
outputs remain charged until the transaction which removes their allocations
commits, so an interrupted recovery can be repeated safely.
"""

import re
import stat

from .derived_cache_resources import MAX_RECORDS, _regular, _sync
from .derived_publication import ArtifactPin, _encode, _inputs
from .download import file_hash
from .ranking_staging_manifest import MAX_BYTES, decode_manifest


def _manifest(resources, relative, expected_context):
    if not isinstance(relative, str) or not re.fullmatch(
        r"staging/[a-f0-9]{32}", relative
    ):
        raise ValueError("Invalid failed staging manifest path")
    path = resources._path(relative)
    info = _regular(path)
    if info.st_size > MAX_BYTES:
        raise ValueError("Oversized failed staging manifest")
    encoded = path.read_bytes()
    body = decode_manifest(encoded)
    if body["allocations"]["manifest"]["path"] != relative or _encode(
        body["context"]
    ) != _encode(_inputs("ranking_staging_context", expected_context)):
        raise ValueError("Failed staging recovery context mismatch")
    return path, encoded, body


def _tree(path, maximum):
    """Return a fully authenticated, deepest-first owned tree."""
    if not path.exists() and not path.is_symlink():
        return []
    pending, found, size = [path], [], 0
    while pending:
        current = pending.pop()
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Symlink in failed staging output")
        if stat.S_ISDIR(info.st_mode):
            children = list(current.iterdir())
            pending.extend(children)
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            size += info.st_size
        else:
            raise ValueError("Unsafe failed staging output")
        found.append((current, info.st_dev, info.st_ino, info.st_mode, info.st_nlink))
        if len(found) + len(pending) > MAX_RECORDS or size > maximum:
            raise ValueError("Failed staging output exceeds reservation")
    return sorted(found, key=lambda row: len(row[0].parts), reverse=True)


def _verify_tree(rows):
    for path, device, inode, mode, links in rows:
        info = path.lstat()
        if (info.st_dev, info.st_ino, info.st_mode, info.st_nlink) != (
            device,
            inode,
            mode,
            links,
        ):
            raise ValueError("Failed staging output changed")


def _remove_tree(rows):
    _verify_tree(rows)
    for path, *_ in rows:
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            path.rmdir()
        else:
            path.unlink()
        _sync(path.parent)


def _ledger(resources, body):
    with resources._connect() as db:
        rows = {}
        for role, allocation in body["allocations"].items():
            row = db.execute(
                "SELECT * FROM allocations WHERE token=?", (allocation["token"],)
            ).fetchone()
            rows[role] = row
    return rows


def recover(resources, manifest, expected_context):
    """Recover one exact failed invocation; safe to repeat after completion."""
    resources.lease.check()
    path, encoded, body = _manifest(resources, manifest, expected_context)
    allocations = body["allocations"]
    ledger = _ledger(resources, body)

    trees = {}
    for role, allocation in allocations.items():
        row = ledger[role]
        expected = (
            allocation["token"],
            allocation["path"],
            allocation["maximum"],
            allocation["purpose"],
        )
        if role == "manifest" and row is not None and row[:4] == expected:
            if row[4:] not in (
                ("pending", None, None),
                ("retained", len(encoded), file_hash(path)),
            ):
                raise ValueError("Changed failed staging manifest allocation")
        elif role != "manifest" and row is not None:
            if row != (*expected, "pending", None, None):
                raise ValueError("Changed failed staging allocation")
        elif role == "manifest":
            raise ValueError("Missing failed staging manifest allocation")
        trees[role] = _tree(resources._path(allocation["path"]), allocation["maximum"])

    # Authenticate every target before the first durable state transition.
    if path.read_bytes() != encoded:
        raise ValueError("Failed staging manifest changed")
    for rows in trees.values():
        _verify_tree(rows)

    manifest_token = allocations["manifest"]["token"]
    if ledger["manifest"][4] == "pending":
        resources.settle(manifest_token)
    # Publication payloads are intentionally constrained to artifacts/.  The
    # retained, content-addressed staging manifest is itself the recovery
    # receipt and no longer blocks future work because it is not pending.
    receipt = ArtifactPin(manifest_token, manifest, len(encoded), file_hash(path))

    released = []
    for role in ("scratch", "metrics", "scores", "ranking"):
        allocation = allocations[role]
        with resources._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT * FROM allocations WHERE token=?", (allocation["token"],)
            ).fetchone()
            if current is None:
                if resources._path(allocation["path"]).exists():
                    raise ValueError("Released failed staging path reappeared")
                released.append(role)
                continue
            expected = (
                allocation["token"],
                allocation["path"],
                allocation["maximum"],
                allocation["purpose"],
                "pending",
                None,
                None,
            )
            if current != expected:
                raise ValueError("Failed staging allocation changed during recovery")
            current_tree = _tree(
                resources._path(allocation["path"]), allocation["maximum"]
            )
            _remove_tree(current_tree)
            db.execute("DELETE FROM allocations WHERE token=?", (allocation["token"],))
            resources._audit(db)
        released.append(role)

    resources.audit()
    if path.read_bytes() != encoded:
        raise ValueError("Retained failed staging manifest changed")
    return dict(receipt=receipt, released_roles=released)
