"""Local derived-cache obligations, separate from lifetime acquisition spending.

Callers must cap their own writers within reservations. This ledger never deletes
payloads, adopts unknown files, or refunds unfinished work on process exit.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import uuid

from .download import file_hash

MAX_BYTES = 8 * 1024**3
METADATA_BYTES = 32 * 1024**2
FREE_RESERVE = 64 * 1024**2
MAX_RECORDS = 100_000
NAMESPACES = ("staging", "artifacts", "scratch")


def _regular(path):
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError("Missing/inaccessible cache file") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Unsafe cache metadata/payload file")
    return info


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CacheResources:
    @classmethod
    def create(cls, lease, identity, *, limit_bytes=MAX_BYTES):
        cls._arguments(lease, identity, limit_bytes)
        root = lease.root
        if {p.name for p in root.iterdir()} != {".resource.lock"}:
            raise ValueError("Cache creation requires pristine root")
        if shutil.disk_usage(root).free < METADATA_BYTES + FREE_RESERVE:
            raise ValueError("Insufficient cache free space")
        marker = dict(
            version=2,
            identity=identity,
            limit_bytes=limit_bytes,
            nonce=uuid.uuid4().hex,
            root=[root.stat().st_dev, root.stat().st_ino],
        )
        with (root / ".initialized").open("x") as handle:
            json.dump(marker, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        _sync(root)
        path = root / "resources.sqlite3"
        with path.open("xb"):
            pass
        db = sqlite3.connect(path)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA page_size=4096")
            db.execute("PRAGMA max_page_count=4000")
            with db:
                db.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
                db.execute(
                    "INSERT INTO metadata VALUES (?)",
                    (json.dumps(marker, sort_keys=True),),
                )
                db.execute(
                    "CREATE TABLE allocations (token TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, "
                    "maximum INTEGER NOT NULL CHECK(maximum>0), purpose TEXT NOT NULL, "
                    "state TEXT NOT NULL, bytes INTEGER, sha256 TEXT)"
                )
                db.execute(
                    "CREATE TABLE publications (key TEXT PRIMARY KEY, descriptor TEXT NOT NULL, sha256 TEXT NOT NULL)"
                )
                db.execute("PRAGMA user_version=2")
        finally:
            db.close()
        for name in NAMESPACES:
            (root / name).mkdir()
        _sync(root)
        return cls(lease, identity, limit_bytes=limit_bytes)

    @staticmethod
    def _arguments(lease, identity, limit_bytes):
        lease.check()
        if (
            not isinstance(identity, str)
            or not 1 <= len(identity) <= 256
            or type(limit_bytes) is not int
            or not METADATA_BYTES < limit_bytes <= MAX_BYTES
        ):
            raise ValueError("Invalid derived cache identity/budget")

    def __init__(self, lease, identity, *, limit_bytes=MAX_BYTES):
        self._arguments(lease, identity, limit_bytes)
        self.lease, self.root = lease, lease.root
        self.identity, self.limit = identity, limit_bytes
        self.path = self.root / "resources.sqlite3"
        self.marker = self.root / ".initialized"
        self.audit()

    def _metadata(self):
        self.lease.check()
        try:
            if not 0 < _regular(self.marker).st_size <= 4096:
                raise ValueError("Invalid initialization marker")
            marker = json.loads(self.marker.read_text())
            if (
                set(marker) != {"version", "identity", "limit_bytes", "nonce", "root"}
                or marker["version"] != 2
                or marker["identity"] != self.identity
                or marker["limit_bytes"] != self.limit
                or not re.fullmatch("[a-f0-9]{32}", marker["nonce"])
                or marker["root"] != [self.root.stat().st_dev, self.root.stat().st_ino]
            ):
                raise ValueError("Changed cache identity/budget")
            if not 0 < _regular(self.path).st_size <= METADATA_BYTES // 2:
                raise ValueError("Missing/oversized cache catalog")
            journal = self.path.with_name(self.path.name + "-journal")
            if journal.exists() or journal.is_symlink():
                if _regular(journal).st_size > METADATA_BYTES // 2:
                    raise ValueError("Oversized cache journal")
            allowed = {
                ".resource.lock",
                ".initialized",
                "resources.sqlite3",
                "resources.sqlite3-journal",
                *NAMESPACES,
            }
            if any(p.name not in allowed for p in self.root.iterdir()):
                raise ValueError("Unknown cache root file")
            for name in NAMESPACES:
                path = self.root / name
                if path.is_symlink() or not path.is_dir():
                    raise ValueError("Unsafe cache namespace")
            return json.dumps(marker, sort_keys=True)
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid existing cache metadata") from exc

    @contextmanager
    def _connect(self):
        marker = self._metadata()
        try:
            db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True)
            try:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("PRAGMA foreign_keys=ON")
                # Leave room for journal headers and the initialization marker
                # within the combined 32-MiB metadata allocation.
                if (
                    db.execute("PRAGMA page_size").fetchone()[0] != 4096
                    or db.execute("PRAGMA journal_mode").fetchone()[0] != "delete"
                ):
                    raise ValueError("Unsupported cache metadata storage settings")
                db.execute("PRAGMA max_page_count=4000")
                if (
                    db.execute("PRAGMA user_version").fetchone()[0] != 2
                    or {
                        r[0]
                        for r in db.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }
                    != {"metadata", "allocations", "publications"}
                    or db.execute("SELECT value FROM metadata").fetchall()
                    != [(marker,)]
                ):
                    raise ValueError("Changed cache catalog schema/identity")
                yield db
            finally:
                db.close()
        except sqlite3.DatabaseError as exc:
            raise ValueError("Invalid existing cache catalog") from exc

    def _path(self, relative):
        if not isinstance(relative, str) or not re.fullmatch(
            r"(?:staging|artifacts|scratch)/[a-f0-9]{32}(?:\.parquet)?", relative
        ):
            raise ValueError("Invalid cache allocation path")
        return self.root / relative

    def _size(self, path):
        if not path.exists() and not path.is_symlink():
            return 0
        count, size, pending = 0, 0, [path]
        while pending:
            current = pending.pop()
            count += 1
            if count > MAX_RECORDS:
                raise ValueError("Cache tree metadata limit exceeded")
            info = current.lstat()
            if stat.S_ISDIR(info.st_mode):
                for child in current.iterdir():
                    pending.append(child)
                    if len(pending) + count > MAX_RECORDS:
                        raise ValueError("Cache tree metadata limit exceeded")
            else:
                size += _regular(current).st_size
        return size

    def _audit(self, db):
        reserved = retained = unwritten = count = 0
        for token, relative, maximum, purpose, state, size, digest in db.execute(
            "SELECT * FROM allocations"
        ):
            count += 1
            if (
                count > MAX_RECORDS
                or not re.fullmatch("[a-f0-9]{32}", token)
                or type(maximum) is not int
                or not 0 < maximum <= self.limit
                or purpose not in ("payload", "scratch")
                or state not in ("pending", "retained")
            ):
                raise ValueError("Invalid cache allocation record")
            path = self._path(relative)
            actual = self._size(path)
            if actual > maximum:
                raise ValueError("Cache allocation exceeded reservation")
            if state == "pending":
                if size is not None or digest is not None:
                    raise ValueError("Invalid pending cache record")
                reserved += maximum
                unwritten += maximum - actual
            else:
                if (
                    purpose != "payload"
                    or type(size) is not int
                    or not 0 <= size <= maximum
                ):
                    raise ValueError("Invalid retained cache record")
                if _regular(path).st_size != size or file_hash(path) != digest:
                    raise ValueError("Changed retained cache payload")
                retained += size
        for namespace in NAMESPACES:
            for count, path in enumerate((self.root / namespace).iterdir(), 1):
                if (
                    count > MAX_RECORDS
                    or db.execute(
                        "SELECT 1 FROM allocations WHERE path=?",
                        (f"{namespace}/{path.name}",),
                    ).fetchone()
                    is None
                ):
                    raise ValueError("Unknown cache allocation output")
        total = reserved + retained + METADATA_BYTES
        if total > self.limit:
            raise ValueError("Cache budget exceeded")
        if (
            shutil.disk_usage(self.root).free
            < unwritten + METADATA_BYTES + FREE_RESERVE
        ):
            raise ValueError("Insufficient cache free space")
        return dict(
            reserved_bytes=reserved,
            retained_bytes=retained,
            metadata_bytes=METADATA_BYTES,
            total_bytes=total,
        ), unwritten

    def audit(self):
        with self._connect() as db:
            return self._audit(db)[0]

    def reserve(self, relative_path, max_bytes, purpose):
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            totals, unwritten = self._audit(db)
            path = self._path(relative_path)
            if (
                type(max_bytes) is not int
                or max_bytes <= 0
                or purpose not in ("payload", "scratch")
                or path.exists()
                or path.is_symlink()
            ):
                raise ValueError("Invalid/preexisting cache allocation")
            if totals["total_bytes"] + max_bytes > self.limit:
                raise ValueError("Cache budget exceeded")
            if (
                shutil.disk_usage(self.root).free
                < unwritten + max_bytes + METADATA_BYTES + FREE_RESERVE
            ):
                raise ValueError("Insufficient cache free space")
            if (
                db.execute("SELECT count(*) FROM allocations").fetchone()[0]
                >= MAX_RECORDS
            ):
                raise ValueError("Cache record limit exceeded")
            token = uuid.uuid4().hex
            db.execute(
                "INSERT INTO allocations VALUES (?,?,?,?, 'pending',NULL,NULL)",
                (token, relative_path, max_bytes, purpose),
            )
            return token

    def _pending(self, db, token):
        row = db.execute(
            "SELECT path,maximum,purpose,state FROM allocations WHERE token=?", (token,)
        ).fetchone()
        if row is None or row[3] != "pending":
            raise ValueError("Expected pending allocation")
        return row

    def settle(self, token):
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            self._audit(db)
            relative, maximum, purpose, _ = self._pending(db, token)
            path = self._path(relative)
            if purpose != "payload":
                raise ValueError("Scratch cannot be settled")
            size = _regular(path).st_size
            if size > maximum:
                raise ValueError("Payload exceeds reservation")
            digest = file_hash(path)
            _sync(path)
            _sync(path.parent)
            db.execute(
                "UPDATE allocations SET state='retained',bytes=?,sha256=? WHERE token=?",
                (size, digest, token),
            )

    def release_missing(self, token):
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            relative, _, _, _ = self._pending(db, token)
            path = self._path(relative)
            if path.exists() or path.is_symlink():
                raise ValueError("Existing cache allocation cannot be refunded")
            _sync(path.parent)
            db.execute("DELETE FROM allocations WHERE token=?", (token,))
