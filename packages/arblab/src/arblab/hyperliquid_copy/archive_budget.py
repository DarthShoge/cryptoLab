"""Durable pre-request byte reservations across frozen archive batches.

Creating a budget is an explicit caller action, not permission to spend. Reserved
bytes are never refunded after failure. No retry or raw-file cleanup lives here.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .download import BUCKET
from .proxy_compact import _sync

MAX_JOB_BYTES = 512 * 1024**3
MAX_OBJECTS = 732 * 24 + 1


class ArchiveBudget:
    def __init__(self, path, objects, max_bytes, *, must_exist=False):
        if type(must_exist) is not bool:
            raise ValueError("Invalid existing-budget requirement")
        self.must_exist = must_exist
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_JOB_BYTES:
            raise ValueError("Invalid lifetime byte budget")
        if not isinstance(objects, list) or not 0 < len(objects) <= MAX_OBJECTS:
            raise ValueError("Invalid archive scope")
        self.objects = {}
        for obj in objects:
            if (
                set(obj) != {"key", "bytes", "etag"}
                or not isinstance(obj["key"], str)
                or not re.fullmatch(
                    r"node_fills(?:_by_block)?/hourly/\d{8}/(?:[0-9]|1[0-9]|2[0-3])\.lz4",
                    obj["key"],
                )
                or type(obj["bytes"]) is not int
                or not 0 < obj["bytes"] <= 6 * 1024**3
                or not isinstance(obj["etag"], str)
                or not 0 < len(obj["etag"]) <= 200
                or obj["key"] in self.objects
            ):
                raise ValueError("Invalid frozen archive object")
            self.objects[obj["key"]] = (obj["bytes"], obj["etag"])
        frozen = sorted((key, *value) for key, value in self.objects.items())
        self.identity = hashlib.sha256(
            json.dumps(frozen, separators=(",", ":")).encode()
        ).hexdigest()
        self.limit = max_bytes
        self.path = Path(path).absolute()
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise ValueError("Symlink archive budget path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            tables = {
                r[0]
                for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not tables:
                if self.must_exist:
                    raise ValueError("Missing existing archive budget state")
                db.execute(
                    "CREATE TABLE metadata (scope TEXT NOT NULL, max_bytes INTEGER NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE objects (key TEXT PRIMARY KEY, bytes INTEGER NOT NULL, etag TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE reservations (key TEXT PRIMARY KEY REFERENCES objects(key), bytes INTEGER NOT NULL CHECK(bytes > 0))"
                )
                db.execute(
                    "INSERT INTO metadata VALUES (?,?)", (self.identity, self.limit)
                )
                db.executemany("INSERT INTO objects VALUES (?,?,?)", frozen)
                db.execute("PRAGMA user_version=1")
            elif (
                tables != {"metadata", "objects", "reservations"}
                or db.execute("PRAGMA user_version").fetchone()[0] != 1
            ):
                raise ValueError("Unsupported archive budget database")
            self._identity(db)
            if db.execute(
                "SELECT key,bytes,etag FROM objects ORDER BY key"
            ).fetchall() != [tuple(r) for r in frozen]:
                raise ValueError("Archive budget scope changed")
        _sync(self.path.parent)

    @contextmanager
    def _connect(self):
        if (
            self.path.is_symlink()
            or self.path.with_name(self.path.name + "-journal").is_symlink()
        ):
            raise ValueError("Symlink archive budget database")
        try:
            db = sqlite3.connect(
                self.path.as_uri() + "?mode=rw" if self.must_exist else self.path,
                uri=self.must_exist,
                timeout=10,
            )
        except sqlite3.OperationalError as exc:
            raise ValueError("Cannot open existing archive budget") from exc
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA max_page_count=4096")
            yield db
        finally:
            db.close()

    def _identity(self, db):
        if db.execute("SELECT scope,max_bytes FROM metadata").fetchall() != [
            (self.identity, self.limit)
        ]:
            raise ValueError("Archive budget scope/limit changed")

    @property
    def reserved_bytes(self):
        with self._connect() as db:
            self._identity(db)
            return db.execute(
                "SELECT coalesce(sum(bytes),0) FROM reservations"
            ).fetchone()[0]

    def reserve(self, key):
        if key not in self.objects:
            raise ValueError("Archive key outside frozen scope")
        size, etag = self.objects[key]
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            self._identity(db)
            if db.execute(
                "SELECT bytes,etag FROM objects WHERE key=?", (key,)
            ).fetchone() != (size, etag):
                raise ValueError("Archive object identity changed")
            if db.execute("SELECT 1 FROM reservations WHERE key=?", (key,)).fetchone():
                raise ValueError("Archive object already reserved; no automatic retry")
            spent = db.execute(
                "SELECT coalesce(sum(bytes),0) FROM reservations"
            ).fetchone()[0]
            if spent + size > self.limit:
                raise ValueError("Lifetime archive budget exhausted")
            db.execute("INSERT INTO reservations VALUES (?,?)", (key, size))


class BudgetedArchiveSource:
    def __init__(self, source, budget):
        if source.meta.config.retries.get("total_max_attempts") != 1:
            raise ValueError("Archive client must disable automatic retries")
        self.source, self.budget = source, budget

    def _object(self, args, *, get):
        fields = {"Bucket", "Key", "RequestPayer"} | ({"IfMatch"} if get else set())
        if (
            set(args) != fields
            or args["Bucket"] != BUCKET
            or args["RequestPayer"] != "requester"
            or args["Key"] not in self.budget.objects
        ):
            raise ValueError("Unscoped archive request")
        size, etag = self.budget.objects[args["Key"]]
        if get and args["IfMatch"] != etag:
            raise ValueError("Archive request identity changed")
        return size, etag

    def head_object(self, **args):
        size, etag = self._object(args, get=False)
        response = self.source.head_object(**args)
        if (response.get("ContentLength"), response.get("ETag")) != (size, etag):
            raise ValueError("Frozen archive identity changed")
        return response

    def get_object(self, **args):
        self._object(args, get=True)
        self.budget.reserve(args["Key"])  # Commit before the network call.
        return self.source.get_object(**args)
