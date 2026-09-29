"""Owned SQLite job metadata and artifact transitions, independent of acquisition."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .archive_cache import _safe
from .download import file_hash
from .proxy_compact import _sync

STAGES = ("raw", "normalized", "compact", "qualified")
MAX_METADATA_BYTES = 16 * 1024**2


class ArchiveJobStore:
    @classmethod
    def create(cls, root, metadata):
        root = Path(root).absolute()
        _safe(root)
        if ".." in root.parts or root.exists():
            raise ValueError("Archive job directory already exists or is unsafe")
        frozen = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        if len(frozen.encode()) > MAX_METADATA_BYTES:
            raise ValueError("Job metadata byte limit exceeded")
        root.mkdir(parents=True)
        path = root / "job.sqlite3"
        path.touch(exist_ok=False)
        store = cls.__new__(cls)
        store.root, store.path = root, path
        with store._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE metadata (frozen TEXT NOT NULL, sha256 TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE artifacts (batch INTEGER NOT NULL, stage TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(batch,stage))"
            )
            db.execute(
                "CREATE TABLE cleanup (batch INTEGER NOT NULL, stage TEXT NOT NULL, path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, qualification_sha256 TEXT NOT NULL, status TEXT NOT NULL)"
            )
            db.execute(
                "INSERT INTO metadata VALUES (?,?)",
                (frozen, hashlib.sha256(frozen.encode()).hexdigest()),
            )
            db.execute("PRAGMA user_version=2")
        _sync(root)
        _sync(root.parent)
        return cls(root)

    def __init__(self, root):
        self.root = Path(root).absolute()
        _safe(self.root)
        if ".." in self.root.parts:
            raise ValueError("Unsafe archive job path")
        self.path = self.root / "job.sqlite3"
        with self._connect() as db:
            tables = {
                r[0]
                for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if (
                tables != {"metadata", "artifacts", "cleanup"}
                or db.execute("PRAGMA user_version").fetchone()[0] != 2
            ):
                raise ValueError("Unsupported archive job database")
            rows = db.execute("SELECT frozen,sha256 FROM metadata").fetchall()
        if (
            len(rows) != 1
            or len(rows[0][0].encode()) > MAX_METADATA_BYTES
            or hashlib.sha256(rows[0][0].encode()).hexdigest() != rows[0][1]
        ):
            raise ValueError("Job metadata identity mismatch")
        self.frozen = rows[0][0]
        self.metadata = json.loads(self.frozen)

    @contextmanager
    def _connect(self):
        for path in (
            self.root,
            self.path,
            self.path.with_name(self.path.name + "-journal"),
        ):
            _safe(path)
        if not self.path.is_file():
            raise ValueError("Missing archive job database")
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA max_page_count=16384")
            yield db
        finally:
            db.close()

    @contextmanager
    def locked(self):
        path = self.root / ".job.lock"
        _safe(path)
        with path.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Archive job is already running") from exc
            with self._connect() as db:
                if db.execute("SELECT frozen FROM metadata").fetchall() != [
                    (self.frozen,)
                ]:
                    raise ValueError("Archive job scope changed")
            yield

    def stage_root(self, batch, stage):
        if (
            type(batch) is not int
            or not 0 <= batch < len(self.metadata["plan"]["batches"])
            or stage not in STAGES
        ):
            raise ValueError("Invalid archive job stage")
        path = self.root / "batches" / f"{batch:04d}" / stage
        _safe(path)
        return path

    def records(self):
        with self._connect() as db:
            rows = db.execute(
                "SELECT batch,stage,path,sha256 FROM artifacts ORDER BY batch"
            ).fetchall()
        result = {}
        for batch, stage, relative, identity in rows:
            root = self.stage_root(batch, stage)
            path = self.root / relative
            _safe(path)
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or path.name != "manifest.json"
                or path.parent.parent != root
                or not re.fullmatch(r"[a-f0-9]{64}", identity)
            ):
                raise ValueError("Invalid recorded artifact identity/path")
            result[batch, stage] = dict(path=path, sha256=identity)
        for batch, stage in result:
            if any(
                (batch, earlier) not in result
                for earlier in STAGES[: STAGES.index(stage)]
            ) or any((earlier, "qualified") not in result for earlier in range(batch)):
                raise ValueError("Archive job stage sequence changed")
        return result

    def find(self, batch, stage):
        root = self.stage_root(batch, stage)
        record = self.records().get((batch, stage))
        children = list(root.iterdir()) if root.exists() else []
        if not children and record is None:
            return None
        if len(children) != 1:
            raise ValueError("Ambiguous or missing archive stage artifacts")
        path = children[0] / "manifest.json"
        _safe(path)
        if not children[0].is_dir() or not path.is_file():
            raise ValueError("Incomplete local archive stage; recovery required")
        if record and (record["path"] != path or file_hash(path) != record["sha256"]):
            raise ValueError("Recorded archive artifact identity mismatch")
        return path

    def record(self, batch, stage, path, identity):
        if self.find(batch, stage) != path or file_hash(path) != identity:
            raise ValueError("Archive artifact changed before recording")
        relative = str(path.relative_to(self.root))
        _sync(path)
        # Persist every new directory entry before SQLite can reference it.
        for parent in path.parents:
            _sync(parent)
            if parent == self.root:
                break
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT path,sha256 FROM artifacts WHERE batch=? AND stage=?",
                (batch, stage),
            ).fetchone()
            if previous is not None and previous != (relative, identity):
                raise ValueError("Archive artifact transition changed")
            db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?)",
                (batch, stage, relative, identity),
            )
        self.records()  # Enforce the causal stage sequence.
