"""Disposable causal seed caches over a frozen, conflict-validated input set.

SQLite holds metadata; canonical seeds remain Parquet. Initial validation retains
the existing 8 GiB/5000-file ceiling. Reopen/advance need only seeds and intervening
partitions, not expired source files. This is not archive-coverage qualification,
incremental acquisition, or authority to delete any source history.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import fcntl
import json
from pathlib import Path
import re
import sqlite3
import shutil
from uuid import uuid4

import duckdb
import pyarrow

from .compact_catalog import _micros
from .checkpoint_io import write_seed
from .contracts import utc, semantic_hash
from .download import file_hash
from .proxy_activity import ProxyActivity, REVERSE_ORDER
from .proxy_compact import COLUMNS, MAX_BYTES, _sync


MAX_METADATA_BYTES = 4 * 1024**2
MAX_SEED_BYTES = 512 * 1024**2
MAX_SEED_ROWS = 2_000_000
MAX_CACHE_BYTES = 8 * 1024**3
FREE_RESERVE_BYTES = 64 * 1024**2


def _engine():
    root = Path(__file__).parent
    return {
        "schema": 1,
        "duckdb": duckdb.__version__,
        "pyarrow": pyarrow.__version__,
        "code": {
            name: file_hash(root / name)
            for name in (
                "activity_checkpoints.py",
                "proxy_activity.py",
                "contracts.py",
                "proxy_compact.py",
                "compact_catalog.py",
                "proxy_dataset.py",
                "proxy_availability.py",
                "registered_activity.py",
                "sharded_validation.py",
                "checkpoint_io.py",
            )
        },
    }


def _verify(entries):
    for entry in entries:
        path = Path(entry["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_hash(path) != entry["sha256"]
        ):
            raise ValueError("Checkpoint input identity changed")


def _limit(limits):
    size = limits.get("max_input_bytes", MAX_BYTES)
    if type(size) is not int or not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid checkpoint byte limit")
    if set(limits) - {
        "max_input_bytes",
        "memory_mb",
        "temp_disk_mb",
        "max_wallet_fills",
        "max_candidates",
    }:
        raise ValueError("Unknown checkpoint resource limit")
    return size


class ActivityCheckpoints:
    def __init__(self, root):
        root = Path(root)
        if root.is_symlink():
            raise ValueError("Symlink checkpoint root")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve(strict=True)
        self.path = self.root / "checkpoints.sqlite3"
        if self.path.is_symlink():
            raise ValueError("Symlink checkpoint database")
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                r[0]
                for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if version == 0 and not tables:
                db.execute(
                    "CREATE TABLE checkpoints (id TEXT PRIMARY KEY, metadata TEXT NOT NULL, sha256 TEXT NOT NULL)"
                )
                db.execute("PRAGMA user_version=1")
            elif version != 1 or tables != {"checkpoints"}:
                raise ValueError("Unsupported checkpoint database")
            db.execute(
                "CREATE INDEX IF NOT EXISTS checkpoint_cache_key "
                "ON checkpoints(json_extract(metadata, '$.cache_key'))"
            )

    @contextmanager
    def _connect(self):
        if self.path.is_symlink():
            raise ValueError("Symlink checkpoint database")
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def metadata(self, identity):
        if not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{32}", identity):
            raise ValueError("Invalid checkpoint identity")
        with self._connect() as db:
            row = db.execute(
                "SELECT metadata,sha256 FROM checkpoints WHERE id=?", (identity,)
            ).fetchone()
        if row is None:
            raise ValueError("Unknown checkpoint identity")
        encoded = row[0].encode()
        if (
            len(encoded) > MAX_METADATA_BYTES
            or hashlib.sha256(encoded).hexdigest() != row[1]
        ):
            raise ValueError("Checkpoint metadata identity changed")
        meta = json.loads(encoded)
        if meta.get("engine") != _engine():
            raise ValueError(
                "Checkpoint metadata engine/version changed; rebuild required"
            )
        return meta

    def lookup(self, key):
        with self._connect() as db:
            row = db.execute(
                "SELECT id FROM checkpoints WHERE json_extract(metadata, '$.cache_key')=? ORDER BY id LIMIT 1",
                (key,),
            ).fetchone()
        if row is None:
            return None
        self.metadata(row[0])  # Verify metadata checksum and engine before reuse.
        return row[0]

    def _commit(self, identity, meta):
        encoded = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > MAX_METADATA_BYTES:
            raise ValueError("Checkpoint metadata byte limit exceeded")
        with self._connect() as db, db:
            db.execute(
                "INSERT INTO checkpoints VALUES (?,?,?)",
                (
                    identity,
                    encoded.decode(),
                    hashlib.sha256(encoded).hexdigest(),
                ),
            )
        _sync(self.root)

    def _publish(
        self, activity, cutoff, entries, manifests, parent=None, cache_key=None
    ):
        # Serialize writers across API previews/jobs so the aggregate check is
        # not merely a per-writer promise. Failed staging is counted, never deleted.
        lock = self.root / ".publication.lock"
        if lock.is_symlink():
            raise ValueError("Symlink checkpoint publication lock")
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if cache_key is not None:
                existing = self.lookup(cache_key)
                if existing is not None:
                    return existing
            reserve = MAX_SEED_BYTES + 2 * MAX_METADATA_BYTES
            used = 0
            for path in self.root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Symlink checkpoint cache path")
                if path.is_file():
                    used += path.stat().st_size
            if used + reserve > MAX_CACHE_BYTES:
                raise ValueError("Checkpoint total cache byte limit exceeded")
            if shutil.disk_usage(self.root).free < reserve + FREE_RESERVE_BYTES:
                raise ValueError("Insufficient checkpoint publication free space")
            return self._publish_locked(
                activity, cutoff, entries, manifests, parent, cache_key
            )

    def _publish_locked(self, activity, cutoff, entries, manifests, parent, cache_key):
        rows = activity.db.execute(
            'SELECT count(DISTINCT ("user",coin)) FROM fills WHERE exchange_time < ?',
            [cutoff],
        ).fetchone()[0]
        if rows > MAX_SEED_ROWS:
            raise ValueError("Checkpoint seed row limit exceeded")
        identity = uuid4().hex
        stage = self.root / (identity + ".partial")
        stage.mkdir()
        seed = stage / "seeds.parquet"
        query = activity.db.execute(
            f"SELECT {', '.join(chr(34) + c + chr(34) for c in COLUMNS)} FROM fills "
            "WHERE exchange_time < ? "
            f"QUALIFY row_number() OVER (PARTITION BY user,coin ORDER BY {REVERSE_ORDER})=1",
            [cutoff],
        )
        write_seed(query, seed, MAX_SEED_BYTES)
        if seed.stat().st_size > MAX_SEED_BYTES:
            raise ValueError("Checkpoint seed byte limit exceeded")
        _sync(seed)
        _sync(stage)
        meta = dict(
            engine=_engine(),
            cutoff=cutoff.isoformat(),
            parent=parent,
            manifest_ids=list(manifests),
            partitions=entries,
            seed=dict(rows=rows, bytes=seed.stat().st_size, sha256=file_hash(seed)),
            cache_key=cache_key,
        )
        stage.rename(self.root / identity)
        _sync(self.root)
        self._commit(identity, meta)
        return identity

    def build(self, catalog, manifest_ids, cutoff, *, temp_root, **limits):
        cutoff = utc(cutoff)
        manifests = list(manifest_ids)
        entries = catalog.partitions(
            manifests,
            datetime.min.replace(tzinfo=timezone.utc),
            datetime.max.replace(tzinfo=timezone.utc),
        )
        paths = [e["path"] for e in entries]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Distinct nonempty checkpoint input partitions required")
        if sum(e["bytes"] for e in entries) > _limit(limits):
            raise ValueError("Checkpoint build input byte limit exceeded")
        _verify(entries)
        # Whole-source duplicate checks happen before window pruning. Future rows
        # are validated too, so this lineage may replay them without a blind spot
        # for conflicts whose earlier event has subsequently left the seed set.
        with ProxyActivity(
            paths,
            temp_root=temp_root,
            query_window=(cutoff, cutoff + timedelta(microseconds=1)),
            **limits,
        ) as activity:
            _verify(entries)
            return self._publish(activity, cutoff, entries, manifests)

    def open(self, identity, start, end, *, temp_root, **limits):
        meta = self.metadata(identity)
        cutoff, start, end = (
            utc(datetime.fromisoformat(meta["cutoff"])),
            utc(start),
            utc(end),
        )
        if start < cutoff or start >= end:
            raise ValueError(
                "Increasing query window at or after checkpoint cutoff required"
            )
        entries = [
            e
            for e in meta["partitions"]
            if e["max_time"] >= _micros(cutoff) and e["min_time"] < _micros(end)
        ]
        directory = self.root / identity
        if directory.is_symlink():
            raise ValueError("Checkpoint directory identity changed")
        seed = dict(meta["seed"], path=str(directory / "seeds.parquet"))
        if sum(e["bytes"] for e in [seed, *entries]) > _limit(limits):
            raise ValueError("Checkpoint replay input byte limit exceeded")
        _verify([seed, *entries])
        activity = ProxyActivity(
            [e["path"] for e in entries],
            temp_root=temp_root,
            query_window=(start, end),
            seed_path=seed["path"],
            seed_cutoff=cutoff,
            **limits,
        )
        try:
            _verify([seed, *entries])
        except BaseException:
            activity.close()
            raise
        return activity

    def advance(self, identity, cutoff, *, temp_root, **limits):
        meta = self.metadata(identity)
        old, cutoff = utc(datetime.fromisoformat(meta["cutoff"])), utc(cutoff)
        if cutoff <= old:
            raise ValueError("Checkpoint cutoff must advance")
        key = semantic_hash(
            dict(
                kind="advanced",
                engine=meta["engine"],
                cutoff=cutoff,
                manifests=meta["manifest_ids"],
                partitions=meta["partitions"],
            )
        )
        existing = self.lookup(key)
        if existing is not None:
            # Reuse must fail on corruption rather than silently rebuilding it.
            with self.open(
                existing,
                cutoff,
                cutoff + timedelta(microseconds=1),
                temp_root=temp_root,
                **limits,
            ):
                pass
            return existing
        with self.open(
            identity, old, cutoff, temp_root=temp_root, **limits
        ) as activity:
            return self._publish(
                activity,
                cutoff,
                meta["partitions"],
                meta["manifest_ids"],
                identity,
                key,
            )
