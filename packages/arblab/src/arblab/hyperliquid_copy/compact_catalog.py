"""Transactional metadata index; bulk events remain immutable Parquet.

Window lookup only returns candidate files. It does not establish coverage, apply
row filters/deduplication, or recover dormant positions before the query window.
Callers must verify returned hashes before use; catalog entries are not file locks.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .contracts import utc
from .download import file_hash
from .proxy_compact import COLUMNS, MAX_BYTES, MAX_PARTITIONS


def _micros(value):
    delta = utc(value) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _bounds(path, entry):
    with pq.ParquetFile(path) as reader:
        if (
            reader.schema_arrow.names != COLUMNS
            or reader.metadata.num_rows != entry["rows"]
        ):
            raise ValueError("Invalid compact schema/row count")
        strings = {
            "event_id",
            "user",
            "coin",
            "side",
            "direction",
            "fee_token",
            "tx_hash",
            "source_key",
        }
        integers = {"block_number", "source_line", "event_index", "tid", "oid"}
        numbers = {"px", "sz", "start_position", "post_position", "closed_pnl", "fee"}
        optional = {"block_number", "block_time", "fee_token"}
        for field in reader.schema_arrow:
            kind = field.type
            if field.name in optional and pa.types.is_null(kind):
                continue
            if field.name in strings:
                valid = pa.types.is_string(kind)
            elif field.name in integers:
                valid = kind == pa.int64()
            elif field.name in numbers:
                valid = kind in (pa.float64(), pa.int64())
            elif field.name in {"crossed", "liquidation"}:
                valid = pa.types.is_boolean(kind)
            else:
                valid = (
                    pa.types.is_timestamp(kind)
                    and kind.tz is not None
                    and kind.unit == "us"
                )
            if not valid:
                raise ValueError(f"Invalid canonical type: {field.name}")
    return event_bounds(path)


def event_bounds(path):
    """Exact event-time bounds; callers separately validate row economics/schema."""
    low = high = None
    with pq.ParquetFile(path) as reader:
        kind = reader.schema_arrow.field("exchange_time").type
        if not pa.types.is_timestamp(kind) or kind.tz is None or kind.unit != "us":
            raise ValueError("Expected timezone-aware microsecond event timestamps")
        for batch in reader.iter_batches(
            batch_size=4096, columns=["exchange_time"], use_threads=False
        ):
            column = batch.column(0)
            if column.null_count:
                raise ValueError("Null event timestamps")
            bounds = pc.min_max(column).as_py()
            if bounds["min"] is not None:
                a, b = _micros(bounds["min"]), _micros(bounds["max"])
                low = a if low is None else min(low, a)
                high = b if high is None else max(high, b)
    return low, high


def _validated(manifest):
    if manifest.stat().st_size > 1_000_000:
        raise ValueError("Compact manifest byte limit exceeded")
    identity = file_hash(manifest)
    text = manifest.read_text()
    data = json.loads(text)
    if (
        data.get("schema") != "hyperliquid_compact_history_v1"
        or data.get("complete") is not True
        or data.get("projection_version") != 1
        or data.get("retained_columns") != COLUMNS
    ):
        raise ValueError("Unsupported/incomplete compact manifest")
    files = data.get("files", [])
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_PARTITIONS:
        raise ValueError("Invalid compact partition count")
    records, seen, size, rows = [], set(), 0, 0
    for f in files:
        name = f["name"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".parquet")
        ):
            raise ValueError("Invalid compact partition path")
        path = manifest.parent / name
        if path in seen or path.is_symlink() or not path.is_file():
            raise ValueError("Missing/duplicate/symlink compact partition")
        seen.add(path)
        if type(f["bytes"]) is not int or type(f["rows"]) is not int or f["rows"] < 0:
            raise ValueError("Invalid partition counts")
        size += f["bytes"]
        rows += f["rows"]
        if (
            not 0 < f["bytes"] <= MAX_BYTES
            or size > MAX_BYTES
            or path.stat().st_size != f["bytes"]
        ):
            raise ValueError("Compact byte identity/limit mismatch")
        if file_hash(path) != f["sha256"]:
            raise ValueError("Compact hash mismatch")
        low, high = _bounds(path, f)
        if file_hash(path) != f["sha256"]:
            raise ValueError("Compact file changed during validation")
        records.append(
            (identity, str(path), f["sha256"], f["bytes"], f["rows"], low, high)
        )
    if rows != data.get("rows") or size != data.get("output_bytes"):
        raise ValueError("Compact manifest total rows/bytes mismatch")
    if file_hash(manifest) != identity:
        raise ValueError("Compact manifest changed during validation")
    return identity, text, records


class CompactCatalog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                r[0]
                for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if (
                version not in (0, 1)
                or (version == 0 and tables)
                or (version == 1 and tables != {"manifests", "partitions"})
            ):
                raise ValueError("Not a supported compact catalog database")
            if version == 0:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute(
                        "CREATE TABLE manifests (id TEXT PRIMARY KEY, path TEXT NOT NULL, frozen_json TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE partitions (manifest_id TEXT NOT NULL REFERENCES manifests(id), path TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, rows INTEGER NOT NULL, min_time INTEGER, max_time INTEGER, PRIMARY KEY(manifest_id,path))"
                    )
                    db.execute(
                        "CREATE INDEX partition_times ON partitions(manifest_id,min_time,max_time)"
                    )
                    db.execute("PRAGMA user_version=1")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def register(self, manifest_path):
        manifest = Path(manifest_path).resolve(strict=True)
        identity, frozen, records = _validated(manifest)
        with self._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT path FROM manifests WHERE id=?", (identity,)
            ).fetchone()
            if previous:
                if previous[0] != str(manifest):
                    raise ValueError(
                        "Manifest identity already registered at another path"
                    )
                return identity
            db.execute(
                "INSERT INTO manifests VALUES (?,?,?)",
                (identity, str(manifest), frozen),
            )
            db.executemany("INSERT INTO partitions VALUES (?,?,?,?,?,?,?)", records)
        return identity

    def partitions(self, manifest_ids, start, end):
        ids = list(manifest_ids)
        low, high = _micros(start), _micros(end)
        if not 1 <= len(ids) <= 1000 or len(set(ids)) != len(ids) or low >= high:
            raise ValueError("Explicit unique manifests and increasing window required")
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as db:
            known = db.execute(
                f"SELECT count(*) FROM manifests WHERE id IN ({placeholders})", ids
            ).fetchone()[0]
            if known != len(ids):
                raise ValueError("unknown compact manifest identity")
            db.row_factory = sqlite3.Row
            rows = db.execute(
                f"SELECT * FROM partitions WHERE manifest_id IN ({placeholders}) AND max_time >= ? AND min_time < ? ORDER BY min_time,manifest_id,path LIMIT 5001",
                [*ids, low, high],
            ).fetchall()
        if len(rows) > 5000:
            raise ValueError("Candidate partition limit exceeded; no truncation")
        return [dict(r) for r in rows]
