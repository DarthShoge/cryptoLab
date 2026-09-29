"""Explicit requester-pays retry outside the immutable archive engine."""

from contextlib import closing
import hashlib
import json
import os
import sqlite3

from .archive_cache import _safe
from .download import BUCKET, file_hash
from .proxy_archive_download import _checkpoint
from .proxy_compact import _sync


def _connect(path):
    db = sqlite3.connect(path, timeout=10)
    db.execute("PRAGMA synchronous=FULL")
    return db


def _ledger(job):
    path = job.store.root / "retry_budget.sqlite3"
    _safe(path)
    identity = hashlib.sha256(job.store.frozen.encode()).hexdigest()
    base = job.metadata["plan"]["total_bytes"] - sum(
        item["bytes"] for item in job.metadata["cache"]
    )
    limit = job.metadata["max_download_bytes"]
    with _connect(path) as db, db:
        db.execute("BEGIN IMMEDIATE")
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not tables:
            db.execute(
                "CREATE TABLE metadata (job_sha256 TEXT NOT NULL, base_bytes INTEGER NOT NULL, max_bytes INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE attempts (batch INTEGER NOT NULL, object_index INTEGER NOT NULL, attempt INTEGER NOT NULL, key TEXT NOT NULL, bytes INTEGER NOT NULL, etag TEXT NOT NULL, status TEXT NOT NULL, sha256 TEXT, PRIMARY KEY(batch,object_index,attempt))"
            )
            db.execute("INSERT INTO metadata VALUES (?,?,?)", (identity, base, limit))
            db.execute("PRAGMA user_version=1")
        elif tables != {"metadata", "attempts"}:
            raise ValueError("Unsupported retry budget database")
        if db.execute("PRAGMA user_version").fetchone()[0] != 1 or db.execute(
            "SELECT job_sha256,base_bytes,max_bytes FROM metadata"
        ).fetchall() != [(identity, base, limit)]:
            raise ValueError("Retry budget scope changed")
    _sync(path.parent)
    return path, base, limit


def recover_job_archive_request(
    root, *, batch: int, object_index: int, approved_bytes: int, source
):
    """Charge and execute exactly one explicitly approved retry."""
    from .archive_job import ArchiveJob

    job = ArchiveJob(root)
    if job.budget is None or source.meta.config.retries.get("total_max_attempts") != 1:
        raise ValueError("Retry requires the bounded no-retry archive client")
    with job.store.locked():
        records = job.store.records()
        completed = sum(stage == "qualified" for _, stage in records)
        if batch != completed or (batch, "raw") in records:
            raise ValueError("Retry is not for the current unpublished raw batch")
        manifest = job.store.find(batch, "raw")
        if manifest is None:
            raise ValueError("Missing interrupted raw acquisition")
        data = json.loads(manifest.read_text())
        objects = data.get("objects")
        if (
            not isinstance(objects, list)
            or type(object_index) is not int
            or not 0 <= object_index < len(objects)
        ):
            raise ValueError("Invalid archive object index")
        obj = objects[object_index]
        if (
            type(approved_bytes) is not int
            or approved_bytes != obj.get("bytes")
            or [
                i for i, item in enumerate(objects) if item.get("status") == "requested"
            ]
            != [object_index]
        ):
            raise ValueError("Retry approval does not match interrupted request")
        target = manifest.parent / obj["file"]
        temporary = target.with_suffix(".partial")
        _safe(target)
        _safe(temporary)
        if target.exists() or temporary.exists():
            raise ValueError("Interrupted retry payload requires recovery inspection")

        ledger, base, limit = _ledger(job)
        with _connect(ledger) as db, db:
            db.execute("BEGIN IMMEDIATE")
            retry_bytes = db.execute(
                "SELECT coalesce(sum(bytes),0) FROM attempts"
            ).fetchone()[0]
            if base + retry_bytes + approved_bytes > limit:
                raise ValueError("Lifetime archive budget exhausted by retry")
            previous = db.execute(
                "SELECT attempt,status FROM attempts WHERE batch=? AND object_index=? ORDER BY attempt DESC LIMIT 1",
                (batch, object_index),
            ).fetchone()
            if previous and previous[1] == "approved":
                attempt = previous[0]
            else:
                attempt = 1 if previous is None else previous[0] + 1
                db.execute(
                    "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,NULL)",
                    (
                        batch,
                        object_index,
                        attempt,
                        obj["key"],
                        approved_bytes,
                        obj["etag"],
                        "approved",
                    ),
                )
        _sync(ledger.parent)
        with _connect(ledger) as db, db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE attempts SET status='requested' WHERE batch=? AND object_index=? AND attempt=? AND status='approved'",
                (batch, object_index, attempt),
            ).rowcount
            if changed != 1:
                raise ValueError("Retry attempt is not approved")

        response = source.get_object(
            Bucket=BUCKET,
            Key=obj["key"],
            RequestPayer="requester",
            IfMatch=obj["etag"],
        )
        with closing(response["Body"]) as body:
            if (response.get("ContentLength"), response.get("ETag")) != (
                obj["bytes"],
                obj["etag"],
            ):
                raise ValueError("Archive identity changed during approved retry")
            digest, count = hashlib.sha256(), 0
            with temporary.open("xb") as stream:
                while chunk := body.read(min(1024**2, obj["bytes"] - count + 1)):
                    count += len(chunk)
                    if count > obj["bytes"]:
                        raise ValueError("Archive retry length exceeded")
                    stream.write(chunk)
                    digest.update(chunk)
                if count != obj["bytes"]:
                    raise ValueError("Archive retry length shorter than frozen size")
                stream.flush()
                os.fsync(stream.fileno())
        identity = digest.hexdigest()
        with _connect(ledger) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE attempts SET status='payload_ready',sha256=? WHERE batch=? AND object_index=? AND attempt=? AND status='requested'",
                (identity, batch, object_index, attempt),
            )
        temporary.replace(target)
        _sync(target.parent)
        obj.update(status="downloaded", sha256=identity)
        _checkpoint(manifest, data)
        with _connect(ledger) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE attempts SET status='completed' WHERE batch=? AND object_index=? AND attempt=? AND status='payload_ready'",
                (batch, object_index, attempt),
            )
        return {
            "batch": batch,
            "object_index": object_index,
            "key": obj["key"],
            "bytes": approved_bytes,
            "attempt": attempt,
            "sha256": identity,
            "state": "completed",
        }
