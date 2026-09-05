"""Local experiment identities and immutable inputs; annotations are separate."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from arblab.hyperliquid_copy.contracts import semantic_hash


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class ExperimentStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "experiments.sqlite3"
        with self.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, status TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL, error TEXT, run_id TEXT,
                artifact_hashes TEXT NOT NULL DEFAULT '{}', needs_resume INTEGER NOT NULL DEFAULT 0)""")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def unpack(self, row):
        if row is None:
            raise ValueError("Unknown experiment")
        output = dict(row)
        output.update(json.loads(output.pop("payload")))
        output["artifact_hashes"] = json.loads(output["artifact_hashes"])
        output["needs_resume"] = bool(output["needs_resume"])
        return output

    def get(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Unknown experiment")
        with self.connection() as db:
            return self.unpack(
                db.execute(
                    "SELECT * FROM experiments WHERE id=?", [identifier]
                ).fetchone()
            )

    def list(self, *, kind="backtest", limit=200):
        with self.connection() as db:
            return [
                self.unpack(r)
                for r in db.execute(
                    "SELECT * FROM experiments WHERE kind=? ORDER BY created_at DESC,id LIMIT ?",
                    [kind, limit],
                )
            ]

    def create(
        self,
        name,
        dataset_id,
        config,
        provenance,
        *,
        kind="backtest",
        preview_date=None,
        preview_scope=None,
        parent_id=None,
    ):
        identifier, now = uuid4().hex, timestamp()
        payload = dict(
            dataset_id=dataset_id,
            config=config,
            config_hash=semantic_hash(config),
            provenance=provenance,
            preview_date=preview_date,
            preview_scope=preview_scope,
            parent_id=parent_id,
        )
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute(
                    "SELECT count(*) FROM experiments WHERE status IN ('queued','running')"
                ).fetchone()[0]
                >= 32
            ):
                raise ValueError("Queue limit: at most 32 outstanding jobs")
            db.execute(
                "INSERT INTO experiments(id,name,notes,created_at,updated_at,status,kind,payload) VALUES(?,?,?,?,?,?,?,?)",
                [
                    identifier,
                    name,
                    "",
                    now,
                    now,
                    "queued",
                    kind,
                    json.dumps(payload, allow_nan=False),
                ],
            )
        return self.get(identifier)

    def annotate(self, identifier, name, notes):
        self.get(identifier)
        with self.connection() as db:
            db.execute(
                "UPDATE experiments SET name=?,notes=?,updated_at=? WHERE id=?",
                [name, notes, timestamp(), identifier],
            )
        return self.get(identifier)

    def transition(
        self,
        identifier,
        expected,
        status,
        *,
        error=None,
        run_id=None,
        artifact_hashes=None,
    ):
        allowed = {
            "queued": {"running", "cancelled"},
            "running": {"completed", "failed", "cancelled"},
        }
        if status not in allowed.get(expected, set()):
            raise ValueError("Invalid job transition")
        with self.connection() as db:
            cursor = db.execute(
                "UPDATE experiments SET status=?,error=?,run_id=?,artifact_hashes=?,updated_at=? WHERE id=? AND status=?",
                [
                    status,
                    error,
                    run_id,
                    json.dumps(artifact_hashes or {}),
                    timestamp(),
                    identifier,
                    expected,
                ],
            )
            if cursor.rowcount != 1:
                raise ValueError("Job state changed")
        return self.get(identifier)

    def recover(self):
        with self.connection() as db:
            db.execute(
                "UPDATE experiments SET status='failed',error='Interrupted before validated publication',updated_at=? WHERE status='running'",
                [timestamp()],
            )
            db.execute("UPDATE experiments SET needs_resume=1 WHERE status='queued'")

    def resume(self, identifier):
        with self.connection() as db:
            if (
                db.execute(
                    "UPDATE experiments SET needs_resume=0 WHERE id=? AND status='queued'",
                    [identifier],
                ).rowcount
                != 1
            ):
                raise ValueError("Only queued jobs can be resumed")
        return self.get(identifier)

    def next_queued(self):
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM experiments WHERE status='queued' AND needs_resume=0 ORDER BY created_at,id LIMIT 1"
            ).fetchone()
            return self.unpack(row) if row else None
