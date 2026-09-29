"""Exact, journaled disposal of qualified job-owned staging payloads only.

Caller holds the job lock and verifies its frozen engine. Manifests and canonical
history are never targets. Deleted payloads are not claimed to be rehashed later.
"""

from pathlib import Path

from .archive_cache import _safe
from .archive_job_artifacts import read
from .archive_job_qualification import pin, verify
from .download import file_hash
from .proxy_compact import _sync

FIELDS = ("batch", "stage", "path", "sha256", "bytes", "qualification_sha256")


def _targets(store, completed):
    records = store.records()
    verify(store, completed - 1, pin(records[completed - 1, "qualified"]))
    targets = []
    for batch in range(completed):
        qualification = records[batch, "qualified"]["sha256"]
        for stage, collection, key, suffix in (
            ("raw", "objects", "file", ".lz4"),
            ("normalized", "files", "name", ".parquet"),
        ):
            manifest = store.find(batch, stage)
            if manifest is None:
                raise ValueError("Cleanup source manifest missing")
            data = read(manifest)
            entries = data[collection]
            if not 1 <= len(entries) <= 169:
                raise ValueError("Invalid cleanup target count")
            for entry in entries:
                name = entry[key]
                if (
                    not isinstance(name, str)
                    or Path(name).name != name
                    or not name.endswith(suffix)
                ):
                    raise ValueError("Unsafe cleanup target path")
                path = manifest.parent / name
                _safe(path)
                targets.append(
                    dict(
                        batch=batch,
                        stage=stage,
                        path=str(path.relative_to(store.root)),
                        sha256=entry["sha256"],
                        bytes=entry["bytes"],
                        qualification_sha256=qualification,
                    )
                )
            if file_hash(manifest) != records[batch, stage]["sha256"]:
                raise ValueError("Cleanup source manifest identity changed")
    if len({e["path"] for e in targets}) != len(targets):
        raise ValueError("Duplicate cleanup targets")
    return targets


def _journal(store):
    with store._connect() as db:
        rows = db.execute(
            "SELECT batch,stage,path,sha256,bytes,qualification_sha256,status FROM cleanup"
        ).fetchall()
    return {r[2]: dict(zip((*FIELDS, "status"), r)) for r in rows}


def _check_payload(store, target, status):
    path = store.root / target["path"]
    _safe(path)
    if status == "deleted":
        if path.exists():
            raise ValueError("A deleted cleanup target was recreated")
        return
    if not path.exists():
        if status != "intent":
            raise ValueError("Cleanup target missing before durable intent")
        return
    if (
        not path.is_file()
        or path.stat().st_size != target["bytes"]
        or file_hash(path) != target["sha256"]
    ):
        raise ValueError("Cleanup payload identity mismatch")


def dispose_prefix(store, completed, progress=None):
    if not completed:
        return
    targets = _targets(store, completed)
    expected = {e["path"]: e for e in targets}
    journal = _journal(store)
    if set(journal) - set(expected):
        raise ValueError("Unexpected cleanup journal targets")
    journaled_batches = {e["batch"] for e in journal.values()}
    for target in targets:
        saved = journal.get(target["path"])
        if saved is None and target["batch"] in journaled_batches:
            raise ValueError("Incomplete cleanup journal scope")
        if saved is not None and (
            {k: saved[k] for k in FIELDS} != target
            or saved["status"] not in ("intent", "deleted")
        ):
            raise ValueError("Cleanup journal identity/status mismatch")
        _check_payload(store, target, saved["status"] if saved else None)
    with store._connect() as db, db:
        db.execute("BEGIN IMMEDIATE")
        for target in targets:
            if target["path"] not in journal:
                db.execute(
                    "INSERT INTO cleanup VALUES (?,?,?,?,?,?,?)",
                    (*[target[k] for k in FIELDS], "intent"),
                )
    if progress:
        progress(dict(cleanup="intent_committed", targets=len(targets)))
    # Recheck canonical evidence after a callback/pause and before any unlink.
    _targets(store, completed)
    for target in targets:
        if journal.get(target["path"], {}).get("status") == "deleted":
            continue
        _check_payload(store, target, "intent")
        path = store.root / target["path"]
        if path.exists():
            path.unlink()
            _sync(path.parent)
            if progress:
                progress(
                    dict(
                        cleanup="payload_unlinked",
                        path=str(path),
                        bytes=target["bytes"],
                    )
                )
        else:
            _sync(path.parent)  # Resolve an interrupted unlink before recording it.
        with store._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE cleanup SET status='deleted' WHERE path=? AND status='intent'",
                (target["path"],),
            )


def summary(store):
    entries = list(_journal(store).values())
    deleted = [e for e in entries if e["status"] == "deleted"]
    return dict(
        policy="qualified_job_owned_payloads_only",
        deleted_files=len(deleted),
        deleted_bytes=sum(e["bytes"] for e in deleted),
        raw_payloads_retained=not entries or len(deleted) != len(entries),
        recovery="Original external cache may remain; otherwise raw reconstruction requires redownload. Lifetime spending is not refunded.",
    )
