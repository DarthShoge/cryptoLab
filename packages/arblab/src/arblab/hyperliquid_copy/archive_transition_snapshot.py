"""Offline preservation of a stopped engine; never activation or spend authority.

Preparation requires the original engine. Reading a pinned snapshot does not:
the later transition can verify preserved state after source code changes.
"""

import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .archive_cache import _safe
from .archive_job_store import ArchiveJobStore, STAGES
from .download import file_hash
from .proxy_compact import _sync

MAX_BYTES = 16 * 1024**2
DIRECTORY = "engine_transition_v2"


def _encoded(value):
    result = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(result) > MAX_BYTES:
        raise ValueError("Transition metadata byte limit exceeded")
    return result


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid transition digest")
    return value


def _budget(store):
    path = store.root / "budget.sqlite3"
    _safe(path)
    _safe(path.with_name(path.name + "-journal"))
    if not store.metadata["max_download_bytes"] or not path.is_file():
        raise ValueError("Transition requires original lifetime budget")
    expected = sorted(
        (o["key"], o["bytes"], o["etag"])
        for b in store.metadata["plan"]["batches"]
        for o in b["objects"]
    )
    scope = hashlib.sha256(
        json.dumps(expected, separators=(",", ":")).encode()
    ).hexdigest()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if (
            tables != {"metadata", "objects", "reservations"}
            or db.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise ValueError("Unsupported transition budget")
        if db.execute("SELECT scope,max_bytes FROM metadata").fetchall() != [
            (scope, store.metadata["max_download_bytes"])
        ]:
            raise ValueError("Transition budget identity changed")
        objects = db.execute(
            "SELECT key,bytes,etag FROM objects ORDER BY key LIMIT ?",
            [len(expected) + 1],
        ).fetchall()
        reservations = db.execute(
            "SELECT key,bytes FROM reservations ORDER BY key LIMIT ?",
            [len(expected) + 1],
        ).fetchall()
    sizes = {key: size for key, size, _ in expected}
    if (
        objects != expected
        or len(reservations) > len(expected)
        or any(
            type(n) is not int or n <= 0 or sizes.get(k) != n for k, n in reservations
        )
    ):
        raise ValueError("Transition budget objects/reservations changed")
    total = sum(n for _, n in reservations)
    if total > store.metadata["max_download_bytes"]:
        raise ValueError("Transition budget exhausted")
    return dict(
        path=str(path),
        scope=scope,
        max_bytes=store.metadata["max_download_bytes"],
        reservations=[list(r) for r in reservations],
        reserved_bytes=total,
    )


def _state(store):
    if ArchiveJobStore(store.root).frozen != store.frozen:
        raise ValueError("Transition job metadata changed")
    records = store.records()
    with store._connect() as db:
        rows = db.execute(
            "SELECT batch,stage,path,sha256,bytes,qualification_sha256,status FROM cleanup ORDER BY path"
        ).fetchall()
    return dict(
        metadata_sha256=hashlib.sha256(store.frozen.encode()).hexdigest(),
        records=[
            dict(batch=n, stage=s, path=str(r["path"]), sha256=r["sha256"])
            for (n, s), r in sorted(records.items())
        ],
        cleanup=[list(r) for r in rows],
        budget=_budget(store),
    )


def _code(engine):
    result = dict(engine["code"])
    for name, digest in engine["qualification"]["code"].items():
        if name in result and result[name] != digest:
            raise ValueError("Inconsistent old engine source identity")
        result[name] = digest
    if not 1 <= len(result) <= 100:
        raise ValueError("Transition source count exceeded")
    for name, digest in result.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z_]+\.py", name):
            raise ValueError("Unsafe transition source name")
        _digest(digest)
    return result


def _relative(root, relative):
    name = Path(relative)
    if not isinstance(relative, str) or name.is_absolute() or ".." in name.parts:
        raise ValueError("Unsafe transition evidence path")
    path = root / name
    _safe(path)
    return path


def _pin(root, path):
    _safe(path)
    if not path.is_file():
        raise ValueError("Missing transition evidence")
    return dict(
        path=str(path.relative_to(root)),
        bytes=path.stat().st_size,
        sha256=file_hash(path),
    )


def _verify_pin(root, pin):
    path = _relative(root, pin["path"])
    _digest(pin["sha256"])
    if (
        type(pin["bytes"]) is not int
        or pin["bytes"] < 0
        or not path.is_file()
        or path.stat().st_size != pin["bytes"]
        or file_hash(path) != pin["sha256"]
    ):
        raise ValueError("Transition evidence identity mismatch")


def _validated_inputs(job):
    from . import archive_job_artifacts as artifacts
    from . import archive_job_cleanup as cleanup
    from . import archive_job_qualification as qualification

    store, records = job.store, job.store.records()
    boundary = sum(stage == "qualified" for _, stage in records)
    expected = {(n, s) for n in range(boundary) for s in STAGES}
    expected |= {(boundary, s) for s in STAGES[:-1]}
    if boundary < 1 or set(records) != expected:
        raise ValueError(
            "Transition requires qualified prefix/pending compact sequence"
        )
    allowed = {f"{n:04d}" for n in range(boundary + 1)}
    for batch_root in (store.root / "batches").iterdir():
        _safe(batch_root)
        if not batch_root.is_dir() or batch_root.name not in allowed:
            raise ValueError("Unexpected transition batch stage")
        for stage_root in batch_root.iterdir():
            _safe(stage_root)
            if not stage_root.is_dir() or stage_root.name not in STAGES:
                raise ValueError("Unexpected transition artifact stage")
    for (n, stage), record in records.items():
        if store.find(n, stage) != record["path"]:
            raise ValueError("Transition artifact path changed")
    for n in range(boundary + 1):
        artifacts.compact(
            records[n, "compact"]["path"],
            job.batches[n],
            job.metadata["coins"],
            records[n, "raw"],
            records[n, "normalized"],
        )
        if n < boundary:
            qualification.verify(
                store,
                n,
                qualification.pin(records[n, "qualified"]),
                recheck_content=n == boundary - 1,
            )
    if store.stage_root(boundary, "qualified").exists() and list(
        store.stage_root(boundary, "qualified").iterdir()
    ):
        raise ValueError("Uncommitted qualification requires separate recovery")
    artifacts.raw(records[boundary, "raw"]["path"], job.batches[boundary])
    artifacts.normalized(
        records[boundary, "normalized"]["path"],
        job.batches[boundary],
        job.metadata["coins"],
        records[boundary, "raw"],
    )
    targets = cleanup._targets(store, boundary)
    journal = cleanup._journal(store)
    expected_targets = {e["path"]: e for e in targets}
    if set(journal) - set(expected_targets):
        raise ValueError("Unexpected transition cleanup targets")
    journaled_batches = {e["batch"] for e in journal.values()}
    for target in targets:
        saved = journal.get(target["path"])
        if saved is None and target["batch"] in journaled_batches:
            raise ValueError("Incomplete transition cleanup journal")
        if saved and (
            {k: saved[k] for k in cleanup.FIELDS} != target
            or saved["status"] != "deleted"
        ):
            raise ValueError("Unfinished or changed transition cleanup journal")
        cleanup._check_payload(store, target, saved["status"] if saved else None)
    deleted = {p for p, e in journal.items() if e["status"] == "deleted"}
    pins = []
    for (_, stage), record in sorted(records.items()):
        path = record["path"]
        pins.append(_pin(store.root, path))
        if stage == "qualified":
            continue
        data = artifacts.read(path)
        for entry in data["objects" if stage == "raw" else "files"]:
            name = entry["file" if stage == "raw" else "name"]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Unsafe transition payload name")
            payload = path.parent / name
            if str(payload.relative_to(store.root)) not in deleted:
                pin = _pin(store.root, payload)
                if (pin["bytes"], pin["sha256"]) != (entry["bytes"], entry["sha256"]):
                    raise ValueError("Transition payload identity mismatch")
                pins.append(pin)
    return boundary, pins


def load_snapshot(root, sha256):
    """Verify immutable snapshot/source bytes only; not current-state authority."""
    _digest(sha256)
    store = ArchiveJobStore(root)
    folder = store.root / DIRECTORY
    path = folder / "manifest.json"
    _safe(path)
    try:
        if path.stat().st_size > MAX_BYTES or file_hash(path) != sha256:
            raise ValueError("Transition snapshot identity mismatch")
        data = json.loads(path.read_bytes())
        if (
            set(data)
            != {
                "schema",
                "boundary",
                "old_engine",
                "metadata_sha256",
                "records",
                "cleanup",
                "budget",
                "sources",
                "retained",
            }
            or data["schema"] != "hyperliquid_engine_preparation_v1"
        ):
            raise ValueError("Unsupported transition snapshot")
        if (
            data["old_engine"] != store.metadata["engine"]
            or data["metadata_sha256"]
            != hashlib.sha256(store.frozen.encode()).hexdigest()
        ):
            raise ValueError("Transition original metadata changed")
        code = _code(data["old_engine"])
        if {e["name"]: e["sha256"] for e in data["sources"]} != code or len(
            data["sources"]
        ) != len(code):
            raise ValueError("Transition source inventory changed")
        if sum(e["bytes"] for e in data["sources"]) > MAX_BYTES:
            raise ValueError("Transition source bytes exceeded")
        for entry in data["sources"]:
            _verify_pin(
                folder / "source",
                dict(path=entry["name"], bytes=entry["bytes"], sha256=entry["sha256"]),
            )
        if file_hash(path) != sha256:
            raise ValueError("Transition changed during verification")
        return data
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid transition snapshot") from exc


def read_snapshot(root, sha256):
    """Verify exact prepared state and retained bytes under the caller's lock."""
    store = ArchiveJobStore(root)
    before = _state(store)
    data = load_snapshot(root, sha256)
    if _encoded(before) != _encoded({k: data[k] for k in before}):
        raise ValueError("Transition prepared state changed")
    for entry in data["retained"]:
        _verify_pin(store.root, entry)
    for row in data["cleanup"]:
        if row[-1] == "deleted" and _relative(store.root, row[2]).exists():
            raise ValueError("Deleted transition payload recreated")
    if _encoded(_state(store)) != _encoded(before):
        raise ValueError("Transition changed during verification")
    load_snapshot(root, sha256)
    return data


def prepare_transition(root):
    """Capture a matching stopped engine. No activation, network, or cleanup."""
    from .archive_job import ArchiveJob, _engine

    job = ArchiveJob(root)
    store = job.store
    with store.locked():
        final = store.root / DIRECTORY
        pending = store.root / (DIRECTORY + ".partial")
        _safe(final)
        _safe(pending)
        if pending.exists():
            raise ValueError("Incomplete transition preparation requires recovery")
        if final.exists():
            path = final / "manifest.json"
            _safe(path)
            digest = file_hash(path)
            saved = read_snapshot(store.root, digest)
            boundary, retained = _validated_inputs(job)
            if (
                type(saved["boundary"]) is not int
                or saved["boundary"] != boundary
                or _encoded(saved["retained"]) != _encoded(retained)
            ):
                raise ValueError("Untrusted transition preparation membership changed")
            return dict(path=str(path), sha256=digest)
        before = _state(store)
        boundary, retained = _validated_inputs(job)
        code = _code(job.metadata["engine"])
        source_root = Path(__file__).parent
        sources = []
        content = {}
        total = 0
        for name, digest in sorted(code.items()):
            path = source_root / name
            _safe(path)
            total += path.stat().st_size
            if total > MAX_BYTES:
                raise ValueError("Transition source byte limit exceeded")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Old engine source changed")
            sources.append(dict(name=name, bytes=len(raw), sha256=digest))
            content[name] = raw
        data = dict(
            schema="hyperliquid_engine_preparation_v1",
            boundary=boundary,
            old_engine=job.metadata["engine"],
            **before,
            sources=sources,
            retained=retained,
        )
        encoded = _encoded(data)
        if (
            _encoded(_state(store)) != _encoded(before)
            or _engine() != job.metadata["engine"]
        ):
            raise ValueError("Transition state/engine changed before preservation")
        pending.mkdir()
        (pending / "source").mkdir()
        for name, raw in content.items():
            with (pending / "source" / name).open("xb") as stream:
                stream.write(raw)
            _sync(pending / "source" / name)
        with (pending / "manifest.json").open("xb") as stream:
            stream.write(encoded)
        _sync(pending / "manifest.json")
        _sync(pending / "source")
        _sync(pending)
        pending.rename(final)
        _sync(store.root)
        digest = hashlib.sha256(encoded).hexdigest()
        read_snapshot(store.root, digest)
        return dict(path=str(final / "manifest.json"), sha256=digest)
