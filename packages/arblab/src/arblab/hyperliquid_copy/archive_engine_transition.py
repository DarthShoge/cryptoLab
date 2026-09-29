"""One explicit baseline-backed engine transition, never download authority."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .archive_cache import _safe
from .archive_job_store import ArchiveJobStore
from .archive_transition_snapshot import (
    DIRECTORY,
    MAX_BYTES,
    _encoded,
    _state,
    load_snapshot,
    read_snapshot,
)
from .archive_transition_evidence import verify_progress
from .download import file_hash
from . import prefix_qualification as prefix
from .proxy_compact import _sync


def _location(store):
    path = store.root / DIRECTORY / "activation.json"
    _safe(path)
    return path


def _generation(old, new):
    versions = (
        old["qualification"].get("version"),
        new["qualification"].get("version"),
    )
    if tuple(map(type, versions)) != (int, int) or versions != (1, 2):
        raise ValueError("Unsupported archive qualification generation transition")


def active_transition(store, *, recheck_content=False, current_engine=None):
    """Return verified active evidence, or None; no writes and no relaxed default."""
    from .archive_job import _engine
    from .archive_job_qualification import verify_report

    path = _location(store)
    if not path.exists():
        return None
    try:
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Activation metadata byte limit exceeded")
        digest = file_hash(path)
        data = json.loads(path.read_bytes())
        if (
            set(data) != {"schema", "snapshot", "baseline", "new_engine", "boundary"}
            or data["schema"] != "hyperliquid_engine_activation_v1"
        ):
            raise ValueError("Unsupported archive engine activation")
        current_engine = _engine() if current_engine is None else current_engine
        if _encoded(data["new_engine"]) != _encoded(current_engine):
            raise ValueError("Activated archive engine changed")
        expected_snapshot = str(path.parent / "manifest.json")
        if (
            set(data["snapshot"]) != {"path", "sha256"}
            or data["snapshot"]["path"] != expected_snapshot
        ):
            raise ValueError("Activation snapshot path changed")
        snapshot = load_snapshot(store.root, data["snapshot"]["sha256"])
        _generation(snapshot["old_engine"], current_engine)
        boundary = data["boundary"]
        if (
            type(boundary) is not int
            or not 1 <= boundary < len(store.metadata["plan"]["batches"])
            or boundary != snapshot["boundary"]
        ):
            raise ValueError("Activation boundary changed")
        baseline_path = Path(data["baseline"]["path"])
        if (
            baseline_path.name != "manifest.json"
            or baseline_path.parent.parent != path.parent / "baseline"
        ):
            raise ValueError("Activation baseline path changed")
        verify_report(
            store,
            boundary - 1,
            data["baseline"],
            engine=prefix._engine(),
            previous=None,
            recheck_content=recheck_content,
        )
        before = verify_progress(
            store, snapshot, data["baseline"], recheck_content=recheck_content
        )
        if (
            file_hash(path) != digest
            or _encoded(_engine()) != _encoded(current_engine)
            or _encoded(_state(store)) != _encoded(before)
        ):
            raise ValueError("Activation changed during verification")
        return data | {
            "old_engine": snapshot["old_engine"],
            "pin": dict(path=str(path), sha256=digest),
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid archive activation") from exc


def _baseline(store, snapshot, progress):
    from .archive_job_qualification import verify_report

    root = store.root / DIRECTORY / "baseline"
    _safe(root)
    children = list(root.iterdir()) if root.exists() else []
    paths = [store.records()[n, "compact"]["path"] for n in range(snapshot["boundary"])]
    if not children:
        result = prefix.qualify_prefix(
            paths,
            previous=None,
            output_root=root,
            temp_root=store.root,
            progress=progress,
        )
    else:
        if len(children) != 1 or not (children[0] / "manifest.json").is_file():
            raise ValueError("Incomplete or ambiguous baseline requires recovery")
        path = children[0] / "manifest.json"
        _safe(path)
        with TemporaryDirectory(prefix="baseline_recheck_", dir=store.root) as scratch:
            repeated = prefix.qualify_prefix(
                paths,
                previous=None,
                output_root=scratch,
                temp_root=scratch,
                progress=progress,
            )
            if file_hash(path) != repeated["sha256"]:
                raise ValueError("Orphan baseline differs from full revalidation")
        result = dict(path=str(path), sha256=file_hash(path))
    verify_report(
        store,
        snapshot["boundary"] - 1,
        result,
        engine=prefix._engine(),
        previous=None,
        recheck_content=True,
    )
    return result


def activate_transition(root, *, snapshot_sha256, progress=None):
    """Requalify baseline and publish under the original lock, entirely offline."""
    from .archive_job import _engine
    from .archive_job_qualification import verify_report

    store = ArchiveJobStore(root)
    with store.locked():
        current = _engine()
        active = active_transition(store, recheck_content=True, current_engine=current)
        if active:
            if active["snapshot"]["sha256"] != snapshot_sha256:
                raise ValueError("Activation preparation pin changed")
            return active["pin"]
        snapshot = read_snapshot(store.root, snapshot_sha256)
        _generation(snapshot["old_engine"], current)
        baseline = _baseline(store, snapshot, progress)
        if progress:
            progress(dict(transition="baseline_verified", baseline=baseline))
        read_snapshot(store.root, snapshot_sha256)
        if _encoded(_engine()) != _encoded(current):
            raise ValueError("Archive engine changed during activation")
        verify_report(
            store,
            snapshot["boundary"] - 1,
            baseline,
            engine=prefix._engine(),
            previous=None,
            recheck_content=True,
        )
        path = _location(store)
        encoded = _encoded(
            dict(
                schema="hyperliquid_engine_activation_v1",
                snapshot=dict(
                    path=str(path.parent / "manifest.json"), sha256=snapshot_sha256
                ),
                baseline=baseline,
                boundary=snapshot["boundary"],
                new_engine=current,
            )
        )
        pending = path.with_name("activation.pending")
        _safe(pending)
        if pending.exists():
            if (
                pending.stat().st_size != len(encoded)
                or pending.read_bytes() != encoded
            ):
                raise ValueError("Interrupted activation content changed")
        else:
            with pending.open("xb") as stream:
                stream.write(encoded)
        _sync(pending)
        os.link(
            pending, path
        )  # Atomic exclusive publication; cannot replace an activation.
        _sync(path.parent)
        pending.unlink()
        _sync(path.parent)
        verified = active_transition(
            store, recheck_content=True, current_engine=current
        )
        return verified["pin"]
