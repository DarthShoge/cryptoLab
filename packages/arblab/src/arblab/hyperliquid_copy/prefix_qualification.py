"""Durable source/canonical evidence; not event-completeness or cleanup authority."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import duckdb
import pyarrow

from .activity_checkpoints import _verify
from .archive_cache import _safe
from .download import file_hash
from .proxy_compact import _sync
from .qualification_bounds import collect, read_json, trade_bounds, overlaps
from .sharded_validation import qualified_seed

MAX_REPORT_BYTES = 16 * 1024**2


def _engine():
    root = Path(__file__).parent
    names = (
        "prefix_qualification.py",
        "qualification_bounds.py",
        "sharded_validation.py",
        "proxy_activity.py",
        "contracts.py",
        "compact_catalog.py",
        "archive.py",
        "activity_checkpoints.py",
        "checkpoint_io.py",
        "proxy_availability.py",
    )
    return dict(
        version=2,
        duckdb=duckdb.__version__,
        pyarrow=pyarrow.__version__,
        code={n: file_hash(root / n) for n in names},
    )


def _previous(pin, engine):
    if pin is None:
        return None
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        raise ValueError("Pinned previous report identity required")
    path = Path(pin["path"]).absolute()
    _safe(path)
    if file_hash(path) != pin["sha256"]:
        raise ValueError("Previous report identity mismatch")
    data = read_json(path, MAX_REPORT_BYTES)
    if file_hash(path) != pin["sha256"]:
        raise ValueError("Previous report identity mismatch")
    if (
        data.get("schema") != "hyperliquid_canonical_prefix_v1"
        or data.get("status") != "canonical_prefix_validated"
        or data.get("engine") != engine
    ):
        raise ValueError("Previous prefix qualification engine/status mismatch")
    return data


def _verify_all(manifests, files, raw_sources, previous):
    _verify(files)
    for entry in [*manifests, *raw_sources, *([previous] if previous else [])]:
        path = Path(entry["path"])
        _safe(path)
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Qualification input or previous report identity changed")


def _publish(report, output_root):
    encoded = json.dumps(report, sort_keys=True, indent=2).encode()
    if len(encoded) > MAX_REPORT_BYTES:
        raise ValueError("Qualification metadata byte limit exceeded")
    root = Path(output_root).absolute()
    _safe(root)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="qualification_", suffix=".partial", dir=root))
    pending = stage / "manifest.pending"
    with pending.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    final = stage.with_suffix("")
    if final.exists():
        raise ValueError("Qualification publication destination exists")
    stage.rename(final)
    path = final / "manifest.json"
    (final / "manifest.pending").rename(path)
    _sync(final)
    _sync(root)
    _sync(root.parent)
    return dict(path=str(path), sha256=hashlib.sha256(encoded).hexdigest())


def qualify_prefix(
    manifest_paths, *, output_root, temp_root, previous=None, buckets=32, progress=None
):
    engine = _engine()
    prior = _previous(previous, engine)
    manifests, files, sources, coins = collect(manifest_paths)
    old_count = 0
    if prior is not None:
        old_count = len(prior["manifests"])
        if old_count >= len(manifests) or manifests[:old_count] != prior["manifests"]:
            raise ValueError("Qualification requires an exact strict prefix extension")
        if prior["coins"] != coins:
            raise ValueError("Previous prefix market scope changed")
    old_files = {e["path"]: e for e in prior["files"]} if prior else {}
    new_files, older = [], []
    for entry in files:
        old = old_files.get(entry["path"])
        if old is not None:
            if any(old.get(k) != v for k, v in entry.items()):
                raise ValueError("Previous prefix file identity/schema changed")
            entry["trade_bounds"] = old["trade_bounds"]
            older.append(entry)
        else:
            entry["trade_bounds"] = trade_bounds(entry, coins)
            new_files.append(entry)
    candidates = [
        e
        for e in older
        if any(overlaps(e["trade_bounds"], n["trade_bounds"]) for n in new_files)
    ] + new_files
    observed = [e for e in files if e["rows"]]
    if not observed:
        raise ValueError("No canonical rows to qualify")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    start = epoch + timedelta(microseconds=min(e["min_time"] for e in observed))
    end = epoch + timedelta(microseconds=max(e["max_time"] for e in observed) + 1)
    report = dict(
        schema="hyperliquid_canonical_prefix_v1",
        status="canonical_prefix_validated",
        engine=engine,
        coins=coins,
        manifests=manifests,
        files=files,
        raw_sources=sources,
        source_start=sources[0]["start"],
        source_end=sources[-1]["end"],
        observed_start=start.isoformat(),
        observed_end_exclusive=end.isoformat(),
        coverage_basis="complete_frozen_source_partitions_and_exact_event_bounds_only",
        research_eligible=False,
        raw_disposal_authorized=False,
        rows=sum(e["rows"] for e in files),
        bytes=sum(e["bytes"] for e in files),
        candidate_files=len(candidates),
        skipped_old_files=len(files) - len(candidates),
        previous=previous,
        buckets=buckets,
    )
    with qualified_seed(
        candidates,
        coins,
        start,
        end,
        start,
        temp_root=temp_root,
        buckets=buckets,
        progress=progress,
    ):
        _verify_all(manifests, files, sources, previous)
        if _engine() != engine:
            raise ValueError("Qualification engine changed during validation")
        return _publish(report, output_root)
