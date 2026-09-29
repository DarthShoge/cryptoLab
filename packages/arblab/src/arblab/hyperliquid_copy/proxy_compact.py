"""Local canonical projection proof. Never downloads, deletes, or grants coverage."""

from dataclasses import fields
import json
import os
from pathlib import Path
import resource
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import FillEvent
from .download import file_hash


MAX_BYTES = 8 * 1024**3
MAX_PARTITIONS = 7 * 24 + 1  # Both hour-8 objects on the 2025-07-27 handoff.
MAX_BATCH_BYTES = 64 * 1024**2
COLUMNS = [f.name for f in fields(FillEvent) if f.name != "raw_details_json"]


def _sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _batches(path):
    with pq.ParquetFile(path) as reader:
        for batch in reader.iter_batches(
            batch_size=2048, columns=COLUMNS, use_threads=False
        ):
            if batch.nbytes > MAX_BATCH_BYTES:
                raise ValueError("Decoded compact batch byte limit exceeded")
            yield batch


def _equal_projection(source, target):
    _equal_batches(_batches(source), _batches(target))


def _equal_batches(left, right):
    # Arrow row-group boundaries need not survive a rewrite. Align row slices,
    # not record-batch boundaries; at most one batch from each side is held.
    left, right = iter(left), iter(right)
    a, b = next(left, None), next(right, None)
    while a is not None and b is not None:
        n = min(a.num_rows, b.num_rows)
        if not a.slice(0, n).equals(b.slice(0, n)):
            raise ValueError("Canonical projection differs")
        a = a.slice(n) if n < a.num_rows else next(left, None)
        b = b.slice(n) if n < b.num_rows else next(right, None)
    if a is not None or b is not None:
        raise ValueError("Canonical projection row count differs")


def _inputs(source, data):
    if (
        data.get("schema") != "hyperliquid_proxy_activity_v1"
        or data.get("complete") is not True
    ):
        raise ValueError("Require complete normalized activity manifest")
    if data.get("scope") != "all_wallets_for_declared_markets":
        raise ValueError("Unsupported activity scope")
    entries = data.get("files", [])
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_PARTITIONS:
        raise ValueError(f"Require 1..{MAX_PARTITIONS} source partitions")
    paths, rows, size = [], 0, 0
    for entry in entries:
        name = entry["name"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".parquet")
        ):
            raise ValueError("Invalid partition path")
        path = source.parent / name
        if path.is_symlink() or not path.is_file() or path in paths:
            raise ValueError("Missing, symlink or duplicate partition")
        actual = path.stat().st_size
        if actual != entry["bytes"] or actual <= 0 or actual > MAX_BYTES:
            raise ValueError("Partition byte identity/limit mismatch")
        size += actual
        if size > MAX_BYTES:
            raise ValueError("Input byte limit exceeded")
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Partition hash identity mismatch")
        with pq.ParquetFile(path) as reader:
            names = reader.schema_arrow.names
            if len(set(names)) != len(names) or set(names) not in (
                set(COLUMNS),
                set(COLUMNS) | {"raw_details_json"},
            ):
                raise ValueError("Unsupported canonical columns")
            if reader.metadata.num_rows != entry["rows"]:
                raise ValueError("Partition row count mismatch")
            rows += reader.metadata.num_rows
        paths.append(path)
    if rows != data.get("rows") or rows <= 0:
        raise ValueError("Manifest row count mismatch")
    if size != data.get("output_bytes"):
        raise ValueError("Manifest byte count mismatch")
    return paths, size, rows


def compact_history(manifest_path, output_root, *, partitioning="source_file"):
    """Lossless typed projection into a new immutable directory.

    Input must already be parser-normalized. This validates projection and source
    identities, not all economics or archive completeness. Batch byte checks occur
    after Arrow decoding; reported RSS is measured, not a hard allocator ceiling.
    Unpublished stages remain for inspection. A failure during final publication
    can leave manifest.pending or a visible manifest whose durability is unconfirmed;
    this proof tool has no resume/cleanup authorization based on either state.
    """
    started = time.monotonic()
    if partitioning not in ("source_file", "source_day"):
        raise ValueError("Invalid compact partitioning")
    source = Path(manifest_path).resolve(strict=True)
    if source.stat().st_size > 1_000_000:
        raise ValueError("Manifest byte limit exceeded")
    source_hash = file_hash(source)
    data = json.loads(source.read_text())
    paths, input_bytes, rows = _inputs(source, data)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="compact_", suffix=".partial", dir=root))
    if partitioning == "source_day":
        from .daily_compact import daily_projection

        result_files, output_bytes = daily_projection(paths, data, stage)
    else:
        result_files, output_bytes = _file_projection(paths, data, stage)
    if file_hash(source) != source_hash:
        raise ValueError("Source manifest changed during compaction")
    result = dict(
        schema="hyperliquid_compact_history_v1",
        complete=True,
        source_manifest=str(source),
        source_manifest_sha256=source_hash,
        source_evidence=data,
        projection_version=1,
        partitioning=partitioning,
        retained_columns=COLUMNS,
        discarded_columns=["raw_details_json"],
        validation="exact_canonical_projection_only",
        rows=rows,
        files=result_files,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        output_input_ratio=output_bytes / input_bytes,
        elapsed_seconds=time.monotonic() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    pending = stage / "manifest.pending"
    with pending.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    final = stage.with_suffix("")
    # Unique mkdtemp identity; refuse any pre-existing final destination.
    if final.exists():
        raise ValueError("Compact publication target already exists")
    stage.rename(final)
    (final / "manifest.pending").rename(final / "manifest.json")
    _sync(final)
    _sync(root)
    return final / "manifest.json"


def _file_projection(paths, data, stage):
    result_files, output_bytes = [], 0
    for index, (path, entry) in enumerate(zip(paths, data["files"])):
        target = stage / f"fills-{index:04d}.parquet"
        with pq.ParquetFile(path) as reader:
            schema = pa.schema([reader.schema_arrow.field(c) for c in COLUMNS])
        with pq.ParquetWriter(
            target, schema, compression="zstd", use_dictionary=True
        ) as writer:
            for batch in _batches(path):
                writer.write_batch(batch)
                if output_bytes + target.stat().st_size > MAX_BYTES:
                    raise ValueError("Output byte limit exceeded")
        _sync(target)
        output_bytes += target.stat().st_size
        if output_bytes > MAX_BYTES:
            raise ValueError("Output byte limit exceeded")
        _equal_projection(path, target)
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Source partition changed during compaction")
        result_files.append(
            dict(
                name=target.name,
                rows=entry["rows"],
                bytes=target.stat().st_size,
                sha256=file_hash(target),
                source_name=path.name,
                source_sha256=entry["sha256"],
                source_bytes=entry["bytes"],
            )
        )
    return result_files, output_bytes
