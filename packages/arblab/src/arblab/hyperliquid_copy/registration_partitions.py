"""Bounded interior projection into caller-owned, unpublished dataset staging."""

from datetime import datetime, timedelta, timezone
import errno
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .compact_catalog import event_bounds
from .checkpoint_io import _CappedOutput
from .contracts import utc
from .download import file_hash
from .market_bundle_inputs import pin_file, recheck
from .proxy_compact import _sync

MAX_FILES = 5000
MAX_BYTES = 64 * 1024**3
MAX_BATCH_BYTES = 64 * 1024**2
RESERVE_BYTES = 64 * 1024**2
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def free_space(stage, required):
    if shutil.disk_usage(stage).free < required + RESERVE_BYTES:
        raise ValueError("Insufficient staging free space")


def link_or_copy(source, destination, *, expected_bytes=None):
    maximum = source.stat().st_size if expected_bytes is None else expected_bytes
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
            raise
        free_space(destination.parent, maximum)
        with source.open("rb") as reader, destination.open("xb") as writer:
            remaining = maximum
            while remaining:
                block = reader.read(min(remaining, 1024**2))
                if not block:
                    raise ValueError("Source size changed during copy")
                writer.write(block)
                remaining -= len(block)
            if reader.read(1):
                raise ValueError("Source exceeds pinned copy byte limit")


def retain_full_source(files, stage):
    """Retain canonical bytes, including boundary spill, in unpublished staging."""
    files = list(files)
    if not 1 <= len(files) <= MAX_FILES - 2:
        raise ValueError("Partition file bound exceeded")
    stage = Path(stage).absolute()
    if not stage.is_dir() or any(p.is_symlink() for p in (stage, *stage.parents)):
        raise ValueError("Expected existing safe staging directory")
    pins, prepared, total, schema = {}, [], 0, None
    for index, entry in enumerate(files):
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise ValueError("Invalid partition byte identity")
        total += entry["bytes"]
        if total > MAX_BYTES:
            raise ValueError("Partition input byte bound exceeded")
        source = pin_file(
            entry["path"], pins, digest=entry["sha256"], size=entry["bytes"]
        )
        destination = stage / f"fills-{index:04d}.parquet"
        if destination.exists() or destination.is_symlink():
            raise ValueError("Partition output already exists")
        with pq.ParquetFile(source) as reader:
            current = reader.schema_arrow
            if (
                type(entry.get("rows")) is not int
                or reader.metadata.num_rows != entry["rows"]
            ):
                raise ValueError("Partition row identity changed")
            if schema is not None and not schema.equals(current, check_metadata=True):
                raise ValueError("Partition schema mismatch")
            schema = current
            kind = schema.field("exchange_time").type
            if not pa.types.is_timestamp(kind) or kind.unit != "us" or kind.tz != "UTC":
                raise ValueError("Expected UTC microsecond event timestamp")
        prepared.append((source, destination, dict(entry)))
    if len(pins) != len(files):
        raise ValueError("Duplicate source partition")
    outputs = []
    for source, destination, entry in prepared:
        free_space(stage, 0)
        link_or_copy(source, destination, expected_bytes=entry["bytes"])
        pin_file(destination, {}, digest=entry["sha256"], size=entry["bytes"])
        with pq.ParquetFile(destination) as reader:
            if reader.metadata.num_rows != entry["rows"] or not schema.equals(
                reader.schema_arrow, check_metadata=True
            ):
                raise ValueError("Retained partition identity changed")
        _sync(destination)
        outputs.append(
            dict(
                name=destination.name,
                **{k: entry[k] for k in ("bytes", "rows", "sha256")},
            )
        )
    recheck(pins)
    _sync(stage)
    return outputs


def filter_partition(source, destination, schema, start, end, remaining):
    with destination.open("xb") as stream:
        with pq.ParquetWriter(
            _CappedOutput(stream, remaining), schema, compression="zstd"
        ) as writer:
            with pq.ParquetFile(source) as reader:
                for batch in reader.iter_batches(batch_size=4096, use_threads=False):
                    if batch.nbytes > MAX_BATCH_BYTES:
                        raise ValueError("Decoded partition batch bound exceeded")
                    times = batch.column(batch.schema.get_field_index("exchange_time"))
                    if times.null_count:
                        raise ValueError("Null event timestamp")
                    mask = pc.and_(pc.greater_equal(times, start), pc.less(times, end))
                    kept = batch.filter(mask)
                    free_space(destination.parent, kept.nbytes)
                    writer.write_batch(kept)
                    if stream.tell() > remaining:
                        raise ValueError("Retained partition byte bound exceeded")


def publish_interior(files, start, end, stage):
    """Keep every canonical column/duplicate in [start,end); never delete inputs.

    The caller owns staging and must not publish it after any failure. Interior
    hardlinks remain protected by the registered dataset's immutable hash checks.
    """
    start, end = utc(start), utc(end)
    if not start < end or end - start > timedelta(days=732):
        raise ValueError("Invalid interior interval")
    files = list(files)
    if not 1 <= len(files) <= MAX_FILES:
        raise ValueError("Partition file bound exceeded")
    stage = Path(stage).absolute()
    if not stage.is_dir() or any(p.is_symlink() for p in (stage, *stage.parents)):
        raise ValueError("Expected existing safe staging directory")
    pins, prepared, total = {}, [], 0
    schema = None
    for index, entry in enumerate(files):
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise ValueError("Invalid partition byte identity")
        total += entry["bytes"]
        if total > MAX_BYTES:
            raise ValueError("Partition input byte bound exceeded")
        source = pin_file(
            entry["path"], pins, digest=entry["sha256"], size=entry["bytes"]
        )
        destination = stage / f"fills-{index:04d}.parquet"
        if destination.exists() or destination.is_symlink():
            raise ValueError("Partition output already exists")
        with pq.ParquetFile(source) as reader:
            current = reader.schema_arrow
            if (
                type(entry.get("rows")) is not int
                or reader.metadata.num_rows != entry["rows"]
            ):
                raise ValueError("Partition row identity changed")
            if schema is not None and not schema.equals(current, check_metadata=True):
                raise ValueError("Partition schema mismatch")
            schema = current
            kind = schema.field("exchange_time").type
            if not pa.types.is_timestamp(kind) or kind.unit != "us" or kind.tz != "UTC":
                raise ValueError("Expected UTC microsecond event timestamp")
        prepared.append((source, destination, event_bounds(source)))
    if len(pins) != len(files):
        raise ValueError("Duplicate source partition")
    lower, upper = ((at - EPOCH) // timedelta(microseconds=1) for at in (start, end))
    outputs, retained = [], 0
    for source, destination, (low, high) in prepared:
        free_space(stage, 0)
        if low is not None and lower <= low <= high < upper:
            if source.stat().st_size > MAX_BYTES - retained:
                raise ValueError("Retained partition byte bound exceeded")
            link_or_copy(source, destination)
        else:
            filter_partition(
                source, destination, schema, start, end, MAX_BYTES - retained
            )
        size = destination.stat().st_size
        retained += size
        if retained > MAX_BYTES:
            raise ValueError("Retained partition byte bound exceeded")
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
        outputs.append(
            dict(
                name=destination.name,
                bytes=size,
                rows=pq.ParquetFile(destination).metadata.num_rows,
                sha256=file_hash(destination),
            )
        )
    recheck(pins)
    _sync(stage)
    return outputs
