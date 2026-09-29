"""Exact canonical source-day projection; never infer event-time coverage."""

from contextlib import ExitStack
from datetime import timedelta
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from .archive import archive_keys
from .checkpoint_io import _CappedOutput
from .download import file_hash
from .lab_config import day
from .proxy_compact import COLUMNS, MAX_BYTES, _batches, _equal_batches, _sync


class _SharedOutput(_CappedOutput):
    def __init__(self, output, limit, used):
        super().__init__(output, limit)
        self.used = used

    def write(self, value):
        if self.used[0] + len(value) > self.limit:
            raise ValueError("Daily compact total byte limit exceeded")
        count = super().write(value)
        self.used[0] += count
        return count


def _row_days(batch, key_days):
    keys = batch.column(batch.schema.get_field_index("source_key")).to_pylist()
    if any(key not in key_days for key in keys):
        raise ValueError("Undeclared canonical source key")
    return [key_days[key] for key in keys]


def _selected(paths, key_days, date):
    for path in paths:
        for batch in _batches(path):
            days = _row_days(batch, key_days)
            selected = batch.filter(pa.array([value == date for value in days]))
            if selected.num_rows:
                yield selected


def daily_projection(paths, data, stage):
    start, end = day(data["start"]), day(data["end"])
    if not 0 < (end - start).days <= 7:
        raise ValueError("Daily compaction requires 1–7 source days")
    dates = [
        (start + timedelta(days=i)).date().isoformat()
        for i in range((end - start).days)
    ]
    key_days = {key: date for date in dates for key in archive_keys(date)}
    declared = data.get("source_keys", [])
    if (
        not isinstance(declared, list)
        or len(declared) != len(key_days)
        or set(declared) != set(key_days)
    ):
        raise ValueError("Incomplete declared source keys for daily compaction")
    if shutil.disk_usage(stage).free < MAX_BYTES + 64 * 1024**2:
        raise ValueError("Insufficient daily compaction free space")
    schema = None
    for path in paths:
        with pq.ParquetFile(path) as reader:
            current = pa.schema([reader.schema_arrow.field(c) for c in COLUMNS])
        if schema is not None and not current.equals(schema, check_metadata=True):
            raise ValueError("Daily compaction requires identical canonical schemas")
        schema = current
    targets = {date: stage / f"fills-{date.replace('-', '')}.parquet" for date in dates}
    owners = {date: [] for date in dates}
    rows = {date: 0 for date in dates}
    used = [0]
    with ExitStack() as stack:
        writers = {}
        for date, target in targets.items():
            stream = stack.enter_context(target.open("xb"))
            writers[date] = stack.enter_context(
                pq.ParquetWriter(
                    _SharedOutput(stream, MAX_BYTES, used),
                    schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            )
        for path in paths:
            for batch in _batches(path):
                days = _row_days(batch, key_days)
                for date in dict.fromkeys(days):
                    selected = batch.filter(pa.array([value == date for value in days]))
                    writers[date].write_batch(selected)
                    rows[date] += selected.num_rows
                    if path not in owners[date]:
                        owners[date].append(path)
    if sum(rows.values()) != data["rows"]:
        raise ValueError("Daily canonical row count changed")
    files = []
    for date, target in targets.items():
        _sync(target)
        _equal_batches(_selected(owners[date], key_days, date), _batches(target))
        files.append(
            dict(
                name=target.name,
                source_day=date,
                rows=rows[date],
                bytes=target.stat().st_size,
                sha256=file_hash(target),
                source_names=[path.name for path in owners[date]],
            )
        )
    for path, entry in zip(paths, data["files"]):
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Source partition changed during daily compaction")
    return files, sum(f["bytes"] for f in files)
