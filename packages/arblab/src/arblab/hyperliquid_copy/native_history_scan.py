"""Bounded observed-time evidence; not listing or funding availability metadata."""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .contracts import symbol
from .download import file_hash
from .prefix_qualification import _engine, _previous, _verify_all

MAX_FILES = 5000
MAX_BYTES = 64 * 1024**3
MAX_BATCH_BYTES = 64 * 1024**2


def scan_qualified_history(report_pin):
    """Scan all pinned rows, including duplicates and source-boundary spill."""
    report_pin = dict(report_pin)
    engine = _engine()
    own_hash = file_hash(Path(__file__))
    report = _previous(report_pin, engine)
    if report is None:
        raise ValueError("Qualified report pin required")
    files, coins = report["files"], report["coins"]
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("Native-history file bound exceeded")
    if (
        not isinstance(coins, list)
        or not 1 <= len(coins) <= 50
        or len(set(coins)) != len(coins)
    ):
        raise ValueError("Invalid native-history scope")
    for coin in coins:
        symbol(coin)
    if any(type(f.get("bytes")) is not int or f["bytes"] < 0 for f in files):
        raise ValueError("Invalid native-history file size")
    if sum(f["bytes"] for f in files) > MAX_BYTES:
        raise ValueError("Native-history byte bound exceeded")
    if len({f["path"] for f in files}) != len(files):
        raise ValueError("Duplicate native-history file")
    _verify_all(report["manifests"], files, report["raw_sources"], report_pin)
    stats = {coin: dict(rows=0, first_event=None, last_event=None) for coin in coins}
    total = 0
    for entry in files:
        count = 0
        with pq.ParquetFile(entry["path"]) as reader:
            for batch in reader.iter_batches(
                batch_size=4096, columns=["coin", "exchange_time"], use_threads=False
            ):
                if batch.nbytes > MAX_BATCH_BYTES:
                    raise ValueError("Native-history decoded batch bound exceeded")
                if batch.schema.names != ["coin", "exchange_time"]:
                    raise ValueError("Missing native-history columns")
                markets, times = batch.columns
                if (
                    not pa.types.is_string(markets.type)
                    or not pa.types.is_timestamp(times.type)
                    or times.type.unit != "us"
                    or times.type.tz != "UTC"
                    or markets.null_count
                    or times.null_count
                ):
                    raise ValueError("Invalid native-history timestamp or market type")
                for coin in pc.unique(markets).to_pylist():
                    if coin not in stats:
                        raise ValueError("Unexpected native-history market")
                    selected = pc.filter(times, pc.equal(markets, coin))
                    bounds = pc.min_max(selected).as_py()
                    row = stats[coin]
                    row["rows"] += len(selected)
                    low, high = bounds["min"], bounds["max"]
                    row["first_event"] = (
                        min(row["first_event"], low)
                        if row["first_event"] is not None
                        else low
                    )
                    row["last_event"] = (
                        max(row["last_event"], high)
                        if row["last_event"] is not None
                        else high
                    )
                count += batch.num_rows
        if type(entry.get("rows")) is not int or count != entry["rows"]:
            raise ValueError("Native-history file row count changed")
        total += count
    if total != report["rows"]:
        raise ValueError("Native-history total row count changed")
    _verify_all(report["manifests"], files, report["raw_sources"], report_pin)
    if _engine() != engine or file_hash(Path(__file__)) != own_hash:
        raise ValueError("Native-history scan engine changed")
    for row in stats.values():
        for field in ("first_event", "last_event"):
            if row[field] is not None:
                row[field] = row[field].isoformat()
    return dict(
        schema="hyperliquid_observed_native_history_v1",
        qualification=report_pin,
        source_start=report["source_start"],
        source_end=report["source_end"],
        rows=total,
        markets=stats,
        native_availability_qualified=False,
        row_semantics="physical canonical rows including duplicates and boundary spill",
        engine=dict(scan_sha256=own_hash, qualification=engine),
    )
