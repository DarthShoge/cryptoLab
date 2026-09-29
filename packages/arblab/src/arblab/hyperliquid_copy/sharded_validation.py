"""Full-corpus validation with bounded deterministic trade groups and causal seeds.

No qualification is yielded until all groups pass and source hashes still match.
Hashing partitions work only; exact equality remains ProxyActivity's responsibility.
"""

from contextlib import contextmanager
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import duckdb
import pyarrow.parquet as pq

from .activity_checkpoints import _verify, MAX_SEED_BYTES, MAX_SEED_ROWS
from .checkpoint_io import write_seed
from .contracts import utc
from .proxy_activity import ProxyActivity, REVERSE_ORDER
from .proxy_availability import validate_native_fills
from .proxy_compact import COLUMNS

MAX_CORPUS_BYTES = 64 * 1024**3
MAX_BUCKET_BYTES = 2 * 1024**3
SPILL_BYTES = 2 * 1024**3


def _merge_seed(activity, previous, target, cutoff):
    columns = ",".join(f'"{name}"' for name in COLUMNS)
    candidates = f"SELECT {columns} FROM fills WHERE exchange_time < ?"
    if previous is not None:
        activity.db.read_parquet(str(previous), hive_partitioning=False).create_view(
            "previous_seed"
        )
        candidates += f" UNION ALL SELECT {columns} FROM previous_seed"
    prefix = f"WITH candidates AS ({candidates}) "
    count = activity.db.execute(
        prefix + "SELECT count(DISTINCT (user,coin)) FROM candidates", [cutoff]
    ).fetchone()[0]
    if count > MAX_SEED_ROWS:
        raise ValueError("Qualification seed row limit exceeded")
    query = activity.db.execute(
        prefix
        + f"SELECT {columns} FROM candidates QUALIFY row_number() OVER (PARTITION BY user,coin ORDER BY {REVERSE_ORDER})=1",
        [cutoff],
    )
    write_seed(query, target, MAX_SEED_BYTES)


@contextmanager
def qualified_seed(
    entries,
    coins,
    start,
    end,
    cutoff,
    *,
    temp_root,
    buckets=32,
    native_starts=None,
    progress=None,
):
    """Yield an owned temporary seed, never a durable coverage certificate.

    Caller must bind frozen source entries and engine versions to any publication.
    One DuckDB connection is open at a time. Disk exhaustion/skew fails closed;
    interruption restarts validation, without deleting canonical input history.
    """
    start, end, cutoff = utc(start), utc(end), utc(cutoff)
    if not start <= cutoff <= end or start >= end:
        raise ValueError("Invalid qualification bounds")
    if type(buckets) is not int or buckets not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("Invalid qualification bucket count")
    entries = [dict(e) for e in entries]
    if not 0 < len(entries) <= 5000 or len({e["path"] for e in entries}) != len(
        entries
    ):
        raise ValueError("Invalid qualification source partitions")
    if (
        any(type(e["bytes"]) is not int or e["bytes"] <= 0 for e in entries)
        or sum(e["bytes"] for e in entries) > MAX_CORPUS_BYTES
    ):
        raise ValueError("Qualification corpus byte limit exceeded")
    _verify(entries)
    required = MAX_BUCKET_BYTES + 2 * MAX_SEED_BYTES + SPILL_BYTES + 64 * 1024**2
    if shutil.disk_usage(temp_root).free < required:
        raise ValueError("Insufficient qualification scratch space")
    paths = [e["path"] for e in entries]
    expected_rows = sum(pq.read_metadata(p).num_rows for p in paths)
    processed_rows = 0
    columns = ",".join(f'"{name}"' for name in COLUMNS)
    with TemporaryDirectory(prefix="qualified_seed_", dir=temp_root) as scratch:
        root, previous = Path(scratch), None
        for bucket in range(buckets):
            shard = root / "bucket.parquet"
            # Streaming the global union preserves the original common numeric
            # types. Do not invent per-file economic fingerprint normalization.
            with duckdb.connect(
                config={
                    "memory_limit": "256MB",
                    "max_temp_directory_size": "2GB",
                    "temp_directory": str(root / "spill"),
                    "threads": 1,
                    "TimeZone": "UTC",
                    "preserve_insertion_order": False,
                }
            ) as scan:
                scan.read_parquet(
                    paths, union_by_name=True, hive_partitioning=False
                ).create_view("all_fills")
                query = scan.execute(
                    f"SELECT {columns} FROM all_fills WHERE hash(coin,tid) % ? = ?",
                    [buckets, bucket],
                )
                write_seed(query, shard, MAX_BUCKET_BYTES)
            processed_rows += pq.read_metadata(shard).num_rows
            target = root / f"seed-{bucket}.parquet"
            with ProxyActivity([shard], temp_root=root) as activity:
                activity.validate_registered_scope(coins, start, end)
                validate_native_fills(activity, native_starts or {})
                _merge_seed(activity, previous, target, cutoff)
            if previous is not None:
                previous.unlink()  # Owned disposable seed; original data untouched.
            previous = target
            shard.unlink()
            if progress:
                progress(
                    dict(
                        completed_buckets=bucket + 1,
                        buckets=buckets,
                        source_rows=processed_rows,
                    )
                )
        if processed_rows != expected_rows:
            raise ValueError("Qualification source row count changed")
        _verify(entries)
        yield previous
