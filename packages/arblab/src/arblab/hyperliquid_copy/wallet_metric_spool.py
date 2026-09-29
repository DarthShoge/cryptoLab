"""Bounded ephemeral numeric observations; ordered sums and exact disk medians.

This primitive is not a shared cache. Its caller must reserve spool and query
resources together and release source-query resources before querying medians.
"""

from math import isfinite
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import shutil

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .checkpoint_io import _CappedOutput

MAX_BYTES = 512 * 1024**2
RESERVE_BYTES = 64 * 1024**2
SPILL_BYTES = 2 * 1024**3
MAX_ROW_GROUPS = 4096
SCHEMA = pa.schema(
    [("kind", pa.string()), *[(n, pa.float64()) for n in ("a", "b", "c")]]
)


class MetricSpool:
    def __init__(
        self,
        temp_root,
        *,
        buffer_rows=4096,
        max_bytes=MAX_BYTES,
        max_row_groups=MAX_ROW_GROUPS,
    ):
        if (
            type(buffer_rows) is not int
            or not 1 <= buffer_rows <= 4096
            or type(max_bytes) is not int
            or not 0 < max_bytes <= MAX_BYTES
            or type(max_row_groups) is not int
            or not 0 < max_row_groups <= MAX_ROW_GROUPS
        ):
            raise ValueError("Invalid metric spool resource limit")
        self.root = Path(temp_root).absolute()
        if not self.root.is_dir() or any(
            p.is_symlink() for p in (self.root, *self.root.parents)
        ):
            raise ValueError("Expected safe existing metric scratch root")
        self.buffer_rows, self.max_bytes = buffer_rows, max_bytes
        self.max_row_groups, self._row_groups = max_row_groups, 0
        self._state = "new"
        self._buffer = []
        self._counts = {"fill": 0, "episode": 0}
        self._writer = self._handle = None
        self.disk_bytes = self.peak_buffered_rows = 0

    @property
    def counts(self):
        return dict(self._counts)

    def __enter__(self):
        if self._state != "new":
            raise ValueError("Metric spool cannot be reopened")
        self._space(0)
        self._temp = TemporaryDirectory(prefix="wallet_metrics_", dir=self.root)
        self.directory = Path(self._temp.name)
        self.path = self.directory / "observations.parquet"
        self._state = "open"
        return self

    def _space(self, size):
        if shutil.disk_usage(self.root).free < size + RESERVE_BYTES:
            raise ValueError("Insufficient metric spool free space")

    def add(self, kind, a, b, c):
        if self._state != "open":
            raise ValueError("Metric spool is not open or is finalized/sealed")
        if kind not in self._counts or any(
            type(v) not in (int, float) or not isfinite(v) for v in (a, b, c)
        ):
            raise ValueError("Invalid numeric metric observation")
        self._buffer.append(dict(kind=kind, a=float(a), b=float(b), c=float(c)))
        self._counts[kind] += 1
        self.peak_buffered_rows = max(self.peak_buffered_rows, len(self._buffer))
        if len(self._buffer) == self.buffer_rows:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        if self._row_groups >= self.max_row_groups:
            raise ValueError("Metric spool row-group metadata limit exceeded")
        table = pa.Table.from_pylist(self._buffer, schema=SCHEMA)
        if table.nbytes > RESERVE_BYTES:
            raise ValueError("Metric decoded batch byte limit exceeded")
        self._space(table.nbytes)
        if self._writer is None:
            self._handle = self.path.open("xb")
            self._writer = pq.ParquetWriter(
                _CappedOutput(self._handle, self.max_bytes), SCHEMA, compression="zstd"
            )
        self._writer.write_table(table)
        self._row_groups += 1
        self.disk_bytes = self._handle.tell()
        self._buffer.clear()

    def _seal(self):
        if self._state not in ("open", "sealed"):
            raise ValueError("Metric spool is not open")
        if self._state == "open":
            if self._writer is not None:
                self._flush()
                writer, self._writer = self._writer, None
                try:
                    writer.close()
                finally:
                    self._handle.close()
                    self._handle = None
                self.disk_bytes = self.path.stat().st_size
            self._state = "sealed"

    def _query(self, kind, column):
        if kind not in self._counts or column not in ("a", "b", "c"):
            raise ValueError("Invalid metric query field")
        self._seal()

    def _values(self, kind, column):
        if not self.disk_bytes:
            yield from (row[column] for row in self._buffer if row["kind"] == kind)
            return
        with pq.ParquetFile(self.path) as reader:
            for batch in reader.iter_batches(
                batch_size=4096, columns=["kind", column], use_threads=False
            ):
                if batch.nbytes > RESERVE_BYTES:
                    raise ValueError("Metric decoded query byte limit exceeded")
                selected = batch.filter(pc.equal(batch.column(0), kind))
                yield from selected.column(1).to_pylist()

    def total(self, kind, column, *, transform=None):
        self._query(kind, column)
        values = self._values(kind, column)
        return sum(values if transform is None else (transform(v) for v in values))

    def median(self, kind, column):
        self._query(kind, column)
        count = self._counts[kind]
        if not count:
            return None
        if not self.disk_bytes:
            return median(self._values(kind, column))
        self._space(SPILL_BYTES)
        db = duckdb.connect(
            config={
                "memory_limit": "256MB",
                "max_temp_directory_size": "2048MB",
                "temp_directory": str(self.directory / "spill"),
                "threads": 1,
                "preserve_insertion_order": False,
            }
        )
        try:
            db.execute("SET enable_progress_bar=false")
            db.read_parquet(str(self.path), hive_partitioning=False).create_view(
                "observations"
            )
            rows = db.execute(
                f"SELECT {column} FROM observations WHERE kind=? ORDER BY {column} LIMIT ? OFFSET ?",
                [kind, 2 if count % 2 == 0 else 1, (count - 1) // 2],
            ).fetchall()
            return median(row[0] for row in rows)
        finally:
            db.close()

    def __exit__(self, kind, *_):
        try:
            if self._writer is not None:
                self._writer.close()
        except BaseException:
            if kind is None:
                raise
        finally:
            if self._handle is not None:
                self._handle.close()
            self._writer = self._handle = None
            self._buffer.clear()
            self._state = "closed"
            self._temp.cleanup()
