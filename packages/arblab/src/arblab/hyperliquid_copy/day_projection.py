"""Bounded native-order projections; caller supplies pinned qualified day input."""

from datetime import timedelta
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .checkpoint_io import _CappedOutput
from .proxy_activity import IDENTITY, ORDER
from .qualified_day import micros

COLUMNS = (
    "user",
    "coin",
    "exchange_time",
    "px",
    "sz",
    "side",
    "start_position",
    "post_position",
    "closed_pnl",
    "fee",
    "fee_token",
    "crossed",
    "tid",
    "oid",
)
MAX_ROWS = 250_000
MAX_PARTS = 4096
MAX_BYTES = 512 * 1024**2


class ProjectionQuery:
    def __init__(self, day, scratch):
        self.day, self.scratch = day, Path(scratch)
        self.db = None

    def __enter__(self):
        if self.db is not None:
            raise ValueError("Projection query already open")
        self.db = duckdb.connect(
            config={
                "memory_limit": "256MB",
                "max_temp_directory_size": "2048MB",
                "temp_directory": str(self.scratch),
                "threads": 1,
                "TimeZone": "UTC",
                "preserve_insertion_order": False,
            }
        )
        try:
            self.db.execute("SET enable_progress_bar=false")
            entries = self.day.entries or (self.day.witness,)
            self.db.read_parquet(
                [str(e.path) for e in entries], hive_partitioning=False
            ).create_view("source_files")
            self.db.execute(
                "CREATE VIEW source AS SELECT * FROM source_files"
                + (" WHERE FALSE" if not self.day.entries else "")
            )
            return self
        except BaseException:
            self.__exit__(None)
            raise

    def _bounds(self, start, end):
        if self.db is None or not self.day.start <= start < end <= self.day.end:
            raise ValueError("Expected open query and interval within pinned day")

    def intervals(self, max_rows=MAX_ROWS):
        if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
            raise ValueError("Invalid projection row bound")
        self._bounds(self.day.start, self.day.end)
        pending, result = [(self.day.start, self.day.end)], []
        while pending:
            start, end = pending.pop()
            count = self.db.execute(
                "SELECT count(*) FROM source WHERE exchange_time>=? AND exchange_time<?",
                [start, end],
            ).fetchone()[0]
            if count <= max_rows:
                result.append((start, end))
            else:
                width = micros(end) - micros(start)
                if width <= 1:
                    raise ValueError(
                        f"indivisible projection timestamp has {count} physical rows; limit {max_rows}"
                    )
                middle = start + timedelta(microseconds=width // 2)
                pending.extend(((middle, end), (start, middle)))
            if len(result) + len(pending) > MAX_PARTS:
                raise ValueError("Projection partition limit exceeded")
        return tuple(result)

    def write(self, start, end, path, max_bytes=MAX_BYTES):
        self._bounds(start, end)
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError("Invalid projection byte limit")
        query = self.db.execute(
            f"WITH unique_fills AS (SELECT * FROM source WHERE exchange_time>=? AND exchange_time<? "
            f"QUALIFY row_number() OVER (PARTITION BY {IDENTITY} ORDER BY {ORDER})=1) "
            f"SELECT {','.join(COLUMNS)}, row_number() OVER (ORDER BY user,{ORDER}) AS native_order "
            f"FROM unique_fills ORDER BY user,{ORDER}",
            [start, end],
        )
        rows = groups = 0
        with query.to_arrow_reader(4096) as batches, Path(path).open("xb") as handle:
            with pq.ParquetWriter(
                _CappedOutput(handle, max_bytes), batches.schema, compression="zstd"
            ) as writer:
                for batch in batches:
                    groups += 1
                    if batch.nbytes > 64 * 1024**2 or groups > 4096:
                        raise ValueError(
                            "Projection decoded batch/metadata limit exceeded"
                        )
                    writer.write_batch(batch)
                    rows += batch.num_rows
        return rows

    def __exit__(self, *_):
        if self.db is not None:
            self.db.close()
            self.db = None
