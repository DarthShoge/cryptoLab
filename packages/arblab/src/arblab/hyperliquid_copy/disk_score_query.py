"""Exact percentile/score pass with bounded query and Arrow buffers.

The orchestration caller must hold the shared resource lease, reserve query spill
and output growth, verify source pins before/after, and publish only on success.
This primitive does not independently qualify candidate provenance or publish.
"""

from math import isclose
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint_io import _CappedOutput
from .disk_metric_rows import (
    METRIC_ROWS_SCHEMA,
    MAX_BYTES,
    MAX_ROWS,
    MAX_GROUPS,
    _record,
)
from .lab_config import METRICS

SCORED_SCHEMA = pa.schema(
    [
        *METRIC_ROWS_SCHEMA,
        pa.field("percentiles", METRIC_ROWS_SCHEMA.field("metrics").type),
        pa.field("score", pa.float64()),
    ]
)


def scoring_terms(config):
    """Ordered identity: a sorted JSON mapping alone loses score addition order."""
    weights = getattr(config, "metric_weights", None)
    directions = getattr(config, "metric_directions", None)
    if (
        type(weights) is not dict
        or not 1 <= len(weights) <= len(METRICS)
        or type(directions) is not dict
        or set(directions) != set(weights)
    ):
        raise ValueError("Invalid effective scoring configuration")
    terms = []
    for name, weight in weights.items():
        if (
            type(name) is not str
            or name not in METRICS
            or type(weight) not in (int, float)
            or not 0 < weight <= 1
            or directions[name] not in ("asc", "desc")
        ):
            raise ValueError("Invalid effective scoring term")
        terms.append((name, float(weight), directions[name]))
    if not isclose(sum(t[1] for t in terms), 1, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Expected normalized scoring weights")
    return tuple(terms)


class ScoreQuery:
    def __init__(self, source, scratch, config):
        self.source, self.scratch = Path(source), Path(scratch)
        self.terms = scoring_terms(config)
        self.db = None
        self.counts = None

    def __enter__(self):
        if self.db is not None:
            raise ValueError("Scoring query already open")
        if (
            self.source.is_symlink()
            or not self.source.is_file()
            or not 0 < self.source.stat().st_size <= MAX_BYTES
        ):
            raise ValueError("Invalid metric source file")
        with pq.ParquetFile(self.source) as source:
            if (
                source.schema_arrow != METRIC_ROWS_SCHEMA
                or source.metadata.num_rows > MAX_ROWS
                or source.metadata.num_row_groups > MAX_GROUPS
            ):
                raise ValueError("Invalid metric source schema/metadata limits")
            for batch in source.iter_batches(batch_size=4096):
                if batch.nbytes > 64 * 1024**2:
                    raise ValueError("Metric decoded batch byte limit exceeded")
                for record in batch.to_pylist():
                    _record(record, tuple(t[0] for t in self.terms))
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
            self.db.read_parquet(str(self.source), hive_partitioning=False).create_view(
                "source"
            )
            total, unique, eligible = self.db.execute(
                "SELECT count(*),count(DISTINCT user),count(*) FILTER (WHERE len(exclusions)=0) FROM source"
            ).fetchone()
            if total != unique:
                raise ValueError("Duplicate candidate users")
            self.counts = total, eligible
            return self
        except BaseException:
            self.__exit__(None)
            raise

    def write_scores(self, path, *, max_bytes=MAX_BYTES, on_created=None):
        if self.db is None:
            raise ValueError("Expected open scoring query")
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError("Invalid score output byte limit")
        if on_created is not None and not callable(on_created):
            raise ValueError("Invalid score output creation hook")
        columns = []
        for index, (name, _, _) in enumerate(self.terms):
            columns.append(
                f"CASE WHEN count(*) OVER ()=1 THEN 0.5 ELSE "
                f"(2*(rank() OVER (ORDER BY metrics.{name})-1)+"
                f"count(*) OVER (PARTITION BY metrics.{name})-1)/2.0/"
                f"(count(*) OVER ()-1) END AS p{index}"
            )
        query = self.db.execute(
            "SELECT *,"
            + ",".join(columns)
            + " FROM source WHERE len(exclusions)=0 UNION ALL "
            "SELECT *,"
            + ",".join("NULL" for _ in columns)
            + " FROM source WHERE len(exclusions)>0"
        )
        count = groups = 0
        with query.to_arrow_reader(4096) as batches, Path(path).open("x+b") as handle:
            if on_created is not None:
                on_created(handle.fileno())
            with pq.ParquetWriter(
                _CappedOutput(handle, max_bytes), SCORED_SCHEMA, compression="zstd"
            ) as writer:
                for batch in batches:
                    groups += 1
                    if batch.nbytes > 64 * 1024**2 or groups > MAX_GROUPS:
                        raise ValueError("Scoring batch/metadata limit exceeded")
                    rows = []
                    for record in batch.to_pylist():
                        percentiles, score = {}, None
                        for index, (name, weight, direction) in enumerate(self.terms):
                            percentile = record.pop(f"p{index}")
                            if not record["exclusions"]:
                                percentile = (
                                    1 - percentile if direction == "asc" else percentile
                                )
                                percentiles[name] = percentile
                                score = (score or 0) + weight * percentile
                        rows.append(record | dict(percentiles=percentiles, score=score))
                    table = pa.Table.from_pylist(rows, schema=SCORED_SCHEMA)
                    if table.nbytes > 64 * 1024**2:
                        raise ValueError("Scoring decoded output limit exceeded")
                    writer.write_table(table)
                    count += len(rows)
        if count != self.counts[0]:
            raise ValueError("Scored candidate count changed")
        return count

    def __exit__(self, *_):
        if self.db is not None:
            self.db.close()
            self.db = None
