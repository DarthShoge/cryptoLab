"""Disk ordering and bounded selected cohort from a verified score-pass artifact.

Caller owns shared reservations, source/config pins and atomic publication. This
query primitive must not be exposed as an unbudgeted production entry point.
"""

from math import ceil, isfinite
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint_io import _CappedOutput
from .contracts import utc
from .disk_metric_rows import MAX_BYTES, MAX_ROWS, MAX_GROUPS
from .disk_score_query import SCORED_SCHEMA
from .ranking_artifact import RANKING_SCHEMA


class CohortQuery:
    def __init__(self, source, scratch, config, decision, scope):
        if (
            config.selection not in ("n", "fraction")
            or type(config.min_cohort) is not int
            or type(config.max_cohort) is not int
            or not 1 <= config.min_cohort <= config.max_cohort <= 250
            or config.aggregation
            not in ("direction_equal", "direction_score_weighted", "conviction_trimmed")
        ):
            raise ValueError("Invalid effective cohort configuration")
        if config.selection == "n":
            if type(config.top_n) is not int or not 1 <= config.top_n <= 250:
                raise ValueError("Invalid top-N configuration")
            self.fraction, self.top_n = None, config.top_n
        else:
            if (
                type(config.top_fraction) not in (int, float)
                or not isfinite(config.top_fraction)
                or not 0 < config.top_fraction <= 1
            ):
                raise ValueError("Invalid fraction configuration")
            self.fraction, self.top_n = config.top_fraction, None
        if scope is not None and (type(scope) is not str or not 1 <= len(scope) <= 100):
            raise ValueError("Invalid ranking scope")
        self.source, self.scratch = Path(source), Path(scratch)
        self.decision, self.scope = utc(decision), scope
        self.minimum, self.maximum, self.aggregation = (
            config.min_cohort,
            config.max_cohort,
            config.aggregation,
        )
        self.db = None

    def __enter__(self):
        if self.db is not None:
            raise ValueError("Cohort query already open")
        if (
            self.source.is_symlink()
            or not self.source.is_file()
            or not 0 < self.source.stat().st_size <= MAX_BYTES
        ):
            raise ValueError("Invalid score source")
        with pq.ParquetFile(self.source) as source:
            if (
                source.schema_arrow != SCORED_SCHEMA
                or source.metadata.num_rows > MAX_ROWS
                or source.metadata.num_row_groups > MAX_GROUPS
            ):
                raise ValueError("Invalid score schema/metadata bounds")
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
                "scores"
            )
            total, unique, eligible, invalid = self.db.execute(
                "SELECT count(*), count(DISTINCT user), count(score), "
                "count(*) FILTER (WHERE (score IS NULL) != (len(exclusions)>0) OR NOT isfinite(score)) FROM scores"
            ).fetchone()
            if total != unique or invalid:
                raise ValueError("Invalid duplicate/eligible score evidence")
            self.candidates, self.eligible = total, eligible
            self.requested = (
                ceil(self.fraction * eligible)
                if self.fraction is not None
                else self.top_n
            )
            self.size = (
                min(self.maximum, max(self.minimum, self.requested))
                if eligible >= self.minimum
                else 0
            )
            self.db.execute(
                "CREATE VIEW ranked AS SELECT *,row_number() OVER (ORDER BY score DESC,user) AS rank "
                "FROM scores WHERE score IS NOT NULL UNION ALL SELECT *,NULL AS rank FROM scores WHERE score IS NULL"
            )
            cursor = self.db.execute(
                "SELECT * FROM ranked WHERE rank<=? ORDER BY rank LIMIT 250",
                [self.size],
            )
            names = [c[0] for c in cursor.description]
            selected = [dict(zip(names, r)) for r in cursor.fetchall()]
            total_score = sum(r["score"] for r in selected)
            self.weights = {
                r["user"]: r["score"] / total_score
                if total_score and self.aggregation != "direction_equal"
                else 1 / len(selected)
                for r in selected
            }
            self.selected = tuple(self._row(r) for r in selected)
            return self
        except BaseException:
            self.__exit__(None)
            raise

    def _row(self, record):
        eligible = record["score"] is not None
        selected = record["user"] in self.weights
        exclusions = record["exclusions"]
        reasons = (
            exclusions
            if not eligible or selected
            else ["insufficient_cohort" if not self.size else "rank_below_cutoff"]
        )
        return dict(
            user=record["user"],
            decision_time=self.decision,
            coin=self.scope,
            metrics=record["metrics"],
            percentiles={
                k: v for k, v in record["percentiles"].items() if v is not None
            },
            score=record["score"],
            rank=record["rank"],
            selected=selected,
            eligible=eligible,
            reasons=list(reasons),
            exclusions=list(exclusions),
            weight=self.weights.get(record["user"], 0.0),
        )

    def write_rankings(self, path, *, max_bytes=MAX_BYTES, on_created=None):
        if self.db is None:
            raise ValueError("Expected open cohort query")
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError("Invalid ranking output byte limit")
        if on_created is not None and not callable(on_created):
            raise ValueError("Invalid ranking output creation hook")
        query = self.db.execute(
            "SELECT * FROM ranked ORDER BY score IS NULL,score DESC,user"
        )
        count = groups = 0
        with query.to_arrow_reader(4096) as batches, Path(path).open("x+b") as handle:
            if on_created is not None:
                on_created(handle.fileno())
            with pq.ParquetWriter(
                _CappedOutput(handle, max_bytes), RANKING_SCHEMA, compression="zstd"
            ) as writer:
                for batch in batches:
                    groups += 1
                    if batch.nbytes > 64 * 1024**2 or groups > MAX_GROUPS:
                        raise ValueError("Ranking batch/metadata limit exceeded")
                    rows = [self._row(r) for r in batch.to_pylist()]
                    table = pa.Table.from_pylist(rows, schema=RANKING_SCHEMA)
                    if table.nbytes > 64 * 1024**2:
                        raise ValueError("Ranking decoded output limit exceeded")
                    writer.write_table(table)
                    count += len(rows)
        if count != self.candidates:
            raise ValueError("Ranking candidate count changed")
        return dict(
            candidate_count=count,
            eligible_count=self.eligible,
            requested_count=self.requested,
            selected_count=len(self.selected),
            cutoff_address=self.selected[-1]["user"] if self.selected else None,
            selected=self.selected,
        )

    def __exit__(self, *_):
        if self.db is not None:
            self.db.close()
            self.db = None
