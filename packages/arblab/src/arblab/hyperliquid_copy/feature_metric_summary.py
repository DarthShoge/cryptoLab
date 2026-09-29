"""Bounded per-day compaction for rolling feature-backed trader metrics."""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint_io import _CappedOutput
from .feature_window import FeatureWindow
from .feature_metric_result import candidate_table, metric_row
from .download import file_hash

MAX_DAILY_BYTES = 512 * 1024**2
MAX_EPISODE_BYTES = 512 * 1024**2
DAILY_SCHEMA = pa.schema(
    [
        ("user", pa.string()),
        ("coin", pa.string()),
        ("activity_day", pa.date32()),
        ("fill_count", pa.int64()),
        ("pnl", pa.float64()),
        ("notional", pa.float64()),
        ("volume", pa.float64()),
        ("takers", pa.int64()),
        ("first_start", pa.float64()),
    ]
)
EPISODE_SCHEMA = pa.schema(
    [
        ("user", pa.string()),
        ("coin", pa.string()),
        ("exchange_time", pa.timestamp("us", tz="UTC")),
        ("block_number", pa.int64()),
        ("source_key", pa.string()),
        ("source_line", pa.int64()),
        ("event_index", pa.int64()),
        ("event_id", pa.string()),
        ("pnl", pa.float64()),
        ("minutes", pa.float64()),
        ("fragments", pa.float64()),
    ]
)


@dataclass(frozen=True)
class MetricSummary:
    daily_path: Path
    episode_path: Path
    daily_rows: int
    episode_rows: int
    bytes: int


def _append(query, writer, schema):
    count = 0
    with query.to_arrow_reader(4096) as batches:
        for batch in batches:
            if batch.nbytes > 64 * 1024**2:
                raise ValueError("Metric summary decoded batch byte limit exceeded")
            batch = batch.cast(schema)
            writer.write_batch(batch)
            count += batch.num_rows
    return count


def stage_metric_summaries(
    window,
    scratch,
    *,
    max_daily_bytes=MAX_DAILY_BYTES,
    max_episode_bytes=MAX_EPISODE_BYTES,
):
    if not isinstance(window, FeatureWindow):
        raise ValueError("Qualified feature window required")
    for value, maximum in (
        (max_daily_bytes, MAX_DAILY_BYTES),
        (max_episode_bytes, MAX_EPISODE_BYTES),
    ):
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError("Invalid metric summary byte limit")
    scratch = Path(scratch).absolute()
    spill = scratch / "spill"
    if not scratch.is_dir() or not spill.is_dir():
        raise ValueError("Expected reserved metric scratch and spill directories")
    daily_path = scratch / "metric_daily.parquet"
    episode_path = scratch / "metric_episodes.parquet"
    if daily_path.exists() or episode_path.exists():
        raise ValueError("Metric summary target already exists")
    window.verify()
    db = duckdb.connect(
        config={
            "memory_limit": "256MB",
            "max_temp_directory_size": "1GB",
            "temp_directory": str(spill),
            "threads": 1,
            "TimeZone": "UTC",
            "preserve_insertion_order": False,
        }
    )
    daily_rows = episode_rows = 0
    try:
        db.execute("SET enable_progress_bar=false")
        with (
            daily_path.open("xb") as daily_handle,
            episode_path.open("xb") as episode_handle,
            pq.ParquetWriter(
                _CappedOutput(daily_handle, max_daily_bytes),
                DAILY_SCHEMA,
                compression="zstd",
            ) as daily_writer,
            pq.ParquetWriter(
                _CappedOutput(episode_handle, max_episode_bytes),
                EPISODE_SCHEMA,
                compression="zstd",
            ) as episode_writer,
        ):
            for day in window.days:
                db.read_parquet(
                    [
                        str(window.resources.root / pin.path)
                        for pin in day._observations
                    ],
                    hive_partitioning=False,
                ).create_view("features", replace=True)
                parameters = [window.start, window.end, list(window.coins)]
                daily_rows += _append(
                    db.execute(
                        """
                        WITH fills AS (
                          SELECT user,coin,exchange_time::DATE AS activity_day,
                            exchange_time,block_number,source_key,source_line,event_index,event_id,
                            json_extract_string(decode(payload),'$.net_pnl')::DOUBLE AS pnl,
                            json_extract_string(decode(payload),'$.closed_notional')::DOUBLE AS notional,
                            json_extract_string(decode(payload),'$.gross_volume')::DOUBLE AS volume,
                            json_extract_string(decode(payload),'$.crossed')::BOOLEAN::INTEGER AS taker,
                            json_extract_string(decode(payload),'$.start_position')::DOUBLE AS start_position
                          FROM features
                          WHERE kind='fill' AND exchange_time>=? AND exchange_time<?
                            AND coin IN (SELECT unnest(?::VARCHAR[]))
                        )
                        SELECT user,coin,activity_day,count(*)::BIGINT AS fill_count,
                          sum(pnl)::DOUBLE AS pnl,sum(notional)::DOUBLE AS notional,
                          sum(volume)::DOUBLE AS volume,sum(taker)::BIGINT AS takers,
                          first(start_position ORDER BY exchange_time,block_number,source_key,
                            source_line,event_index,event_id)::DOUBLE AS first_start
                        FROM fills GROUP BY user,coin,activity_day
                        """,
                        parameters,
                    ),
                    daily_writer,
                    DAILY_SCHEMA,
                )
                episode_rows += _append(
                    db.execute(
                        """
                        SELECT user,coin,exchange_time,block_number,source_key,source_line,
                          event_index,event_id,
                          json_extract_string(decode(payload),'$.pnl')::DOUBLE AS pnl,
                          json_extract_string(decode(payload),'$.minutes')::DOUBLE AS minutes,
                          json_extract_string(decode(payload),'$.fragments')::DOUBLE AS fragments
                        FROM features
                        WHERE kind='episode' AND exchange_time>=? AND exchange_time<?
                          AND coin IN (SELECT unnest(?::VARCHAR[]))
                        """,
                        parameters,
                    ),
                    episode_writer,
                    EPISODE_SCHEMA,
                )
    finally:
        db.close()
        window.verify()
    daily_bytes, episode_bytes = daily_path.stat().st_size, episode_path.stat().st_size
    if daily_bytes > max_daily_bytes or episode_bytes > max_episode_bytes:
        raise ValueError("Metric summary byte limit exceeded")
    return MetricSummary(
        daily_path,
        episode_path,
        daily_rows,
        episode_rows,
        daily_bytes + episode_bytes,
    )


def _summary_stats(summary):
    values = []
    for path in (summary.daily_path, summary.episode_path):
        info = path.lstat()
        if not path.is_file() or path.is_symlink():
            raise ValueError("Metric summary payload identity changed")
        values.append((info.st_dev, info.st_ino, info.st_size, file_hash(path)))
    return tuple(values)


def summary_metric_rows(summary, candidates, config, *, temp_root):
    if not isinstance(summary, MetricSummary):
        raise ValueError("Expected bounded metric summary")
    scratch = Path(temp_root).absolute()
    if (
        summary.daily_path.parent != scratch
        or summary.episode_path.parent != scratch
        or not (scratch / "spill").is_dir()
    ):
        raise ValueError("Metric summary outside owned scratch")
    if (
        pq.read_schema(summary.daily_path) != DAILY_SCHEMA
        or pq.read_schema(summary.episode_path) != EPISODE_SCHEMA
        or pq.read_metadata(summary.daily_path).num_rows != summary.daily_rows
        or pq.read_metadata(summary.episode_path).num_rows != summary.episode_rows
        or summary.bytes
        != summary.daily_path.stat().st_size + summary.episode_path.stat().st_size
    ):
        raise ValueError("Metric summary schema/count changed")
    before = _summary_stats(summary)
    candidates = candidate_table(candidates)
    db = duckdb.connect(
        config={
            "memory_limit": "256MB",
            "max_temp_directory_size": "1GB",
            "temp_directory": str(scratch / "spill"),
            "threads": 1,
            "TimeZone": "UTC",
            "preserve_insertion_order": False,
        }
    )
    try:
        db.execute("SET enable_progress_bar=false")
        db.register("candidates", candidates)
        db.read_parquet(str(summary.daily_path), hive_partitioning=False).create_view(
            "coin_daily"
        )
        db.read_parquet(str(summary.episode_path), hive_partitioning=False).create_view(
            "episode_rows"
        )
        query = db.execute(
            """
            WITH first_fills AS (
              SELECT user,coin,first(first_start ORDER BY activity_day) AS first_start
              FROM coin_daily GROUP BY user,coin
            ), episode_numbered AS (
              SELECT *,row_number() OVER (PARTITION BY user,coin ORDER BY
                exchange_time,block_number,source_key,source_line,event_index,event_id)
                AS episode_number
              FROM episode_rows
            ), episodes AS (
              SELECT e.* FROM episode_numbered e
              JOIN first_fills f USING(user,coin)
              WHERE abs(f.first_start)<=1e-9 OR e.episode_number>1
            ), episode_stats AS (
              SELECT user,count(*) AS episodes,
                sum(greatest(pnl,0)) AS positive_episode_pnl,
                -sum(least(pnl,0)) AS negative_episode_pnl,
                median(minutes) AS duration,median(fragments) AS fragmentation
              FROM episodes GROUP BY user
            ), daily AS (
              SELECT user,activity_day,sum(pnl) AS day_pnl,
                sum(fill_count)::BIGINT AS fill_count,
                sum(notional) AS notional,sum(volume) AS volume,
                sum(takers)::BIGINT AS takers
              FROM coin_daily GROUP BY user,activity_day
            ), curves AS (
              SELECT *,sum(day_pnl) OVER (PARTITION BY user ORDER BY activity_day) AS curve
              FROM daily
            ), drawdowns AS (
              SELECT user,max(greatest(0.0,peak)-curve) AS drawdown
              FROM (
                SELECT *,max(curve) OVER (PARTITION BY user ORDER BY activity_day) AS peak
                FROM curves
              ) q GROUP BY user
            ), fill_stats AS (
              SELECT user,sum(fill_count)::BIGINT AS fill_count,sum(day_pnl) AS pnl,
                sum(notional) AS notional,sum(volume) AS volume,
                sum(takers)::BIGINT AS takers,count(*)::BIGINT AS active_days,
                count(*) FILTER (WHERE day_pnl>0)::BIGINT AS positive_days
              FROM daily GROUP BY user
            )
            SELECT c.user,fs.fill_count,fs.pnl,fs.notional,fs.volume,fs.takers,
              fs.active_days,fs.positive_days,dd.drawdown,es.episodes,
              es.positive_episode_pnl,es.negative_episode_pnl,es.duration,
              es.fragmentation
            FROM candidates c
            LEFT JOIN fill_stats fs USING(user)
            LEFT JOIN drawdowns dd USING(user)
            LEFT JOIN episode_stats es USING(user)
            ORDER BY c.user
            """
        )
        while batch := query.fetchmany(4096):
            for row in batch:
                yield metric_row(row, config)
    finally:
        db.close()
        if _summary_stats(summary) != before:
            raise ValueError("Metric summary changed during reduction")


def discard_metric_summary(summary):
    """Remove only a fully verified invocation-owned summary after consumption."""

    if not isinstance(summary, MetricSummary):
        raise ValueError("Expected bounded metric summary")
    before = _summary_stats(summary)
    if before != _summary_stats(summary):
        raise ValueError("Metric summary changed before cleanup")
    for path, expected in zip((summary.daily_path, summary.episode_path), before):
        info = path.lstat()
        if (info.st_dev, info.st_ino, info.st_size, file_hash(path)) != expected:
            raise ValueError("Metric summary changed before cleanup")
        path.unlink()
