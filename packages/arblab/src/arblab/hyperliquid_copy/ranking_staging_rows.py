"""Write charged pending metric rows; caller still verifies causal completeness."""

import os

import pyarrow.parquet as pq

from .checkpoint_io import _CappedOutput
from .disk_metric_rows import (
    MAX_BYTES,
    MAX_ROWS,
    METRIC_ROWS_SCHEMA,
    _record,
    _write_batch,
)
from .lab_config import METRICS
from .ranking_staging_owner import RankingStagingOwner
from .ranking_staging_artifact import StagingArtifact


def write_pending_metrics(
    owner, rows, required_metrics, *, max_rows=MAX_ROWS, max_bytes=MAX_BYTES
):
    if (
        type(owner) is not RankingStagingOwner
        or type(max_rows) is not int
        or not 0 < max_rows <= MAX_ROWS
        or type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_BYTES
        or type(required_metrics) not in (list, tuple)
        or not 1 <= len(required_metrics) <= len(METRICS)
        or any(type(key) is not str or key not in METRICS for key in required_metrics)
        or len(set(required_metrics)) != len(required_metrics)
    ):
        raise ValueError("Invalid pending metric writer bounds/configuration")
    required = tuple(required_metrics)
    owner.verify()
    path = owner.path("metrics")
    count = groups = 0
    batch = []
    with path.open("x+b") as handle:
        owner.capture_created_fd("metrics", handle.fileno())
        with pq.ParquetWriter(
            _CappedOutput(handle, max_bytes), METRIC_ROWS_SCHEMA, compression="zstd"
        ) as writer:
            for row in rows:
                count += 1
                if count > max_rows:
                    raise ValueError("Metric artifact row limit exceeded")
                batch.append(_record(row, required))
                if len(batch) == 4096:
                    groups += 1
                    _write_batch(writer, batch, groups)
                    batch.clear()
            if batch:
                _write_batch(writer, batch, groups + 1)
        handle.flush()
        os.fsync(handle.fileno())
        artifact = StagingArtifact.capture(path, handle.fileno(), maximum=max_bytes)
    owner.verify()
    artifact.verify()
    # No settle/publication/refund or causal completeness claim here.
    return artifact
