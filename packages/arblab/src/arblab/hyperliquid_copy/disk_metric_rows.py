"""Capped candidate-metric staging; caller owns causal provenance/publication.

Records are not a validated universe until the scoring layer verifies unique
users, source completeness and the decision/configuration publication identity.
"""

from math import isfinite
import re
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint_io import _CappedOutput
from .lab_config import METRICS

MAX_ROWS = 50_000_000
MAX_BYTES = 512 * 1024**2
MAX_GROUPS = 4096
METRIC_ROWS_SCHEMA = pa.schema(
    [
        pa.field("user", pa.string(), nullable=False),
        pa.field(
            "metrics",
            pa.struct([pa.field(k, pa.float64()) for k in METRICS]),
            nullable=False,
        ),
        pa.field("exclusions", pa.list_(pa.string()), nullable=False),
    ]
)


def _record(row, required):
    if type(row) is not dict or set(row) != set(METRIC_ROWS_SCHEMA.names):
        raise ValueError("Invalid metric record fields")
    user, metrics, reasons = row["user"], row["metrics"], row["exclusions"]
    if type(user) is not str or not re.fullmatch(r"0x[0-9a-f]{40}", user):
        raise ValueError("Invalid metric user address")
    if type(metrics) is not dict or set(metrics) != set(METRICS):
        raise ValueError("Invalid metric fields")
    if (
        type(reasons) not in (list, tuple)
        or len(reasons) > 32
        or any(
            type(r) is not str or not 0 < len(r) <= 256 or len(r.encode()) > 256
            for r in reasons
        )
    ):
        raise ValueError("Invalid metric exclusions")
    values = {}
    for name, value in metrics.items():
        if value is None:
            values[name] = None
            continue
        if type(value) not in (int, float):
            raise ValueError("Invalid metric numeric type")
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ValueError("Lossy metric conversion") from exc
        if not isfinite(converted) or converted != value:
            raise ValueError("Nonfinite or lossy metric conversion")
        values[name] = converted
    if not reasons and any(values[name] is None for name in required):
        raise ValueError("Missing required metric for eligible candidate")
    return dict(user=user, metrics=values, exclusions=list(reasons))


def write_metric_rows(
    resources, rows, required_metrics, *, max_rows=MAX_ROWS, max_bytes=MAX_BYTES
):
    """Return a settled, unpublished token; failures retain their full reservation.

    Duplicate users are intentionally left for the disk scoring validation gate,
    not tracked with an unbounded Python set. No source/config identity is inferred.
    """
    resources.lease.check()
    if (
        type(max_rows) is not int
        or not 0 < max_rows <= MAX_ROWS
        or type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_BYTES
        or type(required_metrics) not in (list, tuple)
        or not 1 <= len(required_metrics) <= len(METRICS)
        or any(type(k) is not str or k not in METRICS for k in required_metrics)
        or len(set(required_metrics)) != len(required_metrics)
    ):
        raise ValueError("Invalid metric writer bounds/configuration")
    required = tuple(required_metrics)
    relative = f"artifacts/{uuid.uuid4().hex}.parquet"
    token = resources.reserve(relative, max_bytes, "payload")
    count = groups = 0
    batch = []
    with (resources.root / relative).open("xb") as handle:
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
    resources.settle(token)
    return token


def _write_batch(writer, batch, groups):
    if groups > MAX_GROUPS:
        raise ValueError("Metric artifact row-group limit exceeded")
    table = pa.Table.from_pylist(batch, schema=METRIC_ROWS_SCHEMA)
    if table.nbytes > 64 * 1024**2:
        raise ValueError("Metric decoded batch byte limit exceeded")
    writer.write_table(table)
