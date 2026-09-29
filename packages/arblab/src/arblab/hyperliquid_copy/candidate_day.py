"""Reusable first observations from a complete, verified canonical event day.

No listing inference or candidate lookback filtering. Caller owns the shared
cache lease; only completed publications may feed a candidate-history query.
"""

from dataclasses import asdict
from pathlib import Path
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .annual_execution_policy import execution_policy
from .checkpoint_io import _CappedOutput
from .derived_day_builder import _release_empty_scratch
from .derived_publication import PublishedArtifacts, _encode
from .download import file_hash
from .qualified_day import QualifiedDay

MAX_BYTES = 512 * 1024**2
MAX_ROWS = 50_000_000
MAX_GROUPS = 4096
SPILL_BYTES = 2 * 1024**3
SCHEMA = pa.schema(
    [
        pa.field("user", pa.string()),
        pa.field("coin", pa.string()),
        pa.field("first_observed", pa.timestamp("us", tz="UTC")),
    ]
)


def _engine():
    root = Path(__file__).parent
    names = (
        "candidate_day.py",
        "qualified_day.py",
        "derived_day_builder.py",
        "checkpoint_io.py",
        "derived_publication.py",
        "derived_cache_resources.py",
        "derived_cache_lease.py",
    )
    return dict(
        schema=2,
        duckdb=duckdb.__version__,
        pyarrow=pa.__version__,
        code={n: file_hash(root / n) for n in names},
    )


def _write(source, scratch, target, max_bytes, max_rows):
    with duckdb.connect(
        config={
            "memory_limit": "256MB",
            "max_temp_directory_size": "2GB",
            "temp_directory": str(scratch),
            "threads": 1,
            "TimeZone": "UTC",
            "preserve_insertion_order": False,
        }
    ) as db:
        db.execute("SET enable_progress_bar=false")
        paths = [str(e.path) for e in source.entries or (source.witness,)]
        db.read_parquet(paths, hive_partitioning=False).create_view("fills")
        query = db.execute(
            "SELECT user,coin,min(exchange_time) AS first_observed FROM fills "
            "WHERE exchange_time>=? AND exchange_time<? "
            + ("AND FALSE " if not source.entries else "")
            + "GROUP BY user,coin ORDER BY user,coin",
            [source.start, source.end],
        )
        rows = groups = 0
        with query.to_arrow_reader(4096) as batches, target.open("xb") as output:
            with pq.ParquetWriter(
                _CappedOutput(output, max_bytes), SCHEMA, compression="zstd"
            ) as writer:
                for batch in batches:
                    rows += batch.num_rows
                    groups += 1
                    if rows > max_rows or groups > MAX_GROUPS:
                        raise ValueError("Candidate day row/metadata limit exceeded")
                    if batch.nbytes > 64 * 1024**2:
                        raise ValueError("Candidate day decoded batch limit exceeded")
                    writer.write_batch(batch.cast(SCHEMA))


def build_candidate_day(
    resources,
    report_pin,
    day,
    *,
    max_bytes=MAX_BYTES,
    max_rows=MAX_ROWS,
    execution_policy_name=None,
):
    resources.lease.check()
    policy = execution_policy(execution_policy_name)
    expected_bytes = MAX_BYTES if policy is None else policy.candidate_day_bytes
    if (
        type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_BYTES
        or policy is not None
        and max_bytes != expected_bytes
        or type(max_rows) is not int
        or not 0 < max_rows <= MAX_ROWS
    ):
        raise ValueError("Invalid candidate day resource bounds")
    source = QualifiedDay(report_pin, day)
    engine = _engine()
    inputs = dict(
        **source.inputs(),
        engine=engine,
        max_bytes=max_bytes,
        max_rows=max_rows,
    )
    if policy is not None:
        inputs["execution_policy"] = policy.name
    publications = PublishedArtifacts(resources)

    def verify():
        source.verify()
        if _engine() != engine:
            raise ValueError("Candidate day source/engine changed")

    existing = publications.lookup("candidate_day", inputs)
    if existing is not None:
        verify()
        return existing
    relative = f"scratch/{uuid.uuid4().hex}"
    scratch_token = resources.reserve(relative, SPILL_BYTES, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    info = scratch.stat()
    identity = info.st_dev, info.st_ino
    try:
        relative = f"artifacts/{uuid.uuid4().hex}.parquet"
        token = resources.reserve(relative, max_bytes, "payload")
        _write(
            source,
            scratch,
            resources.root / relative,
            max_bytes,
            max_rows,
        )
        resources.settle(token)
        verify()
        if policy is not None:
            with resources._connect() as db:
                pins = publications._records(db, [token])
            descriptor = _encode(
                dict(
                    schema=1,
                    kind="candidate_day",
                    inputs=inputs,
                    artifacts=[asdict(pin) for pin in pins],
                )
            )
            if len(descriptor) > policy.candidate_day_descriptor_bytes:
                raise ValueError("Candidate day publication descriptor limit exceeded")
        return publications.publish("candidate_day", inputs, [token])
    finally:
        _release_empty_scratch(resources, scratch_token, scratch, identity)
