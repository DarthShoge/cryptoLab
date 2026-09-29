"""Publish one verified event-day projection under the shared cache budget."""

from pathlib import Path
import uuid

import duckdb
import pyarrow

from .day_projection import ProjectionQuery, MAX_ROWS, MAX_BYTES
from .derived_publication import PublishedArtifacts
from .download import file_hash
from .qualified_day import QualifiedDay

SPILL_BYTES = 2 * 1024**3


def _engine():
    root = Path(__file__).parent
    names = (
        "qualified_day.py",
        "day_projection.py",
        "derived_day_builder.py",
        "derived_publication.py",
        "derived_cache_resources.py",
        "derived_cache_lease.py",
    )
    return dict(
        schema=1,
        duckdb=duckdb.__version__,
        pyarrow=pyarrow.__version__,
        code={name: file_hash(root / name) for name in names},
    )


def _release_empty_scratch(resources, token, path, identity):
    resources.lease.check()
    info = path.lstat()
    if path.is_symlink() or (info.st_dev, info.st_ino) != identity or not path.is_dir():
        raise ValueError("Projection scratch identity changed; retained reservation")
    if any(path.iterdir()):
        return  # Unknown/unremoved spill remains charged; no recursive deletion.
    path.rmdir()  # Only this invocation's verified, empty scratch directory.
    resources.release_missing(token)  # Fsyncs parent before refunding local storage.


def build_projection_day(
    resources,
    report_pin,
    day,
    *,
    max_partition_rows=MAX_ROWS,
    max_output_bytes=MAX_BYTES,
):
    resources.lease.check()
    if (
        type(max_partition_rows) is not int
        or not 1 <= max_partition_rows <= MAX_ROWS
        or type(max_output_bytes) is not int
        or not 0 < max_output_bytes <= MAX_BYTES
    ):
        raise ValueError("Invalid day projection resource bounds")
    source = QualifiedDay(report_pin, day)
    engine = _engine()
    inputs = dict(
        **source.inputs(),
        engine=engine,
        max_partition_rows=max_partition_rows,
        max_output_bytes=max_output_bytes,
    )
    publications = PublishedArtifacts(resources)
    existing = publications.lookup("qualified_day", inputs)
    if existing is not None:
        source.verify()
        if _engine() != engine:
            raise ValueError("Projection engine changed during reuse")
        return existing
    relative = f"scratch/{uuid.uuid4().hex}"
    scratch_token = resources.reserve(relative, SPILL_BYTES, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    info = scratch.stat()
    identity = info.st_dev, info.st_ino
    try:
        tokens = []
        with ProjectionQuery(source, scratch) as query:
            for start, end in query.intervals(max_partition_rows):
                relative = f"artifacts/{uuid.uuid4().hex}.parquet"
                token = resources.reserve(relative, max_output_bytes, "payload")
                query.write(start, end, resources.root / relative, max_output_bytes)
                resources.settle(token)
                tokens.append(token)
        source.verify()
        if _engine() != engine:
            raise ValueError("Projection engine changed during build")
        return publications.publish("qualified_day", inputs, tokens)
    finally:
        _release_empty_scratch(resources, scratch_token, scratch, identity)
