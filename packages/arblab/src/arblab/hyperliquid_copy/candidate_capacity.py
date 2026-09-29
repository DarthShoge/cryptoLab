"""Complete source-origin candidate counts; no eligibility or lookback filtering."""

from pathlib import Path
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint_io import _CappedOutput
from .contracts import semantic_hash, utc
from .derived_day_builder import _release_empty_scratch
from .derived_publication import PublishedArtifacts, _encode, _inputs
from .download import file_hash
from .prefix_qualification import _previous, _engine as qualification_engine
from .qualified_source_session import QualifiedSourceSession, _identity
from .candidate_capacity_records import SCHEMA, read_counts

KIND = "candidate_capacity"
MAX_BYTES = 8 * 1024**2
MAX_ROWS = 732 * 51
SPILL_BYTES = 2 * 1024**3


def _engine():
    root = Path(__file__).parent
    return semantic_hash(
        dict(
            duckdb=duckdb.__version__,
            pyarrow=pa.__version__,
            code={
                name: file_hash(root / name)
                for name in (
                    "candidate_capacity.py",
                    "candidate_capacity_records.py",
                    "qualified_source_session.py",
                    "derived_day_builder.py",
                    "checkpoint_io.py",
                    "derived_publication.py",
                )
            },
        )
    )


def _context(source, max_bytes, max_rows):
    if type(source) is not QualifiedSourceSession:
        raise ValueError("Verified qualified source session required")
    if (
        not 0 < (source.finish - source.origin).days <= 732
        or type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_BYTES
        or type(max_rows) is not int
        or not 0 < max_rows <= MAX_ROWS
    ):
        raise ValueError("Invalid candidate capacity source/output bounds")
    source.verify()
    result = dict(
        source=source.inputs(), engine=_engine(), max_bytes=max_bytes, max_rows=max_rows
    )
    source.verify_identity()
    return result


def _write(source, scratch, target, max_bytes, max_rows):
    report = _previous(source.inputs()["pin"], qualification_engine())
    paths = [entry["path"] for entry in report["files"]]
    with duckdb.connect(
        config={
            "memory_limit": "256MB",
            "threads": 1,
            "TimeZone": "UTC",
            "max_temp_directory_size": "2GB",
            "temp_directory": str(scratch),
            "preserve_insertion_order": False,
        }
    ) as db:
        query = db.execute(
            """
            WITH firsts AS (
                SELECT "user", coin, min(exchange_time) AS first_time
                FROM read_parquet(?, hive_partitioning=false)
                WHERE exchange_time >= ? AND exchange_time < ?
                GROUP BY "user", coin
            ), pooled AS (
                SELECT "user", min(first_time) AS first_time FROM firsts GROUP BY "user"
            ), counts AS (
                SELECT CAST(first_time AS DATE) AS day, coin, count(*) AS entrants
                FROM firsts GROUP BY day, coin
                UNION ALL
                SELECT CAST(first_time AS DATE) AS day, NULL AS coin, count(*) AS entrants
                FROM pooled GROUP BY day
            ) SELECT day, coin, entrants FROM counts ORDER BY coin NULLS FIRST, day
        """,
            [paths, source.origin, source.finish],
        )
        count = 0
        with query.to_arrow_reader(4096) as batches, target.open("xb") as output:
            with pq.ParquetWriter(
                _CappedOutput(output, max_bytes), SCHEMA, compression="zstd"
            ) as writer:
                for batch in batches:
                    count += batch.num_rows
                    if count > max_rows or batch.nbytes > 64 * 1024**2:
                        raise ValueError("Candidate capacity row/batch limit exceeded")
                    writer.write_batch(batch.cast(SCHEMA))


def build_candidate_capacity(
    resources, source_session, *, max_bytes=MAX_BYTES, max_rows=MAX_ROWS
):
    inputs = _context(source_session, max_bytes, max_rows)
    resources.lease.check()
    publications = PublishedArtifacts(resources)
    if publications.lookup(KIND, inputs) is not None:
        CandidateCapacity(resources, source_session, inputs)
        return inputs
    relative = f"scratch/{uuid.uuid4().hex}"
    scratch_token = resources.reserve(relative, SPILL_BYTES, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    info = scratch.stat()
    identity = info.st_dev, info.st_ino
    try:
        relative = f"artifacts/{uuid.uuid4().hex}.parquet"
        token = resources.reserve(relative, max_bytes, "payload")
        _write(source_session, scratch, resources.root / relative, max_bytes, max_rows)
        payload_identity = _identity(resources.root / relative)
        read_counts(
            resources.root / relative,
            source_session,
            max_bytes=max_bytes,
            max_rows=max_rows,
        )
        if _identity(resources.root / relative) != payload_identity:
            raise ValueError("Staged candidate capacity changed during validation")
        resources.settle(token)
        if (
            _context(source_session, max_bytes, max_rows) != inputs
            or _identity(resources.root / relative) != payload_identity
        ):
            raise ValueError("Candidate capacity source/engine changed")
        publications.publish(KIND, inputs, [token])
        CandidateCapacity(resources, source_session, inputs)
        return inputs
    finally:
        _release_empty_scratch(resources, scratch_token, scratch, identity)


class CandidateCapacity:
    def __init__(self, resources, source_session, inputs):
        normalized = _inputs(KIND, inputs)
        if set(normalized) != {"source", "engine", "max_bytes", "max_rows"}:
            raise ValueError("Invalid candidate capacity inputs")
        self.resources, self.source = resources, source_session
        self._caller, self.inputs = inputs, normalized
        self._frozen = _encode(normalized)
        self._verify()
        publication = PublishedArtifacts(resources).lookup(KIND, normalized)
        if publication is None or len(publication.artifacts) != 1:
            raise ValueError("Missing candidate capacity publication")
        self.publication = publication
        self.path = resources.root / publication.artifacts[0].path
        self._identity = _identity(self.path)
        self._rows = read_counts(
            self.path,
            source_session,
            max_bytes=normalized["max_bytes"],
            max_rows=normalized["max_rows"],
        )
        self._verify_artifact()

    def _verify(self):
        self.resources.lease.check()
        if (
            _encode(self._caller) != self._frozen
            or _encode(
                _context(self.source, self.inputs["max_bytes"], self.inputs["max_rows"])
            )
            != self._frozen
        ):
            raise ValueError("Candidate capacity context changed")
        self.source.verify_identity()
        if _encode(self._caller) != self._frozen:
            raise ValueError("Candidate capacity caller changed")
        self.resources.lease.check()

    def _verify_artifact(self):
        self._verify()
        current = PublishedArtifacts(self.resources).lookup(KIND, self.inputs)
        self._verify()
        if current != self.publication or _identity(self.path) != self._identity:
            raise ValueError("Candidate capacity artifact changed")
        self.source.verify_identity()

    def upper_bound(self, decision, coins, scope):
        self._verify_artifact()
        decision = utc(decision)
        if (
            not self.source.origin <= decision <= self.source.finish
            or any(
                (decision.hour, decision.minute, decision.second, decision.microsecond)
            )
            or type(coins) not in (list, tuple)
            or any(type(c) is not str for c in coins)
            or len(set(coins)) != len(coins)
            or not set(coins) <= set(self.source.coins)
            or scope is not None
            and scope not in coins
        ):
            raise ValueError("Invalid candidate capacity decision/market scope")
        totals = {coin: 0 for coin in (*self.source.coins, None)}
        for date, coin, entrants in self._rows:
            if date < decision.date():
                totals[coin] += entrants
        return (
            totals[scope]
            if scope is not None
            else min(sum(totals[coin] for coin in coins), totals[None])
        )
