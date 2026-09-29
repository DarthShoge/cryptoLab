"""Complete causal candidate history; never an eligible-only or lookback universe.

Use the completed Parquet publication for metric production so candidate SQL
resources are closed before wallet metric queries start.
"""

from datetime import timedelta
import hashlib
from pathlib import Path
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from . import candidate_day
from .annual_execution_policy import execution_policy, require_descriptor_bound
from .checkpoint_io import _CappedOutput
from .contracts import utc, symbol
from .derived_day_builder import _release_empty_scratch
from .derived_publication import PublishedArtifacts, _encode
from .download import file_hash
from .prefix_qualification import _previous, _engine as qualification_engine
from .qualified_day import QualifiedDay, _day

MAX_BYTES = 512 * 1024**2
MAX_ROWS = 50_000_000
MAX_GROUPS = 4096
SPILL_BYTES = 2 * 1024**3
SCHEMA = pa.schema([pa.field("user", pa.string())])


def _engine():
    return dict(
        schema=1,
        candidate_day=candidate_day._engine(),
        code_sha256=file_hash(Path(__file__)),
        contracts_sha256=file_hash(Path(__file__).with_name("contracts.py")),
    )


class CandidateHistory:
    def __init__(
        self,
        resources,
        report_pin,
        source_start,
        decision,
        coins,
        scope=None,
        *,
        execution_policy_name=None,
    ):
        resources.lease.check()
        if type(report_pin) is not dict or set(report_pin) != {"path", "sha256"}:
            raise ValueError("Pinned candidate qualification required")
        self.resources, self.pin = resources, dict(report_pin)
        report = _previous(self.pin, qualification_engine())
        if source_start != report["source_start"]:
            raise ValueError("Candidate origin must equal qualified source origin")
        self.start, self.decision = _day(source_start), utc(decision)
        if not self.start <= self.decision <= _day(report["source_end"]):
            raise ValueError("Candidate decision outside qualified history")
        if (
            type(coins) not in (list, tuple)
            or not 1 <= len(coins) <= 50
            or any(type(c) is not str for c in coins)
        ):
            raise ValueError("Invalid candidate market scope")
        self.coins = tuple(sorted(symbol(c) for c in coins))
        if (
            len(set(self.coins)) != len(self.coins)
            or not set(self.coins) <= set(report["coins"])
            or scope is not None
            and scope not in self.coins
        ):
            raise ValueError("Candidate scope outside qualified markets")
        self.scope = scope
        policy = execution_policy(execution_policy_name)
        self.execution_policy = None if policy is None else policy.name
        self.candidate_day_bytes = (
            candidate_day.MAX_BYTES if policy is None else policy.candidate_day_bytes
        )
        span = self.decision - self.start
        day_count = span.days + bool(span.seconds or span.microseconds)
        if day_count > 732:
            raise ValueError("Candidate history day limit exceeded")
        self.days = tuple(
            (self.start + timedelta(days=n)).date().isoformat()
            for n in range(day_count)
        )
        self.anchor = QualifiedDay(self.pin, source_start)
        self.engine = _engine()
        self._frozen_context = self._context()
        self._expected_publications = None
        self.publications = ()
        self.db = self.reader = self.scratch = None
        self.completed = self.started = self.active = self.entered = False

    def __enter__(self):
        if self.entered:
            raise ValueError("Candidate history cannot be reopened")
        self._check_context()
        self.entered = self.active = True
        try:
            self.publications = tuple(
                candidate_day.build_candidate_day(
                    self.resources,
                    self.pin,
                    day,
                    max_bytes=self.candidate_day_bytes,
                    execution_policy_name=self.execution_policy,
                )
                for day in self.days
            )
            if sum(len(p.artifacts) for p in self.publications) > 5000:
                raise ValueError("Candidate history artifact limit exceeded")
            self._expected_publications = self.publications
            self.verify()
            return self
        except BaseException:
            self.__exit__(None)
            raise

    def _context(self):
        return _encode(
            dict(
                pin=self.pin,
                start=self.start.isoformat(),
                decision=self.decision.isoformat(),
                coins=self.coins,
                scope=self.scope,
                days=self.days,
                engine=self.engine,
                execution_policy=self.execution_policy,
                candidate_day_bytes=self.candidate_day_bytes,
            )
        )

    def _check_context(self):
        if self._context() != self._frozen_context:
            raise ValueError("Candidate history context changed")
        if (
            self._expected_publications is not None
            and self.publications != self._expected_publications
        ):
            raise ValueError("Candidate history chain changed")

    def verify(self):
        self._check_context()
        if not self.active:
            raise ValueError("Candidate history is not open")
        self.resources.lease.check()
        self.anchor.verify()
        for day, publication in zip(self.days, self.publications, strict=True):
            if (
                candidate_day.build_candidate_day(
                    self.resources,
                    self.pin,
                    day,
                    max_bytes=self.candidate_day_bytes,
                    execution_policy_name=self.execution_policy,
                )
                != publication
            ):
                raise ValueError("Candidate history day identity changed")
        if _engine() != self.engine:
            raise ValueError("Candidate history engine changed")

    def inputs(self):
        self._check_context()
        if not self.active:
            raise ValueError("Candidate history is not open")
        chain = list(zip(self.days, (p.key for p in self.publications), strict=True))
        result = dict(
            schema=1,
            report_sha256=self.pin["sha256"],
            source_start=self.start.date().isoformat(),
            decision=self.decision.isoformat(),
            coins=list(self.coins),
            scope=self.scope,
            day_count=len(chain),
            daily_chain_sha256=hashlib.sha256(_encode(chain)).hexdigest(),
            engine=self.engine,
        )
        if self.execution_policy is not None:
            result.update(
                execution_policy=self.execution_policy,
                candidate_day_bytes=self.candidate_day_bytes,
            )
        return result

    def _open_query(self):
        self._check_context()
        relative = f"scratch/{uuid.uuid4().hex}"
        self.scratch_token = self.resources.reserve(relative, SPILL_BYTES, "scratch")
        self.scratch = self.resources.root / relative
        self.scratch.mkdir()
        info = self.scratch.stat()
        self.scratch_identity = info.st_dev, info.st_ino
        self.db = duckdb.connect(
            config={
                "memory_limit": "256MB",
                "max_temp_directory_size": "2GB",
                "temp_directory": str(self.scratch),
                "threads": 1,
                "TimeZone": "UTC",
                "preserve_insertion_order": False,
            }
        )
        self.db.execute("SET enable_progress_bar=false")
        paths = [
            str(self.resources.root / a.path)
            for p in self.publications
            for a in p.artifacts
        ]
        if not paths:
            return self.db.execute("SELECT CAST(NULL AS VARCHAR) AS user WHERE FALSE")
        self.db.read_parquet(paths, hive_partitioning=False).create_view("observations")
        coins = (self.scope,) if self.scope else self.coins
        marks = ",".join("?" for _ in coins)
        return self.db.execute(
            "SELECT DISTINCT user FROM observations WHERE first_observed < ? "
            f"AND coin IN ({marks}) ORDER BY user",
            [self.decision, *coins],
        )

    def rows(self):
        self._check_context()
        if not self.active or self.started:
            raise ValueError("Expected a fresh open candidate stream")
        self.started = True
        try:
            query = self._open_query()
            count = 0
            self.reader = query.to_arrow_reader(4096)
            for batch in self.reader:
                count += batch.num_rows
                if count > MAX_ROWS or batch.nbytes > 64 * 1024**2:
                    raise ValueError("Candidate history row/batch limit exceeded")
                for user in batch.column(0).to_pylist():
                    if not self.active:
                        raise ValueError("Candidate stream consumed after close")
                    yield user
            self._close_query()
            self.verify()
            self.completed = True
        finally:
            self._close_query()

    def _close_query(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.scratch is not None:
            _release_empty_scratch(
                self.resources, self.scratch_token, self.scratch, self.scratch_identity
            )
            self.scratch = None

    def __exit__(self, *_):
        self._close_query()
        self.active = False


def build_candidate_history(
    resources,
    report_pin,
    source_start,
    decision,
    coins,
    scope=None,
    *,
    max_bytes=MAX_BYTES,
    execution_policy_name=None,
):
    policy = execution_policy(execution_policy_name)
    expected = MAX_BYTES if policy is None else policy.candidate_history_bytes
    if (
        type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_BYTES
        or policy is not None
        and max_bytes != expected
    ):
        raise ValueError("Invalid candidate history byte limit")
    with CandidateHistory(
        resources,
        report_pin,
        source_start,
        decision,
        coins,
        scope,
        execution_policy_name=execution_policy_name,
    ) as history:
        publications = PublishedArtifacts(resources)
        inputs = dict(**history.inputs(), max_bytes=max_bytes)
        existing = publications.lookup("candidate_history", inputs)
        if existing is not None:
            history.verify()
            return existing
        relative = f"artifacts/{uuid.uuid4().hex}.parquet"
        token = resources.reserve(relative, max_bytes, "payload")
        values, groups = [], 0
        with (resources.root / relative).open("xb") as output:
            with pq.ParquetWriter(
                _CappedOutput(output, max_bytes), SCHEMA, compression="zstd"
            ) as writer:
                for user in history.rows():
                    values.append(user)
                    if len(values) == 4096:
                        groups += 1
                        _write(writer, values, groups)
                        values.clear()
                if values:
                    _write(writer, values, groups + 1)
        if not history.completed or history.db is not None:
            raise ValueError("Incomplete candidate history cannot publish")
        resources.settle(token)
        history.verify()
        if policy is not None:
            require_descriptor_bound(
                resources,
                "candidate_history",
                inputs,
                [token],
                policy.active_ranking_descriptor_bytes,
            )
        return publications.publish("candidate_history", inputs, [token])


def _write(writer, values, groups):
    if groups > MAX_GROUPS:
        raise ValueError("Candidate history rowgroup limit exceeded")
    batch = pa.record_batch([pa.array(values, type=pa.string())], schema=SCHEMA)
    if batch.nbytes > 64 * 1024**2:
        raise ValueError("Candidate history decoded batch limit exceeded")
    writer.write_batch(batch)
