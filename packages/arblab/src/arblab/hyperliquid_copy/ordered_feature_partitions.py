"""Bounded feature ordering; caller owns complete consumption and scratch disposal."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from . import feature_publication, qualified_window
from .archive_cache import _safe
from .checkpoint_io import _CappedOutput
from .contracts import semantic_hash
from .derived_publication import _encode
from .download import file_hash
from .feature_order import ObservationOrder
from .feature_records import OBSERVATION_SCHEMA, KEY_FIELDS, decode_observation
from .feature_window import FeatureWindow
from .ordered_wallet_partitions import AddressPartition, OrderedArtifact
from .query_directory_pin import pin_directory
from .wallet_partition_plan import counted_plan
from .wallet_replay_groups import coalesce_plan

MAX_BYTES = 512 * 1024**2
MAX_GROUPS = 4096


class OrderedFeaturePartitions:
    def __init__(self, window, scratch):
        if not isinstance(window, FeatureWindow):
            raise ValueError("Qualified feature window required")
        self.window, self.scratch = window, Path(scratch).absolute()
        self.db = self._plan = self._batch = None
        self._reservation()
        self._identities = tuple(self._identity(p) for p in self._directories)
        self._snapshot = self._context()
        self._check()

    @property
    def _directories(self):
        return self.scratch, self.scratch / "spill"

    @staticmethod
    def _identity(path):
        _safe(path)
        if not path.is_dir():
            raise ValueError("Owned query directory required")
        info = path.stat()
        return info.st_dev, info.st_ino

    def _reservation(self):
        resources = self.window.resources
        resources.lease.check()
        if self.scratch.parent != resources.root / "scratch":
            raise ValueError("Scratch outside reserved cache namespace")
        relative = str(self.scratch.relative_to(resources.root))
        with resources._connect() as db:
            row = db.execute(
                "SELECT maximum,purpose,state FROM allocations WHERE path=?", [relative]
            ).fetchone()
        if row is None or row[0] < 3 * 1024**3 or row[1:] != ("scratch", "pending"):
            raise ValueError("Pending combined scratch reservation required")

    def _context(self):
        root = Path(__file__).parent
        return _encode(
            dict(
                window=self.window.inputs(),
                duckdb=duckdb.__version__,
                pyarrow=pa.__version__,
                code={
                    name: file_hash(root / name)
                    for name in (
                        "ordered_feature_partitions.py",
                        "wallet_partition_plan.py",
                        "wallet_replay_groups.py",
                        "query_directory_pin.py",
                        "feature_window.py",
                    )
                },
                feature_engine=feature_publication.feature_engine(),
                qualification_engine=semantic_hash(qualified_window._engine()),
            )
        )

    def _check_runtime(self):
        self._reservation()
        if (
            tuple(self._identity(p) for p in self._directories) != self._identities
            or self._context() != self._snapshot
        ):
            raise ValueError("Ordered feature context/directory changed")

    def _check(self):
        self._check_runtime()
        if self._batch is None:
            self.window.verify()
        elif self.window._stats() != self._batch:
            raise ValueError("Feature window changed during query batch")
        self._check_runtime()

    @contextmanager
    def _pinned(self):
        with (
            pin_directory(self.scratch, self._identities[0]),
            pin_directory(self.scratch / "spill", self._identities[1]),
        ):
            self._check()
            try:
                yield
            finally:
                self._check()

    @contextmanager
    def verified_batch(self):
        if self._batch is not None:
            raise ValueError("Feature verification batch already active")
        self._check()
        self._batch = self.window._stats()
        try:
            with self._pinned():
                yield self
        finally:
            self._batch = None
            self._check()

    @contextmanager
    def _query(self):
        if self.db is not None:
            raise ValueError("Feature query already active")
        with self._pinned():
            self.db = duckdb.connect(
                config={
                    "memory_limit": "256MB",
                    "max_temp_directory_size": "2GB",
                    "temp_directory": str(self.scratch / "spill"),
                    "threads": 1,
                    "TimeZone": "UTC",
                    "preserve_insertion_order": False,
                }
            )
            try:
                self.db.execute("SET enable_progress_bar=false")
                self.db.read_parquet(
                    [
                        str(self.window.resources.root / p.path)
                        for p in self.window.observation_pins
                    ],
                    hive_partitioning=False,
                ).create_view("features")
                quote = lambda value: "'" + value.replace("'", "''") + "'"
                start, end = (
                    quote(t.isoformat()) for t in (self.window.start, self.window.end)
                )
                coins = ",".join(quote(c) for c in self.window.coins)
                self.db.execute(
                    f"CREATE TEMP VIEW source AS SELECT * FROM features "
                    f"WHERE exchange_time>=TIMESTAMPTZ {start} AND exchange_time<TIMESTAMPTZ {end} "
                    f"AND coin IN ({coins})"
                )
                yield self.db
            finally:
                self.db.close()
                self.db = None

    def plan(self, *, max_rows=250000):
        self._check()
        if type(max_rows) is not int or not 1 <= max_rows <= 250000:
            raise ValueError("Invalid feature partition row bound")
        if self._plan is not None:
            if max_rows != self._max_rows:
                raise ValueError("Feature partition bound changed")
            return self._plan
        with self._query() as db:
            parts = coalesce_plan(
                counted_plan(db, max_rows=max_rows), max_rows=max_rows
            )
        self._check()
        self._plan = tuple(AddressPartition(*part) for part in parts)
        self._max_rows = max_rows
        return self._plan

    def write(self, partition, path, *, max_bytes=MAX_BYTES):
        self._check()
        if self._plan is None or partition not in self._plan:
            raise ValueError("Feature partition outside complete plan")
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError("Invalid ordered feature byte bound")
        path = Path(path).absolute()
        _safe(path)
        if path.parent != self.scratch or {p.name for p in self.scratch.iterdir()} != {
            "spill"
        }:
            raise ValueError("One owned ordered artifact at a time required")
        count = groups = 0
        with self._pinned(), path.open("xb") as output:
            identity = os.fstat(output.fileno())
            with self._query() as db:
                query = db.execute(
                    "SELECT * FROM source WHERE user>=? AND user<? ORDER BY user,"
                    + ",".join(KEY_FIELDS)
                    + ",CASE kind WHEN 'episode' THEN 0 ELSE 1 END",
                    [partition.lower, partition.upper],
                )
                with (
                    query.to_arrow_reader(4096) as batches,
                    pq.ParquetWriter(
                        _CappedOutput(output, max_bytes),
                        OBSERVATION_SCHEMA,
                        compression="zstd",
                    ) as writer,
                ):
                    for batch in batches:
                        groups += 1
                        if groups > MAX_GROUPS or batch.nbytes > 64 * 1024**2:
                            raise ValueError(
                                "Ordered feature batch/metadata limit exceeded"
                            )
                        writer.write_batch(batch.cast(OBSERVATION_SCHEMA))
                        count += batch.num_rows
            output.flush()
            if count != partition.physical_rows:
                raise ValueError("Ordered feature physical count mismatch")
            digest = file_hash(path)
            info = path.stat()
            if (info.st_dev, info.st_ino) != (identity.st_dev, identity.st_ino):
                raise ValueError("Ordered feature output identity changed")
            result = OrderedArtifact(
                path,
                info.st_size,
                digest,
                count,
                partition,
                hashlib.sha256(self._snapshot).hexdigest(),
            )
        return result

    def _verify_artifact(self, artifact):
        if (
            not isinstance(artifact, OrderedArtifact)
            or artifact.context != hashlib.sha256(self._snapshot).hexdigest()
            or self._plan is None
            or artifact.partition not in self._plan
            or artifact.path.parent != self.scratch
            or artifact.bytes > MAX_BYTES
            or artifact.rows != artifact.partition.physical_rows
        ):
            raise ValueError("Ordered feature artifact context changed")
        _safe(artifact.path)
        if (
            artifact.path.stat().st_size != artifact.bytes
            or file_hash(artifact.path) != artifact.sha256
        ):
            raise ValueError("Ordered feature artifact identity changed")

    def read(self, artifact):
        self._check()
        self._verify_artifact(artifact)
        with self._pinned():
            try:
                order, count = ObservationOrder(), 0
                with pq.ParquetFile(artifact.path) as reader:
                    if (
                        reader.schema_arrow != OBSERVATION_SCHEMA
                        or reader.metadata.num_rows != artifact.rows
                        or reader.metadata.num_row_groups > MAX_GROUPS
                    ):
                        raise ValueError(
                            "Ordered feature artifact schema/count changed"
                        )
                    for batch in reader.iter_batches(batch_size=256, use_threads=False):
                        if batch.nbytes > 32 * 1024**2:
                            raise ValueError(
                                "Ordered feature decoded batch limit exceeded"
                            )
                        for row in batch.to_pylist():
                            observation = decode_observation(row)
                            order.add(row)
                            if (
                                not artifact.partition.lower
                                <= observation.user
                                < artifact.partition.upper
                                or not self.window.start
                                <= observation.order_key[0]
                                < self.window.end
                                or observation.coin not in self.window.coins
                            ):
                                raise ValueError(
                                    "Ordered feature observation scope changed"
                                )
                            count += 1
                            yield observation
                order.finish()
                if count != artifact.rows:
                    raise ValueError("Incomplete ordered feature consumption")
            finally:
                self._verify_artifact(artifact)
                self._check()
