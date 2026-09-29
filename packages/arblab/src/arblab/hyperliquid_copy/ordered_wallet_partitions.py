"""Bounded ordered replay primitive; caller reserves scratch and owns disposal.

No publication or completeness certificate is emitted here. The metric producer
must consume every plan interval and verify candidate/source evidence afterward.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .archive_cache import _safe
from .checkpoint_io import _CappedOutput
from .contracts import FillEvent, symbol
from .derived_publication import _encode
from .download import file_hash
from .proxy_activity import IDENTITY, ORDER
from .proxy_compact import COLUMNS
from .qualified_window import QualifiedWindow
from .wallet_partition_plan import counted_plan
from .wallet_replay_groups import coalesce_plan

MAX_ROWS = 250_000
MAX_PARTS = 5000
MAX_BYTES = 512 * 1024**2
MAX_GROUPS = 4096


@dataclass(frozen=True)
class AddressPartition:
    lower: str
    upper: str
    physical_rows: int
    single_wallet: str | None


@dataclass(frozen=True)
class OrderedArtifact:
    path: Path
    bytes: int
    sha256: str
    rows: int
    partition: AddressPartition
    context: str


class OrderedWalletPartitions:
    def __init__(self, window, coins, scratch, scope=None):
        if not isinstance(window, QualifiedWindow):
            raise ValueError("Qualified window required for ordered replay")
        if (
            type(coins) not in (list, tuple)
            or not 1 <= len(coins) <= 50
            or any(type(c) is not str for c in coins)
        ):
            raise ValueError("Invalid ordered replay markets")
        coins = tuple(sorted(symbol(c) for c in coins))
        if (
            len(set(coins)) != len(coins)
            or not set(coins) <= set(window.coins)
            or scope is not None
            and scope not in coins
        ):
            raise ValueError("Ordered replay scope outside source")
        self.window, self.coins, self.scope = window, coins, scope
        self.scratch = Path(scratch).absolute()
        _safe(self.scratch)
        if not self.scratch.is_dir():
            raise ValueError("Owned reserved replay scratch required")
        scratch_stat = self.scratch.stat()
        self._scratch_identity = (
            self.scratch,
            scratch_stat.st_dev,
            scratch_stat.st_ino,
        )
        self.db = None
        self._plan = None
        self._batch_state = None
        self._snapshot = self._context()

    def _context(self):
        return _encode(
            dict(
                source=self.window.inputs(),
                coins=self.coins,
                scope=self.scope,
                code_sha256=file_hash(Path(__file__)),
                planner_code={
                    name: file_hash(Path(__file__).with_name(name))
                    for name in ("wallet_partition_plan.py", "wallet_replay_groups.py")
                },
                duckdb=duckdb.__version__,
                pyarrow=pa.__version__,
            )
        )

    def _check_identity(self):
        if self._context() != self._snapshot:
            raise ValueError("Ordered replay context/engine changed")
        _safe(self.scratch)
        scratch_stat = self.scratch.stat()
        if (
            not self.scratch.is_dir()
            or (self.scratch, scratch_stat.st_dev, scratch_stat.st_ino)
            != self._scratch_identity
        ):
            raise ValueError("Owned reserved replay scratch changed")

    def verify(self):
        self._check_identity()
        self.window.verify()

    def _source_stats(self):
        entries = self.window.entries or (self.window.witness,)
        paths = (self.window._anchor.report_path, *(e.path for e in entries))
        result = []
        for path in paths:
            _safe(path)
            info = path.stat()
            result.append(
                (
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_nlink,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            )
        return tuple(result)

    def _check(self):
        if self._batch_state is None:
            self.verify()
        else:
            self._check_identity()
            if self._source_stats() != self._batch_state:
                raise ValueError("Ordered replay source changed during verified batch")

    @contextmanager
    def verified_batch(self):
        """Amortize source hashing across an unpublished replay, never skip it.

        Full hashes are mandatory on entry and exit (including exceptional exit).
        Each operation still checks pinned context, scratch and source stat
        identities. Callers may publish only after this context exits normally.
        """
        if self._batch_state is not None:
            raise ValueError("Ordered verification batch already active")
        self.verify()
        self._batch_state = self._source_stats()
        try:
            yield self
        finally:
            self._batch_state = None
            self.verify()

    @contextmanager
    def _query(self):
        if self.db is not None:
            raise ValueError("Ordered replay query is already open")
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
            entries = self.window.entries or (self.window.witness,)
            self.db.read_parquet(
                [str(e.path) for e in entries], hive_partitioning=False
            ).create_view("all_fills")
            coins = (self.scope,) if self.scope else self.coins
            # DuckDB .sql(..., params=...).create_view eagerly materializes the
            # result outside the query buffer limit. A literal SQL view stays
            # lazy and allows column/row predicates to reach the Parquet scan.
            # Values are validated timestamps/symbols and SQL literals are escaped.
            quote = lambda value: "'" + value.replace("'", "''") + "'"
            lower, upper = (
                quote(at.isoformat()) for at in (self.window.start, self.window.end)
            )
            markets = ",".join(quote(coin) for coin in coins)
            self.db.execute(
                f"CREATE TEMP VIEW source AS SELECT * FROM all_fills "
                f"WHERE exchange_time>=TIMESTAMPTZ {lower} AND exchange_time<TIMESTAMPTZ {upper} "
                f"AND coin IN ({markets})"
                + (" AND FALSE" if not self.window.entries else "")
            )
            yield self.db
        finally:
            self.db.close()
            self.db = None

    def plan(self, *, max_rows=MAX_ROWS):
        if type(max_rows) is not int or not 0 < max_rows <= MAX_ROWS:
            raise ValueError("Invalid ordered replay row bound")
        self._check()
        if self._plan is not None:
            if max_rows != self._max_rows:
                raise ValueError("Ordered replay plan bound changed")
            return self._plan
        with self._query() as db:
            physical = counted_plan(db, max_rows=max_rows)
            if len(physical) > MAX_PARTS:
                raise ValueError("Ordered replay partition limit exceeded")
            result = tuple(
                AddressPartition(*part)
                for part in coalesce_plan(physical, max_rows=max_rows)
            )
        self._check()
        self._plan, self._max_rows = tuple(result), max_rows
        return self._plan

    def write(self, partition, path, *, max_bytes=MAX_BYTES):
        self._check()
        if self._plan is None or partition not in self._plan:
            raise ValueError("Partition outside complete replay plan")
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError("Invalid ordered replay byte limit")
        path = Path(path).absolute()
        _safe(path)
        if path.parent != self.scratch:
            raise ValueError("Ordered output must be directly inside owned scratch")
        rows = groups = 0
        with self._query() as db:
            count = db.execute(
                "SELECT count(*) FROM source WHERE user>=? AND user<?",
                [partition.lower, partition.upper],
            ).fetchone()[0]
            if count != partition.physical_rows:
                raise ValueError("Ordered replay source count changed")
            query = db.execute(
                "WITH unique_fills AS (SELECT * FROM source WHERE user>=? AND user<? "
                f"QUALIFY row_number() OVER (PARTITION BY {IDENTITY} ORDER BY {ORDER})=1) "
                f"SELECT {','.join(COLUMNS)} FROM unique_fills ORDER BY user,{ORDER}",
                [partition.lower, partition.upper],
            )
            with query.to_arrow_reader(4096) as batches, path.open("xb") as output:
                with pq.ParquetWriter(
                    _CappedOutput(output, max_bytes), batches.schema, compression="zstd"
                ) as writer:
                    for batch in batches:
                        groups += 1
                        if groups > MAX_GROUPS or batch.nbytes > 64 * 1024**2:
                            raise ValueError(
                                "Ordered replay batch/metadata limit exceeded"
                            )
                        writer.write_batch(batch)
                        rows += batch.num_rows
        self._check()
        return OrderedArtifact(
            path,
            path.stat().st_size,
            file_hash(path),
            rows,
            partition,
            hashlib.sha256(self._snapshot).hexdigest(),
        )

    def _verify_artifact(self, artifact):
        if (
            not isinstance(artifact, OrderedArtifact)
            or artifact.context != hashlib.sha256(self._snapshot).hexdigest()
            or self._plan is None
            or artifact.partition not in self._plan
            or artifact.path.parent != self.scratch
        ):
            raise ValueError("Ordered artifact context changed")
        _safe(artifact.path)
        if (
            artifact.path.stat().st_size != artifact.bytes
            or file_hash(artifact.path) != artifact.sha256
        ):
            raise ValueError("Ordered artifact identity changed")

    def read(self, artifact):
        self._check()
        self._verify_artifact(artifact)
        previous, count = None, 0
        with pq.ParquetFile(artifact.path) as reader:
            if (
                reader.metadata.num_rows != artifact.rows
                or reader.metadata.num_row_groups > MAX_GROUPS
                or reader.schema_arrow.names != COLUMNS
            ):
                raise ValueError("Ordered artifact schema/count changed")
            for batch in reader.iter_batches(batch_size=4096, use_threads=False):
                if batch.nbytes > 64 * 1024**2:
                    raise ValueError("Ordered replay decoded batch limit exceeded")
                for row in batch.to_pylist():
                    fill = FillEvent(**row)
                    key = (fill.user, *fill.order_key)
                    if (
                        previous is not None
                        and key < previous
                        or not artifact.partition.lower
                        <= fill.user
                        < artifact.partition.upper
                        or not self.window.start <= fill.exchange_time < self.window.end
                        or fill.coin not in self.coins
                        or self.scope is not None
                        and fill.coin != self.scope
                    ):
                        raise ValueError("Ordered replay event scope/order changed")
                    previous = key
                    count += 1
                    yield fill
        if count != artifact.rows:
            raise ValueError("Incomplete ordered replay artifact")
        self._verify_artifact(artifact)
        self._check()
