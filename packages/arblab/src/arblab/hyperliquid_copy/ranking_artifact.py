"""Bounded disk ranking history; publication copies files, never all-row lists."""

from dataclasses import dataclass
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from .download import file_hash
from .lab_config import METRICS
from .lab_market_evidence import EMPTY_RANKINGS

MAX_RANKING_ROWS = 250_000_000
MAX_RANKING_BYTES = 16 * 1024**3
RESERVE_BYTES = 64 * 1024**2
METRIC_TYPE = pa.struct([pa.field(k, pa.float64()) for k in METRICS])
RANKING_SCHEMA = pa.schema(
    [
        *EMPTY_RANKINGS,
        pa.field("metrics", METRIC_TYPE),
        pa.field("percentiles", METRIC_TYPE),
        pa.field("exclusions", pa.list_(pa.string())),
        pa.field("weight", pa.float64()),
    ]
)


def _space(directory, size):
    if shutil.disk_usage(directory).free < size + RESERVE_BYTES:
        raise ValueError("Insufficient ranking artifact disk space")


@dataclass(frozen=True)
class RankingFile:
    path: Path
    rows: int
    bytes: int
    sha256: str

    def __len__(self):
        return self.rows

    def verify(self):
        if (
            self.path.is_symlink()
            or not self.path.is_file()
            or self.path.stat().st_size != self.bytes
            or file_hash(self.path) != self.sha256
        ):
            raise ValueError("Ranking artifact identity changed")

    def copy_to(self, destination):
        self.verify()
        destination = Path(destination)
        _space(destination.parent, self.bytes)
        with self.path.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024**2)
        self.verify()
        if file_hash(destination) != self.sha256:
            raise ValueError("Copied ranking artifact identity mismatch")


class RankingSink:
    def __init__(self, path, *, max_rows=MAX_RANKING_ROWS, max_bytes=MAX_RANKING_BYTES):
        if (
            type(max_rows) is not int
            or not 0 < max_rows <= MAX_RANKING_ROWS
            or type(max_bytes) is not int
            or not 0 < max_bytes <= MAX_RANKING_BYTES
        ):
            raise ValueError("Invalid ranking artifact limits")
        self.path, self.max_rows, self.max_bytes = Path(path), max_rows, max_bytes
        self.rows = 0
        self._artifact = None
        self._writer = None
        self._handle = None
        self._failed = False

    def __len__(self):
        return self.rows

    def __enter__(self):
        _space(self.path.parent, RESERVE_BYTES)
        self._handle = self.path.open("xb")
        try:
            self._writer = pq.ParquetWriter(
                self._handle, RANKING_SCHEMA, compression="zstd"
            )
        except BaseException:
            self._handle.close()
            raise
        return self

    def extend(self, rows):
        if self._writer is None or self._failed:
            raise ValueError("Ranking sink is not open")
        if len(rows) > 100_000 or self.rows + len(rows) > self.max_rows:
            raise ValueError("Ranking artifact row limit exceeded")
        for offset in range(0, len(rows), 4096):
            batch = rows[offset : offset + 4096]
            for row in batch:
                if set(row) - set(RANKING_SCHEMA.names):
                    raise ValueError("Unknown ranking evidence field")
                for name in ("metrics", "percentiles"):
                    if set(row.get(name) or {}) - set(METRICS):
                        raise ValueError("Unknown ranking metric field")
            cleaned = [
                {k: None if v == {} else v for k, v in row.items()} for row in batch
            ]
            table = pa.Table.from_pylist(cleaned, schema=RANKING_SCHEMA)
            if table.nbytes > RESERVE_BYTES:
                raise ValueError("Ranking decoded batch byte limit exceeded")
            _space(self.path.parent, table.nbytes)
            self._writer.write_table(table)
            self.rows += len(batch)
            if self._handle.tell() > self.max_bytes:
                raise ValueError("Ranking artifact byte limit exceeded")

    def abort(self):
        """Poison this sink so caught append failures cannot be retried in-place."""
        self._failed = True

    def __exit__(self, kind, *_):
        try:
            self._writer.close()
        finally:
            self._handle.close()
            self._writer = None
        if kind is None and not self._failed:
            size = self.path.stat().st_size
            if size > self.max_bytes:
                raise ValueError("Ranking artifact byte limit exceeded")
            self._artifact = RankingFile(
                self.path.resolve(), self.rows, size, file_hash(self.path)
            )

    @property
    def artifact(self):
        if self._artifact is None:
            raise ValueError("Ranking artifact is not successfully finalized")
        return self._artifact
