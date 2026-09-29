"""Pinned event-day view of qualified source partitions, not listing/coverage proof."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import re

import pyarrow.parquet as pq

from .archive_cache import _safe
from .contracts import semantic_hash, symbol
from .download import file_hash
from .prefix_qualification import _engine, _previous


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("Invalid qualified source digest")
    return value


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Expected UTC calendar day")
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def micros(value):
    delta = value - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


@dataclass(frozen=True)
class QualifiedFile:
    path: Path
    sha256: str
    bytes: int
    rows: int
    min_time: int | None
    max_time: int | None
    schema_sha256: str

    @classmethod
    def parse(cls, entry):
        path = Path(entry["path"]).absolute()
        _safe(path)
        size, rows = entry["bytes"], entry["rows"]
        low, high = entry["min_time"], entry["max_time"]
        if (
            type(size) is not int
            or not 0 < size <= 64 * 1024**3
            or type(rows) is not int
            or not 0 <= rows <= 2**63 - 1
        ):
            raise ValueError("Invalid qualified source counts")
        if rows:
            if (
                type(low) is not int
                or type(high) is not int
                or not -(2**63) <= low <= high < 2**63
            ):
                raise ValueError("Invalid qualified event bounds")
        elif low is not None or high is not None:
            raise ValueError("Empty source must have null bounds")
        return cls(
            path,
            _digest(entry["sha256"]),
            size,
            rows,
            low,
            high,
            _digest(entry["schema_sha256"]),
        )

    def verify(self):
        _safe(self.path)
        if (
            not self.path.is_file()
            or self.path.stat().st_size != self.bytes
            or file_hash(self.path) != self.sha256
        ):
            raise ValueError("Qualified source identity changed")
        with pq.ParquetFile(self.path) as reader:
            schema_hash = hashlib.sha256(
                reader.schema_arrow.serialize().to_pybytes()
            ).hexdigest()
            if (
                reader.metadata.num_rows != self.rows
                or schema_hash != self.schema_sha256
            ):
                raise ValueError("Qualified source schema/row count changed")


@dataclass(frozen=True, init=False)
class QualifiedDay:
    report_path: Path
    report_sha256: str
    engine_sha256: str
    start: datetime
    end: datetime
    coins: tuple[str, ...]
    entries: tuple[QualifiedFile, ...]
    witness: QualifiedFile

    def __init__(self, report_pin, day):
        try:
            if type(report_pin) is not dict or set(report_pin) != {"path", "sha256"}:
                raise ValueError("Pinned qualification report required")
            digest = _digest(report_pin["sha256"])
            engine = _engine()
            report = _previous(dict(report_pin), engine)
            start, end = _day(day), _day(day) + timedelta(days=1)
            if (
                not _day(report["source_start"])
                <= start
                < end
                <= _day(report["source_end"])
            ):
                raise ValueError("Day outside qualified source partitions")
            entries = report["files"]
            if type(entries) is not list or not 1 <= len(entries) <= 5000:
                raise ValueError("Qualified source file count exceeded")
            files = tuple(QualifiedFile.parse(entry) for entry in entries)
            if (
                len({f.path for f in files}) != len(files)
                or sum(f.bytes for f in files) > 64 * 1024**3
                or len({f.schema_sha256 for f in files}) != 1
            ):
                raise ValueError("Invalid qualified corpus identity/size/schema")
            coins = report["coins"]
            if (
                type(coins) is not list
                or not 1 <= len(coins) <= 50
                or coins != sorted({symbol(coin) for coin in coins})
            ):
                raise ValueError("Invalid qualified market scope")
            selected = tuple(
                f
                for f in files
                if f.rows and f.max_time >= micros(start) and f.min_time < micros(end)
            )
            for name, value in dict(
                report_path=Path(report_pin["path"]).absolute(),
                report_sha256=digest,
                engine_sha256=semantic_hash(engine),
                start=start,
                end=end,
                coins=tuple(coins),
                entries=selected,
                witness=selected[0] if selected else files[0],
            ).items():
                object.__setattr__(self, name, value)
            self.verify()
        except (KeyError, TypeError, OSError) as exc:
            raise ValueError("Invalid qualified day source") from exc

    def verify(self):
        _safe(self.report_path)
        if (
            file_hash(self.report_path) != self.report_sha256
            or semantic_hash(_engine()) != self.engine_sha256
        ):
            raise ValueError("Qualification report/engine identity changed")
        for entry in self.entries or (self.witness,):
            entry.verify()
        if file_hash(self.report_path) != self.report_sha256:
            raise ValueError("Qualification changed during source verification")

    def inputs(self):
        return dict(
            report_sha256=self.report_sha256,
            qualification_engine=self.engine_sha256,
            day=self.start.date().isoformat(),
            coins=list(self.coins),
            schema_sha256=self.witness.schema_sha256,
        )
