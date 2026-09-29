"""Immutable, qualified source view for exact causal lookback queries.

File pruning is conservative. Consumers must still apply the half-open interval
to rows and reverify this view before publishing any derived result.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from datetime import datetime

from .contracts import utc, semantic_hash
from .derived_publication import _encode
from .download import file_hash
from .prefix_qualification import _previous, _engine
from .qualified_day import QualifiedDay, QualifiedFile, _day, micros


def _code():
    root = Path(__file__).parent
    return tuple(
        (name, file_hash(root / name))
        for name in (
            "qualified_window.py",
            "qualified_day.py",
            "contracts.py",
            "derived_publication.py",
        )
    )


@dataclass(frozen=True, init=False)
class QualifiedWindow:
    start: datetime
    end: datetime
    entries: tuple[QualifiedFile, ...]
    witness: QualifiedFile
    coins: tuple[str, ...]
    _anchor: QualifiedDay
    _code: tuple

    def __init__(self, report_pin, start, end):
        start, end = utc(start), utc(end)
        if start >= end:
            raise ValueError("Increasing qualified window required")
        # QualifiedDay validates bounded report membership, paths, whole-corpus
        # schema consistency, market scope and a first-day schema witness.
        code = _code()
        anchor = QualifiedDay(report_pin, start.date().isoformat())
        report = _previous(
            dict(path=str(anchor.report_path), sha256=anchor.report_sha256), _engine()
        )
        if (
            not _day(report["source_start"])
            <= start
            < end
            <= _day(report["source_end"])
        ):
            raise ValueError("Window outside qualified source days")
        files = tuple(QualifiedFile.parse(entry) for entry in report["files"])
        selected = tuple(
            f
            for f in files
            if f.rows and f.max_time >= micros(start) and f.min_time < micros(end)
        )
        for name, value in dict(
            start=start,
            end=end,
            entries=selected,
            witness=selected[0] if selected else anchor.witness,
            coins=anchor.coins,
            _anchor=anchor,
            _code=code,
        ).items():
            object.__setattr__(self, name, value)
        self.verify()

    def verify(self):
        self._anchor.verify()
        for entry in self.entries or (self.witness,):
            entry.verify()
        if (
            _code() != self._code
            or semantic_hash(_engine()) != self._anchor.engine_sha256
            or file_hash(self._anchor.report_path) != self._anchor.report_sha256
        ):
            raise ValueError("Qualified window engine/report changed")

    def inputs(self):
        membership = [
            (e.sha256, e.bytes, e.rows, e.min_time, e.max_time, e.schema_sha256)
            for e in self.entries
        ]
        return dict(
            schema=1,
            report_sha256=self._anchor.report_sha256,
            qualification_engine=self._anchor.engine_sha256,
            start=self.start.isoformat(),
            end=self.end.isoformat(),
            coins=list(self.coins),
            schema_sha256=self.witness.schema_sha256,
            file_count=len(self.entries),
            file_membership_sha256=hashlib.sha256(_encode(membership)).hexdigest(),
            derivation_code=dict(self._code),
        )
