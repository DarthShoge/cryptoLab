"""Immutable qualified binding for a complete calendar-day feature chain.

Queries must still enforce the exact intraday interval and selected market scope.
This descriptor performs no allocation, feature building or ranking query.
"""

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
from pathlib import Path

from . import feature_publication, qualified_window
from .contracts import semantic_hash, symbol, utc
from .derived_publication import _encode, _inputs
from .download import file_hash
from .feature_publication import FeatureDay
from .qualified_day import micros
from .qualified_window import QualifiedWindow


@dataclass(frozen=True, init=False)
class FeatureWindow:
    resources: object
    source: QualifiedWindow
    days: tuple
    start: object
    end: object
    coins: tuple
    semantics: str
    _encoded: bytes

    def __init__(self, resources, report_pin, days, start, end, coins, semantics):
        resources.lease.check()
        start, end = utc(start), utc(end)
        if not start < end or end - start > timedelta(days=732):
            raise ValueError("Invalid bounded feature window")
        if type(days) not in (list, tuple) or not 1 <= len(days) <= 733:
            raise ValueError("Expected bounded complete feature day chain")
        if (
            type(coins) not in (list, tuple)
            or not 1 <= len(coins) <= 50
            or list(coins) != sorted({symbol(c) for c in coins})
        ):
            raise ValueError("Invalid feature window market subset")
        first = start.replace(hour=0, minute=0, second=0, microsecond=0)
        last = end.replace(hour=0, minute=0, second=0, microsecond=0)
        if last != end:
            last += timedelta(days=1)
        if len(days) != (last - first).days:
            raise ValueError("Incomplete feature day coverage")
        source = QualifiedWindow(report_pin, first, last)
        for name, value in dict(
            resources=resources,
            source=source,
            days=tuple(days),
            start=start,
            end=end,
            coins=tuple(coins),
            semantics=semantics,
        ).items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_encoded", _encode(self._binding()))
        self.verify()

    def _day_source(self, day):
        end = day + timedelta(days=1)
        selected = tuple(
            entry
            for entry in self.source.entries
            if entry.rows
            and entry.max_time >= micros(day)
            and entry.min_time < micros(end)
        )
        membership = [
            (e.sha256, e.bytes, e.rows, e.min_time, e.max_time, e.schema_sha256)
            for e in selected
        ]
        return dict(
            self.source.inputs(),
            start=day.isoformat(),
            end=end.isoformat(),
            file_count=len(selected),
            file_membership_sha256=hashlib.sha256(_encode(membership)).hexdigest(),
        )

    def _binding(self):
        expected = self.source.start
        common, previous, artifacts = None, None, 0
        keys = []
        for day in self.days:
            if not isinstance(day, FeatureDay) or day.resources is not self.resources:
                raise ValueError("Feature day requires current resource lease")
            data = day.inputs
            current = (data["origin"], data["coins"], data["semantics"], data["engine"])
            if (
                day.day != expected
                or day.cutoff != expected + timedelta(days=1)
                or data["semantics"] != self.semantics
                or not set(self.coins) <= set(data["coins"])
                or data["source"] != self._day_source(expected)
                or common is not None
                and current != common
                or previous is not None
                and data["previous"] != previous
            ):
                raise ValueError("Feature day chain/source/scope mismatch")
            artifacts += len(day._observations)
            if artifacts > 5000:
                raise ValueError("Feature window observation artifact limit exceeded")
            common, previous, expected = current, day.publication.key, day.cutoff
            keys.append(previous)
        if expected != self.source.end:
            raise ValueError("Incomplete feature day coverage")
        return _inputs(
            "qualified_feature_window",
            dict(
                schema=1,
                source=self.source.inputs(),
                start=self.start.isoformat(),
                end=self.end.isoformat(),
                coins=list(self.coins),
                semantics=self.semantics,
                days=keys,
                code_sha256=file_hash(Path(__file__)),
            ),
        )

    @property
    def observation_pins(self):
        return tuple(pin for day in self.days for pin in day._observations)

    def inputs(self):
        return json.loads(self._encoded)

    def _stats(self):
        def paths():
            yield self.resources.path
            yield self.resources.marker
            yield self.source._anchor.report_path
            yield self.source._anchor.witness.path
            for entry in self.source.entries or (self.source.witness,):
                yield entry.path
            for day in self.days:
                for pin in day.publication.artifacts:
                    yield self.resources.root / pin.path

        digest = hashlib.sha256()
        try:
            for path in paths():
                info = path.lstat()
                digest.update(
                    _encode(
                        (
                            str(path),
                            info.st_dev,
                            info.st_ino,
                            info.st_mode,
                            info.st_nlink,
                            info.st_size,
                            info.st_mtime_ns,
                            info.st_ctime_ns,
                        )
                    )
                )
        except OSError as exc:
            raise ValueError("Feature window input unavailable") from exc
        return digest.digest()

    def verify(self):
        self.resources.lease.check()
        before = self._stats()
        self.source.verify()
        for day in self.days:
            day.verify()
        if (
            _encode(self._binding()) != self._encoded
            or self._stats() != before
            or semantic_hash(qualified_window._engine())
            != self.source._anchor.engine_sha256
            or feature_publication.feature_engine() != self.days[0].inputs["engine"]
        ):
            raise ValueError("Feature window context/engine changed")
