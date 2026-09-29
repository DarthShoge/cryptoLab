"""Causal OHLC lookups for approximate execution, never synthetic order books.

Session/gap validation belongs to the source layer. These lookups only expose
actual bar opens and completed closes within caller-supplied time bounds.
"""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime

from .contracts import finite, symbol, utc


def _positive(value):
    number = finite(value)
    if number <= 0:
        raise ValueError("Expected positive finite value")
    return number


@dataclass(frozen=True)
class ProxyBar:
    instrument_id: str
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self):
        symbol(self.instrument_id)
        for name in ("start", "end"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        if not 0 < (self.end - self.start).total_seconds() <= 3600:
            raise ValueError("Expected positive bar duration at most one hour")
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _positive(getattr(self, name)))
        if (
            not self.low
            <= min(self.open, self.close)
            <= max(self.open, self.close)
            <= self.high
        ):
            raise ValueError("Invalid OHLC ordering")


@dataclass(frozen=True)
class ProxyMark:
    price: float
    time: datetime
    age_seconds: float


class ProxyBars:
    def __init__(self, bars):
        unique = {}
        for bar in bars:
            if not isinstance(bar, ProxyBar):
                raise TypeError("Expected validated ProxyBar")
            key = bar.instrument_id, bar.start
            if key in unique and unique[key] != bar:
                raise ValueError("conflicting proxy bars")
            unique[key] = bar
        grouped = {}
        for key, bar in sorted(unique.items()):
            rows = grouped.setdefault(key[0], [])
            if rows and rows[-1].end > bar.start:
                raise ValueError("overlapping proxy bars")
            rows.append(bar)
        self._bars = {coin: tuple(rows) for coin, rows in grouped.items()}
        self._starts = {
            coin: tuple(b.start for b in rows) for coin, rows in self._bars.items()
        }
        self._ends = {
            coin: tuple(b.end for b in rows) for coin, rows in self._bars.items()
        }

    def next_open(self, instrument_id, due, *, max_wait_seconds):
        due = utc(due)
        limit = _positive(max_wait_seconds)
        rows = self._bars.get(instrument_id, ())
        index = bisect_right(self._starts.get(instrument_id, ()), due)
        if index == len(rows):
            return None
        bar = rows[index]
        return bar if (bar.start - due).total_seconds() <= limit else None

    def completed_mark(self, instrument_id, at, *, max_age_seconds):
        at = utc(at)
        limit = _positive(max_age_seconds)
        index = bisect_right(self._ends.get(instrument_id, ()), at) - 1
        if index < 0:
            raise ValueError(f"No completed proxy bar: {instrument_id}")
        bar = self._bars[instrument_id][index]
        age = (at - bar.end).total_seconds()
        if age > limit:
            raise ValueError(f"stale proxy mark: {instrument_id}, age={age}s")
        return ProxyMark(bar.close, bar.end, age)
