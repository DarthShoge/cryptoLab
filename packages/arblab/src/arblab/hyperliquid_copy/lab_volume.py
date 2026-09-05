"""Nonoverlapping daily market turnover; never inferred from tracked-wallet fills."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from .contracts import symbol, utc, finite
from .lab_instruments import timestamp


@dataclass(frozen=True)
class VolumeBucket:
    instrument_id: str
    interval_start: datetime
    interval_end: datetime
    available_at: datetime
    notional_usd: float

    def __post_init__(self):
        symbol(self.instrument_id)
        for name in ("interval_start", "interval_end", "available_at"):
            object.__setattr__(self, name, timestamp(getattr(self, name)))
        start = self.interval_start
        if (
            start.hour
            or start.minute
            or start.second
            or start.microsecond
            or self.interval_end - start != timedelta(days=1)
        ):
            raise ValueError("Market volume requires complete UTC daily buckets")
        if self.available_at < self.interval_end or finite(self.notional_usd) < 0:
            raise ValueError("Invalid market volume or publication time")


class MarketVolume:
    def __init__(self, rows, *, snapshot_at=None, instrument_ids=None):
        self.buckets = {}
        for row in rows:
            bucket = VolumeBucket(**row)
            key = bucket.instrument_id, bucket.interval_start
            if key in self.buckets:
                raise ValueError("Duplicate market volume day")
            if snapshot_at is not None and bucket.available_at > timestamp(snapshot_at):
                raise ValueError("Volume published after dataset snapshot")
            if (
                instrument_ids is not None
                and bucket.instrument_id not in instrument_ids
            ):
                raise ValueError("Unknown volume instrument")
            self.buckets[key] = bucket

    def trailing(self, identifier, decision, days, lag=1):
        end = decision - timedelta(days=lag)
        at = end - timedelta(days=days)
        total = 0.0
        while at < end:
            row = self.buckets.get((identifier, at))
            if row is None or row.available_at >= decision:
                return None
            total += row.notional_usd
            at += timedelta(days=1)
        return total
