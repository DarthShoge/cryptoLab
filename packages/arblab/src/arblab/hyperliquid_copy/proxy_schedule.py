"""UTC target decisions independent of the hourly valuation/funding clock."""

from datetime import timedelta
from .contracts import utc


def decision_times(start, end, frequency):
    start, end = utc(start), utc(end)
    if frequency not in ("hourly", "daily", "weekly") or start >= end:
        raise ValueError("Invalid proxy decision schedule")
    if (
        start.minute
        or start.second
        or start.microsecond
        or end.minute
        or end.second
        or end.microsecond
    ):
        raise ValueError("Proxy schedule requires whole UTC hours")
    at = start
    while at < end:
        if frequency == "hourly" or (
            at.hour == 0 and (frequency == "daily" or at.weekday() == 0)
        ):
            yield at
        at += timedelta(hours=1)
