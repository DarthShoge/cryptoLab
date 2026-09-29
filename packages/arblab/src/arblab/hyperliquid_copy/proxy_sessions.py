"""Versioned exchange schedules, independent of observed price availability."""

from datetime import timedelta
from importlib.metadata import version

import exchange_calendars

from .contracts import utc


CALENDARS = ("24/7", "XNYS")


def hourly_windows(calendar, start, end):
    start, end = utc(start), utc(end)
    if calendar not in CALENDARS:
        raise ValueError("Unsupported proxy calendar")
    if not 0 < (end - start).total_seconds() <= 366 * 86400:
        raise ValueError("Proxy calendar range must be positive and at most 366 days")
    windows = {}
    if calendar == "24/7":
        at = start.replace(minute=0, second=0, microsecond=0)
        if at < start:
            at += timedelta(hours=1)
        while at < end:
            windows[at] = at + timedelta(hours=1)
            at += timedelta(hours=1)
        return windows
    schedule = exchange_calendars.get_calendar(
        calendar,
        start=(start - timedelta(days=7)).date().isoformat(),
        end=(end + timedelta(days=7)).date().isoformat(),
    ).schedule
    for row in schedule.itertuples():
        opened, closed = row.open.to_pydatetime(), row.close.to_pydatetime()
        at = opened
        while at < closed:
            finish = min(at + timedelta(hours=1), closed)
            if start <= at < end:
                windows[at] = finish
            at = finish
    return windows


def calendar_provenance():
    return {
        "library": "exchange-calendars",
        "version": version("exchange-calendars"),
        "convention": "regular-session, start-anchored hourly bars; short closing bar",
    }
