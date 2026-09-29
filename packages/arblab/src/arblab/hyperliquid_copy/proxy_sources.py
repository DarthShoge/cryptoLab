"""Pure, unadjusted source normalization with explicit missing-bar evidence.

Yahoo session windows must come from separately validated session evidence.
No market closure is inferred from the absence of a quote.
"""

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .contracts import utc
from .proxy_bars import ProxyBar, ProxyBars


@dataclass(frozen=True)
class NormalizedBars:
    bars: tuple[ProxyBar, ...]
    missing_starts: tuple[datetime, ...]


def _range(start, end):
    start, end = utc(start), utc(end)
    if start >= end:
        raise ValueError("Invalid requested price range")
    return start, end


def _finish(bars, expected):
    ProxyBars(bars)  # Reject conflicting duplicates and overlaps.
    unique = {bar.start: bar for bar in bars}
    return NormalizedBars(
        tuple(unique[t] for t in sorted(unique)),
        tuple(sorted(set(expected) - unique.keys())),
    )


def binance_bars(text, instrument_id, *, start, end):
    """Normalize USD-M 1h CSV (millisecond timestamps, inclusive close time)."""
    start, end = _range(start, end)
    expected = []
    at = start.replace(minute=0, second=0, microsecond=0)
    if at < start:
        at += timedelta(hours=1)
    while at + timedelta(hours=1) <= end:
        expected.append(at)
        at += timedelta(hours=1)
    bars = []
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] == "open_time":
            continue
        if len(row) != 12:
            raise ValueError("Invalid Binance kline row")
        opened = datetime.fromtimestamp(int(row[0]) / 1000, timezone.utc)
        closed = datetime.fromtimestamp((int(row[6]) + 1) / 1000, timezone.utc)
        if not start <= opened < end or closed > end:
            continue
        if opened not in expected or closed - opened != timedelta(hours=1):
            raise ValueError("Invalid Binance hourly window")
        bars.append(ProxyBar(instrument_id, opened, closed, *map(float, row[1:5])))
    return _finish(bars, expected)


def _qualify_yahoo(data, start, end):
    if data["meta"].get("instrumentType") not in {"EQUITY", "ETF", "INDEX"}:
        raise ValueError(
            "Unsupported or unqualified Yahoo instrument type; rolls are not modeled"
        )
    events = data.get("events", {})
    if not isinstance(events, dict) or set(events) - {
        "splits",
        "dividends",
        "capitalGains",
    }:
        raise ValueError("Invalid Yahoo corporate action evidence")
    for records in events.values():
        if not isinstance(records, dict):
            raise ValueError("Invalid Yahoo corporate action evidence")
        for record in records.values():
            date = record.get("date") if isinstance(record, dict) else None
            if type(date) is not int:
                raise ValueError("Invalid Yahoo corporate action timestamp")
            if start.timestamp() <= date < end.timestamp():
                raise ValueError("Unmodeled Yahoo corporate action in requested period")


def yahoo_bars(payload, instrument_id, *, start, end, windows):
    """Normalize raw USD Yahoo OHLC against supplied half-open session windows."""
    start, end = _range(start, end)
    schedule = {}
    for opened, closed in windows.items():
        opened, closed = utc(opened), utc(closed)
        if not 0 < (closed - opened).total_seconds() <= 3600:
            raise ValueError("Invalid Yahoo session window")
        if start <= opened < end:
            schedule[opened] = closed
    expected = {t for t, close in schedule.items() if close <= end}
    chart = payload["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError("Yahoo source returned an error or no result")
    if len(chart["result"]) != 1:
        raise ValueError("Ambiguous Yahoo source result")
    data = chart["result"][0]
    if data["meta"].get("currency") != "USD":
        raise ValueError("Unsupported Yahoo quote currency")
    _qualify_yahoo(data, start, end)
    timestamps = data.get("timestamp", [])
    quote = data["indicators"]["quote"][0]
    fields = ("open", "high", "low", "close")
    if any(len(quote.get(field, [])) != len(timestamps) for field in fields):
        raise ValueError("Mismatched Yahoo price arrays")
    bars = []
    for index, timestamp in enumerate(timestamps):
        opened = datetime.fromtimestamp(timestamp, timezone.utc)
        if not start <= opened < end:
            continue  # Yahoo may append a quote outside the requested range.
        if opened not in schedule:
            raise ValueError("Yahoo quote outside supplied session windows")
        if opened not in expected:
            continue  # Never use the final, incomplete bar.
        prices = [quote[field][index] for field in fields]
        if any(value is None for value in prices):
            continue  # Missing evidence, not permission to forward-fill prices.
        bars.append(ProxyBar(instrument_id, opened, schedule[opened], *prices))
    return _finish(bars, expected)
