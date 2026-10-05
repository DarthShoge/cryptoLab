"""UTC calendar candles and causal daily/weekly/monthly Supertrend consensus."""

from datetime import datetime, timedelta, timezone

from .indicators import supertrend

HISTORY_START = 1577836800  # 2020-01-01 UTC, bounds remote history requests.
INTERVALS = {"1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1M": None}


def candle_start(timestamp, interval):
    if interval == "1M":
        date = datetime.fromtimestamp(timestamp, timezone.utc)
        return int(
            date.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        )
    if interval == "1w":
        date = datetime.fromtimestamp(timestamp, timezone.utc)
        monday = date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=date.weekday()
        )
        return int(monday.timestamp())
    return int(timestamp) // INTERVALS[interval] * INTERVALS[interval]


def candle_end(timestamp, interval):
    start = candle_start(timestamp, interval)
    if interval == "1M":
        date = datetime.fromtimestamp(start, timezone.utc)
        year, month = (
            (date.year + 1, 1) if date.month == 12 else (date.year, date.month + 1)
        )
        return int(date.replace(year=year, month=month).timestamp())
    return start + INTERVALS[interval]


def aggregate_candles(source, interval, as_of, source_interval="1d"):
    """Only emit complete periods containing every required source bar.

    No partial week/month, missing daily/hourly interval, or repeated bar can
    silently create a complete higher-timeframe candle.
    """
    by_time = {
        c["time"]: c for c in source if candle_end(c["time"], source_interval) <= as_of
    }
    buckets = sorted({candle_start(t, interval) for t in by_time})
    result = []
    step = INTERVALS[source_interval]
    for start in buckets:
        end = candle_end(start, interval)
        expected = list(range(start, end, step))
        if end > as_of or not all(t in by_time for t in expected):
            continue
        bars = [by_time[t] for t in expected]
        result.append(
            {
                "time": start,
                "endTime": end,
                "open": bars[0]["open"],
                "high": max(c["high"] for c in bars),
                "low": min(c["low"] for c in bars),
                "close": bars[-1]["close"],
                "volume": sum(c["volume"] for c in bars),
            }
        )
    return result


def timeframe_supertrend(candles, interval, period=10, multiplier=3):
    """Restart Wilder ATR after a missing candle rather than bridging gaps."""
    supertrend([], period, multiplier)
    result, segment = [], []
    for candle in candles:
        if segment and candle_end(segment[-1]["time"], interval) != candle["time"]:
            result.extend(supertrend(segment, period, multiplier))
            segment = []
        segment.append(candle)
    result.extend(supertrend(segment, period, multiplier))
    return result


def combined_supertrend(
    display, daily, display_interval, period=10, multiplier=3, as_of=None
):
    """Equal-weight consensus using the latest expected COMPLETED period.

    A missing expected period is unavailable, never an indefinitely carried
    forward signal. Display candles are evaluated at their calendar close.
    """
    supertrend([], period, multiplier)
    if as_of is None:
        as_of = max((candle_end(c["time"], "1d") for c in daily), default=0)
    daily = sorted(
        (c for c in daily if candle_end(c["time"], "1d") <= as_of),
        key=lambda c: c["time"],
    )
    series = {
        "1d": daily,
        "1w": aggregate_candles(daily, "1w", as_of),
        "1M": aggregate_candles(daily, "1M", as_of),
    }
    signals = {}
    for interval, candles in series.items():
        signals[interval] = {
            candle_end(p["time"], interval): p["direction"]
            for p in timeframe_supertrend(candles, interval, period, multiplier)
        }
    result = []
    for candle in display:
        close = candle_end(candle["time"], display_interval)
        directions = {
            interval: signals[interval].get(candle_start(close, interval))
            for interval in ("1d", "1w", "1M")
        }
        bulls = sum(direction == "bullish" for direction in directions.values())
        bears = sum(direction == "bearish" for direction in directions.values())
        result.append(
            {
                "time": candle["time"],
                "score": (bulls - bears) / 3 if bulls + bears == 3 else None,
                "bullishCount": bulls,
                "bearishCount": bears,
                "directions": directions,
            }
        )
    return result
