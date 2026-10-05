import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


def timestamp(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def days(start, count):
    return [
        {
            "time": start + i * 86400,
            "open": 10 + i,
            "high": 12 + i,
            "low": 8 + i,
            "close": 11 + i,
            "volume": 2,
        }
        for i in range(count)
    ]


def test_weekly_candles_start_monday_and_exclude_partial_weeks():
    from api.timeframes import aggregate_candles

    source = days(timestamp("2024-02-01"), 29)
    result = aggregate_candles(source, "1w", timestamp("2024-03-01"))
    assert [c["time"] for c in result] == [
        timestamp("2024-02-05"),
        timestamp("2024-02-12"),
        timestamp("2024-02-19"),
    ]
    assert result[0]["endTime"] == timestamp("2024-02-12")
    assert result[0]["volume"] == 14
    assert result[0]["open"] == source[4]["open"]
    assert result[0]["close"] == source[10]["close"]


def test_calendar_months_include_leap_day_and_never_use_30_day_approximation():
    from api.timeframes import aggregate_candles, candle_end

    source = days(timestamp("2024-01-01"), 91)
    result = aggregate_candles(source, "1M", timestamp("2024-04-01"))
    assert [c["time"] for c in result] == [
        timestamp("2024-01-01"),
        timestamp("2024-02-01"),
        timestamp("2024-03-01"),
    ]
    assert [c["volume"] for c in result] == [62, 58, 62]
    assert candle_end(timestamp("2024-02-01"), "1M") == timestamp("2024-03-01")


def test_missing_daily_bar_invalidates_week_and_month():
    from api.timeframes import aggregate_candles

    source = days(timestamp("2024-02-01"), 29)
    source = [c for c in source if c["time"] != timestamp("2024-02-14")]
    assert aggregate_candles(source, "1M", timestamp("2024-03-01")) == []
    weeks = aggregate_candles(source, "1w", timestamp("2024-03-01"))
    assert timestamp("2024-02-12") not in [c["time"] for c in weeks]


def test_consensus_requires_monthly_warmup_and_uses_closed_signals_only():
    from api.timeframes import combined_supertrend

    start = timestamp("2024-01-01")
    daily = days(start, 91)
    display = [
        {
            "time": timestamp("2024-02-01"),
            "open": 10,
            "high": 20,
            "low": 5,
            "close": 15,
        },
        {
            "time": timestamp("2024-03-01"),
            "open": 10,
            "high": 20,
            "low": 5,
            "close": 15,
        },
    ]
    result = combined_supertrend(
        display, daily, "1d", period=2, multiplier=3, as_of=timestamp("2024-04-01")
    )
    # Feb1's daily close sees only January's completed monthly candle, not Feb's.
    assert result[0]["directions"]["1M"] is None
    assert result[0]["score"] is None
    assert result[1]["directions"]["1M"] is not None
    assert result[1]["score"] == pytest.approx(
        (result[1]["bullishCount"] - result[1]["bearishCount"]) / 3
    )


def test_missing_latest_daily_signal_is_unavailable_instead_of_carried_forward():
    from api.timeframes import combined_supertrend

    daily = days(timestamp("2024-01-01"), 91)
    missing = timestamp("2024-03-10")
    daily = [c for c in daily if c["time"] != missing]
    display = [
        {"time": missing, "open": 1, "high": 2, "low": 1, "close": 2},
        {"time": missing + 86400, "open": 1, "high": 2, "low": 1, "close": 2},
    ]
    result = combined_supertrend(
        display, daily, "1d", period=2, multiplier=3, as_of=timestamp("2024-04-01")
    )
    assert result[0]["directions"]["1d"] is None
    assert result[0]["score"] is None
    # The first daily candle after a gap must warm its ATR again.
    assert result[1]["directions"]["1d"] is None


def test_monthly_chart_prefetches_indicator_warmup_and_reuses_combined_daily_source(
    monkeypatch,
):
    from api import server

    calls = []
    end = timestamp("2026-09-30")
    source = days(timestamp("2020-01-01"), (end - timestamp("2020-01-01")) // 86400)

    def fetch(asset, interval, days, end_time=None, start_time=None):
        calls.append((interval, days, start_time))
        return source

    server._candles_cache.clear()
    monkeypatch.setattr(server, "market_candles", fetch)
    response = server.chart_data(
        "live", "SOL", "1M", 90, 10, 3, end_time=end, combined=True
    )
    assert len(response["candles"]) == 2  # July/August; September has not closed.
    assert all(point["value"] is not None for point in response["indicator"])
    assert all(point["score"] is not None for point in response["combined"])
    assert len(calls) == 1
    assert calls[0][0] == "1d"
    server.chart_data("live", "SOL", "1w", 90, 10, 3, end_time=end, combined=True)
    assert len(calls) == 1


def test_combined_off_daily_chart_does_not_fetch_warmup_source(monkeypatch):
    from api import server

    calls = []

    def fetch(asset, interval, days, end_time=None, start_time=None):
        calls.append((interval, days, start_time))
        return []

    server._candles_cache.clear()
    monkeypatch.setattr(server, "market_candles", fetch)
    result = server.chart_data("live", "SOL", "1d", 90, 10, 3)
    assert "combined" not in result
    assert calls == [("1d", 90, None)]


def test_demo_monthly_and_combined_use_same_deterministic_warmup_history():
    from api import server

    result = server.chart_data("demo", "SOL", "1M", 90, 10, 3, combined=True)
    assert len(result["candles"]) == 2
    assert all(p["value"] is not None for p in result["indicator"])
    assert all(p["score"] is not None for p in result["combined"])
    assert len(server.chart_data("demo", "SOL", "1d", 90, 10, 3)["candles"]) == 90


def test_invalid_inception_timestamp_is_rejected_before_warmup_requests(monkeypatch):
    from api import server

    monkeypatch.setattr(
        server, "market_candles", lambda *args: pytest.fail("network invoked")
    )
    with pytest.raises(ValueError):
        server.chart_data("live", "SOL", "1M", 0, 10, 3, start_time=-1, combined=True)


def test_daily_combined_reuses_warmup_source_without_a_second_range_download(
    monkeypatch,
):
    from api import server

    calls = []
    end = timestamp("2026-09-30")
    source = days(timestamp("2020-01-01"), (end - timestamp("2020-01-01")) // 86400)

    def fetch(*args):
        calls.append(args)
        return source

    server._candles_cache.clear()
    monkeypatch.setattr(server, "market_candles", fetch)
    response = server.chart_data(
        "live", "SOL", "1d", 90, 10, 3, end_time=end, combined=True
    )
    assert len(response["candles"]) == 90
    assert len(calls) == 1


def test_future_daily_bars_cannot_change_intraday_or_weekly_consensus():
    from api.timeframes import combined_supertrend

    daily = days(timestamp("2024-01-01"), 121)
    cutoff = timestamp("2024-03-14")
    display = [{"time": cutoff + 12 * 3600}]
    past = [c for c in daily if c["time"] < cutoff]
    future = [
        dict(c, close=0.001, low=0.0001, high=100000, open=1000)
        if c["time"] >= cutoff
        else c
        for c in daily
    ]
    expected = combined_supertrend(
        display, past, "1h", period=2, as_of=cutoff + 13 * 3600
    )
    actual = combined_supertrend(
        display, future, "1h", period=2, as_of=timestamp("2024-05-01")
    )
    assert actual == expected
    assert expected[0]["directions"]["1w"] is not None
    assert expected[0]["directions"]["1M"] is not None
