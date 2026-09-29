from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from arblab.hyperliquid_copy import proxy_bars


def moment(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def bar(start="2026-08-03T13:30", **changes):
    at = moment(start) if isinstance(start, str) else start
    values = dict(
        instrument_id="xyz:TSLA",
        start=at,
        end=at + timedelta(hours=1),
        open=100,
        high=110,
        low=90,
        close=105,
    )
    return proxy_bars.ProxyBar(**(values | changes))


def test_open_strictly_after_due_preserves_half_hour_session():
    first, second = bar(), bar("2026-08-03T14:30")
    prices = proxy_bars.ProxyBars([second, first])
    assert (
        prices.next_open("xyz:TSLA", moment("2026-08-03T13:00"), max_wait_seconds=1800)
        == first
    )
    assert prices.next_open("xyz:TSLA", first.start, max_wait_seconds=3600) == second
    assert prices.next_open("xyz:TSLA", first.start, max_wait_seconds=3599) is None


def test_mark_cannot_see_unfinished_bar_close():
    first, second = bar(), bar("2026-08-03T14:30", close=109)
    prices = proxy_bars.ProxyBars([first, second])
    with pytest.raises(ValueError, match="completed"):
        prices.completed_mark("xyz:TSLA", first.start, max_age_seconds=3600)
    marked = prices.completed_mark("xyz:TSLA", first.end, max_age_seconds=3600)
    assert (marked.price, marked.time, marked.age_seconds) == (105, first.end, 0)
    assert (
        prices.completed_mark(
            "xyz:TSLA", second.start + timedelta(minutes=30), max_age_seconds=3600
        ).price
        == 105
    )


def test_weekend_has_no_synthetic_execution_and_mark_age_is_visible():
    friday = bar("2026-08-07T19:30", end=moment("2026-08-07T20:00"))
    monday = bar("2026-08-10T13:30")
    prices = proxy_bars.ProxyBars([friday, monday])
    due = moment("2026-08-08T10:00")
    assert prices.next_open("xyz:TSLA", due, max_wait_seconds=3600) is None
    assert prices.next_open("xyz:TSLA", due, max_wait_seconds=3 * 86400) == monday
    assert (
        prices.completed_mark("xyz:TSLA", due, max_age_seconds=86400).age_seconds
        == 14 * 3600
    )
    with pytest.raises(ValueError, match="stale"):
        prices.completed_mark("xyz:TSLA", due, max_age_seconds=3600)


@pytest.mark.parametrize(
    "changes",
    [
        {"open": 0},
        {"close": float("nan")},
        {"high": float("inf")},
        {"low": True},
        {"close": 111},
        {"open": 89},
        {"high": 80},
        {"start": datetime(2026, 8, 3, 13, 30)},
        {"end": moment("2026-08-03T13:30")},
        {"end": moment("2026-08-03T15:00")},
        {"instrument_id": ""},
    ],
)
def test_rejects_invalid_bar(changes):
    with pytest.raises(ValueError):
        bar(**changes)


def test_duplicate_identical_allowed_but_conflicts_and_overlaps_rejected():
    first = bar()
    assert (
        proxy_bars.ProxyBars([first, first]).next_open(
            "xyz:TSLA", first.start - timedelta(seconds=1), max_wait_seconds=1
        )
        == first
    )
    with pytest.raises(ValueError, match="conflict"):
        proxy_bars.ProxyBars([first, replace(first, close=104)])
    with pytest.raises(ValueError, match="overlap"):
        proxy_bars.ProxyBars([first, bar("2026-08-03T14:00")])


@pytest.mark.parametrize("bound", [-1, 0, True, float("nan"), float("inf")])
def test_lookup_bounds_must_be_positive_finite(bound):
    prices = proxy_bars.ProxyBars([bar()])
    with pytest.raises(ValueError):
        prices.next_open("xyz:TSLA", moment("2026-08-03T13:00"), max_wait_seconds=bound)
    with pytest.raises(ValueError):
        prices.completed_mark(
            "xyz:TSLA", moment("2026-08-03T15:00"), max_age_seconds=bound
        )


def test_unknown_instrument_has_no_execution_or_mark():
    prices = proxy_bars.ProxyBars([bar()])
    assert (
        prices.next_open("BTC", moment("2026-08-03T13:00"), max_wait_seconds=3600)
        is None
    )
    with pytest.raises(ValueError, match="completed"):
        prices.completed_mark("BTC", moment("2026-08-03T15:00"), max_age_seconds=3600)
