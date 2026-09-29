from datetime import datetime, timezone

import pytest


def at(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def test_us_equity_session_has_half_hour_last_bar():
    from arblab.hyperliquid_copy.proxy_sessions import hourly_windows

    windows = hourly_windows("XNYS", at("2026-08-03"), at("2026-08-04"))
    assert len(windows) == 7
    assert min(windows) == at("2026-08-03T13:30")
    assert windows[at("2026-08-03T19:30")] == at("2026-08-03T20:00")


def test_holiday_weekend_and_early_close_are_not_missing_quotes():
    from arblab.hyperliquid_copy.proxy_sessions import hourly_windows

    assert hourly_windows("XNYS", at("2026-07-03"), at("2026-07-06")) == {}
    windows = hourly_windows("XNYS", at("2026-11-27"), at("2026-11-28"))
    assert len(windows) == 4
    assert windows[max(windows)] == at("2026-11-27T18:00")


def test_dst_changes_utc_open_but_not_local_session():
    from arblab.hyperliquid_copy.proxy_sessions import hourly_windows

    assert min(hourly_windows("XNYS", at("2026-03-06"), at("2026-03-07"))) == at(
        "2026-03-06T14:30"
    )
    assert min(hourly_windows("XNYS", at("2026-03-09"), at("2026-03-10"))) == at(
        "2026-03-09T13:30"
    )


def test_crypto_weekend_remains_open_and_mid_bar_range_is_not_reanchored():
    from arblab.hyperliquid_copy.proxy_sessions import hourly_windows

    windows = hourly_windows("24/7", at("2026-08-08T00:15"), at("2026-08-09"))
    assert len(windows) == 23
    assert min(windows) == at("2026-08-08T01:00")


@pytest.mark.parametrize("calendar", ["", "made_up", "CMES"])
def test_unqualified_calendars_rejected(calendar):
    from arblab.hyperliquid_copy.proxy_sessions import hourly_windows

    with pytest.raises(ValueError, match="calendar"):
        hourly_windows(calendar, at("2026-08-03"), at("2026-08-04"))


def test_invalid_or_unbounded_ranges_rejected():
    from arblab.hyperliquid_copy.proxy_sessions import hourly_windows

    with pytest.raises(ValueError):
        hourly_windows("24/7", at("2026-08-04"), at("2026-08-03"))
    with pytest.raises(ValueError):
        hourly_windows("24/7", at("2020-01-01"), at("2026-01-01"))
