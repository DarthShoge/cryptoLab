from datetime import datetime, timedelta, timezone

import pytest

from arblab.hyperliquid_copy import proxy_sources


START = datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc)


def test_binance_header_ms_and_exclusive_close_normalization():
    ms = int(START.timestamp() * 1000)
    csv = f"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n{ms},100,110,90,105,10,{ms + 3599999},1000,2,5,500,0\n"
    # Crypto bar starts must be hourly UTC, so use an aligned source sample.
    csv = csv.replace(str(ms), str(ms - 1800000)).replace(
        str(ms + 3599999), str(ms + 1799999)
    )
    begin = START - timedelta(minutes=30)
    result = proxy_sources.binance_bars(
        csv, "BTC", start=begin, end=begin + timedelta(hours=1)
    )
    assert len(result.bars) == 1
    assert result.bars[0].end == begin + timedelta(hours=1)
    assert result.bars[0].close == 105


def yahoo(timestamps, closes):
    n = len(timestamps)
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"currency": "USD", "instrumentType": "EQUITY"},
                    "timestamp": [int(t.timestamp()) for t in timestamps],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100] * n,
                                "high": [110] * n,
                                "low": [90] * n,
                                "close": closes,
                            }
                        ]
                    },
                }
            ],
        }
    }


def test_yahoo_uses_explicit_session_windows_and_drops_appended_future_row():
    payload = yahoo([START, START + timedelta(days=20)], [105, 109])
    result = proxy_sources.yahoo_bars(
        payload,
        "xyz:TSLA",
        start=START,
        end=START + timedelta(hours=1),
        windows={START: START + timedelta(minutes=30)},
    )
    assert len(result.bars) == 1
    assert result.bars[0].end == START + timedelta(minutes=30)
    assert result.missing_starts == ()


def test_null_bar_is_missing_not_forward_filled():
    result = proxy_sources.yahoo_bars(
        yahoo([START], [None]),
        "xyz:TSLA",
        start=START,
        end=START + timedelta(hours=1),
        windows={START: START + timedelta(hours=1)},
    )
    assert result.bars == ()
    assert result.missing_starts == (START,)


def test_unfinished_bar_is_not_accepted():
    result = proxy_sources.yahoo_bars(
        yahoo([START], [105]),
        "xyz:TSLA",
        start=START,
        end=START + timedelta(minutes=15),
        windows={START: START + timedelta(hours=1)},
    )
    assert result.bars == ()
    assert result.missing_starts == ()


def test_unscheduled_bar_and_non_usd_currency_rejected():
    with pytest.raises(ValueError, match="session"):
        proxy_sources.yahoo_bars(
            yahoo([START], [105]),
            "xyz:TSLA",
            start=START,
            end=START + timedelta(hours=1),
            windows={},
        )
    payload = yahoo([START], [105])
    payload["chart"]["result"][0]["meta"]["currency"] = "GBP"
    with pytest.raises(ValueError, match="currency"):
        proxy_sources.yahoo_bars(
            payload,
            "xyz:TSLA",
            start=START,
            end=START + timedelta(hours=1),
            windows={START: START + timedelta(hours=1)},
        )


def test_expected_missing_hour_and_duplicate_conflict_are_visible():
    windows = {
        START: START + timedelta(hours=1),
        START + timedelta(hours=1): START + timedelta(hours=2),
    }
    result = proxy_sources.yahoo_bars(
        yahoo([START], [105]),
        "xyz:TSLA",
        start=START,
        end=START + timedelta(hours=2),
        windows=windows,
    )
    assert result.missing_starts == (START + timedelta(hours=1),)
    with pytest.raises(ValueError, match="conflict"):
        proxy_sources.yahoo_bars(
            yahoo([START, START], [105, 106]),
            "xyz:TSLA",
            start=START,
            end=START + timedelta(hours=2),
            windows=windows,
        )


@pytest.mark.parametrize("kind", ["splits", "dividends", "capitalGains"])
def test_yahoo_corporate_action_in_window_rejects_unmodeled_return(kind):
    payload = yahoo([START], [105])
    payload["chart"]["result"][0]["events"] = {
        kind: {"event": {"date": int(START.timestamp()), "amount": 1}}
    }
    with pytest.raises(ValueError, match="corporate action"):
        proxy_sources.yahoo_bars(
            payload,
            "xyz:TSLA",
            start=START,
            end=START + timedelta(hours=1),
            windows={START: START + timedelta(hours=1)},
        )


@pytest.mark.parametrize("events", [None, [], {"splits": {"bad": {}}}, {"unknown": {}}])
def test_yahoo_malformed_action_evidence_is_not_treated_as_no_actions(events):
    payload = yahoo([START], [105])
    payload["chart"]["result"][0]["events"] = events
    with pytest.raises(ValueError, match="corporate action"):
        proxy_sources.yahoo_bars(
            payload,
            "xyz:TSLA",
            start=START,
            end=START + timedelta(hours=1),
            windows={START: START + timedelta(hours=1)},
        )


def test_yahoo_action_at_exclusive_end_does_not_contaminate_period():
    payload = yahoo([START], [105])
    end = START + timedelta(hours=1)
    payload["chart"]["result"][0]["events"] = {
        "splits": {"later": {"date": int(end.timestamp())}}
    }
    assert (
        len(
            proxy_sources.yahoo_bars(
                payload, "xyz:TSLA", start=START, end=end, windows={START: end}
            ).bars
        )
        == 1
    )


@pytest.mark.parametrize("kind", [None, "FUTURE", "OPTION", "CURRENCY"])
def test_yahoo_unqualified_or_rolling_instrument_rejected(kind):
    payload = yahoo([START], [105])
    payload["chart"]["result"][0]["meta"]["instrumentType"] = kind
    with pytest.raises(ValueError, match="instrument type"):
        proxy_sources.yahoo_bars(
            payload,
            "xyz:TSLA",
            start=START,
            end=START + timedelta(hours=1),
            windows={START: START + timedelta(hours=1)},
        )
