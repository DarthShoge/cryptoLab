"""Tests for Hyperliquid read-only market data ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arblab.backtest.hyperliquid_data import (
    HyperliquidDataClient,
    interval_to_milliseconds,
    safe_market_filename,
)


class StubInfoApi:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, payload):
        self.requests.append(payload)
        if not self.responses:
            raise AssertionError("unexpected Hyperliquid request")
        return self.responses.pop(0)


def make_candle(ts: int, close: str = "101.5") -> dict:
    return {
        "t": ts,
        "T": ts + 3_599_999,
        "s": "BTC",
        "i": "1h",
        "o": "100.0",
        "h": "102.0",
        "l": "99.0",
        "c": close,
        "v": "1234.5",
        "n": 42,
    }


def test_interval_to_milliseconds_supports_hyperliquid_intervals():
    assert interval_to_milliseconds("1m") == 60_000
    assert interval_to_milliseconds("15m") == 900_000
    assert interval_to_milliseconds("1h") == 3_600_000
    assert interval_to_milliseconds("1d") == 86_400_000

    with pytest.raises(ValueError, match="Unsupported"):
        interval_to_milliseconds("7m")


def test_safe_market_filename_handles_hip3_namespaces():
    assert safe_market_filename("BTC") == "BTC"
    assert safe_market_filename("xyz:AAPL") == "xyz_AAPL"
    assert safe_market_filename("PURR/USDC") == "PURR_USDC"


def test_fetch_perp_universe_normalizes_meta_response():
    stub = StubInfoApi(
        [
            {
                "universe": [
                    {"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
                    {"name": "DOGE", "szDecimals": 0, "maxLeverage": 10, "isDelisted": True},
                ]
            }
        ]
    )
    client = HyperliquidDataClient(post_info=stub)

    universe = client.fetch_perp_universe()

    assert stub.requests == [{"type": "meta"}]
    assert universe == [
        {
            "coin": "BTC",
            "sz_decimals": 5,
            "max_leverage": 50,
            "is_delisted": False,
        },
        {
            "coin": "DOGE",
            "sz_decimals": 0,
            "max_leverage": 10,
            "is_delisted": True,
        },
    ]


def test_fetch_all_mids_casts_prices_to_float():
    stub = StubInfoApi([{"BTC": "68000.5", "ETH": "3500"}])
    client = HyperliquidDataClient(post_info=stub)

    mids = client.fetch_all_mids()

    assert stub.requests == [{"type": "allMids"}]
    assert mids == {"BTC": 68000.5, "ETH": 3500.0}


def test_fetch_perp_categories_normalizes_category_pairs():
    stub = StubInfoApi(
        [
            [
                ["xyz:AAPL", "stocks"],
                ["BTC", "crypto"],
                ["xyz:US500", "indices"],
            ]
        ]
    )
    client = HyperliquidDataClient(post_info=stub)

    categories = client.fetch_perp_categories()

    assert stub.requests == [{"type": "perpCategories"}]
    assert categories == {
        "xyz:AAPL": "stocks",
        "BTC": "crypto",
        "xyz:US500": "indices",
    }


def test_fetch_assets_by_category_returns_matching_coins_sorted():
    stub = StubInfoApi(
        [
            [
                ["xyz:TSLA", "stocks"],
                ["BTC", "crypto"],
                ["xyz:AAPL", "stocks"],
            ]
        ]
    )
    client = HyperliquidDataClient(post_info=stub)

    assert client.fetch_assets_by_category("stocks") == ["xyz:AAPL", "xyz:TSLA"]


def test_fetch_all_perp_metas_tags_dex_names():
    stub = StubInfoApi(
        [
            [
                {"universe": [{"name": "BTC", "maxLeverage": 40}]},
                {"dex": "xyz", "universe": [{"name": "AAPL", "maxLeverage": 5}]},
            ]
        ]
    )
    client = HyperliquidDataClient(post_info=stub)

    metas = client.fetch_all_perp_metas()

    assert stub.requests == [{"type": "allPerpMetas"}]
    assert metas == [
        {"dex": "", "coin": "BTC", "raw_name": "BTC", "max_leverage": 40},
        {"dex": "xyz", "coin": "xyz:AAPL", "raw_name": "AAPL", "max_leverage": 5},
    ]


def test_fetch_candles_sends_expected_payload_and_normalizes_rows():
    start = 1_704_067_200_000
    candles = [
        make_candle(start + 3_600_000, close="102.0"),
        make_candle(start, close="101.0"),
        make_candle(start, close="101.5"),
    ]
    stub = StubInfoApi([candles])
    client = HyperliquidDataClient(post_info=stub)

    df = client.fetch_candles("BTC", "1h", start, start + 3_600_000)

    assert stub.requests == [
        {
            "type": "candleSnapshot",
            "req": {
                "coin": "BTC",
                "interval": "1h",
                "startTime": start,
                "endTime": start + 3_600_000,
            },
        }
    ]
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "num_trades",
        "coin",
        "interval",
        "source",
    ]
    assert df["timestamp"].tolist() == [
        pd.Timestamp(start, unit="ms", tz="UTC"),
        pd.Timestamp(start + 3_600_000, unit="ms", tz="UTC"),
    ]
    assert df["close"].tolist() == [101.5, 102.0]
    assert df["coin"].tolist() == ["BTC", "BTC"]
    assert df["interval"].tolist() == ["1h", "1h"]
    assert df["source"].tolist() == ["hyperliquid", "hyperliquid"]


def test_fetch_candles_returns_empty_canonical_frame_for_empty_response():
    stub = StubInfoApi([[]])
    client = HyperliquidDataClient(post_info=stub)

    df = client.fetch_candles("BTC", "1h", 1_704_067_200_000, 1_704_070_800_000)

    assert df.empty
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "num_trades",
        "coin",
        "interval",
        "source",
    ]


def test_backfill_candles_paginates_and_writes_csv(tmp_path: Path):
    start = 1_704_067_200_000
    end = start + 2 * 3_600_000
    stub = StubInfoApi(
        [
            [make_candle(start), make_candle(start + 3_600_000)],
            [make_candle(start + 2 * 3_600_000)],
        ]
    )
    client = HyperliquidDataClient(post_info=stub)

    df = client.backfill_candles("BTC", "1h", start, end, tmp_path)

    assert len(stub.requests) == 2
    assert stub.requests[0]["req"]["startTime"] == start
    assert stub.requests[1]["req"]["startTime"] == start + 2 * 3_600_000
    assert len(df) == 3

    out_file = tmp_path / "1h" / "BTC.csv"
    assert out_file.exists()
    saved = pd.read_csv(out_file)
    assert len(saved) == 3
    assert saved["coin"].unique().tolist() == ["BTC"]


def test_backfill_candles_chunks_oversized_ranges(tmp_path: Path):
    start = 1_704_067_200_000
    end = start + 4 * 3_600_000
    stub = StubInfoApi(
        [
            [],
            [make_candle(start + 2 * 3_600_000), make_candle(start + 3 * 3_600_000)],
            [make_candle(start + 4 * 3_600_000)],
        ]
    )
    client = HyperliquidDataClient(post_info=stub, max_candles_per_request=2)

    df = client.backfill_candles("BTC", "1h", start, end, tmp_path)

    request_windows = [
        (req["req"]["startTime"], req["req"]["endTime"])
        for req in stub.requests
    ]
    assert request_windows == [
        (start, start + 3_600_000),
        (start + 2 * 3_600_000, start + 3 * 3_600_000),
        (start + 4 * 3_600_000, start + 4 * 3_600_000),
    ]
    assert len(df) == 3
    assert df["timestamp"].min() == pd.Timestamp(start + 2 * 3_600_000, unit="ms", tz="UTC")


def test_backfill_candles_sanitizes_hip3_market_filename(tmp_path: Path):
    start = 1_704_067_200_000
    stub = StubInfoApi([[make_candle(start)]])
    client = HyperliquidDataClient(post_info=stub)

    client.backfill_candles("xyz:AAPL", "1h", start, start, tmp_path)

    assert (tmp_path / "1h" / "xyz_AAPL.csv").exists()
