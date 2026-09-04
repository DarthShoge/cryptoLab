"""Tests for arblab.backtest.data — OHLCV fetching, caching, alignment."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from arblab.backtest.data import OHLCVConfig, fetch_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_candles(
    n: int = 100,
    start_ts: int = 1704067200000,  # 2024-01-01 00:00 UTC
    interval_ms: int = 3_600_000,
    base_price: float = 150.0,
) -> list:
    return [
        [
            start_ts + i * interval_ms,
            base_price + i * 0.1,          # open
            base_price + i * 0.1 + 1,      # high
            base_price + i * 0.1 - 1,      # low
            base_price + i * 0.1,          # close
            1000,                           # volume
        ]
        for i in range(n)
    ]


def _patch_ccxt(mock_exchange: MagicMock):
    """Return a mock ccxt module whose ``binance()`` returns *mock_exchange*."""
    mock_ccxt = MagicMock()
    mock_ccxt.binance.return_value = mock_exchange
    return mock_ccxt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOHLCVConfig:
    def test_construction(self):
        cfg = OHLCVConfig(symbol="SOL/USDT", display_name="SOL")
        assert cfg.symbol == "SOL/USDT"
        assert cfg.display_name == "SOL"

    def test_frozen(self):
        cfg = OHLCVConfig(symbol="SOL/USDT", display_name="SOL")
        with pytest.raises(AttributeError):
            cfg.symbol = "ETH/USDT"


class TestFetchOhlcvBasic:
    """fetch_ohlcv with a single symbol."""

    def test_returns_multiindex_columns(self, tmp_path):
        candles = make_mock_candles(10)
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = candles

        with patch("arblab.backtest.data.ccxt", new=_patch_ccxt(mock_ex), create=True):
            # The function does `import ccxt as _ccxt` locally, so we patch
            # the module-level import that the function will resolve.
            import arblab.backtest.data as data_mod
            with patch.dict("sys.modules", {"ccxt": _patch_ccxt(mock_ex)}):
                df = fetch_ohlcv(
                    symbols=[OHLCVConfig("SOL/USDT", "SOL")],
                    timeframe="1h",
                    start="2024-01-01",
                    cache_dir=tmp_path / "cache",
                    use_cache=False,
                    rate_limit_sleep=0,
                )

        assert isinstance(df.columns, pd.MultiIndex)
        # Top level should be "SOL"
        assert "SOL" in df.columns.get_level_values(0)
        # Second level should contain OHLCV fields
        fields = set(df.columns.get_level_values(1))
        assert {"open", "high", "low", "close", "volume"}.issubset(fields)
        assert len(df) == 10

    def test_close_values_match_candles(self, tmp_path):
        candles = make_mock_candles(5)
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = candles

        with patch.dict("sys.modules", {"ccxt": _patch_ccxt(mock_ex)}):
            df = fetch_ohlcv(
                symbols=[OHLCVConfig("SOL/USDT", "SOL")],
                timeframe="1h",
                start="2024-01-01",
                cache_dir=tmp_path / "cache",
                use_cache=False,
                rate_limit_sleep=0,
            )

        expected_closes = [c[4] for c in candles]
        actual_closes = df[("SOL", "close")].tolist()
        assert actual_closes == pytest.approx(expected_closes)


class TestMultipleSymbols:
    """Fetching two symbols produces aligned outer-join + ffill output."""

    def test_multi_symbol_alignment(self, tmp_path):
        # SOL candles: 10 bars starting at ts 1704067200000
        sol_candles = make_mock_candles(10, base_price=150.0)
        # ETH candles: 8 bars starting at ts 1704067200000 (shorter)
        eth_candles = make_mock_candles(8, base_price=3000.0)

        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.side_effect = [sol_candles, eth_candles]

        with patch.dict("sys.modules", {"ccxt": _patch_ccxt(mock_ex)}):
            df = fetch_ohlcv(
                symbols=[
                    OHLCVConfig("SOL/USDT", "SOL"),
                    OHLCVConfig("ETH/USDT", "ETH"),
                ],
                timeframe="1h",
                start="2024-01-01",
                cache_dir=tmp_path / "cache",
                use_cache=False,
                rate_limit_sleep=0,
            )

        assert "SOL" in df.columns.get_level_values(0)
        assert "ETH" in df.columns.get_level_values(0)
        # After outer join + ffill + dropna, the frame should have no NaNs
        assert not df.isna().any().any()


class TestCSVCaching:
    """CSV round-trip: write on first call, read from cache on second."""

    def test_write_then_read(self, tmp_path):
        candles = make_mock_candles(5)
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = candles

        cache_dir = tmp_path / "cache"
        symbols = [OHLCVConfig("SOL/USDT", "SOL")]

        # First call: fetches and caches
        with patch.dict("sys.modules", {"ccxt": _patch_ccxt(mock_ex)}):
            df1 = fetch_ohlcv(
                symbols=symbols,
                timeframe="1h",
                start="2024-01-01",
                cache_dir=cache_dir,
                use_cache=True,
                rate_limit_sleep=0,
            )

        assert mock_ex.fetch_ohlcv.call_count == 1
        # CSV file should exist
        csv_files = list(cache_dir.glob("*.csv"))
        assert len(csv_files) == 1

        # Second call: reads from cache, no new fetch
        mock_ex2 = MagicMock()
        mock_ex2.fetch_ohlcv.return_value = []  # should NOT be called

        with patch.dict("sys.modules", {"ccxt": _patch_ccxt(mock_ex2)}):
            df2 = fetch_ohlcv(
                symbols=symbols,
                timeframe="1h",
                start="2024-01-01",
                cache_dir=cache_dir,
                use_cache=True,
                rate_limit_sleep=0,
            )

        mock_ex2.fetch_ohlcv.assert_not_called()
        # Data should match
        pd.testing.assert_frame_equal(df1, df2)


class TestDateRangeFiltering:
    """The ``start`` and ``end`` parameters clip the returned data."""

    def test_end_date_clips_rows(self, tmp_path):
        # 100 candles ~ 100 hours from 2024-01-01 00:00 UTC
        candles = make_mock_candles(100)
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = candles

        with patch.dict("sys.modules", {"ccxt": _patch_ccxt(mock_ex)}):
            df = fetch_ohlcv(
                symbols=[OHLCVConfig("SOL/USDT", "SOL")],
                timeframe="1h",
                start="2024-01-01",
                end="2024-01-02",  # only first 24 hours
                cache_dir=tmp_path / "cache",
                use_cache=False,
                rate_limit_sleep=0,
            )

        # Should have at most 25 rows (hour 0..24 inclusive)
        assert len(df) <= 25
        # All timestamps should be <= 2024-01-02T00:00 UTC
        assert df.index.max() <= pd.Timestamp("2024-01-02", tz="UTC")


class TestPagination:
    """When the exchange returns 1000 candles, the fetcher paginates."""

    def test_two_pages(self, tmp_path):
        page1 = make_mock_candles(1000, start_ts=1704067200000)
        last_ts_page1 = page1[-1][0]
        page2 = make_mock_candles(
            200,
            start_ts=last_ts_page1 + 3_600_000,
            base_price=250.0,
        )

        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.side_effect = [page1, page2]

        with patch.dict("sys.modules", {"ccxt": _patch_ccxt(mock_ex)}):
            df = fetch_ohlcv(
                symbols=[OHLCVConfig("SOL/USDT", "SOL")],
                timeframe="1h",
                start="2024-01-01",
                cache_dir=tmp_path / "cache",
                use_cache=False,
                rate_limit_sleep=0,
            )

        # Two calls to fetch_ohlcv on the exchange object
        assert mock_ex.fetch_ohlcv.call_count == 2
        # Should contain all 1200 unique bars
        assert len(df) == 1200


class TestEmptyResult:
    """When the exchange returns no candles, the result is an empty frame."""

    def test_empty_candles(self, tmp_path):
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = []

        with patch.dict("sys.modules", {"ccxt": _patch_ccxt(mock_ex)}):
            df = fetch_ohlcv(
                symbols=[OHLCVConfig("SOL/USDT", "SOL")],
                timeframe="1h",
                start="2024-01-01",
                cache_dir=tmp_path / "cache",
                use_cache=False,
                rate_limit_sleep=0,
            )

        assert len(df) == 0
