"""Tests for research data loading and universe selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arblab.research.data import (
    base_market_name,
    load_hyperliquid_ohlcv,
    select_liquid_markets,
    summarize_ohlcv_coverage,
)


def write_market_csv(root: Path, interval: str, market_file: str, rows: list[dict]) -> None:
    path = root / interval / market_file
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_rows(market: str, closes: list[float], volumes: list[float] | None = None) -> list[dict]:
    volumes = volumes or [1000.0] * len(closes)
    timestamps = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
    return [
        {
            "timestamp": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
            "num_trades": 10,
            "coin": market,
            "interval": "1h",
            "source": "hyperliquid",
        }
        for ts, close, volume in zip(timestamps, closes, volumes)
    ]


def test_load_hyperliquid_ohlcv_uses_safe_market_filenames(tmp_path):
    write_market_csv(tmp_path, "1h", "BTC.csv", make_rows("BTC", [100, 101]))
    write_market_csv(tmp_path, "1h", "xyz_AAPL.csv", make_rows("xyz:AAPL", [200, 201]))

    df = load_hyperliquid_ohlcv(tmp_path, "1h", ["BTC", "xyz:AAPL"])

    assert set(df.columns.get_level_values(0)) == {"BTC", "xyz:AAPL"}
    assert df[("xyz:AAPL", "close")].tolist() == [200.0, 201.0]


def test_summarize_ohlcv_coverage_reports_rows_dates_and_dollar_volume(tmp_path):
    write_market_csv(
        tmp_path,
        "1h",
        "BTC.csv",
        make_rows("BTC", [100, 110, 120], volumes=[10, 20, 30]),
    )

    summary = summarize_ohlcv_coverage(tmp_path, "1h")

    assert summary.loc[0, "market"] == "BTC"
    assert summary.loc[0, "rows"] == 3
    assert summary.loc[0, "avg_dollar_volume"] == (1000 + 2200 + 3600) / 3


def test_select_liquid_markets_filters_history_and_delisted():
    summary = pd.DataFrame(
        [
            {"market": "BTC", "rows": 100, "avg_dollar_volume": 1000.0, "is_delisted": False},
            {"market": "ETH", "rows": 50, "avg_dollar_volume": 2000.0, "is_delisted": False},
            {"market": "DOGE", "rows": 100, "avg_dollar_volume": 3000.0, "is_delisted": True},
            {"market": "SOL", "rows": 100, "avg_dollar_volume": 1500.0, "is_delisted": False},
        ]
    )

    selected = select_liquid_markets(summary, min_rows=80, top_n=2)

    assert selected == ["SOL", "BTC"]


def test_base_market_name_strips_builder_namespace():
    assert base_market_name("BTC") == "BTC"
    assert base_market_name("xyz:AAPL") == "AAPL"


def test_select_liquid_markets_can_dedupe_by_underlying():
    summary = pd.DataFrame(
        [
            {"market": "xyz:COIN", "rows": 100, "avg_dollar_volume": 1000.0},
            {"market": "flx:COIN", "rows": 100, "avg_dollar_volume": 2000.0},
            {"market": "xyz:AAPL", "rows": 100, "avg_dollar_volume": 1500.0},
        ]
    )

    selected = select_liquid_markets(
        summary,
        min_rows=80,
        top_n=3,
        dedupe_underlying=True,
    )

    assert selected == ["flx:COIN", "xyz:AAPL"]
