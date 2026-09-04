"""Tests for reusable multi-asset traffic-light signals."""

from __future__ import annotations

import pandas as pd

from arblab.backtest.traffic_lights import (
    asset_green_counts_by_bar,
    latest_asset_rankings,
)


def _ohlcv(symbol: str, values: list[float]) -> dict[tuple[str, str], list[float]]:
    series = pd.Series(values, dtype=float)
    return {
        (symbol, "open"): series.tolist(),
        (symbol, "high"): (series * 1.01).tolist(),
        (symbol, "low"): (series * 0.99).tolist(),
        (symbol, "close"): series.tolist(),
        (symbol, "volume"): [1_000.0] * len(values),
    }


def _price_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
    uptrend = [100.0 + i for i in range(len(dates))]
    downtrend = [200.0 - i for i in range(len(dates))]
    flat = [50.0] * len(dates)
    data = {}
    data.update(_ohlcv("SOL", uptrend))
    data.update(_ohlcv("ETH", downtrend))
    data.update(_ohlcv("mSOL", flat))
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_asset_green_counts_by_bar_scores_each_symbol_independently():
    scores = asset_green_counts_by_bar(
        _price_data(),
        symbols=["SOL", "ETH"],
        atr_period=3,
        multiplier=1.5,
        timeframes=("1h", "4h"),
    )

    final = scores.iloc[-1]

    assert list(scores.columns) == ["SOL", "ETH"]
    assert int(final["SOL"]) > int(final["ETH"])


def test_latest_asset_rankings_returns_strongest_and_weakest_assets():
    scores = asset_green_counts_by_bar(
        _price_data(),
        symbols=["SOL", "ETH", "mSOL"],
        atr_period=3,
        multiplier=1.5,
        timeframes=("1h", "4h"),
    )

    ranking = latest_asset_rankings(scores)

    assert ranking.strongest == "SOL"
    assert ranking.weakest == "ETH"
    assert ranking.green_counts["SOL"] >= ranking.green_counts["mSOL"]
    assert ranking.green_counts["ETH"] <= ranking.green_counts["mSOL"]
