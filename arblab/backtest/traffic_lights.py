"""Reusable multi-asset traffic-light signal helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from arblab.strategies.sol_supertrend_short import supertrend_direction


@dataclass(frozen=True)
class AssetRanking:
    """Latest strongest/weakest asset view from green-count scores."""

    strongest: str
    weakest: str
    green_counts: dict[str, int]


def _pandas_timeframe(timeframe: str) -> str:
    if timeframe.lower().endswith("w"):
        return timeframe[:-1] + "W"
    return timeframe


def _resample_ohlcv(price_data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    base = price_data[symbol]
    rule = _pandas_timeframe(timeframe)
    return pd.DataFrame(
        {
            "open": base["open"].resample(rule).first(),
            "high": base["high"].resample(rule).max(),
            "low": base["low"].resample(rule).min(),
            "close": base["close"].resample(rule).last(),
            "volume": base["volume"].resample(rule).sum(),
        }
    ).dropna()


def _closed_direction_on_base_index(
    price_data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    atr_period: int,
    multiplier: float,
) -> pd.Series:
    resampled = _resample_ohlcv(price_data, symbol, timeframe)
    if resampled.empty:
        return pd.Series(True, index=price_data.index)
    direction = supertrend_direction(resampled, atr_period, multiplier)
    closed_direction = direction.shift(1)
    mapped = closed_direction.reindex(price_data.index, method="ffill")
    return mapped.where(mapped.notna(), True).astype(bool)


def asset_green_counts_by_bar(
    price_data: pd.DataFrame,
    symbols: list[str],
    atr_period: int,
    multiplier: float,
    timeframes: tuple[str, ...] = ("1h", "4h", "8h", "1d"),
) -> pd.DataFrame:
    """Return per-bar green-light counts for each symbol."""
    scores: dict[str, pd.Series] = {}
    for symbol in symbols:
        directions = [
            _closed_direction_on_base_index(
                price_data,
                symbol,
                timeframe,
                atr_period,
                multiplier,
            )
            for timeframe in timeframes
        ]
        score = sum(direction.astype(int) for direction in directions)
        scores[symbol] = score.astype(int)
    return pd.DataFrame(scores, index=price_data.index)


def latest_asset_rankings(scores: pd.DataFrame) -> AssetRanking:
    """Rank the latest score row, breaking ties by column order."""
    if scores.empty:
        raise ValueError("Cannot rank assets from empty scores")

    latest = scores.iloc[-1].astype(int)
    symbols = list(scores.columns)
    strongest = max(symbols, key=lambda symbol: (int(latest[symbol]), -symbols.index(symbol)))
    weakest = min(symbols, key=lambda symbol: (int(latest[symbol]), symbols.index(symbol)))
    return AssetRanking(
        strongest=strongest,
        weakest=weakest,
        green_counts={symbol: int(latest[symbol]) for symbol in symbols},
    )
