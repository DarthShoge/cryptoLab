"""Tests for lightweight candidate strategy research backtests."""

from __future__ import annotations

import pandas as pd
import pytest

from arblab.research.candidates import (
    backtest_weight_strategy,
    cross_sectional_momentum_weights,
    moving_average_trend_weights,
)


def make_price_frame(closes: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(next(iter(closes.values()))), freq="h", tz="UTC")
    data = {}
    for symbol, values in closes.items():
        series = pd.Series(values, index=index, dtype=float)
        data[(symbol, "open")] = series
        data[(symbol, "high")] = series
        data[(symbol, "low")] = series
        data[(symbol, "close")] = series
        data[(symbol, "volume")] = 1000.0
    df = pd.DataFrame(data, index=index)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_backtest_weight_strategy_shifts_weights_and_charges_turnover_costs():
    prices = make_price_frame({"BTC": [100, 110, 121, 108.9]})
    weights = pd.DataFrame({"BTC": [0.0, 1.0, 1.0, 0.0]}, index=prices.index)

    result = backtest_weight_strategy(prices, weights, fee_bps=10.0, periods_per_year=8760)

    assert result.history["gross_return"].tolist() == pytest.approx([0.0, 0.0, 0.1, -0.1])
    assert result.history["turnover"].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert result.history["net_return"].tolist() == pytest.approx([0.0, 0.0, 0.099, -0.1])
    assert result.metrics.total_return_pct == pytest.approx(-1.09)
    assert result.metrics.total_trades == 1


def test_moving_average_trend_weights_go_long_only_when_fast_above_slow():
    prices = make_price_frame({"BTC": [100, 99, 98, 101, 104, 107]})

    weights = moving_average_trend_weights(prices, symbol="BTC", fast=2, slow=3)

    assert weights.columns.tolist() == ["BTC"]
    assert weights["BTC"].tolist() == pytest.approx([0, 0, 0, 1, 1, 1])


def test_cross_sectional_momentum_weights_select_top_assets_after_lookback():
    prices = make_price_frame(
        {
            "BTC": [100, 102, 104, 106, 108],
            "ETH": [100, 99, 98, 97, 96],
            "SOL": [100, 105, 110, 115, 120],
        }
    )

    weights = cross_sectional_momentum_weights(prices, lookback=2, top_n=1)

    assert weights.loc[prices.index[0]].sum() == 0.0
    assert weights.loc[prices.index[1]].sum() == 0.0
    assert weights.loc[prices.index[2], "SOL"] == 1.0
    assert weights.loc[prices.index[2], ["BTC", "ETH"]].sum() == 0.0


def test_cross_sectional_momentum_weights_hold_between_rebalances():
    prices = make_price_frame(
        {
            "BTC": [100, 120, 121, 122, 90],
            "ETH": [100, 101, 140, 160, 142],
        }
    )

    weights = cross_sectional_momentum_weights(
        prices,
        lookback=1,
        top_n=1,
        rebalance_every=2,
    )

    assert weights.loc[prices.index[1], "BTC"] == 1.0
    assert weights.loc[prices.index[2], "BTC"] == 1.0
    assert weights.loc[prices.index[3], "ETH"] == 1.0
