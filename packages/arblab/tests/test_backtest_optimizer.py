"""Tests for backtest optimizer defaults."""

from __future__ import annotations

import pandas as pd

import inspect

from arblab.backtest.optimizer import grid_search
from arblab.backtest.strategy import BarData, Strategy
from arblab.kamino_risk import AccountSnapshot, CollateralPosition


class ConfigReturnStrategy(Strategy):
    def setup(self, snapshot: AccountSnapshot, config: dict) -> AccountSnapshot:
        return AccountSnapshot(
            collateral=[
                CollateralPosition("SOL", config["final_price"] / 100.0, 100.0, 0.75, 0.80)
            ],
            debt=[],
        )

    def on_bar(self, snapshot: AccountSnapshot, bar: BarData) -> list[dict]:
        return []


def test_grid_search_defaults_to_sortino_ratio():
    assert inspect.signature(grid_search).parameters["sort_metric"].default == "sortino_ratio"

    dates = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    price_data = pd.DataFrame(
        {
            ("SOL", "open"): [100.0, 100.0],
            ("SOL", "high"): [100.0, 100.0],
            ("SOL", "low"): [100.0, 100.0],
            ("SOL", "close"): [100.0, 100.0],
            ("SOL", "volume"): [1_000.0, 1_000.0],
        },
        index=dates,
    )
    price_data.columns = pd.MultiIndex.from_tuples(price_data.columns)

    result = grid_search(
        strategy=ConfigReturnStrategy(),
        price_data=price_data,
        param_grid={"final_price": [90.0, 110.0]},
    )

    assert result.comparison_df.columns.tolist().index("sortino_ratio") > -1
