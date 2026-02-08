"""Grid search parameter optimizer for backtest strategies."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from arblab.backtest.engine import BacktestEngine, EngineConfig
from arblab.backtest.market import MarketParams
from arblab.backtest.results import BacktestResult
from arblab.backtest.strategy import Strategy


@dataclass(frozen=True)
class OptimizationResult:
    results: List[BacktestResult]
    param_grid: List[Dict[str, Any]]
    best_result: BacktestResult
    comparison_df: pd.DataFrame


def grid_search(
    strategy: Strategy,
    price_data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    base_config: Dict[str, Any] | None = None,
    engine_config: EngineConfig | None = None,
    market_params: MarketParams | None = None,
    sort_metric: str = "sharpe_ratio",
    sort_ascending: bool = False,
) -> OptimizationResult:
    """Run a grid search over strategy parameter combinations.

    *param_grid* maps parameter names to lists of values.  Every combination
    in the Cartesian product is tested.
    """
    base_config = base_config or {}
    engine_config = engine_config or EngineConfig()
    market_params = market_params or MarketParams.kamino_defaults()

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    results: List[BacktestResult] = []
    grid: List[Dict[str, Any]] = []

    for combo in combos:
        params = dict(zip(keys, combo))
        config = {**base_config, **params}
        grid.append(params)

        engine = BacktestEngine(strategy, engine_config)
        result = engine.run(price_data, config, market_params)
        results.append(result)

    # Build comparison table
    rows = []
    for params, result in zip(grid, results):
        m = result.metrics
        row = {**params}
        row["total_return_pct"] = m.total_return_pct
        row["max_drawdown_pct"] = m.max_drawdown_pct
        row["sharpe_ratio"] = m.sharpe_ratio
        row["sortino_ratio"] = m.sortino_ratio
        row["min_health_factor"] = m.min_health_factor
        row["total_liquidations"] = m.total_liquidations
        row["total_interest_paid"] = m.total_interest_paid
        row["total_lst_yield_earned"] = m.total_lst_yield_earned
        row["liquidated"] = result.liquidated
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(
        sort_metric, ascending=sort_ascending
    ).reset_index(drop=True)

    best_idx = comparison_df.index[0]
    best_result = results[grid.index(dict(zip(keys, combos[best_idx])))]

    return OptimizationResult(
        results=results,
        param_grid=grid,
        best_result=best_result,
        comparison_df=comparison_df,
    )
