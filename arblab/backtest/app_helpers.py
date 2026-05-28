"""Helper functions shared by the Streamlit backtester app and tests."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from arblab.backtest.data import OHLCVConfig
from arblab.backtest.engine import BacktestEngine
from arblab.backtest.market import MarketParams
from arblab.backtest.optimizer import OptimizationResult, grid_search
from arblab.backtest.results import BacktestResult
from arblab.backtest.strategy import Strategy
from arblab.strategies.leverage_loop import LeverageLoopStrategy
from arblab.strategies.sol_supertrend_short import SolSupertrendShortStrategy


LEVERAGE_LOOP_STRATEGY = "Leverage Loop"
SOL_SUPERTREND_SHORT_STRATEGY = "SOL Supertrend Short"
DEFAULT_STRATEGY = SOL_SUPERTREND_SHORT_STRATEGY

EXCHANGE_SYMBOLS = {
    "SOL": "SOL/USDT",
    "JitoSOL": "JITOSOL/USDT",
    "mSOL": "MSOL/USDT",
    "ETH": "ETH/USDT",
}


def visible_strategy_controls(strategy_name: str) -> List[str]:
    """Return the strategy-specific sidebar controls for a strategy."""
    if strategy_name == SOL_SUPERTREND_SHORT_STRATEGY:
        return [
            "Initial SOL Collateral",
            "Supertrend ATR Period",
            "Supertrend Multiplier",
            "Target Bullish HF",
            "Minimum Rebalance HF",
            "Max USDC Debt / Equity",
            "Rebalance Threshold",
            "Cooldown Bars",
            "Full Short Lower Bound",
            "Full Short Upper Bound",
        ]
    return [
        "Initial Collateral",
        "Collateral Asset",
        "Debt Asset",
        "Leverage Loops",
        "Loop Utilization",
        "Target HF",
        "Delever Trigger",
        "Re-lever Trigger",
    ]


def build_price_configs(
    strategy_name: str,
    collateral_symbol: str,
) -> List[OHLCVConfig]:
    """Return exchange symbols required by the selected strategy."""
    if strategy_name == SOL_SUPERTREND_SHORT_STRATEGY:
        return [
            OHLCVConfig(symbol=EXCHANGE_SYMBOLS["SOL"], display_name="SOL"),
            OHLCVConfig(symbol=EXCHANGE_SYMBOLS["ETH"], display_name="ETH"),
        ]
    return [
        OHLCVConfig(
            symbol=EXCHANGE_SYMBOLS.get(
                collateral_symbol, f"{collateral_symbol}/USDT"
            ),
            display_name=collateral_symbol,
        )
    ]


def strategy_for_name(strategy_name: str) -> Strategy:
    """Instantiate the selected backtest strategy."""
    if strategy_name == SOL_SUPERTREND_SHORT_STRATEGY:
        return SolSupertrendShortStrategy()
    return LeverageLoopStrategy()


def position_value_chart_data(history: pd.DataFrame) -> pd.DataFrame:
    """Build mark-to-market position chart data from backtest history.

    Collateral values are positive, debt values are negative, and net
    portfolio value is included so the components reconcile visually.
    """
    data: dict[str, pd.Series] = {}
    for col in history.columns:
        if not col.endswith("_value"):
            continue
        if col in {"collateral_value", "debt_value", "portfolio_value"}:
            continue
        if col.startswith("collateral_"):
            sym = col.removeprefix("collateral_").removesuffix("_value")
            data[f"Collateral {sym}"] = history[col]
        elif col.startswith("debt_"):
            sym = col.removeprefix("debt_").removesuffix("_value")
            data[f"Debt {sym}"] = -history[col]

    if "portfolio_value" in history:
        data["Net Portfolio"] = history["portfolio_value"]
    return pd.DataFrame(data, index=history.index)


def build_sol_supertrend_short_config(
    price_data: pd.DataFrame,
    initial_sol_collateral: float,
    supertrend_atr_period: int,
    supertrend_multiplier: float,
    target_bullish_hf: float,
    min_rebalance_hf: float,
    max_usdc_debt_to_equity: float,
    rebalance_threshold: float,
    rebalance_cooldown_bars: int,
    swap_fee_bps: float,
    full_short_lower_bound: float,
    full_short_upper_bound: float,
) -> Dict[str, float | int]:
    """Build strategy config for the SOL Supertrend short strategy."""
    return {
        "initial_sol_collateral": initial_sol_collateral,
        "initial_sol_price": float(price_data.iloc[0][("SOL", "close")]),
        "initial_eth_price": float(price_data.iloc[0][("ETH", "close")]),
        "supertrend_atr_period": supertrend_atr_period,
        "supertrend_multiplier": supertrend_multiplier,
        "target_bullish_hf": target_bullish_hf,
        "min_rebalance_hf": min_rebalance_hf,
        "max_usdc_debt_to_equity": max_usdc_debt_to_equity,
        "rebalance_threshold": rebalance_threshold,
        "rebalance_cooldown_bars": rebalance_cooldown_bars,
        "swap_fee_bps": swap_fee_bps,
        "full_short_lower_bound": full_short_lower_bound,
        "full_short_upper_bound": full_short_upper_bound,
    }


def run_selected_backtest(
    strategy_name: str,
    price_data: pd.DataFrame,
    strategy_config: Dict[str, object],
    market_params: MarketParams | None = None,
) -> BacktestResult:
    """Run a backtest using the selected strategy."""
    strategy = strategy_for_name(strategy_name)
    engine = BacktestEngine(strategy)
    return engine.run(
        price_data,
        strategy_config,
        market_params or MarketParams.kamino_defaults(),
    )


def run_selected_grid_search(
    strategy_name: str,
    price_data: pd.DataFrame,
    param_grid: Dict[str, list],
    base_config: Dict[str, object],
    market_params: MarketParams | None = None,
    sort_metric: str = "sortino_ratio",
) -> OptimizationResult:
    """Run grid search using the selected strategy."""
    return grid_search(
        strategy=strategy_for_name(strategy_name),
        price_data=price_data,
        param_grid=param_grid,
        base_config=base_config,
        market_params=market_params or MarketParams.kamino_defaults(),
        sort_metric=sort_metric,
    )
