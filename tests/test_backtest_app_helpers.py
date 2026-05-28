"""Tests for backtester UI helper functions."""

from __future__ import annotations

import pandas as pd

from arblab.backtest.app_helpers import (
    DEFAULT_STRATEGY,
    LEVERAGE_LOOP_STRATEGY,
    SOL_SUPERTREND_SHORT_STRATEGY,
    build_price_configs,
    build_sol_supertrend_short_config,
    run_selected_grid_search,
    run_selected_backtest,
    position_value_chart_data,
    strategy_for_name,
    visible_strategy_controls,
)
from arblab.strategies.leverage_loop import LeverageLoopStrategy
from arblab.strategies.sol_supertrend_short import SolSupertrendShortStrategy


def _price_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            ("SOL", "open"): [100.0, 101.0],
            ("SOL", "high"): [101.0, 102.0],
            ("SOL", "low"): [99.0, 100.0],
            ("SOL", "close"): [100.0, 101.0],
            ("SOL", "volume"): [1_000.0, 1_000.0],
            ("ETH", "open"): [2_000.0, 2_010.0],
            ("ETH", "high"): [2_010.0, 2_020.0],
            ("ETH", "low"): [1_990.0, 2_000.0],
            ("ETH", "close"): [2_000.0, 2_010.0],
            ("ETH", "volume"): [1_000.0, 1_000.0],
        },
        index=dates,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_build_price_configs_requires_sol_and_eth_for_supertrend_short():
    configs = build_price_configs(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        collateral_symbol="SOL",
    )

    assert [(cfg.display_name, cfg.symbol) for cfg in configs] == [
        ("SOL", "SOL/USDT"),
        ("ETH", "ETH/USDT"),
    ]


def test_default_strategy_is_sol_supertrend_short():
    assert DEFAULT_STRATEGY == SOL_SUPERTREND_SHORT_STRATEGY


def test_sol_supertrend_visible_controls_drop_leverage_loop_controls():
    controls = visible_strategy_controls(SOL_SUPERTREND_SHORT_STRATEGY)

    assert "Supertrend ATR Period" in controls
    assert "Supertrend Multiplier" in controls
    assert "Full Short Lower Bound" in controls
    assert "Leverage Loops" not in controls
    assert "Loop Utilization" not in controls
    assert "Collateral Asset" not in controls
    assert "Debt Asset" not in controls


def test_build_price_configs_keeps_single_asset_for_leverage_loop():
    configs = build_price_configs(
        strategy_name=LEVERAGE_LOOP_STRATEGY,
        collateral_symbol="JitoSOL",
    )

    assert [(cfg.display_name, cfg.symbol) for cfg in configs] == [
        ("JitoSOL", "JITOSOL/USDT"),
    ]


def test_strategy_for_name_returns_selected_strategy():
    assert isinstance(strategy_for_name(LEVERAGE_LOOP_STRATEGY), LeverageLoopStrategy)
    assert isinstance(
        strategy_for_name(SOL_SUPERTREND_SHORT_STRATEGY),
        SolSupertrendShortStrategy,
    )


def test_build_sol_supertrend_short_config_uses_initial_prices():
    config = build_sol_supertrend_short_config(
        price_data=_price_data(),
        initial_sol_collateral=50.0,
        supertrend_atr_period=10,
        supertrend_multiplier=3.0,
        target_bullish_hf=1.35,
        min_rebalance_hf=1.25,
        max_usdc_debt_to_equity=1.0,
        rebalance_threshold=0.10,
        rebalance_cooldown_bars=4,
        swap_fee_bps=10.0,
        full_short_lower_bound=1.0,
        full_short_upper_bound=1.5,
    )

    assert config["initial_sol_collateral"] == 50.0
    assert config["initial_sol_price"] == 100.0
    assert config["initial_eth_price"] == 2_000.0
    assert config["target_bullish_hf"] == 1.35
    assert config["min_rebalance_hf"] == 1.25
    assert config["supertrend_atr_period"] == 10
    assert config["supertrend_multiplier"] == 3.0
    assert config["full_short_lower_bound"] == 1.0
    assert config["full_short_upper_bound"] == 1.5


def test_run_selected_backtest_uses_sol_supertrend_short_strategy():
    result = run_selected_backtest(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        price_data=_price_data(),
        strategy_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "signal_by_bar": {0: {"green": 1}},
            "rebalance_cooldown_bars": 0,
        },
    )

    assert result.strategy_events
    assert result.strategy_events[0]["reason"] == "hedge_up"


def test_run_selected_grid_search_uses_sortino_for_supertrend_short():
    result = run_selected_grid_search(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        price_data=_price_data(),
        param_grid={"rebalance_threshold": [0.05, 0.10]},
        base_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "signal_by_bar": {0: {"green": 1}},
            "rebalance_cooldown_bars": 0,
        },
    )

    assert "sortino_ratio" in result.comparison_df.columns
    assert len(result.comparison_df) == 2


def test_position_value_chart_data_uses_mark_to_market_values_only():
    history = pd.DataFrame(
        {
            "collateral_value": [20_000.0],
            "debt_value": [5_000.0],
            "portfolio_value": [15_000.0],
            "collateral_SOL": [100.0],
            "collateral_SOL_value": [10_000.0],
            "collateral_USDC": [10_000.0],
            "collateral_USDC_value": [10_000.0],
            "debt_ETH": [2.5],
            "debt_ETH_value": [5_000.0],
        }
    )

    chart = position_value_chart_data(history)

    assert chart.to_dict("list") == {
        "Collateral SOL": [10_000.0],
        "Collateral USDC": [10_000.0],
        "Debt ETH": [-5_000.0],
        "Net Portfolio": [15_000.0],
    }
