"""Tests for backtester UI helper functions."""

from __future__ import annotations

import pandas as pd

from arblab.backtest.app_helpers import (
    DEFAULT_STRATEGY,
    LEVERAGE_LOOP_STRATEGY,
    SOL_SUPERTREND_SHORT_STRATEGY,
    benchmark_tier,
    benchmark_tier_rank,
    build_price_configs,
    build_sol_supertrend_short_config,
    build_sol_supertrend_signal_by_bar,
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


def _long_price_data(n: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    sol = pd.Series(range(100, 100 + n), index=dates, dtype=float)
    eth = pd.Series(range(2_000, 2_000 + n), index=dates, dtype=float)
    df = pd.DataFrame(
        {
            ("SOL", "open"): sol,
            ("SOL", "high"): sol * 1.01,
            ("SOL", "low"): sol * 0.99,
            ("SOL", "close"): sol,
            ("SOL", "volume"): 1_000.0,
            ("ETH", "open"): eth,
            ("ETH", "high"): eth * 1.01,
            ("ETH", "low"): eth * 0.99,
            ("ETH", "close"): eth,
            ("ETH", "volume"): 1_000.0,
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
    assert "Enable USDC Releverage" in controls
    assert "Enable Full Short Mode" in controls
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
        enable_full_short_mode=True,
        enable_usdc_releverage=False,
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
    assert config["enable_full_short_mode"] is True
    assert config["enable_usdc_releverage"] is False
    assert isinstance(config["signal_by_bar"], dict)


def test_build_sol_supertrend_signal_by_bar_precomputes_all_base_bars():
    price_data = _long_price_data()

    signals = build_sol_supertrend_signal_by_bar(
        price_data,
        atr_period=7,
        multiplier=2.0,
    )

    assert set(signals.keys()) == set(range(len(price_data)))
    assert all(0 <= signal["green"] <= 4 for signal in signals.values())
    assert all("bearish_3d" in signal for signal in signals.values())
    assert all("bearish_1w" in signal for signal in signals.values())


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
        price_data=_long_price_data(),
        param_grid={
            "supertrend_atr_period": [7, 10],
            "supertrend_multiplier": [2.0],
        },
        base_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "rebalance_cooldown_bars": 0,
        },
    )

    assert "sortino_ratio" in result.comparison_df.columns
    assert len(result.comparison_df) == 2
    for result_item in result.results:
        assert "signal_by_bar" in result_item.strategy_config
        assert result_item.strategy_config["signal_by_bar"]


def test_benchmark_tier_uses_sol_relative_usd_compounding_rules():
    assert (
        benchmark_tier(
            strategy_return_pct=320.0,
            strategy_max_drawdown_pct=50.0,
            sol_benchmark_return_pct=300.0,
            sol_benchmark_max_drawdown_pct=70.0,
        )
        == "pass"
    )
    assert (
        benchmark_tier(
            strategy_return_pct=285.0,
            strategy_max_drawdown_pct=40.0,
            sol_benchmark_return_pct=300.0,
            sol_benchmark_max_drawdown_pct=70.0,
        )
        == "acceptable"
    )
    assert (
        benchmark_tier(
            strategy_return_pct=150.0,
            strategy_max_drawdown_pct=35.0,
            sol_benchmark_return_pct=300.0,
            sol_benchmark_max_drawdown_pct=70.0,
        )
        == "capital_preservation"
    )
    assert (
        benchmark_tier(
            strategy_return_pct=150.0,
            strategy_max_drawdown_pct=95.0,
            sol_benchmark_return_pct=300.0,
            sol_benchmark_max_drawdown_pct=70.0,
        )
        == "reject"
    )


def test_benchmark_tier_rank_orders_normal_passes_before_defensive_tiers():
    assert benchmark_tier_rank("pass") < benchmark_tier_rank("acceptable")
    assert benchmark_tier_rank("acceptable") < benchmark_tier_rank("capital_preservation")
    assert benchmark_tier_rank("capital_preservation") < benchmark_tier_rank("reject")


def test_sol_supertrend_grid_search_adds_benchmark_tier_columns():
    result = run_selected_grid_search(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        price_data=_long_price_data(),
        param_grid={
            "supertrend_atr_period": [10],
            "supertrend_multiplier": [3.0],
        },
        base_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "rebalance_cooldown_bars": 0,
            "max_usdc_debt_to_equity": 0.0,
            "enable_usdc_releverage": False,
            "enable_full_short_mode": False,
            "full_short_lower_bound": 0.0,
            "full_short_upper_bound": 0.0,
        },
    )

    row = result.comparison_df.iloc[0]
    assert "sol_benchmark_return_pct" in result.comparison_df.columns
    assert "tracking_gap_vs_sol_pct" in result.comparison_df.columns
    assert "drawdown_improvement_vs_sol_pct" in result.comparison_df.columns
    assert "benchmark_tier" in result.comparison_df.columns
    assert "experiment_group" in result.comparison_df.columns
    assert row["benchmark_tier"] in {
        "pass",
        "acceptable",
        "capital_preservation",
        "reject",
    }
    assert row["experiment_group"] == "core_hedge"


def test_sol_supertrend_grid_search_groups_full_short_only_when_enabled():
    price_data = _long_price_data()

    hedge_only = run_selected_grid_search(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        price_data=price_data,
        param_grid={"supertrend_atr_period": [10]},
        base_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "signal_by_bar": build_sol_supertrend_signal_by_bar(price_data, 10, 3.0),
            "enable_full_short_mode": False,
            "full_short_lower_bound": 1.0,
            "full_short_upper_bound": 1.5,
            "enable_usdc_releverage": False,
            "max_usdc_debt_to_equity": 0.0,
        },
    )
    full_short = run_selected_grid_search(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        price_data=price_data,
        param_grid={"supertrend_atr_period": [10]},
        base_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "signal_by_bar": build_sol_supertrend_signal_by_bar(price_data, 10, 3.0),
            "enable_full_short_mode": True,
            "full_short_lower_bound": 1.0,
            "full_short_upper_bound": 1.5,
            "enable_usdc_releverage": False,
            "max_usdc_debt_to_equity": 0.0,
        },
    )

    assert hedge_only.comparison_df.iloc[0]["experiment_group"] == "core_hedge"
    assert full_short.comparison_df.iloc[0]["experiment_group"] == "core_full_short"


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
