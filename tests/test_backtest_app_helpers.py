"""Tests for backtester UI helper functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arblab.backtest.app_helpers import (
    DEFAULT_END_DATE,
    DEFAULT_STRATEGY,
    DEFAULT_START_DATE,
    LEVERAGE_LOOP_STRATEGY,
    SOFT_HEDGE_LADDER,
    SOL_SUPERTREND_BEST_IN_CLASS_DEFAULTS,
    SOL_SUPERTREND_SHORT_STRATEGY,
    benchmark_tier,
    benchmark_tier_rank,
    build_price_configs,
    build_sol_supertrend_short_config,
    build_sol_supertrend_signal_by_bar,
    final_position_summary,
    hedge_pnl_chart_data,
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


def _drawdown_price_data(n: int = 2_201) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    sol = pd.Series([150.0] * (n - 1) + [100.0], index=dates)
    eth = pd.Series([2_000.0] * n, index=dates)
    df = pd.DataFrame(
        {
            ("SOL", "open"): sol,
            ("SOL", "high"): 150.0,
            ("SOL", "low"): sol,
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


def test_default_backtest_date_range_matches_long_report_window():
    assert DEFAULT_START_DATE == pd.Timestamp("2021-01-01")
    assert DEFAULT_END_DATE == pd.Timestamp("2025-12-31")


def test_sol_supertrend_best_in_class_defaults_match_scientific_report_winner():
    assert SOL_SUPERTREND_BEST_IN_CLASS_DEFAULTS == {
        "supertrend_atr_period": 10,
        "supertrend_multiplier": 3.0,
        "hedge_ladder": SOFT_HEDGE_LADDER,
        "enable_usdc_releverage": False,
        "max_usdc_debt_to_equity": 0.0,
        "target_bullish_hf": 1.35,
        "min_rebalance_hf": 2.00,
        "rebalance_threshold": 0.10,
        "rebalance_cooldown_bars": 4,
        "enable_surplus_usdc_reinvestment": True,
        "realized_hedge_profit_gate_pct": 0.10,
        "surplus_reinvestment_ladder": {3: 0.25, 4: 0.50},
        "max_surplus_reinvestment_pct_of_sol_collateral": 0.05,
        "surplus_reinvestment_min_hf": 2.0,
        "enable_full_short_mode": True,
        "full_short_lower_bound": 0.75,
        "full_short_upper_bound": 1.25,
        "enable_crisis_mode": True,
        "crisis_sol_drawdown_threshold": 0.25,
        "crisis_portfolio_drawdown_threshold": 0.20,
        "crisis_hedge_floor_base": 0.75,
        "crisis_hedge_floor_3d": 1.0,
        "crisis_hedge_floor_3d_1w": 1.25,
        "crisis_exit_sol_equiv_recovery_gap": 0.10,
        "partial_fill_min_hf": 2.50,
        "crisis_partial_fill_budget_pct": 0.00,
        "enable_profit_lock": True,
        "profit_lock_metric": "portfolio",
        "profit_lock_min_gain_pct": 0.25,
        "profit_lock_drawdown_threshold": 0.05,
        "profit_lock_near_high_threshold": 0.01,
        "profit_lock_stateful": True,
        "profit_lock_stateful_exit_gap": 0.05,
        "profit_lock_hedge_floor": 0.35,
        "profit_lock_max_green": 3,
        "enable_fast_break_overlay": True,
        "fast_break_return_lookback_bars": 24,
        "fast_break_return_threshold": -0.08,
        "fast_break_use_donchian_break": False,
        "fast_break_donchian_lookback_bars": 7 * 24,
        "fast_break_vol_lookback_bars": 24,
        "fast_break_vol_median_bars": 30 * 24,
        "fast_break_vol_multiplier": 2.5,
        "fast_break_max_green": 3,
        "fast_break_hedge_floor": 1.0,
        "fast_break_hold_bars": 72,
        "fast_break_exit_min_green": 4,
        "fast_break_add_min_hf": 2.50,
        "fast_break_decay_enabled": True,
        "fast_break_decay_floors": [0.75, 0.35],
        "enable_fast_break_partial_fill": False,
        "fast_break_partial_fill_requires_crisis": False,
        "fast_break_partial_fill_max_green": None,
        "fast_break_partial_fill_min_hf": 2.50,
        "fast_break_partial_fill_budget_pct": 0.25,
        "enable_weekly_bearish_reserve": False,
        "weekly_bearish_reserve_sell_fraction": 0.10,
        "weekly_bearish_reserve_max_fraction": 0.30,
        "weekly_bearish_reserve_min_sol_collateral": 100.0,
        "weekly_bearish_reserve_rebuy_fraction": 0.50,
        "enable_profit_lock_reserve": False,
        "profit_lock_reserve_sell_fraction": 0.10,
        "profit_lock_reserve_escalation_sell_fraction": 0.10,
        "profit_lock_reserve_max_fraction": 0.30,
        "profit_lock_reserve_min_sol_collateral": 100.0,
        "profit_lock_reserve_min_gain_pct": 1.00,
        "profit_lock_reserve_near_high_threshold": 0.10,
        "profit_lock_reserve_escalation_drawdown": 0.15,
        "profit_lock_reserve_rebuy_fraction": 0.50,
        "profit_lock_reserve_episode_mode": False,
        "profit_lock_reserve_rebuy_cooldown_bars": 0,
        "profit_lock_reserve_new_high_reset_gap": 0.00,
        "enable_cppi_exposure_cap": False,
        "cppi_activation_gain": 1.50,
        "cppi_protect_pct": 0.65,
        "cppi_cushion_multiplier": 2.00,
        "cppi_core_min_sol_collateral": 100.0,
        "cppi_exposure_buffer_pct": 0.05,
        "cppi_max_sell_fraction_per_bar": 0.10,
        "cppi_rebuy_fraction": 0.50,
        "cppi_rebuy_min_green": 4,
        "enable_hedge_failure_circuit_breaker": False,
        "hedge_failure_lookback_bars": 72,
        "hedge_failure_underperformance_threshold": 0.10,
        "hedge_failure_hold_bars": 168,
        "hedge_failure_sell_fraction": 0.00,
        "hedge_failure_min_sol_collateral": 100.0,
        "enable_traffic_light_governor": False,
        "traffic_light_hedge_floors": {
            4: 0.0,
            3: 0.10,
            2: 0.35,
            1: 0.75,
            0: 1.0,
        },
        "traffic_light_add_min_hf": None,
        "traffic_light_min_reinvestment_green": 3,
        "traffic_light_min_releverage_green": 4,
        "enable_traffic_light_protected_book": False,
        "traffic_light_protected_book_fractions": {
            4: 0.0,
            3: 0.10,
            2: 0.25,
            1: 0.50,
            0: 1.0,
        },
        "traffic_light_protected_rebuy_min_green": 4,
        "traffic_light_protected_rebuy_fraction": 0.25,
        "traffic_light_protected_rebuy_max_pct_of_sol_collateral": 0.05,
        "traffic_light_protected_rebuy_min_hf": 2.0,
        "enable_protected_book": False,
        "protected_book_realized_pnl_fraction": 0.25,
        "enable_froth_reserve": False,
        "froth_reserve_min_sol_collateral": 100.0,
        "froth_reserve_tiers": {1.0: 0.05, 3.0: 0.05},
        "froth_reserve_rebuy_drawdown_threshold": 0.35,
        "froth_reserve_rebuy_fraction": 0.25,
        "enable_drawdown_containment": False,
        "drawdown_containment_trigger": 0.20,
        "drawdown_containment_exit_gap": 0.10,
        "drawdown_containment_hedge_floor": 0.75,
        "drawdown_containment_block_rebuy": True,
        "drawdown_containment_block_reinvestment": True,
        "drawdown_containment_block_releverage": True,
    }


def test_sol_supertrend_visible_controls_drop_leverage_loop_controls():
    controls = visible_strategy_controls(SOL_SUPERTREND_SHORT_STRATEGY)

    assert "Supertrend ATR Period" in controls
    assert "Supertrend Multiplier" in controls
    assert "Enable USDC Releverage" in controls
    assert "Enable Surplus USDC Reinvestment" in controls
    assert "Realized Hedge Profit Gate" in controls
    assert "Max Surplus Reinvestment / SOL Collateral" in controls
    assert "Surplus Reinvestment Min HF" in controls
    assert "Enable Full Short Mode" in controls
    assert "Full Short Lower Bound" in controls
    assert "Enable Crisis Mode" in controls
    assert "Crisis SOL Drawdown Threshold" in controls
    assert "Crisis Base Hedge Floor" in controls
    assert "Crisis 3d Hedge Floor" in controls
    assert "Crisis 3d+1w Hedge Floor" in controls
    assert "Crisis Exit Recovery Gap" in controls
    assert "Partial Fill Min HF" in controls
    assert "Crisis Partial Fill Budget" in controls
    assert "Enable Profit Lock" in controls
    assert "Profit Lock Metric" in controls
    assert "Profit Lock Min Gain" in controls
    assert "Profit Lock Drawdown" in controls
    assert "Profit Lock Near High" in controls
    assert "Stateful Profit Lock" in controls
    assert "Profit Lock Exit Gap" in controls
    assert "Profit Lock Hedge Floor" in controls
    assert "Profit Lock Max Green" in controls
    assert "Enable Fast Break Overlay" in controls
    assert "Fast Break Return Threshold" in controls
    assert "Fast Break Vol Multiplier" in controls
    assert "Fast Break Hedge Floor" in controls
    assert "Fast Break Hold Bars" in controls
    assert "Fast Break Add Min HF" in controls
    assert "Fast Break Staged Decay" in controls
    assert "Fast Break Partial Fill" in controls
    assert "Fast Break Partial Fill Min HF" in controls
    assert "Fast Break Partial Fill Budget" in controls
    assert "Enable Weekly Bearish Reserve" in controls
    assert "Weekly Bearish Reserve Sell Fraction" in controls
    assert "Weekly Bearish Reserve Max Fraction" in controls
    assert "Weekly Bearish Reserve Min SOL" in controls
    assert "Weekly Bearish Reserve Rebuy Fraction" in controls
    assert "Enable Profit Lock Reserve" in controls
    assert "Profit Lock Reserve Sell Fraction" in controls
    assert "Profit Lock Reserve Escalation Fraction" in controls
    assert "Profit Lock Reserve Max Fraction" in controls
    assert "Profit Lock Reserve Min SOL" in controls
    assert "Profit Lock Reserve Min Gain" in controls
    assert "Profit Lock Reserve Near High" in controls
    assert "Profit Lock Reserve Escalation Drawdown" in controls
    assert "Profit Lock Reserve Rebuy Fraction" in controls
    assert "Enable Froth Reserve" in controls
    assert "Froth Reserve Min SOL" in controls
    assert "Froth Reserve Rebuy Drawdown" in controls
    assert "Froth Reserve Rebuy Fraction" in controls
    assert "Enable Drawdown Containment" in controls
    assert "Drawdown Containment Trigger" in controls
    assert "Drawdown Containment Exit Gap" in controls
    assert "Drawdown Containment Hedge Floor" in controls
    assert "Leverage Loops" not in controls
    assert "Loop Utilization" not in controls
    assert "Collateral Asset" not in controls
    assert "Debt Asset" not in controls


def test_froth_reserve_rebuy_fraction_is_rendered_with_froth_reserve_controls():
    app_source = Path("backtest_app.py").read_text()

    froth_section = app_source.index("if enable_froth_reserve:")
    rebuy_fraction = app_source.index('"Froth Reserve Rebuy Fraction"')
    drawdown_section = app_source.index("if enable_drawdown_containment:")

    assert froth_section < rebuy_fraction < drawdown_section


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
        enable_surplus_usdc_reinvestment=True,
        realized_hedge_profit_gate_pct=0.10,
        surplus_reinvestment_ladder={3: 0.25, 4: 0.50},
        max_surplus_reinvestment_pct_of_sol_collateral=0.05,
        surplus_reinvestment_min_hf=2.0,
        enable_crisis_mode=True,
        crisis_sol_drawdown_threshold=0.25,
        crisis_portfolio_drawdown_threshold=0.20,
        crisis_hedge_floor_base=0.75,
        crisis_hedge_floor_3d=1.0,
        crisis_hedge_floor_3d_1w=1.25,
        crisis_exit_sol_equiv_recovery_gap=0.10,
        partial_fill_min_hf=2.50,
        crisis_partial_fill_budget_pct=0.00,
        enable_profit_lock=True,
        profit_lock_metric="portfolio",
        profit_lock_min_gain_pct=0.25,
        profit_lock_drawdown_threshold=0.10,
        profit_lock_near_high_threshold=0.03,
        profit_lock_stateful=True,
        profit_lock_stateful_exit_gap=0.04,
        profit_lock_hedge_floor=0.35,
        profit_lock_max_green=3,
        enable_fast_break_overlay=True,
        fast_break_return_lookback_bars=24,
        fast_break_return_threshold=-0.10,
        fast_break_use_donchian_break=True,
        fast_break_donchian_lookback_bars=120,
        fast_break_vol_lookback_bars=24,
        fast_break_vol_median_bars=480,
        fast_break_vol_multiplier=2.0,
        fast_break_max_green=2,
        fast_break_hedge_floor=1.0,
        fast_break_hold_bars=48,
        fast_break_exit_min_green=4,
        fast_break_add_min_hf=1.50,
        fast_break_decay_enabled=True,
        fast_break_decay_floors=[0.75, 0.35],
        enable_fast_break_partial_fill=True,
        fast_break_partial_fill_requires_crisis=True,
        fast_break_partial_fill_max_green=1,
        fast_break_partial_fill_min_hf=2.50,
        fast_break_partial_fill_budget_pct=0.30,
        enable_weekly_bearish_reserve=True,
        weekly_bearish_reserve_sell_fraction=0.15,
        weekly_bearish_reserve_max_fraction=0.45,
        weekly_bearish_reserve_min_sol_collateral=75.0,
        weekly_bearish_reserve_rebuy_fraction=0.35,
        enable_profit_lock_reserve=True,
        profit_lock_reserve_sell_fraction=0.12,
        profit_lock_reserve_escalation_sell_fraction=0.08,
        profit_lock_reserve_max_fraction=0.32,
        profit_lock_reserve_min_sol_collateral=80.0,
        profit_lock_reserve_min_gain_pct=1.25,
        profit_lock_reserve_near_high_threshold=0.07,
        profit_lock_reserve_escalation_drawdown=0.12,
        profit_lock_reserve_rebuy_fraction=0.40,
        profit_lock_reserve_episode_mode=True,
        profit_lock_reserve_rebuy_cooldown_bars=168,
        profit_lock_reserve_new_high_reset_gap=0.03,
        enable_cppi_exposure_cap=True,
        cppi_activation_gain=2.00,
        cppi_protect_pct=0.60,
        cppi_cushion_multiplier=1.50,
        cppi_core_min_sol_collateral=90.0,
        cppi_exposure_buffer_pct=0.08,
        cppi_max_sell_fraction_per_bar=0.12,
        cppi_rebuy_fraction=0.35,
        cppi_rebuy_min_green=3,
        enable_hedge_failure_circuit_breaker=True,
        hedge_failure_lookback_bars=48,
        hedge_failure_underperformance_threshold=0.12,
        hedge_failure_hold_bars=96,
        hedge_failure_sell_fraction=0.08,
        hedge_failure_min_sol_collateral=85.0,
        enable_protected_book=True,
        protected_book_realized_pnl_fraction=0.40,
        enable_froth_reserve=True,
        froth_reserve_min_sol_collateral=125.0,
        froth_reserve_tiers={1.0: 0.05},
        froth_reserve_rebuy_drawdown_threshold=0.40,
        froth_reserve_rebuy_fraction=0.50,
        enable_drawdown_containment=True,
        drawdown_containment_trigger=0.15,
        drawdown_containment_exit_gap=0.08,
        drawdown_containment_hedge_floor=0.75,
        drawdown_containment_block_rebuy=True,
        drawdown_containment_block_reinvestment=True,
        drawdown_containment_block_releverage=True,
    )

    assert config["initial_sol_collateral"] == 50.0
    assert config["initial_sol_price"] == 100.0
    assert config["initial_eth_price"] == 2_000.0
    assert config["target_bullish_hf"] == 1.35
    assert config["min_rebalance_hf"] == 1.25
    assert config["supertrend_atr_period"] == 10
    assert config["supertrend_multiplier"] == 3.0
    assert config["hedge_ladder"] == SOFT_HEDGE_LADDER
    assert config["full_short_lower_bound"] == 1.0
    assert config["full_short_upper_bound"] == 1.5
    assert config["enable_full_short_mode"] is True
    assert config["enable_usdc_releverage"] is False
    assert config["enable_surplus_usdc_reinvestment"] is True
    assert config["realized_hedge_profit_gate_pct"] == 0.10
    assert config["surplus_reinvestment_ladder"] == {3: 0.25, 4: 0.50}
    assert config["max_surplus_reinvestment_pct_of_sol_collateral"] == 0.05
    assert config["surplus_reinvestment_min_hf"] == 2.0
    assert config["enable_crisis_mode"] is True
    assert config["crisis_sol_drawdown_threshold"] == 0.25
    assert config["crisis_portfolio_drawdown_threshold"] == 0.20
    assert config["crisis_hedge_floor_base"] == 0.75
    assert config["crisis_hedge_floor_3d"] == 1.0
    assert config["crisis_hedge_floor_3d_1w"] == 1.25
    assert config["crisis_exit_sol_equiv_recovery_gap"] == 0.10
    assert config["partial_fill_min_hf"] == 2.50
    assert config["crisis_partial_fill_budget_pct"] == 0.00
    assert config["enable_profit_lock"] is True
    assert config["profit_lock_metric"] == "portfolio"
    assert config["profit_lock_min_gain_pct"] == 0.25
    assert config["profit_lock_drawdown_threshold"] == 0.10
    assert config["profit_lock_near_high_threshold"] == 0.03
    assert config["profit_lock_stateful"] is True
    assert config["profit_lock_stateful_exit_gap"] == 0.04
    assert config["profit_lock_hedge_floor"] == 0.35
    assert config["profit_lock_max_green"] == 3
    assert config["enable_fast_break_overlay"] is True
    assert config["fast_break_return_lookback_bars"] == 24
    assert config["fast_break_return_threshold"] == -0.10
    assert config["fast_break_use_donchian_break"] is True
    assert config["fast_break_donchian_lookback_bars"] == 120
    assert config["fast_break_vol_lookback_bars"] == 24
    assert config["fast_break_vol_median_bars"] == 480
    assert config["fast_break_vol_multiplier"] == 2.0
    assert config["fast_break_max_green"] == 2
    assert config["fast_break_hedge_floor"] == 1.0
    assert config["fast_break_hold_bars"] == 48
    assert config["fast_break_exit_min_green"] == 4
    assert config["fast_break_add_min_hf"] == 1.50
    assert config["fast_break_decay_enabled"] is True
    assert config["fast_break_decay_floors"] == [0.75, 0.35]
    assert config["enable_fast_break_partial_fill"] is True
    assert config["fast_break_partial_fill_requires_crisis"] is True
    assert config["fast_break_partial_fill_max_green"] == 1
    assert config["fast_break_partial_fill_min_hf"] == 2.50
    assert config["fast_break_partial_fill_budget_pct"] == 0.30
    assert config["enable_weekly_bearish_reserve"] is True
    assert config["weekly_bearish_reserve_sell_fraction"] == 0.15
    assert config["weekly_bearish_reserve_max_fraction"] == 0.45
    assert config["weekly_bearish_reserve_min_sol_collateral"] == 75.0
    assert config["weekly_bearish_reserve_rebuy_fraction"] == 0.35
    assert config["enable_profit_lock_reserve"] is True
    assert config["profit_lock_reserve_sell_fraction"] == 0.12
    assert config["profit_lock_reserve_escalation_sell_fraction"] == 0.08
    assert config["profit_lock_reserve_max_fraction"] == 0.32
    assert config["profit_lock_reserve_min_sol_collateral"] == 80.0
    assert config["profit_lock_reserve_min_gain_pct"] == 1.25
    assert config["profit_lock_reserve_near_high_threshold"] == 0.07
    assert config["profit_lock_reserve_escalation_drawdown"] == 0.12
    assert config["profit_lock_reserve_rebuy_fraction"] == 0.40
    assert config["profit_lock_reserve_episode_mode"] is True
    assert config["profit_lock_reserve_rebuy_cooldown_bars"] == 168
    assert config["profit_lock_reserve_new_high_reset_gap"] == 0.03
    assert config["enable_cppi_exposure_cap"] is True
    assert config["cppi_activation_gain"] == 2.00
    assert config["cppi_protect_pct"] == 0.60
    assert config["cppi_cushion_multiplier"] == 1.50
    assert config["cppi_core_min_sol_collateral"] == 90.0
    assert config["cppi_exposure_buffer_pct"] == 0.08
    assert config["cppi_max_sell_fraction_per_bar"] == 0.12
    assert config["cppi_rebuy_fraction"] == 0.35
    assert config["cppi_rebuy_min_green"] == 3
    assert config["enable_hedge_failure_circuit_breaker"] is True
    assert config["hedge_failure_lookback_bars"] == 48
    assert config["hedge_failure_underperformance_threshold"] == 0.12
    assert config["hedge_failure_hold_bars"] == 96
    assert config["hedge_failure_sell_fraction"] == 0.08
    assert config["hedge_failure_min_sol_collateral"] == 85.0
    assert config["enable_protected_book"] is True
    assert config["protected_book_realized_pnl_fraction"] == 0.40
    assert config["enable_froth_reserve"] is True
    assert config["froth_reserve_min_sol_collateral"] == 125.0
    assert config["froth_reserve_tiers"] == {1.0: 0.05}
    assert config["froth_reserve_rebuy_drawdown_threshold"] == 0.40
    assert config["froth_reserve_rebuy_fraction"] == 0.50
    assert config["enable_drawdown_containment"] is True
    assert config["drawdown_containment_trigger"] == 0.15
    assert config["drawdown_containment_exit_gap"] == 0.08
    assert config["drawdown_containment_hedge_floor"] == 0.75
    assert config["drawdown_containment_block_rebuy"] is True
    assert config["drawdown_containment_block_reinvestment"] is True
    assert config["drawdown_containment_block_releverage"] is True
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
    assert all("bearish_1d" in signal for signal in signals.values())
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


def test_run_selected_backtest_gives_crisis_mode_enough_history():
    price_data = _drawdown_price_data()
    result = run_selected_backtest(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        price_data=price_data,
        strategy_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 150.0,
            "initial_eth_price": 2_000.0,
            "enable_crisis_mode": True,
            "signal_by_bar": {
                len(price_data) - 1: {
                    "green": 3,
                    "bearish_1d": True,
                    "bearish_3d": True,
                    "bearish_1w": False,
                }
            },
            "rebalance_cooldown_bars": 0,
        },
    )

    assert result.strategy_events[-1]["reason"] == "crisis_hedge_up"


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
    assert len(result.comparison_df) == 4
    for result_item in result.results:
        assert "signal_by_bar" in result_item.strategy_config
        assert result_item.strategy_config["signal_by_bar"]


def test_sol_supertrend_grid_search_compares_crisis_on_and_off_by_default():
    result = run_selected_grid_search(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        price_data=_long_price_data(),
        param_grid={"supertrend_atr_period": [10]},
        base_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "rebalance_cooldown_bars": 0,
            "enable_full_short_mode": False,
            "enable_usdc_releverage": False,
        },
    )

    assert set(result.comparison_df["enable_crisis_mode"]) == {True, False}
    assert len(result.comparison_df) == 2


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


def test_sol_supertrend_grid_search_adds_final_position_columns():
    result = run_selected_grid_search(
        strategy_name=SOL_SUPERTREND_SHORT_STRATEGY,
        price_data=_long_price_data(),
        param_grid={"supertrend_atr_period": [10]},
        base_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "rebalance_cooldown_bars": 0,
            "enable_usdc_releverage": False,
            "enable_full_short_mode": False,
        },
    )

    row = result.comparison_df.iloc[0]
    expected_columns = {
        "final_portfolio_value_usd",
        "final_sol_equiv",
        "final_collateral_SOL",
        "final_collateral_SOL_value_usd",
        "final_collateral_USDC",
        "final_collateral_USDC_value_usd",
        "final_debt_ETH",
        "final_debt_ETH_value_usd",
        "final_debt_USDC",
        "final_debt_USDC_value_usd",
    }

    assert expected_columns.issubset(result.comparison_df.columns)
    assert row["final_portfolio_value_usd"] == row["final_collateral_SOL_value_usd"]
    assert row["final_sol_equiv"] > 0
    assert row["final_collateral_SOL"] == 100.0
    assert row["final_debt_ETH"] == 0.0
    assert row["final_debt_USDC"] == 0.0


def test_final_position_summary_returns_assets_and_mark_to_market_values():
    history = pd.DataFrame(
        {
            "portfolio_value": [15_000.0],
            "collateral_SOL": [100.0],
            "collateral_SOL_value": [10_000.0],
            "collateral_USDC": [10_000.0],
            "collateral_USDC_value": [10_000.0],
            "debt_ETH": [2.5],
            "debt_ETH_value": [5_000.0],
            "debt_USDC": [0.0],
            "debt_USDC_value": [0.0],
        }
    )

    summary = final_position_summary(history, final_sol_price=100.0)

    assert summary["net"] == {
        "portfolio_value_usd": 15_000.0,
        "sol_equivalent": 150.0,
    }
    assert summary["collateral"] == [
        {"Asset": "SOL", "Amount": 100.0, "Value USD": 10_000.0},
        {"Asset": "USDC", "Amount": 10_000.0, "Value USD": 10_000.0},
    ]
    assert summary["debt"] == [
        {"Asset": "ETH", "Amount": 2.5, "Value USD": 5_000.0},
        {"Asset": "USDC", "Amount": 0.0, "Value USD": 0.0},
    ]


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


def test_hedge_pnl_chart_data_returns_ratio_and_pnl_views():
    events = [
        {
            "timestamp": pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            "target_eth_short_ratio": 0.50,
            "current_eth_short_ratio": 0.25,
            "lifetime_realized_hedge_pnl_usdc": 100.0,
            "spendable_hedge_profit_usdc": 50.0,
            "open_eth_short_amount": 1.5,
            "average_eth_short_basis_usdc": 2_000.0,
        },
        {
            "timestamp": pd.Timestamp("2024-01-01 04:00:00", tz="UTC"),
            "target_eth_short_ratio": 0.00,
            "current_eth_short_ratio": 0.10,
            "lifetime_realized_hedge_pnl_usdc": 175.0,
            "spendable_hedge_profit_usdc": 125.0,
            "open_eth_short_amount": 0.5,
            "average_eth_short_basis_usdc": 2_000.0,
        },
    ]
    history = pd.DataFrame(
        {
            "debt_ETH_value": [1_000.0, 500.0],
            "collateral_SOL_value": [4_000.0, 5_000.0],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-01-01 04:00:00", tz="UTC"),
            ],
            name="timestamp",
        ),
    )

    charts = hedge_pnl_chart_data(events, history)

    assert charts["levels"].to_dict("list") == {
        "Target ETH Short / SOL Collateral": [0.50, 0.00],
        "Actual ETH Short / SOL Collateral": [0.25, 0.10],
        "Mark-to-Market ETH Debt / SOL Collateral": [0.25, 0.10],
    }
    assert charts["pnl"].to_dict("list") == {
        "Lifetime Realized Hedge PnL": [100.0, 175.0],
        "Spendable Hedge Profit": [50.0, 125.0],
        "Open ETH Short Notional Basis": [3_000.0, 1_000.0],
    }


def test_hedge_pnl_chart_data_returns_empty_views_without_events():
    charts = hedge_pnl_chart_data([], pd.DataFrame())

    assert charts["levels"].empty
    assert charts["pnl"].empty
