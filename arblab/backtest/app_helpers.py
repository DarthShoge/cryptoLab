"""Helper functions shared by the Streamlit backtester app and tests."""

from __future__ import annotations

import itertools
from typing import Dict, List

import pandas as pd

from arblab.backtest.data import OHLCVConfig
from arblab.backtest.engine import BacktestEngine, EngineConfig
from arblab.backtest.market import MarketParams
from arblab.backtest.optimizer import OptimizationResult, grid_search
from arblab.backtest.results import BacktestResult
from arblab.backtest.strategy import Strategy
from arblab.strategies.leverage_loop import LeverageLoopStrategy
from arblab.strategies.sol_supertrend_short import (
    SolSupertrendShortStrategy,
    supertrend_direction,
)


LEVERAGE_LOOP_STRATEGY = "Leverage Loop"
SOL_SUPERTREND_SHORT_STRATEGY = "SOL Supertrend Short"
DEFAULT_STRATEGY = SOL_SUPERTREND_SHORT_STRATEGY
DEFAULT_START_DATE = pd.Timestamp("2021-01-01")
DEFAULT_END_DATE = pd.Timestamp("2025-12-31")
SOL_SUPERTREND_LOOKBACK_BARS = 90 * 24

SOFT_HEDGE_LADDER = {4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50}

SOL_SUPERTREND_BEST_IN_CLASS_DEFAULTS = {
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
    "traffic_light_hedge_floors": {4: 0.0, 3: 0.10, 2: 0.35, 1: 0.75, 0: 1.0},
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
    "enable_traffic_light_exposure_reserve": False,
    "traffic_light_exposure_sell_fractions": {2: 0.05, 1: 0.10, 0: 0.15},
    "traffic_light_exposure_max_fraction": 0.30,
    "traffic_light_exposure_min_sol_collateral": 100.0,
    "traffic_light_exposure_rebuy_min_green": 4,
    "traffic_light_exposure_rebuy_fraction": 0.25,
    "traffic_light_exposure_rebuy_min_hf": 2.0,
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
            "Enable USDC Releverage",
            "Target Bullish HF",
            "Minimum Rebalance HF",
            "Max USDC Debt / Equity",
            "Rebalance Threshold",
            "Cooldown Bars",
            "Enable Surplus USDC Reinvestment",
            "Realized Hedge Profit Gate",
            "Max Surplus Reinvestment / SOL Collateral",
            "Surplus Reinvestment Min HF",
            "Enable Full Short Mode",
            "Full Short Lower Bound",
            "Full Short Upper Bound",
            "Enable Crisis Mode",
            "Crisis SOL Drawdown Threshold",
            "Crisis Portfolio Drawdown Threshold",
            "Crisis Base Hedge Floor",
            "Crisis 3d Hedge Floor",
            "Crisis 3d+1w Hedge Floor",
            "Crisis Exit Recovery Gap",
            "Partial Fill Min HF",
            "Crisis Partial Fill Budget",
            "Enable Profit Lock",
            "Profit Lock Metric",
            "Profit Lock Min Gain",
            "Profit Lock Drawdown",
            "Profit Lock Near High",
            "Stateful Profit Lock",
            "Profit Lock Exit Gap",
            "Profit Lock Hedge Floor",
            "Profit Lock Max Green",
            "Enable Fast Break Overlay",
            "Fast Break Return Threshold",
            "Fast Break Vol Multiplier",
            "Fast Break Hedge Floor",
            "Fast Break Hold Bars",
            "Fast Break Add Min HF",
            "Fast Break Staged Decay",
            "Fast Break Partial Fill",
            "Fast Break Partial Fill Max Green",
            "Fast Break Partial Fill Min HF",
            "Fast Break Partial Fill Budget",
            "Enable Weekly Bearish Reserve",
            "Weekly Bearish Reserve Sell Fraction",
            "Weekly Bearish Reserve Max Fraction",
            "Weekly Bearish Reserve Min SOL",
            "Weekly Bearish Reserve Rebuy Fraction",
            "Enable Profit Lock Reserve",
            "Profit Lock Reserve Sell Fraction",
            "Profit Lock Reserve Escalation Fraction",
            "Profit Lock Reserve Max Fraction",
            "Profit Lock Reserve Min SOL",
            "Profit Lock Reserve Min Gain",
            "Profit Lock Reserve Near High",
            "Profit Lock Reserve Escalation Drawdown",
            "Profit Lock Reserve Rebuy Fraction",
            "Profit Lock Reserve Episode Mode",
            "Profit Lock Reserve Rebuy Cooldown",
            "Profit Lock Reserve New High Reset Gap",
            "Enable CPPI Exposure Cap",
            "CPPI Activation Gain",
            "CPPI Protect Pct",
            "CPPI Cushion Multiplier",
            "CPPI Core Min SOL",
            "CPPI Exposure Buffer",
            "CPPI Max Sell Per Bar",
            "CPPI Rebuy Fraction",
            "CPPI Rebuy Min Green",
            "Enable Hedge Failure Circuit Breaker",
            "Hedge Failure Lookback Bars",
            "Hedge Failure Underperformance",
            "Hedge Failure Hold Bars",
            "Hedge Failure Sell Fraction",
            "Hedge Failure Min SOL",
            "Enable Protected Book",
            "Protected Book PnL Fraction",
            "Enable Froth Reserve",
            "Froth Reserve Min SOL",
            "Froth Reserve Rebuy Drawdown",
            "Froth Reserve Rebuy Fraction",
            "Enable Drawdown Containment",
            "Drawdown Containment Trigger",
            "Drawdown Containment Exit Gap",
            "Drawdown Containment Hedge Floor",
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


def hedge_pnl_chart_data(
    strategy_events: list[dict],
    history: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build hedge ratio and realized PnL chart data from strategy events."""
    empty = {"levels": pd.DataFrame(), "pnl": pd.DataFrame()}
    if not strategy_events:
        return empty

    events = pd.DataFrame(strategy_events)
    if "timestamp" not in events:
        return empty
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events = events.set_index("timestamp").sort_index()

    levels = pd.DataFrame(index=events.index)
    levels["Target ETH Short / SOL Collateral"] = events.get(
        "target_eth_short_ratio",
        pd.Series(index=events.index, dtype=float),
    ).astype(float)
    levels["Actual ETH Short / SOL Collateral"] = events.get(
        "current_eth_short_ratio",
        pd.Series(index=events.index, dtype=float),
    ).astype(float)

    if {"debt_ETH_value", "collateral_SOL_value"}.issubset(history.columns):
        aligned = history.reindex(events.index, method="ffill")
        levels["Mark-to-Market ETH Debt / SOL Collateral"] = (
            aligned["debt_ETH_value"] / aligned["collateral_SOL_value"]
        ).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    pnl = pd.DataFrame(index=events.index)
    pnl["Lifetime Realized Hedge PnL"] = events.get(
        "lifetime_realized_hedge_pnl_usdc",
        pd.Series(index=events.index, dtype=float),
    ).astype(float)
    pnl["Spendable Hedge Profit"] = events.get(
        "spendable_hedge_profit_usdc",
        pd.Series(index=events.index, dtype=float),
    ).astype(float)
    pnl["Open ETH Short Notional Basis"] = (
        events.get("open_eth_short_amount", pd.Series(index=events.index, dtype=float))
        .astype(float)
        * events.get(
            "average_eth_short_basis_usdc",
            pd.Series(index=events.index, dtype=float),
        ).astype(float)
    )

    return {"levels": levels, "pnl": pnl}


def _series_max_drawdown_pct(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    running_peak = values.cummax()
    drawdown = (values - running_peak) / running_peak
    return abs(float(drawdown.min())) * 100.0


def benchmark_tier(
    strategy_return_pct: float,
    strategy_max_drawdown_pct: float,
    sol_benchmark_return_pct: float,
    sol_benchmark_max_drawdown_pct: float,
    acceptable_tracking_gap_pct: float = 25.0,
) -> str:
    """Classify a run against the SOL-relative USD compounding objective."""
    tracking_gap = strategy_return_pct - sol_benchmark_return_pct
    drawdown_improvement = sol_benchmark_max_drawdown_pct - strategy_max_drawdown_pct

    if tracking_gap >= 0.0 and drawdown_improvement > 0.0:
        return "pass"
    if tracking_gap >= -acceptable_tracking_gap_pct and drawdown_improvement > 0.0:
        return "acceptable"
    if drawdown_improvement > 0.0:
        return "capital_preservation"
    return "reject"


def benchmark_tier_rank(tier: str) -> int:
    """Return sort precedence for benchmark tiers; lower is better."""
    return {
        "pass": 0,
        "acceptable": 1,
        "capital_preservation": 2,
        "reject": 3,
    }.get(tier, 99)


def _experiment_group(config: Dict[str, object]) -> str:
    if (
        bool(config.get("enable_usdc_releverage", False))
        and float(config.get("max_usdc_debt_to_equity", 0.0)) > 0.0
    ):
        return "usdc_releverage"
    if bool(config.get("enable_full_short_mode", True)):
        return "core_full_short"
    return "core_hedge"


def _final_position_columns(history: pd.DataFrame, final_sol_price: float) -> dict[str, float]:
    if history.empty:
        return {
            "final_portfolio_value_usd": 0.0,
            "final_sol_equiv": 0.0,
        }

    final = history.iloc[-1]
    values = {
        "final_portfolio_value_usd": float(final.get("portfolio_value", 0.0)),
        "final_sol_equiv": (
            float(final.get("portfolio_value", 0.0)) / final_sol_price
            if final_sol_price > 0.0
            else 0.0
        ),
    }

    for symbol in ("SOL", "USDC"):
        values[f"final_collateral_{symbol}"] = float(
            final.get(f"collateral_{symbol}", 0.0)
        )
        values[f"final_collateral_{symbol}_value_usd"] = float(
            final.get(f"collateral_{symbol}_value", 0.0)
        )
    for symbol in ("ETH", "USDC"):
        values[f"final_debt_{symbol}"] = float(final.get(f"debt_{symbol}", 0.0))
        values[f"final_debt_{symbol}_value_usd"] = float(
            final.get(f"debt_{symbol}_value", 0.0)
        )
    return values


def final_position_summary(
    history: pd.DataFrame,
    final_sol_price: float,
) -> dict[str, object]:
    """Return final account positions in native units and USD values."""
    values = _final_position_columns(history, final_sol_price)
    return {
        "net": {
            "portfolio_value_usd": values["final_portfolio_value_usd"],
            "sol_equivalent": values["final_sol_equiv"],
        },
        "collateral": [
            {
                "Asset": symbol,
                "Amount": values[f"final_collateral_{symbol}"],
                "Value USD": values[f"final_collateral_{symbol}_value_usd"],
            }
            for symbol in ("SOL", "USDC")
        ],
        "debt": [
            {
                "Asset": symbol,
                "Amount": values[f"final_debt_{symbol}"],
                "Value USD": values[f"final_debt_{symbol}_value_usd"],
            }
            for symbol in ("ETH", "USDC")
        ],
    }


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
    timeframe: str,
    atr_period: int,
    multiplier: float,
) -> pd.Series:
    resampled = _resample_ohlcv(price_data, "SOL", timeframe)
    if resampled.empty:
        return pd.Series(True, index=price_data.index)
    direction = supertrend_direction(resampled, atr_period, multiplier)
    closed_direction = direction.shift(1)
    mapped = closed_direction.reindex(price_data.index, method="ffill")
    return mapped.where(mapped.notna(), True).astype(bool)


def build_sol_supertrend_signal_by_bar(
    price_data: pd.DataFrame,
    atr_period: int,
    multiplier: float,
    timeframes: tuple[str, ...] = ("1h", "4h", "8h", "1d"),
) -> Dict[int, Dict[str, bool | int]]:
    """Precompute SOL Supertrend votes once for a full price dataset."""
    directions = [
        _closed_direction_on_base_index(
            price_data, timeframe, atr_period, multiplier
        )
        for timeframe in timeframes
    ]
    bearish_1d = ~_closed_direction_on_base_index(
        price_data, "1d", atr_period, multiplier
    )
    bearish_3d = ~_closed_direction_on_base_index(
        price_data, "3d", atr_period, multiplier
    )
    bearish_1w = ~_closed_direction_on_base_index(
        price_data, "1w", atr_period, multiplier
    )

    signals: dict[int, dict[str, bool | int]] = {}
    for idx, timestamp in enumerate(price_data.index):
        green = sum(int(bool(direction.loc[timestamp])) for direction in directions)
        signals[idx] = {
            "green": green,
            "bearish_1d": bool(bearish_1d.loc[timestamp]),
            "bearish_3d": bool(bearish_3d.loc[timestamp]),
            "bearish_1w": bool(bearish_1w.loc[timestamp]),
        }
    return signals


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
    enable_full_short_mode: bool = True,
    enable_usdc_releverage: bool = False,
    enable_surplus_usdc_reinvestment: bool = True,
    realized_hedge_profit_gate_pct: float = 0.10,
    surplus_reinvestment_ladder: dict[int, float] | None = None,
    max_surplus_reinvestment_pct_of_sol_collateral: float = 0.05,
    surplus_reinvestment_min_hf: float = 2.0,
    hedge_ladder: dict[int, float] | None = None,
    enable_crisis_mode: bool = True,
    crisis_sol_drawdown_threshold: float = 0.25,
    crisis_portfolio_drawdown_threshold: float = 0.20,
    crisis_hedge_floor_base: float = 0.75,
    crisis_hedge_floor_3d: float = 1.0,
    crisis_hedge_floor_3d_1w: float = 1.25,
    crisis_exit_sol_equiv_recovery_gap: float = 0.10,
    partial_fill_min_hf: float = 2.50,
    crisis_partial_fill_budget_pct: float = 0.25,
    enable_profit_lock: bool = False,
    profit_lock_metric: str = "portfolio",
    profit_lock_min_gain_pct: float = 0.25,
    profit_lock_drawdown_threshold: float = 0.10,
    profit_lock_near_high_threshold: float = 0.00,
    profit_lock_stateful: bool = False,
    profit_lock_stateful_exit_gap: float = 0.02,
    profit_lock_hedge_floor: float = 0.35,
    profit_lock_max_green: int = 3,
    enable_fast_break_overlay: bool = False,
    fast_break_return_lookback_bars: int = 24,
    fast_break_return_threshold: float = -0.08,
    fast_break_use_donchian_break: bool = False,
    fast_break_donchian_lookback_bars: int = 7 * 24,
    fast_break_vol_lookback_bars: int = 24,
    fast_break_vol_median_bars: int = 30 * 24,
    fast_break_vol_multiplier: float = 1.5,
    fast_break_max_green: int = 3,
    fast_break_hedge_floor: float = 0.75,
    fast_break_hold_bars: int = 72,
    fast_break_exit_min_green: int = 4,
    fast_break_add_min_hf: float | None = None,
    fast_break_decay_enabled: bool = False,
    fast_break_decay_floors: list[float] | None = None,
    enable_fast_break_partial_fill: bool = False,
    fast_break_partial_fill_requires_crisis: bool = False,
    fast_break_partial_fill_max_green: int | None = None,
    fast_break_partial_fill_min_hf: float = 2.50,
    fast_break_partial_fill_budget_pct: float = 0.25,
    enable_weekly_bearish_reserve: bool = False,
    weekly_bearish_reserve_sell_fraction: float = 0.10,
    weekly_bearish_reserve_max_fraction: float = 0.30,
    weekly_bearish_reserve_min_sol_collateral: float = 100.0,
    weekly_bearish_reserve_rebuy_fraction: float = 0.50,
    enable_profit_lock_reserve: bool = False,
    profit_lock_reserve_sell_fraction: float = 0.10,
    profit_lock_reserve_escalation_sell_fraction: float = 0.10,
    profit_lock_reserve_max_fraction: float = 0.30,
    profit_lock_reserve_min_sol_collateral: float = 100.0,
    profit_lock_reserve_min_gain_pct: float = 1.00,
    profit_lock_reserve_near_high_threshold: float = 0.10,
    profit_lock_reserve_escalation_drawdown: float = 0.15,
    profit_lock_reserve_rebuy_fraction: float = 0.50,
    profit_lock_reserve_episode_mode: bool = False,
    profit_lock_reserve_rebuy_cooldown_bars: int = 0,
    profit_lock_reserve_new_high_reset_gap: float = 0.00,
    enable_cppi_exposure_cap: bool = False,
    cppi_activation_gain: float = 1.50,
    cppi_protect_pct: float = 0.65,
    cppi_cushion_multiplier: float = 2.00,
    cppi_core_min_sol_collateral: float = 100.0,
    cppi_exposure_buffer_pct: float = 0.05,
    cppi_max_sell_fraction_per_bar: float = 0.10,
    cppi_rebuy_fraction: float = 0.50,
    cppi_rebuy_min_green: int = 4,
    enable_hedge_failure_circuit_breaker: bool = False,
    hedge_failure_lookback_bars: int = 72,
    hedge_failure_underperformance_threshold: float = 0.10,
    hedge_failure_hold_bars: int = 168,
    hedge_failure_sell_fraction: float = 0.00,
    hedge_failure_min_sol_collateral: float = 100.0,
    enable_traffic_light_governor: bool = False,
    traffic_light_hedge_floors: dict[int, float] | None = None,
    traffic_light_add_min_hf: float | None = None,
    traffic_light_min_reinvestment_green: int = 3,
    traffic_light_min_releverage_green: int = 4,
    enable_traffic_light_protected_book: bool = False,
    traffic_light_protected_book_fractions: dict[int, float] | None = None,
    traffic_light_protected_rebuy_min_green: int = 4,
    traffic_light_protected_rebuy_fraction: float = 0.25,
    traffic_light_protected_rebuy_max_pct_of_sol_collateral: float = 0.05,
    traffic_light_protected_rebuy_min_hf: float = 2.0,
    enable_traffic_light_exposure_reserve: bool = False,
    traffic_light_exposure_sell_fractions: dict[int, float] | None = None,
    traffic_light_exposure_max_fraction: float = 0.30,
    traffic_light_exposure_min_sol_collateral: float = 100.0,
    traffic_light_exposure_rebuy_min_green: int = 4,
    traffic_light_exposure_rebuy_fraction: float = 0.25,
    traffic_light_exposure_rebuy_min_hf: float = 2.0,
    enable_protected_book: bool = False,
    protected_book_realized_pnl_fraction: float = 0.25,
    enable_froth_reserve: bool = False,
    froth_reserve_min_sol_collateral: float = 100.0,
    froth_reserve_tiers: dict[float, float] | None = None,
    froth_reserve_rebuy_drawdown_threshold: float = 0.35,
    froth_reserve_rebuy_fraction: float = 0.25,
    enable_drawdown_containment: bool = False,
    drawdown_containment_trigger: float = 0.20,
    drawdown_containment_exit_gap: float = 0.10,
    drawdown_containment_hedge_floor: float = 0.75,
    drawdown_containment_block_rebuy: bool = True,
    drawdown_containment_block_reinvestment: bool = True,
    drawdown_containment_block_releverage: bool = True,
) -> Dict[str, float | int | bool | dict]:
    """Build strategy config for the SOL Supertrend short strategy."""
    return {
        "initial_sol_collateral": initial_sol_collateral,
        "initial_sol_price": float(price_data.iloc[0][("SOL", "close")]),
        "initial_eth_price": float(price_data.iloc[0][("ETH", "close")]),
        "supertrend_atr_period": supertrend_atr_period,
        "supertrend_multiplier": supertrend_multiplier,
        "signal_by_bar": build_sol_supertrend_signal_by_bar(
            price_data,
            atr_period=supertrend_atr_period,
            multiplier=supertrend_multiplier,
        ),
        "target_bullish_hf": target_bullish_hf,
        "min_rebalance_hf": min_rebalance_hf,
        "max_usdc_debt_to_equity": max_usdc_debt_to_equity,
        "rebalance_threshold": rebalance_threshold,
        "rebalance_cooldown_bars": rebalance_cooldown_bars,
        "swap_fee_bps": swap_fee_bps,
        "hedge_ladder": hedge_ladder or SOFT_HEDGE_LADDER,
        "full_short_lower_bound": full_short_lower_bound,
        "full_short_upper_bound": full_short_upper_bound,
        "enable_full_short_mode": enable_full_short_mode,
        "enable_usdc_releverage": enable_usdc_releverage,
        "enable_surplus_usdc_reinvestment": enable_surplus_usdc_reinvestment,
        "realized_hedge_profit_gate_pct": realized_hedge_profit_gate_pct,
        "surplus_reinvestment_ladder": surplus_reinvestment_ladder or {3: 0.25, 4: 0.50},
        "max_surplus_reinvestment_pct_of_sol_collateral": (
            max_surplus_reinvestment_pct_of_sol_collateral
        ),
        "surplus_reinvestment_min_hf": surplus_reinvestment_min_hf,
        "enable_crisis_mode": enable_crisis_mode,
        "crisis_sol_drawdown_threshold": crisis_sol_drawdown_threshold,
        "crisis_portfolio_drawdown_threshold": crisis_portfolio_drawdown_threshold,
        "crisis_hedge_floor_base": crisis_hedge_floor_base,
        "crisis_hedge_floor_3d": crisis_hedge_floor_3d,
        "crisis_hedge_floor_3d_1w": crisis_hedge_floor_3d_1w,
        "crisis_exit_sol_equiv_recovery_gap": crisis_exit_sol_equiv_recovery_gap,
        "partial_fill_min_hf": partial_fill_min_hf,
        "crisis_partial_fill_budget_pct": crisis_partial_fill_budget_pct,
        "enable_profit_lock": enable_profit_lock,
        "profit_lock_metric": profit_lock_metric,
        "profit_lock_min_gain_pct": profit_lock_min_gain_pct,
        "profit_lock_drawdown_threshold": profit_lock_drawdown_threshold,
        "profit_lock_near_high_threshold": profit_lock_near_high_threshold,
        "profit_lock_stateful": profit_lock_stateful,
        "profit_lock_stateful_exit_gap": profit_lock_stateful_exit_gap,
        "profit_lock_hedge_floor": profit_lock_hedge_floor,
        "profit_lock_max_green": profit_lock_max_green,
        "enable_fast_break_overlay": enable_fast_break_overlay,
        "fast_break_return_lookback_bars": fast_break_return_lookback_bars,
        "fast_break_return_threshold": fast_break_return_threshold,
        "fast_break_use_donchian_break": fast_break_use_donchian_break,
        "fast_break_donchian_lookback_bars": fast_break_donchian_lookback_bars,
        "fast_break_vol_lookback_bars": fast_break_vol_lookback_bars,
        "fast_break_vol_median_bars": fast_break_vol_median_bars,
        "fast_break_vol_multiplier": fast_break_vol_multiplier,
        "fast_break_max_green": fast_break_max_green,
        "fast_break_hedge_floor": fast_break_hedge_floor,
        "fast_break_hold_bars": fast_break_hold_bars,
        "fast_break_exit_min_green": fast_break_exit_min_green,
        "fast_break_add_min_hf": fast_break_add_min_hf,
        "fast_break_decay_enabled": fast_break_decay_enabled,
        "fast_break_decay_floors": fast_break_decay_floors or [0.75, 0.35],
        "enable_fast_break_partial_fill": enable_fast_break_partial_fill,
        "fast_break_partial_fill_requires_crisis": (
            fast_break_partial_fill_requires_crisis
        ),
        "fast_break_partial_fill_max_green": fast_break_partial_fill_max_green,
        "fast_break_partial_fill_min_hf": fast_break_partial_fill_min_hf,
        "fast_break_partial_fill_budget_pct": fast_break_partial_fill_budget_pct,
        "enable_weekly_bearish_reserve": enable_weekly_bearish_reserve,
        "weekly_bearish_reserve_sell_fraction": (
            weekly_bearish_reserve_sell_fraction
        ),
        "weekly_bearish_reserve_max_fraction": weekly_bearish_reserve_max_fraction,
        "weekly_bearish_reserve_min_sol_collateral": (
            weekly_bearish_reserve_min_sol_collateral
        ),
        "weekly_bearish_reserve_rebuy_fraction": (
            weekly_bearish_reserve_rebuy_fraction
        ),
        "enable_profit_lock_reserve": enable_profit_lock_reserve,
        "profit_lock_reserve_sell_fraction": profit_lock_reserve_sell_fraction,
        "profit_lock_reserve_escalation_sell_fraction": (
            profit_lock_reserve_escalation_sell_fraction
        ),
        "profit_lock_reserve_max_fraction": profit_lock_reserve_max_fraction,
        "profit_lock_reserve_min_sol_collateral": (
            profit_lock_reserve_min_sol_collateral
        ),
        "profit_lock_reserve_min_gain_pct": profit_lock_reserve_min_gain_pct,
        "profit_lock_reserve_near_high_threshold": (
            profit_lock_reserve_near_high_threshold
        ),
        "profit_lock_reserve_escalation_drawdown": (
            profit_lock_reserve_escalation_drawdown
        ),
        "profit_lock_reserve_rebuy_fraction": profit_lock_reserve_rebuy_fraction,
        "profit_lock_reserve_episode_mode": profit_lock_reserve_episode_mode,
        "profit_lock_reserve_rebuy_cooldown_bars": (
            profit_lock_reserve_rebuy_cooldown_bars
        ),
        "profit_lock_reserve_new_high_reset_gap": (
            profit_lock_reserve_new_high_reset_gap
        ),
        "enable_cppi_exposure_cap": enable_cppi_exposure_cap,
        "cppi_activation_gain": cppi_activation_gain,
        "cppi_protect_pct": cppi_protect_pct,
        "cppi_cushion_multiplier": cppi_cushion_multiplier,
        "cppi_core_min_sol_collateral": cppi_core_min_sol_collateral,
        "cppi_exposure_buffer_pct": cppi_exposure_buffer_pct,
        "cppi_max_sell_fraction_per_bar": cppi_max_sell_fraction_per_bar,
        "cppi_rebuy_fraction": cppi_rebuy_fraction,
        "cppi_rebuy_min_green": cppi_rebuy_min_green,
        "enable_hedge_failure_circuit_breaker": (
            enable_hedge_failure_circuit_breaker
        ),
        "hedge_failure_lookback_bars": hedge_failure_lookback_bars,
        "hedge_failure_underperformance_threshold": (
            hedge_failure_underperformance_threshold
        ),
        "hedge_failure_hold_bars": hedge_failure_hold_bars,
        "hedge_failure_sell_fraction": hedge_failure_sell_fraction,
        "hedge_failure_min_sol_collateral": hedge_failure_min_sol_collateral,
        "enable_traffic_light_governor": enable_traffic_light_governor,
        "traffic_light_hedge_floors": traffic_light_hedge_floors
        or {4: 0.0, 3: 0.10, 2: 0.35, 1: 0.75, 0: 1.0},
        "traffic_light_add_min_hf": traffic_light_add_min_hf,
        "traffic_light_min_reinvestment_green": traffic_light_min_reinvestment_green,
        "traffic_light_min_releverage_green": traffic_light_min_releverage_green,
        "enable_traffic_light_protected_book": enable_traffic_light_protected_book,
        "traffic_light_protected_book_fractions": (
            traffic_light_protected_book_fractions
            or {4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 1.0}
        ),
        "traffic_light_protected_rebuy_min_green": (
            traffic_light_protected_rebuy_min_green
        ),
        "traffic_light_protected_rebuy_fraction": traffic_light_protected_rebuy_fraction,
        "traffic_light_protected_rebuy_max_pct_of_sol_collateral": (
            traffic_light_protected_rebuy_max_pct_of_sol_collateral
        ),
        "traffic_light_protected_rebuy_min_hf": traffic_light_protected_rebuy_min_hf,
        "enable_traffic_light_exposure_reserve": enable_traffic_light_exposure_reserve,
        "traffic_light_exposure_sell_fractions": (
            traffic_light_exposure_sell_fractions or {2: 0.05, 1: 0.10, 0: 0.15}
        ),
        "traffic_light_exposure_max_fraction": traffic_light_exposure_max_fraction,
        "traffic_light_exposure_min_sol_collateral": (
            traffic_light_exposure_min_sol_collateral
        ),
        "traffic_light_exposure_rebuy_min_green": traffic_light_exposure_rebuy_min_green,
        "traffic_light_exposure_rebuy_fraction": traffic_light_exposure_rebuy_fraction,
        "traffic_light_exposure_rebuy_min_hf": traffic_light_exposure_rebuy_min_hf,
        "enable_protected_book": enable_protected_book,
        "protected_book_realized_pnl_fraction": protected_book_realized_pnl_fraction,
        "enable_froth_reserve": enable_froth_reserve,
        "froth_reserve_min_sol_collateral": froth_reserve_min_sol_collateral,
        "froth_reserve_tiers": froth_reserve_tiers or {1.0: 0.05, 3.0: 0.05},
        "froth_reserve_rebuy_drawdown_threshold": (
            froth_reserve_rebuy_drawdown_threshold
        ),
        "froth_reserve_rebuy_fraction": froth_reserve_rebuy_fraction,
        "enable_drawdown_containment": enable_drawdown_containment,
        "drawdown_containment_trigger": drawdown_containment_trigger,
        "drawdown_containment_exit_gap": drawdown_containment_exit_gap,
        "drawdown_containment_hedge_floor": drawdown_containment_hedge_floor,
        "drawdown_containment_block_rebuy": drawdown_containment_block_rebuy,
        "drawdown_containment_block_reinvestment": (
            drawdown_containment_block_reinvestment
        ),
        "drawdown_containment_block_releverage": drawdown_containment_block_releverage,
    }


def run_selected_backtest(
    strategy_name: str,
    price_data: pd.DataFrame,
    strategy_config: Dict[str, object],
    market_params: MarketParams | None = None,
) -> BacktestResult:
    """Run a backtest using the selected strategy."""
    strategy = strategy_for_name(strategy_name)
    engine_config = (
        EngineConfig(lookback_bars=SOL_SUPERTREND_LOOKBACK_BARS)
        if strategy_name == SOL_SUPERTREND_SHORT_STRATEGY
        else None
    )
    engine = BacktestEngine(strategy, engine_config)
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
    if strategy_name == SOL_SUPERTREND_SHORT_STRATEGY:
        return _run_sol_supertrend_grid_search(
            price_data=price_data,
            param_grid=param_grid,
            base_config=base_config,
            market_params=market_params or MarketParams.kamino_defaults(),
            sort_metric=sort_metric,
        )

    return grid_search(
        strategy=strategy_for_name(strategy_name),
        price_data=price_data,
        param_grid=param_grid,
        base_config=base_config,
        market_params=market_params or MarketParams.kamino_defaults(),
        sort_metric=sort_metric,
    )


def _run_sol_supertrend_grid_search(
    price_data: pd.DataFrame,
    param_grid: Dict[str, list],
    base_config: Dict[str, object],
    market_params: MarketParams,
    sort_metric: str,
) -> OptimizationResult:
    grid_params = dict(param_grid)
    if "enable_crisis_mode" not in grid_params:
        grid_params["enable_crisis_mode"] = [True, False]

    keys = list(grid_params.keys())
    combos = list(itertools.product(*(grid_params[key] for key in keys)))
    results: list[BacktestResult] = []
    grid: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    sol_close = price_data[("SOL", "close")].astype(float)
    sol_benchmark_return_pct = (
        (float(sol_close.iloc[-1]) - float(sol_close.iloc[0]))
        / float(sol_close.iloc[0])
        * 100.0
        if len(sol_close) > 0 and float(sol_close.iloc[0]) != 0.0
        else 0.0
    )
    sol_benchmark_max_drawdown_pct = _series_max_drawdown_pct(sol_close)
    final_sol_price = float(sol_close.iloc[-1]) if len(sol_close) > 0 else 0.0

    for combo in combos:
        params = dict(zip(keys, combo))
        config = {**base_config, **params}
        atr_period = int(config.get("supertrend_atr_period", 10))
        multiplier = float(config.get("supertrend_multiplier", 3.0))
        config["signal_by_bar"] = build_sol_supertrend_signal_by_bar(
            price_data,
            atr_period=atr_period,
            multiplier=multiplier,
        )
        result = BacktestEngine(
            SolSupertrendShortStrategy(),
            EngineConfig(lookback_bars=SOL_SUPERTREND_LOOKBACK_BARS),
        ).run(
            price_data,
            config,
            market_params,
        )
        grid.append(params)
        results.append(result)

        m = result.metrics
        tracking_gap = m.total_return_pct - sol_benchmark_return_pct
        drawdown_improvement = sol_benchmark_max_drawdown_pct - m.max_drawdown_pct
        rows.append(
            {
                **params,
                "_result_index": len(results) - 1,
                "experiment_group": _experiment_group(config),
                "sol_benchmark_return_pct": sol_benchmark_return_pct,
                "tracking_gap_vs_sol_pct": tracking_gap,
                "drawdown_improvement_vs_sol_pct": drawdown_improvement,
                "benchmark_tier": benchmark_tier(
                    strategy_return_pct=m.total_return_pct,
                    strategy_max_drawdown_pct=m.max_drawdown_pct,
                    sol_benchmark_return_pct=sol_benchmark_return_pct,
                    sol_benchmark_max_drawdown_pct=sol_benchmark_max_drawdown_pct,
                ),
                **_final_position_columns(result.history, final_sol_price),
                "total_return_pct": m.total_return_pct,
                "max_drawdown_pct": m.max_drawdown_pct,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "min_health_factor": m.min_health_factor,
                "total_liquidations": m.total_liquidations,
                "total_interest_paid": m.total_interest_paid,
                "total_lst_yield_earned": m.total_lst_yield_earned,
                "liquidated": result.liquidated,
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df["_benchmark_tier_rank"] = comparison_df["benchmark_tier"].map(
        benchmark_tier_rank
    )
    comparison_df = comparison_df.sort_values(
        ["_benchmark_tier_rank", sort_metric],
        ascending=[True, False],
    ).reset_index(drop=True)
    best_idx = int(comparison_df.iloc[0]["_result_index"])
    comparison_df = comparison_df.drop(columns=["_result_index", "_benchmark_tier_rank"])

    return OptimizationResult(
        results=results,
        param_grid=grid,
        best_result=results[best_idx],
        comparison_df=comparison_df,
    )
