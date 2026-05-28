"""Helper functions shared by the Streamlit backtester app and tests."""

from __future__ import annotations

import itertools
from typing import Dict, List

import pandas as pd

from arblab.backtest.data import OHLCVConfig
from arblab.backtest.engine import BacktestEngine
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

SOFT_HEDGE_LADDER = {4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50}

SOL_SUPERTREND_BEST_IN_CLASS_DEFAULTS = {
    "supertrend_atr_period": 10,
    "supertrend_multiplier": 3.0,
    "hedge_ladder": SOFT_HEDGE_LADDER,
    "enable_usdc_releverage": False,
    "max_usdc_debt_to_equity": 0.0,
    "target_bullish_hf": 1.35,
    "min_rebalance_hf": 1.75,
    "rebalance_threshold": 0.10,
    "rebalance_cooldown_bars": 4,
    "enable_full_short_mode": True,
    "full_short_lower_bound": 0.75,
    "full_short_upper_bound": 1.25,
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
            "Enable Full Short Mode",
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
    hedge_ladder: dict[int, float] | None = None,
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
    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[key] for key in keys)))
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
        result = BacktestEngine(SolSupertrendShortStrategy()).run(
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
