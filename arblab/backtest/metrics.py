"""Performance metrics computed from backtest history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    initial_value: float
    final_value: float
    total_return_pct: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    sharpe_ratio: float
    sortino_ratio: float
    min_health_factor: float
    avg_health_factor: float
    bars_below_hf_1_0: int
    bars_below_hf_1_5: int
    total_actions: int
    total_liquidations: int
    total_interest_paid: float
    total_lst_yield_earned: float
    total_liquidation_penalty_paid: float
    pnl_series: pd.Series


def _max_drawdown(values: pd.Series) -> tuple[float, int]:
    """Return (max_drawdown_pct, max_drawdown_duration_bars)."""
    if len(values) < 2:
        return 0.0, 0

    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    max_dd = float(drawdown.min())  # most negative

    # Duration: longest streak below the running peak
    below_peak = values < cummax
    if not below_peak.any():
        return 0.0, 0

    groups = (~below_peak).cumsum()
    dd_groups = below_peak.groupby(groups)
    max_dur = int(dd_groups.sum().max()) if len(dd_groups) > 0 else 0

    return abs(max_dd), max_dur


def compute_metrics(
    history_df: pd.DataFrame,
    initial_value: float,
    periods_per_year: float = 8760.0,
) -> PerformanceMetrics:
    """Compute performance metrics from engine history.

    Expected columns in *history_df*:
    - ``portfolio_value``: net value at each bar
    - ``health_factor``: HF at each bar
    - ``action_count``: number of strategy actions at each bar
    - ``interest_accrued``: USD interest accrued at each bar
    - ``lst_yield``: USD LST yield at each bar
    - ``liquidation_penalty``: USD penalty at each bar (0 if no liquidation)
    """
    values = history_df["portfolio_value"]
    hf = history_df["health_factor"]

    final_value = float(values.iloc[-1]) if len(values) > 0 else initial_value
    total_return = (final_value - initial_value) / initial_value if initial_value != 0 else 0.0

    max_dd, max_dd_dur = _max_drawdown(values)

    # Returns series for Sharpe/Sortino
    pnl = values.diff().fillna(0.0)
    returns = values.pct_change().fillna(0.0)

    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    downside_std = float(returns[returns < 0].std()) if (returns < 0).any() else 0.0

    # Annualized Sharpe (assume risk-free = 0)
    sharpe = (
        mean_ret / std_ret * np.sqrt(periods_per_year)
        if std_ret > 0
        else 0.0
    )
    sortino = (
        mean_ret / downside_std * np.sqrt(periods_per_year)
        if downside_std > 0
        else 0.0
    )

    # Health factor stats — replace inf with NaN for averaging
    hf_finite = hf.replace([np.inf, -np.inf], np.nan)
    min_hf = float(hf_finite.min()) if not hf_finite.isna().all() else float("inf")
    avg_hf = float(hf_finite.mean()) if not hf_finite.isna().all() else float("inf")
    bars_below_1 = int((hf_finite < 1.0).sum())
    bars_below_1_5 = int((hf_finite < 1.5).sum())

    total_actions = int(history_df["action_count"].sum()) if "action_count" in history_df else 0
    total_interest = float(history_df["interest_accrued"].sum()) if "interest_accrued" in history_df else 0.0
    total_yield = float(history_df["lst_yield"].sum()) if "lst_yield" in history_df else 0.0
    total_liq_penalty = float(history_df["liquidation_penalty"].sum()) if "liquidation_penalty" in history_df else 0.0
    total_liqs = int((history_df["liquidation_penalty"] > 0).sum()) if "liquidation_penalty" in history_df else 0

    return PerformanceMetrics(
        initial_value=initial_value,
        final_value=final_value,
        total_return_pct=total_return * 100.0,
        max_drawdown_pct=max_dd * 100.0,
        max_drawdown_duration_bars=max_dd_dur,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        min_health_factor=min_hf,
        avg_health_factor=avg_hf,
        bars_below_hf_1_0=bars_below_1,
        bars_below_hf_1_5=bars_below_1_5,
        total_actions=total_actions,
        total_liquidations=total_liqs,
        total_interest_paid=total_interest,
        total_lst_yield_earned=total_yield,
        total_liquidation_penalty_paid=total_liq_penalty,
        pnl_series=pnl,
    )
