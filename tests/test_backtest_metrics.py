"""Tests for arblab.backtest.metrics — compute_metrics and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arblab.backtest.metrics import PerformanceMetrics, compute_metrics, _max_drawdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    portfolio_values: list[float],
    health_factors: list[float] | None = None,
    action_counts: list[int] | None = None,
    interest_accrued: list[float] | None = None,
    lst_yield: list[float] | None = None,
    liquidation_penalty: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal history DataFrame with the expected columns."""
    n = len(portfolio_values)
    return pd.DataFrame(
        {
            "portfolio_value": portfolio_values,
            "health_factor": health_factors if health_factors is not None else [2.0] * n,
            "action_count": action_counts if action_counts is not None else [0] * n,
            "interest_accrued": interest_accrued if interest_accrued is not None else [0.0] * n,
            "lst_yield": lst_yield if lst_yield is not None else [0.0] * n,
            "liquidation_penalty": liquidation_penalty if liquidation_penalty is not None else [0.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# 1. Flat portfolio — no change in value
# ---------------------------------------------------------------------------

def test_flat_portfolio():
    """A portfolio that never changes should have 0% return, 0 drawdown, 0 Sharpe."""
    df = _make_df([100.0, 100.0, 100.0, 100.0, 100.0])
    m = compute_metrics(df, initial_value=100.0)

    assert m.initial_value == 100.0
    assert m.final_value == 100.0
    assert m.total_return_pct == pytest.approx(0.0)
    assert m.max_drawdown_pct == pytest.approx(0.0)
    assert m.max_drawdown_duration_bars == 0
    assert m.sharpe_ratio == pytest.approx(0.0)
    assert m.sortino_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Positive returns — monotonically increasing portfolio
# ---------------------------------------------------------------------------

def test_positive_returns():
    """Steadily increasing portfolio should have positive return, zero drawdown."""
    df = _make_df([100.0, 105.0, 110.0, 115.0, 120.0])
    m = compute_metrics(df, initial_value=100.0)

    assert m.final_value == 120.0
    assert m.total_return_pct == pytest.approx(20.0)
    assert m.max_drawdown_pct == pytest.approx(0.0)
    assert m.max_drawdown_duration_bars == 0
    assert m.sharpe_ratio > 0
    assert m.sortino_ratio == pytest.approx(0.0)  # no negative returns


# ---------------------------------------------------------------------------
# 3. Drawdown — magnitude and duration
# ---------------------------------------------------------------------------

def test_drawdown_magnitude_and_duration():
    """Peak at 200, dip to 150 (25% drawdown over 3 bars), then recovery."""
    values = [100.0, 200.0, 180.0, 160.0, 150.0, 170.0, 200.0]
    df = _make_df(values)
    m = compute_metrics(df, initial_value=100.0)

    # max drawdown = (200 - 150) / 200 = 25%
    assert m.max_drawdown_pct == pytest.approx(25.0)
    # bars below peak: indices 2,3,4,5 = 4 bars (bar 6 recovers to peak)
    assert m.max_drawdown_duration_bars == 4


def test_drawdown_multiple_dips():
    """Two separate dips — the longer one should determine duration."""
    # Dip 1: 100 -> 95 (1 bar), recover
    # Dip 2: 105 -> 100 -> 98 -> 97 (3 bars), recover
    values = [100.0, 95.0, 100.0, 105.0, 100.0, 98.0, 97.0, 105.0]
    df = _make_df(values)
    m = compute_metrics(df, initial_value=100.0)

    # Largest percentage drawdown: max(5/100, (105-97)/105) = max(5%, 7.62%) = 7.62%
    expected_dd = (105.0 - 97.0) / 105.0 * 100.0
    assert m.max_drawdown_pct == pytest.approx(expected_dd, rel=1e-4)
    # Longest consecutive streak below peak: 3 bars (indices 4,5,6)
    assert m.max_drawdown_duration_bars == 3


# ---------------------------------------------------------------------------
# 4. Sharpe ratio — annualized
# ---------------------------------------------------------------------------

def test_sharpe_ratio_annualized():
    """Verify Sharpe = mean(returns) / std(returns) * sqrt(periods_per_year)."""
    values = [100.0, 102.0, 101.0, 104.0, 103.0]
    df = _make_df(values)
    periods_per_year = 252.0
    m = compute_metrics(df, initial_value=100.0, periods_per_year=periods_per_year)

    returns = pd.Series(values).pct_change().fillna(0.0)
    expected_sharpe = float(returns.mean()) / float(returns.std()) * np.sqrt(periods_per_year)
    assert m.sharpe_ratio == pytest.approx(expected_sharpe, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Sortino ratio — only downside deviation
# ---------------------------------------------------------------------------

def test_sortino_ratio_downside_only():
    """Sortino should use std of negative returns only."""
    values = [100.0, 102.0, 99.0, 101.0, 97.0]
    df = _make_df(values)
    periods_per_year = 252.0
    m = compute_metrics(df, initial_value=100.0, periods_per_year=periods_per_year)

    returns = pd.Series(values).pct_change().fillna(0.0)
    mean_ret = float(returns.mean())
    downside_std = float(returns[returns < 0].std())
    expected_sortino = mean_ret / downside_std * np.sqrt(periods_per_year)
    assert m.sortino_ratio == pytest.approx(expected_sortino, rel=1e-6)


def test_sortino_zero_when_no_negative_returns():
    """If there are no negative returns, Sortino should be 0."""
    df = _make_df([100.0, 105.0, 110.0])
    m = compute_metrics(df, initial_value=100.0)
    assert m.sortino_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Health factor statistics
# ---------------------------------------------------------------------------

def test_health_factor_stats():
    """Verify min, avg, bars below 1.0, and bars below 1.5."""
    health_factors = [2.0, 1.8, 1.4, 0.9, 0.8, 1.2, 1.6]
    df = _make_df(
        portfolio_values=[100.0] * 7,
        health_factors=health_factors,
    )
    m = compute_metrics(df, initial_value=100.0)

    assert m.min_health_factor == pytest.approx(0.8)
    assert m.avg_health_factor == pytest.approx(np.mean(health_factors))
    # bars below 1.0: HF=0.9, HF=0.8 => 2
    assert m.bars_below_hf_1_0 == 2
    # bars below 1.5: HF=1.4, HF=0.9, HF=0.8, HF=1.2 => 4
    assert m.bars_below_hf_1_5 == 4


def test_health_factor_inf_handling():
    """Infinite HF values should be excluded from min and avg."""
    health_factors = [float("inf"), 2.0, 1.5, float("inf")]
    df = _make_df(
        portfolio_values=[100.0] * 4,
        health_factors=health_factors,
    )
    m = compute_metrics(df, initial_value=100.0)

    assert m.min_health_factor == pytest.approx(1.5)
    assert m.avg_health_factor == pytest.approx(1.75)  # mean(2.0, 1.5)
    assert m.bars_below_hf_1_0 == 0
    assert m.bars_below_hf_1_5 == 0  # inf replaced with NaN, so not < 1.5


# ---------------------------------------------------------------------------
# 7. Total interest / yield / penalty tracking
# ---------------------------------------------------------------------------

def test_totals_tracking():
    """Cumulative interest, yield, penalty, actions, and liquidations."""
    df = _make_df(
        portfolio_values=[100.0, 99.0, 98.0, 97.0],
        action_counts=[0, 1, 2, 1],
        interest_accrued=[0.0, 0.5, 0.5, 0.5],
        lst_yield=[0.0, 0.3, 0.3, 0.3],
        liquidation_penalty=[0.0, 0.0, 1.0, 0.0],
    )
    m = compute_metrics(df, initial_value=100.0)

    assert m.total_actions == 4
    assert m.total_interest_paid == pytest.approx(1.5)
    assert m.total_lst_yield_earned == pytest.approx(0.9)
    assert m.total_liquidation_penalty_paid == pytest.approx(1.0)
    assert m.total_liquidations == 1


# ---------------------------------------------------------------------------
# 8. Edge cases: single bar and empty-ish DataFrames
# ---------------------------------------------------------------------------

def test_single_bar():
    """A single-bar history should still produce valid metrics without errors."""
    df = _make_df([100.0])
    m = compute_metrics(df, initial_value=100.0)

    assert m.final_value == 100.0
    assert m.total_return_pct == pytest.approx(0.0)
    assert m.max_drawdown_pct == pytest.approx(0.0)
    assert m.max_drawdown_duration_bars == 0
    assert m.sharpe_ratio == pytest.approx(0.0)
    assert m.sortino_ratio == pytest.approx(0.0)
    assert len(m.pnl_series) == 1


def test_two_bars():
    """Two bars should work for all calculations including drawdown."""
    df = _make_df([100.0, 90.0])
    m = compute_metrics(df, initial_value=100.0)

    assert m.final_value == 90.0
    assert m.total_return_pct == pytest.approx(-10.0)
    assert m.max_drawdown_pct == pytest.approx(10.0)
    assert m.max_drawdown_duration_bars == 1


# ---------------------------------------------------------------------------
# 9. PnL series correctness
# ---------------------------------------------------------------------------

def test_pnl_series():
    """pnl_series should be the diff of portfolio_value, first entry filled with 0."""
    values = [100.0, 105.0, 102.0, 108.0]
    df = _make_df(values)
    m = compute_metrics(df, initial_value=100.0)

    expected_pnl = [0.0, 5.0, -3.0, 6.0]
    assert len(m.pnl_series) == 4
    for i, expected in enumerate(expected_pnl):
        assert m.pnl_series.iloc[i] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 10. Total return percentage
# ---------------------------------------------------------------------------

def test_total_return_percentage():
    """Verify total_return_pct = (final - initial) / initial * 100."""
    df = _make_df([200.0, 250.0])
    m = compute_metrics(df, initial_value=200.0)

    assert m.total_return_pct == pytest.approx(25.0)


def test_total_return_negative():
    """Negative total return when portfolio declines."""
    df = _make_df([1000.0, 800.0])
    m = compute_metrics(df, initial_value=1000.0)

    assert m.total_return_pct == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# 11. _max_drawdown helper directly
# ---------------------------------------------------------------------------

def test_max_drawdown_helper_no_drawdown():
    """Monotonically increasing series has no drawdown."""
    dd_pct, dd_dur = _max_drawdown(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert dd_pct == pytest.approx(0.0)
    assert dd_dur == 0


def test_max_drawdown_helper_single_element():
    """Single element series cannot have a drawdown."""
    dd_pct, dd_dur = _max_drawdown(pd.Series([100.0]))
    assert dd_pct == pytest.approx(0.0)
    assert dd_dur == 0
