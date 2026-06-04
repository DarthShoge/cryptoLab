"""Lightweight candidate strategy backtesting utilities.

These helpers are for exchange-style research strategies where the output is
portfolio weights over OHLCV assets. They intentionally sit outside the Kamino
engine, which models lending positions and liquidation mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateMetrics:
    initial_value: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    avg_turnover: float
    total_trades: int
    exposure_pct: float


@dataclass(frozen=True)
class CandidateBacktestResult:
    history: pd.DataFrame
    weights: pd.DataFrame
    metrics: CandidateMetrics


def close_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    """Extract close prices from a multi-index OHLCV frame."""
    if not isinstance(price_data.columns, pd.MultiIndex):
        raise ValueError("price_data must have MultiIndex columns: (symbol, field)")
    closes = price_data.xs("close", axis=1, level=1)
    return closes.astype(float)


def backtest_weight_strategy(
    price_data: pd.DataFrame,
    weights: pd.DataFrame,
    fee_bps: float = 0.0,
    initial_value: float = 1.0,
    periods_per_year: float = 8760.0,
) -> CandidateBacktestResult:
    """Backtest close-to-close returns from target weights.

    Target weights are shifted by one bar before applying returns, so a signal
    computed at bar *t* is first exposed to returns from *t* to *t+1*.
    """
    closes = close_prices(price_data)
    weights = weights.reindex(closes.index).reindex(columns=closes.columns, fill_value=0.0)
    weights = weights.fillna(0.0).astype(float)

    asset_returns = closes.pct_change().fillna(0.0)
    executed_weights = weights.shift(1).fillna(0.0)
    turnover = executed_weights.diff().abs().sum(axis=1).fillna(0.0)
    gross_return = (executed_weights * asset_returns).sum(axis=1)
    cost = turnover * (fee_bps / 10_000.0)
    net_return = gross_return - cost
    equity = initial_value * (1.0 + net_return).cumprod()

    history = pd.DataFrame(
        {
            "gross_return": gross_return,
            "cost": cost,
            "net_return": net_return,
            "turnover": turnover,
            "equity": equity,
            "exposure": executed_weights.abs().sum(axis=1),
        },
        index=closes.index,
    )

    return CandidateBacktestResult(
        history=history,
        weights=weights,
        metrics=_compute_candidate_metrics(history, initial_value, periods_per_year),
    )


def moving_average_trend_weights(
    price_data: pd.DataFrame,
    symbol: str,
    fast: int,
    slow: int,
) -> pd.DataFrame:
    """Long-only moving-average trend-following weights for one symbol."""
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow windows must be positive")
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")

    closes = close_prices(price_data)
    if symbol not in closes.columns:
        raise ValueError(f"symbol {symbol!r} not found in price_data")

    fast_ma = closes[symbol].rolling(fast, min_periods=fast).mean()
    slow_ma = closes[symbol].rolling(slow, min_periods=slow).mean()
    signal = (fast_ma > slow_ma).astype(float).fillna(0.0)
    return pd.DataFrame({symbol: signal}, index=closes.index)


def cross_sectional_momentum_weights(
    price_data: pd.DataFrame,
    lookback: int,
    top_n: int,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    """Equal-weight the top *n* assets by trailing return."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")

    closes = close_prices(price_data)
    momentum = closes.pct_change(periods=lookback)
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current_weights = pd.Series(0.0, index=closes.columns)

    for bar_index, (timestamp, row) in enumerate(momentum.iterrows()):
        should_rebalance = (
            bar_index >= lookback
            and (bar_index - lookback) % rebalance_every == 0
        )
        if not should_rebalance:
            weights.loc[timestamp] = current_weights
            continue
        valid = row.dropna()
        if len(valid) < top_n:
            weights.loc[timestamp] = current_weights
            continue
        winners = valid.sort_values(ascending=False).head(top_n).index
        current_weights = pd.Series(0.0, index=closes.columns)
        current_weights.loc[winners] = 1.0 / top_n
        weights.loc[timestamp] = current_weights

    return weights


def _compute_candidate_metrics(
    history: pd.DataFrame,
    initial_value: float,
    periods_per_year: float,
) -> CandidateMetrics:
    returns = history["net_return"]
    equity = history["equity"]
    final_value = float(equity.iloc[-1]) if len(equity) else initial_value
    total_return = final_value / initial_value - 1.0 if initial_value else 0.0

    periods = max(len(history) - 1, 1)
    annualized_return = (final_value / initial_value) ** (periods_per_year / periods) - 1.0
    annualized_vol = float(returns.std() * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
        if returns.std() > 0
        else 0.0
    )
    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = (
        float(returns.mean() / downside_std * np.sqrt(periods_per_year))
        if downside_std > 0
        else 0.0
    )

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_drawdown = abs(float(drawdown.min())) if len(drawdown) else 0.0
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    exposure = history["exposure"]
    return CandidateMetrics(
        initial_value=initial_value,
        final_value=final_value,
        total_return_pct=total_return * 100.0,
        annualized_return_pct=annualized_return * 100.0,
        annualized_volatility_pct=annualized_vol * 100.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_drawdown * 100.0,
        calmar_ratio=calmar,
        avg_turnover=float(history["turnover"].mean()),
        total_trades=int((history["turnover"] > 0).sum()),
        exposure_pct=float(exposure.mean() * 100.0),
    )
