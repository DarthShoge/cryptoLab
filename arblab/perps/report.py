"""Reporting helpers for BTC perp signal backtests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_perp_history(
    history: pd.DataFrame,
    starting_equity: float,
) -> dict[str, Any]:
    """Compute report summary metrics from a perp account history."""
    if history.empty:
        raise ValueError("history is required")

    final_equity = float(history["equity"].iloc[-1])
    returns = history["equity"].astype(float).pct_change().dropna()
    periods_per_year = _periods_per_year(history.index)
    total_return = final_equity / starting_equity - 1.0 if starting_equity else 0.0
    annualized_return = _annualized_return(total_return, len(history), periods_per_year)

    rolling_peak = history["equity"].astype(float).cummax()
    drawdown = history["equity"].astype(float) / rolling_peak - 1.0
    max_drawdown = abs(float(drawdown.min())) if not drawdown.empty else 0.0

    exposure = history["exposure"].astype(float)
    long_count = int((exposure > 0.0).sum())
    short_count = int((exposure < 0.0).sum())
    flat_count = int((exposure == 0.0).sum())
    total_count = len(history)

    btc_start = float(history["btc_close"].iloc[0])
    btc_end = float(history["btc_close"].iloc[-1])

    return {
        "starting_equity": float(starting_equity),
        "final_equity": final_equity,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": _sharpe(returns, periods_per_year),
        "sortino": _sortino(returns, periods_per_year),
        "turnover": float(history["traded_notional"].sum()),
        "fee_drag": float(history["fees"].sum()),
        "funding_drag": float(history["funding"].sum()),
        "stop_count": int((history["exit_reason"] == "stop_loss").sum()),
        "take_profit_count": int((history["exit_reason"] == "take_profit").sum()),
        "percent_time_long": long_count / total_count if total_count else 0.0,
        "percent_time_short": short_count / total_count if total_count else 0.0,
        "percent_time_flat": flat_count / total_count if total_count else 0.0,
        "btc_buy_and_hold_return": btc_end / btc_start - 1.0 if btc_start else 0.0,
    }


def write_perp_report(
    output_dir: str | Path,
    signals: pd.DataFrame,
    history: pd.DataFrame,
    starting_equity: float,
) -> Path:
    """Write signals, history, summary JSON, and Markdown report artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    signals.to_csv(output_path / "signals.csv")
    history.to_csv(output_path / "history.csv")

    summary = summarize_perp_history(history, starting_equity=starting_equity)
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "report.md").write_text(_markdown_report(summary), encoding="utf-8")
    return output_path


def _periods_per_year(index: pd.Index) -> float:
    if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
        return 365.0
    seconds = index.to_series().diff().dt.total_seconds().median()
    if not seconds or pd.isna(seconds) or seconds <= 0.0:
        return 365.0
    return 365.25 * 24.0 * 60.0 * 60.0 / float(seconds)


def _annualized_return(
    total_return: float,
    periods: int,
    periods_per_year: float,
) -> float:
    if periods <= 1 or total_return <= -1.0:
        return 0.0
    years = periods / periods_per_year
    if years <= 0.0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float(returns.mean()) / std * periods_per_year**0.5


def _sortino(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0.0]
    downside_std = float(downside.std(ddof=0))
    if downside_std == 0.0:
        return 0.0
    return float(returns.mean()) / downside_std * periods_per_year**0.5


def _markdown_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BTC Pure Perp Signal Backtest",
            "",
            f"- Final equity: ${summary['final_equity']:,.2f}",
            f"- Total return: {summary['total_return']:.2%}",
            f"- BTC buy-and-hold return: {summary['btc_buy_and_hold_return']:.2%}",
            f"- Max drawdown: {summary['max_drawdown']:.2%}",
            f"- Sharpe: {summary['sharpe']:.2f}",
            f"- Sortino: {summary['sortino']:.2f}",
            f"- Turnover: ${summary['turnover']:,.2f}",
            f"- Fees: ${summary['fee_drag']:,.2f}",
            f"- Funding: ${summary['funding_drag']:,.2f}",
            f"- Stop losses: {summary['stop_count']}",
            f"- Take profits: {summary['take_profit_count']}",
            "",
        ]
    )
