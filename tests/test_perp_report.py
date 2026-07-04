from __future__ import annotations

import json

import pandas as pd
import pytest

from arblab.perps.report import summarize_perp_history, write_perp_report


def _history() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "btc_close": [100.0, 105.0, 102.0, 108.0],
            "equity": [10_000.0, 10_500.0, 10_100.0, 10_800.0],
            "position_notional": [10_000.0, 10_500.0, 0.0, -10_000.0],
            "exposure": [1.0, 1.0, 0.0, -0.9259],
            "traded_notional": [10_000.0, 0.0, 10_500.0, 10_000.0],
            "fees": [6.0, 0.0, 6.3, 6.0],
            "funding": [0.0, 1.0, 0.0, 1.0],
            "exit_reason": ["", "", "take_profit", ""],
        },
        index=index,
    )


def _signals() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "btc_close": [100.0, 105.0, 102.0, 108.0],
            "signal": [1.0, 1.0, 0.0, -1.0],
            "raw_score": [1.0, 1.0, 0.0, -1.0],
        },
        index=index,
    )


def test_summarize_perp_history_includes_required_metrics():
    summary = summarize_perp_history(_history(), starting_equity=10_000.0)

    assert {
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe",
        "sortino",
        "turnover",
        "trade_count",
        "max_abs_exposure",
        "average_abs_exposure",
        "fee_drag",
        "funding_drag",
        "stop_count",
        "take_profit_count",
        "percent_time_long",
        "percent_time_short",
        "percent_time_flat",
        "btc_buy_and_hold_return",
    }.issubset(summary)
    assert summary["total_return"] == pytest.approx(0.08)
    assert summary["take_profit_count"] == 1
    assert summary["trade_count"] == 3
    assert summary["max_abs_exposure"] == pytest.approx(1.0)
    assert summary["average_abs_exposure"] == pytest.approx(0.731475)


def test_write_perp_report_creates_required_artifacts(tmp_path):
    output_dir = write_perp_report(
        output_dir=tmp_path / "btc_report",
        signals=_signals(),
        history=_history(),
        starting_equity=10_000.0,
    )

    assert (output_dir / "signals.csv").exists()
    assert (output_dir / "history.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()

    summary = json.loads((output_dir / "summary.json").read_text())

    assert summary["final_equity"] == 10_800.0
    assert summary["trade_count"] == 3
    assert "BTC Pure Perp Signal Backtest" in (output_dir / "report.md").read_text()
