"""Tests for reusable strategy report explorer helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arblab.backtest.report_explorer import (
    build_metric_frame,
    discover_report_dirs,
    load_report_bundle,
    max_drawdown_series,
    slice_regime,
)


def _history(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "portfolio_value": values,
            "health_factor": [2.0] * len(values),
            "target_long_fraction": [1.0] * len(values),
            "target_short_fraction": [0.0] * len(values),
        },
        index=index,
    )


def test_discover_report_dirs_requires_summary_or_report(tmp_path: Path):
    valid = tmp_path / "strategy_comparison_20260625_180556"
    valid.mkdir()
    (valid / "summary.csv").write_text("name,final_portfolio_value_usd\nx,1\n")
    invalid = tmp_path / "raw_files"
    invalid.mkdir()
    (invalid / "notes.txt").write_text("ignore me\n")

    dirs = discover_report_dirs(tmp_path)

    assert dirs == [valid]


def test_load_report_bundle_reads_summary_regimes_markdown_and_histories(tmp_path: Path):
    report = tmp_path / "report_a"
    report.mkdir()
    (report / "summary.csv").write_text("name,final_portfolio_value_usd\nA,100\n")
    (report / "regime_summary.csv").write_text("name,regime,return_pct\nA,full,10\n")
    (report / "report.md").write_text("# Report\n")
    _history([100.0, 120.0]).to_csv(report / "A_history.csv")

    bundle = load_report_bundle(report)

    assert list(bundle.summary["name"]) == ["A"]
    assert list(bundle.regimes["regime"]) == ["full"]
    assert bundle.markdown == "# Report\n"
    assert list(bundle.histories) == ["A"]
    assert bundle.histories["A"].index.tz is not None


def test_max_drawdown_series_returns_positive_percent_drawdown():
    drawdown = max_drawdown_series(pd.Series([100.0, 120.0, 90.0, 150.0]))

    assert drawdown.round(3).tolist() == [0.0, 0.0, 25.0, 0.0]


def test_slice_regime_uses_named_regime_or_custom_dates():
    history = _history([100.0, 90.0, 110.0, 120.0])

    named = slice_regime(history, "2024-01-01", "2024-01-01 01:00:00+00:00")
    custom = slice_regime(history, None, None)

    assert named["portfolio_value"].tolist() == [100.0, 90.0]
    assert len(custom) == 4


def test_build_metric_frame_combines_histories_with_normalized_value_and_drawdown():
    histories = {
        "A": _history([100.0, 120.0, 90.0]),
        "B": _history([50.0, 100.0, 75.0]),
    }

    frame = build_metric_frame(histories, "portfolio_value")

    assert set(frame.columns) == {"A", "B"}
    assert frame.iloc[0]["A"] == 100.0
    normalized = build_metric_frame(histories, "normalized_portfolio_value")
    assert normalized.iloc[-1]["A"] == 90.0
    assert normalized.iloc[-1]["B"] == 150.0
    drawdown = build_metric_frame(histories, "drawdown_pct")
    assert round(float(drawdown.iloc[-1]["A"]), 3) == 25.0
