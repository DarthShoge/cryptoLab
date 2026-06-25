"""Tests for reusable strategy report explorer helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arblab.backtest.report_explorer import (
    build_buy_hold_frame,
    build_composition_frame,
    final_composition_table,
    build_temperature_frame,
    build_timeline_frame,
    build_metric_frame,
    discover_report_dirs,
    history_label_options,
    history_selection_for_summary,
    load_price_cache,
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
            "collateral_SOL_value": values,
            "collateral_USDC_value": [0.0] * len(values),
            "debt_USDC_value": [value * 0.1 for value in values],
            "debt_ETH_value": [0.0] * len(values),
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


def test_history_selection_maps_comparison_aliases_to_summary_names():
    summary = pd.DataFrame(
        {
            "name": [
                "static_long1.50",
                "rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012",
            ]
        }
    )
    histories = {
        "highest_return": _history([100.0, 120.0]),
        "high_sortino": _history([100.0, 110.0]),
    }

    options = history_label_options(summary, histories)
    selected = history_selection_for_summary(
        [
            "static_long1.50",
            "rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012",
        ],
        options,
    )

    assert options["static_long1.50"] == "highest_return"
    assert options["rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012"] == "high_sortino"
    assert selected == ["highest_return", "high_sortino"]


def test_load_price_cache_reads_symbol_close_series(tmp_path: Path):
    cache = tmp_path / ".price_cache"
    cache.mkdir()
    (cache / "SOL_USDT_1h_test.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 00:00:00+00:00,1,1,1,10,100\n"
        "2024-01-01 01:00:00+00:00,1,1,1,12,100\n"
    )

    prices = load_price_cache(cache, ["SOL", "ETH"])

    assert list(prices) == ["SOL"]
    assert prices["SOL"].tolist() == [10, 12]
    assert prices["SOL"].index.tz is not None


def test_build_buy_hold_frame_uses_common_initial_portfolio_value():
    histories = {"strategy": _history([100.0, 110.0, 120.0])}
    prices = {
        "SOL": pd.Series(
            [10.0, 20.0, 15.0],
            index=histories["strategy"].index,
        ),
        "ETH": pd.Series(
            [50.0, 25.0, 100.0],
            index=histories["strategy"].index,
        ),
    }

    frame = build_buy_hold_frame(histories, prices)

    assert frame["Buy & Hold SOL"].tolist() == [100.0, 200.0, 150.0]
    assert frame["Buy & Hold ETH"].tolist() == [100.0, 50.0, 200.0]


def test_build_composition_frame_extracts_collateral_and_debt_values():
    history = _history([100.0, 120.0])

    collateral = build_composition_frame(history, "collateral")
    debt = build_composition_frame(history, "debt")

    assert list(collateral.columns) == ["SOL", "USDC"]
    assert collateral.iloc[-1]["SOL"] == 120.0
    assert list(debt.columns) == ["USDC", "ETH"]
    assert debt.iloc[-1]["USDC"] == 12.0


def test_final_composition_table_returns_latest_values_and_percentages():
    history = _history([100.0, 120.0])

    table = final_composition_table(history, "collateral")

    assert table.iloc[0]["asset"] == "SOL"
    assert table.iloc[0]["value_usd"] == 120.0
    assert table.iloc[0]["share_pct"] == 100.0


def test_final_composition_table_handles_zero_final_debt_values():
    history = _history([100.0, 0.0])

    table = final_composition_table(history, "debt")

    assert list(table.columns) == ["asset", "value_usd", "share_pct"]
    assert table.empty


def test_build_temperature_frame_combines_sol_price_and_net_exposure():
    history = _history([100.0, 120.0, 90.0])
    history["target_long_fraction"] = [1.5, 0.6, 0.2]
    history["target_short_fraction"] = [0.0, 0.1, 0.7]
    prices = {
        "SOL": pd.Series(
            [10.0, 12.0, 8.0],
            index=history.index,
        )
    }

    frame = build_temperature_frame(history, prices)

    assert frame["sol_price"].tolist() == [10.0, 12.0, 8.0]
    assert frame["net_temperature"].round(3).tolist() == [1.5, 0.5, -0.5]


def test_build_timeline_frame_extracts_trade_action_events():
    history = _history([100.0, 120.0, 90.0, 80.0])
    history["selected_long"] = ["SOL", "SOL", "ETH", "ETH"]
    history["target_long_fraction"] = [1.0, 1.5, 0.8, 0.8]
    history["target_short_fraction"] = [0.0, 0.0, 0.2, 0.0]
    history["recovery_boost_active"] = [False, False, True, True]
    history["health_factor"] = [2.0, 1.7, 1.4, 1.6]
    history["liquidation_count"] = [0, 0, 0, 1]
    prices = {
        "SOL": pd.Series(
            [10.0, 12.0, 8.0, 7.0],
            index=history.index,
        )
    }

    frame = build_timeline_frame(history, prices)

    assert set(frame["event_family"]) >= {
        "buy",
        "sell",
        "short",
        "cover",
        "rotation",
        "liquidation",
    }
    assert frame.iloc[0]["event_family"] == "start"
    assert "long target" in " ".join(frame["event"].astype(str))
    assert "health_factor" not in set(frame["event_family"])
    assert "drawdown" not in set(frame["event_family"])


def test_build_timeline_frame_includes_portfolio_snapshot():
    history = _history([100.0, 120.0])
    history["target_long_fraction"] = [1.0, 1.5]
    history["target_short_fraction"] = [0.0, 0.1]
    prices = {
        "SOL": pd.Series(
            [10.0, 12.0],
            index=history.index,
        )
    }

    frame = build_timeline_frame(history, prices)

    snapshot = str(frame.iloc[-1]["portfolio_snapshot"])
    assert "portfolio=$120.00" in snapshot
    assert "collateral SOL=$120.00" in snapshot
    assert "debt USDC=$12.00" in snapshot
