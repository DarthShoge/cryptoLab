from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from arblab.backtest.app_helpers import (
    SOL_SUPERTREND_SHORT_STRATEGY,
    build_price_configs,
    final_position_summary,
    run_selected_backtest,
)
from arblab.backtest.data import fetch_ohlcv
from arblab.backtest.market import MarketParams
from temp.run_drawdown_containment_sweep import END, START, _base_config, _event_count, _fmt, _table
from temp.run_fast_break_crisis_gated_partial_fill_sweep import _drawdown_window
from temp.run_profit_lock_reserve_sweep import CURRENT_FAST_BREAK


def _row(name, overrides, result, events, final_sol_price):
    final_positions = final_position_summary(result.history, final_sol_price)
    recent_dd = _drawdown_window(result.history, "2024-01-01")
    partial_fill_usd = (
        float(result.history["fast_break_partial_fill_added_usd"].max())
        if "fast_break_partial_fill_added_usd" in result.history
        else 0.0
    )
    return {
        "scenario": name,
        "requires_crisis": overrides.get("fast_break_partial_fill_requires_crisis", ""),
        "max_green": overrides.get("fast_break_partial_fill_max_green", ""),
        "partial_fill_min_hf": overrides.get("fast_break_partial_fill_min_hf", ""),
        "partial_fill_budget_pct": overrides.get("fast_break_partial_fill_budget_pct", ""),
        "final_sol_equiv": final_positions["net"]["sol_equivalent"],
        "final_portfolio_value_usd": final_positions["net"]["portfolio_value_usd"],
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_peak_to_trough_pct": recent_dd["drawdown_pct"],
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "fast_break_partial_fill_events": _event_count(events, "fast_break_partial_fill"),
        "max_fast_break_partial_fill_added_usd": partial_fill_usd,
    }


def _report_rows(df: pd.DataFrame):
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "crisis": row["requires_crisis"],
                "max_green": row["max_green"],
                "SOL-eq": _fmt(float(row["final_sol_equiv"])),
                "USD": _fmt(float(row["final_portfolio_value_usd"])),
                "maxDD": _fmt(float(row["max_drawdown_pct"])),
                "2024DD": _fmt(float(row["post_2024_peak_to_trough_pct"])),
                "Sortino": _fmt(float(row["sortino_ratio"])),
                "minHF": _fmt(float(row["min_health_factor"])),
                "events": int(row["fast_break_partial_fill_events"]),
                "maxPF$": _fmt(float(row["max_fast_break_partial_fill_added_usd"])),
            }
        )
    return rows


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/sol_supertrend_3y") / (
        f"confirmed_crash_partial_fill_sweep_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    price_data = fetch_ohlcv(
        symbols=build_price_configs(SOL_SUPERTREND_SHORT_STRATEGY, "SOL"),
        timeframe="1h",
        start=START,
        end=END,
        exchange_id="binance",
        use_cache=True,
    )
    final_sol_price = float(price_data[("SOL", "close")].iloc[-1])
    base_config = _base_config(price_data)
    market_params = MarketParams.kamino_defaults()

    crisis_partial = {
        **CURRENT_FAST_BREAK,
        "enable_fast_break_partial_fill": True,
        "fast_break_partial_fill_requires_crisis": True,
        "fast_break_partial_fill_min_hf": 2.50,
        "fast_break_partial_fill_budget_pct": 0.25,
    }
    confirmed = {
        **crisis_partial,
        "fast_break_partial_fill_max_green": 1,
    }
    scenarios = [
        ("v2_best_no_partial_fill", CURRENT_FAST_BREAK),
        ("crisis_gated_partial_fill", crisis_partial),
        ("confirmed_crash_partial_fill", confirmed),
    ]

    rows = []
    summary = {"window": {"start": START, "end": END, "timeframe": "1h"}, "scenarios": {}}
    for idx, (name, overrides) in enumerate(scenarios, start=1):
        print(f"[{idx}/{len(scenarios)}] {name}", flush=True)
        result = run_selected_backtest(
            SOL_SUPERTREND_SHORT_STRATEGY,
            price_data,
            {**base_config, **overrides},
            market_params,
        )
        scenario_dir = out_dir / name
        scenario_dir.mkdir()
        result.history.to_csv(scenario_dir / "history.csv")
        events = pd.DataFrame(result.strategy_events)
        events.to_csv(scenario_dir / "strategy_events.csv", index=False)
        row = _row(name, overrides, result, events, final_sol_price)
        rows.append(row)
        summary["scenarios"][name] = {
            "config_overrides": overrides,
            "metrics": row,
            "artifact_dir": str(scenario_dir),
        }

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    ranked = comparison.sort_values(
        ["sortino_ratio", "final_sol_equiv"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked.to_csv(out_dir / "ranked.csv", index=False)
    columns = [
        "scenario",
        "crisis",
        "max_green",
        "SOL-eq",
        "USD",
        "maxDD",
        "2024DD",
        "Sortino",
        "minHF",
        "events",
        "maxPF$",
    ]
    report = f"""# Confirmed-Crash Partial Fill Sweep

Window: `{START}` to `{END}` at 1h bars from cached Binance SOL/ETH data.

## Question

Does requiring a confirmed crash state (`crisis=true` and `green <= 1`) rescue fast-break partial fills by preventing early-cycle over-hedging?

## Ranking

{_table(_report_rows(ranked), columns)}

## Readout

TODO: Fill after inspecting rankings and event logs.

## Artifacts

- Comparison CSV: `{out_dir / "comparison.csv"}`
- Ranked CSV: `{out_dir / "ranked.csv"}`
- Summary JSON: `{out_dir / "summary.json"}`
- Scenario folders: `{out_dir}`
"""
    (out_dir / "report.md").write_text(report)
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
