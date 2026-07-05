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
from temp.run_drawdown_containment_sweep import (
    END,
    START,
    _base_config,
    _event_count,
    _fmt,
    _table,
)
from temp.run_fast_break_crisis_gated_partial_fill_sweep import _drawdown_window
from temp.run_profit_lock_reserve_sweep import CURRENT_FAST_BREAK


def _row(name: str, group: str, overrides: dict[str, object], result, events, final_sol_price):
    final_positions = final_position_summary(result.history, final_sol_price)
    history = result.history
    recent_dd = _drawdown_window(history, "2024-01-01")
    cppi_usdc = (
        float(history["cppi_protected_usdc"].max())
        if "cppi_protected_usdc" in history
        else 0.0
    )
    active_pct = (
        float(history["in_cppi_exposure_cap"].astype(bool).mean() * 100.0)
        if "in_cppi_exposure_cap" in history
        else 0.0
    )
    return {
        "scenario": name,
        "group": group,
        "protect_pct": overrides.get("cppi_protect_pct", ""),
        "multiplier": overrides.get("cppi_cushion_multiplier", ""),
        "rebuy_min_green": overrides.get("cppi_rebuy_min_green", ""),
        "rebuy_fraction": overrides.get("cppi_rebuy_fraction", ""),
        "final_sol_equiv": final_positions["net"]["sol_equivalent"],
        "final_portfolio_value_usd": final_positions["net"]["portfolio_value_usd"],
        "final_sol_collateral": final_positions["collateral"][0]["Amount"],
        "final_usdc_collateral_value_usd": final_positions["collateral"][1][
            "Value USD"
        ],
        "final_eth_debt_value_usd": final_positions["debt"][0]["Value USD"],
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "post_2024_peak_to_trough_pct": recent_dd["drawdown_pct"],
        "cppi_active_pct": active_pct,
        "cppi_sell_events": _event_count(events, "cppi_exposure_cap_sell"),
        "cppi_rebuy_events": _event_count(events, "cppi_exposure_cap_rebuy"),
        "max_cppi_protected_usdc": cppi_usdc,
    }


def _report_rows(df: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    rows = []
    for _, row in df.head(limit).iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "protect": _fmt(row["protect_pct"]),
                "mult": _fmt(row["multiplier"]),
                "green": _fmt(row["rebuy_min_green"]),
                "SOL-eq": _fmt(float(row["final_sol_equiv"])),
                "USD": _fmt(float(row["final_portfolio_value_usd"])),
                "maxDD": _fmt(float(row["max_drawdown_pct"])),
                "2024DD": _fmt(float(row["post_2024_peak_to_trough_pct"])),
                "Sortino": _fmt(float(row["sortino_ratio"])),
                "minHF": _fmt(float(row["min_health_factor"])),
                "active%": _fmt(float(row["cppi_active_pct"])),
                "sell/rebuy": (
                    f"{int(row['cppi_sell_events'])}/"
                    f"{int(row['cppi_rebuy_events'])}"
                ),
            }
        )
    return rows


def _overrides(protect_pct: float, multiplier: float, rebuy_min_green: int):
    return {
        **CURRENT_FAST_BREAK,
        "enable_cppi_exposure_cap": True,
        "cppi_activation_gain": 1.50,
        "cppi_protect_pct": protect_pct,
        "cppi_cushion_multiplier": multiplier,
        "cppi_core_min_sol_collateral": 100.0,
        "cppi_exposure_buffer_pct": 0.05,
        "cppi_max_sell_fraction_per_bar": 0.10,
        "cppi_rebuy_fraction": 0.50,
        "cppi_rebuy_min_green": rebuy_min_green,
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/sol_supertrend_3y") / f"cppi_exposure_cap_sweep_{timestamp}"
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

    scenarios = [
        ("incumbent_stateful_best", "control", {}),
        ("v2_best_no_reserve", "control", CURRENT_FAST_BREAK),
    ]
    for protect_pct in (0.55, 0.65, 0.75):
        for multiplier in (1.5, 2.0):
            for rebuy_min_green in (3, 4):
                scenarios.append(
                    (
                        f"cppi_protect{protect_pct:.2f}_mult{multiplier:.1f}_green{rebuy_min_green}",
                        "cppi_exposure_cap",
                        _overrides(protect_pct, multiplier, rebuy_min_green),
                    )
                )

    rows = []
    summary = {"window": {"start": START, "end": END, "timeframe": "1h"}, "scenarios": {}}
    for idx, (name, group, overrides) in enumerate(scenarios, start=1):
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
        row = _row(name, group, overrides, result, events, final_sol_price)
        rows.append(row)
        summary["scenarios"][name] = {
            "group": group,
            "config_overrides": overrides,
            "metrics": row,
            "artifact_dir": str(scenario_dir),
        }

    comparison = pd.DataFrame(rows)
    by_sortino = comparison.sort_values(
        ["sortino_ratio", "final_sol_equiv", "max_drawdown_pct"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    by_drawdown = comparison.sort_values(
        ["max_drawdown_pct", "sortino_ratio", "final_sol_equiv"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    by_2024 = comparison.sort_values(
        ["post_2024_peak_to_trough_pct", "sortino_ratio", "final_sol_equiv"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    by_sol = comparison.sort_values(
        ["final_sol_equiv", "sortino_ratio", "max_drawdown_pct"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    by_sortino.to_csv(out_dir / "comparison.csv", index=False)
    by_drawdown.to_csv(out_dir / "top_by_drawdown.csv", index=False)
    by_2024.to_csv(out_dir / "top_by_2024_drawdown.csv", index=False)
    by_sol.to_csv(out_dir / "top_by_sol_equiv.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    control = comparison[comparison["scenario"] == "v2_best_no_reserve"].iloc[0]
    best_sortino = by_sortino.iloc[0]
    best_dd = by_drawdown.iloc[0]
    best_2024 = by_2024.iloc[0]
    best_sol = by_sol.iloc[0]
    columns = [
        "scenario",
        "protect",
        "mult",
        "green",
        "SOL-eq",
        "USD",
        "maxDD",
        "2024DD",
        "Sortino",
        "minHF",
        "active%",
        "sell/rebuy",
    ]
    report = f"""# CPPI Exposure Cap Sweep

Window: `{START}` to `{END}` at 1h bars from cached Binance SOL/ETH data.

## Question

Can a high-watermark CPPI/TIPP exposure cap reduce drawdown by continuously limiting SOL collateral value to a multiple of the cushion above a protected floor?

## Scenario Set

- Activation gain: `1.5x` initial portfolio value.
- Protected floor grid: `55%`, `65%`, `75%` of portfolio high-watermark.
- Cushion multiplier grid: `1.5`, `2.0`.
- Rebuy confirmation grid: `3` or `4` green votes with no bearish 1d/3d/1w votes.
- Core SOL minimum: `100 SOL`.
- Per-bar sell step cap: `10%` of current SOL collateral.
- Rebuy fraction: `50%` of permitted exposure gap.

## Ranking By Sortino

{_table(_report_rows(by_sortino, 12), columns)}

## Ranking By Max Drawdown

{_table(_report_rows(by_drawdown, 12), columns)}

## Ranking By 2024+ Peak-To-Trough

{_table(_report_rows(by_2024, 12), columns)}

## Ranking By Final SOL-Equivalent

{_table(_report_rows(by_sol, 12), columns)}

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | CPPI sell/rebuy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 no reserve | {control["scenario"]} | {_fmt(float(control["final_sol_equiv"]))} | {_fmt(float(control["final_portfolio_value_usd"]))} | {_fmt(float(control["max_drawdown_pct"]))} | {_fmt(float(control["post_2024_peak_to_trough_pct"]))} | {_fmt(float(control["sortino_ratio"]))} | {_fmt(float(control["min_health_factor"]))} | {int(control["cppi_sell_events"])}/{int(control["cppi_rebuy_events"])} |
| Best Sortino | {best_sortino["scenario"]} | {_fmt(float(best_sortino["final_sol_equiv"]))} | {_fmt(float(best_sortino["final_portfolio_value_usd"]))} | {_fmt(float(best_sortino["max_drawdown_pct"]))} | {_fmt(float(best_sortino["post_2024_peak_to_trough_pct"]))} | {_fmt(float(best_sortino["sortino_ratio"]))} | {_fmt(float(best_sortino["min_health_factor"]))} | {int(best_sortino["cppi_sell_events"])}/{int(best_sortino["cppi_rebuy_events"])} |
| Best Max DD | {best_dd["scenario"]} | {_fmt(float(best_dd["final_sol_equiv"]))} | {_fmt(float(best_dd["final_portfolio_value_usd"]))} | {_fmt(float(best_dd["max_drawdown_pct"]))} | {_fmt(float(best_dd["post_2024_peak_to_trough_pct"]))} | {_fmt(float(best_dd["sortino_ratio"]))} | {_fmt(float(best_dd["min_health_factor"]))} | {int(best_dd["cppi_sell_events"])}/{int(best_dd["cppi_rebuy_events"])} |
| Best 2024 DD | {best_2024["scenario"]} | {_fmt(float(best_2024["final_sol_equiv"]))} | {_fmt(float(best_2024["final_portfolio_value_usd"]))} | {_fmt(float(best_2024["max_drawdown_pct"]))} | {_fmt(float(best_2024["post_2024_peak_to_trough_pct"]))} | {_fmt(float(best_2024["sortino_ratio"]))} | {_fmt(float(best_2024["min_health_factor"]))} | {int(best_2024["cppi_sell_events"])}/{int(best_2024["cppi_rebuy_events"])} |
| Best SOL-eq | {best_sol["scenario"]} | {_fmt(float(best_sol["final_sol_equiv"]))} | {_fmt(float(best_sol["final_portfolio_value_usd"]))} | {_fmt(float(best_sol["max_drawdown_pct"]))} | {_fmt(float(best_sol["post_2024_peak_to_trough_pct"]))} | {_fmt(float(best_sol["sortino_ratio"]))} | {_fmt(float(best_sol["min_health_factor"]))} | {int(best_sol["cppi_sell_events"])}/{int(best_sol["cppi_rebuy_events"])} |

## Readout

TODO: Fill after inspecting rankings and event logs.

## Artifacts

- Comparison CSV: `{out_dir / "comparison.csv"}`
- Top-by-drawdown CSV: `{out_dir / "top_by_drawdown.csv"}`
- Top-by-2024-drawdown CSV: `{out_dir / "top_by_2024_drawdown.csv"}`
- Top-by-SOL-equivalent CSV: `{out_dir / "top_by_sol_equiv.csv"}`
- Summary JSON: `{out_dir / "summary.json"}`
- Scenario folders: `{out_dir}`
"""
    (out_dir / "report.md").write_text(report)
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
