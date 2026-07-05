from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temp.run_adaptive_hedge_gate_sweep import (
    LIGHT_FLOORS,
    REFERENCE,
    SYMBOLS,
    TIMEFRAMES,
    _drawdown_window,
    _max_drawdown_pct,
    _price_data,
    _run_scenario,
    _selected_rows,
    _signal_tiers,
)

import json
from datetime import datetime

import pandas as pd

from arblab.backtest.traffic_lights import asset_green_counts_by_bar


def _base_scenario(name: str) -> dict[str, object]:
    return {
        "name": name,
        "enable_protective_short_hedge": True,
        "protective_short_symbol": "ETH",
        "protective_hedge_floors": LIGHT_FLOORS,
    }


def _scenarios() -> list[dict[str, object]]:
    scenarios = [_base_scenario("static_light_hedge")]
    for min_drawdown, min_worsening in (
        (0.03, 0.0030),
        (0.03, 0.0050),
        (0.03, 0.0075),
        (0.05, 0.0030),
        (0.05, 0.0040),
        (0.05, 0.0050),
        (0.05, 0.0060),
        (0.05, 0.0075),
        (0.07, 0.0030),
        (0.07, 0.0050),
        (0.07, 0.0075),
        (0.10, 0.0030),
        (0.10, 0.0050),
        (0.10, 0.0075),
    ):
        scenario = _base_scenario(
            f"dd_refine_dd{min_drawdown:.3f}_w{min_worsening:.4f}"
        )
        scenario.update(
            {
                "enable_drawdown_slope_hedge_gate": True,
                "drawdown_slope_min_drawdown": min_drawdown,
                "drawdown_slope_min_worsening": min_worsening,
            }
        )
        scenarios.append(scenario)

    for lookback, underperformance in (
        (12, -0.005),
        (12, 0.000),
        (12, 0.005),
        (18, -0.005),
        (18, 0.000),
        (18, 0.005),
        (24, -0.005),
        (24, 0.000),
        (24, 0.005),
        (36, 0.000),
        (48, 0.000),
    ):
        scenario = _base_scenario(
            f"rs_refine_lb{lookback}_u{underperformance:.3f}"
        )
        scenario.update(
            {
                "enable_relative_strength_hedge_gate": True,
                "relative_strength_lookback_bars": lookback,
                "relative_strength_min_underperformance": underperformance,
            }
        )
        scenarios.append(scenario)
    return scenarios


def _write_report(out_dir: Path, rows: list[dict[str, object]]) -> None:
    selected = _selected_rows(rows)
    strict = selected.get("Strict reference improvement")
    lines = [
        "# Adaptive Protective Hedge Gate Refinement",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Refine the useful single-gate region from the first adaptive hedge sweep.",
        "",
        "## Reference",
        "",
        "| Reference | Final USD | Final SOL | Max DD |",
        "| --- | ---: | ---: | ---: |",
        f"| `{REFERENCE['name']}` | ${REFERENCE['final_usd']:,.2f} | {REFERENCE['final_sol']:.3f} | {REFERENCE['max_drawdown_pct']:.3f}% |",
        "",
        "## Winners",
        "",
        "| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Hedge | Gate Active |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in selected.items():
        lines.append(
            f"| {label} | `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['post_2024_drawdown_pct']):.3f}% | "
            f"{float(row['avg_target_short_fraction']):.3f}x | "
            f"{float(row['hedge_gate_active_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            (
                "- No refined adaptive gate beat the static light hedge on both final USD and max DD."
                if strict is None
                else (
                    f"- Strict reference improvement: `{strict['name']}` with "
                    f"${float(strict['final_portfolio_value_usd']):,.2f}, "
                    f"{float(strict['final_sol_equiv']):.3f} SOL, and "
                    f"{float(strict['max_drawdown_pct']):.3f}% max DD."
                )
            ),
            "",
            "## Top Variants By Final USD",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | Avg Hedge | Gate Active |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: float(item["final_portfolio_value_usd"]), reverse=True)[:10]:
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['avg_target_short_fraction']):.3f}x | "
            f"{float(row['hedge_gate_active_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{out_dir / 'summary.csv'}`",
            f"- Summary JSON: `{out_dir / 'summary.json'}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    price_data = _price_data()
    green_scores = asset_green_counts_by_bar(
        price_data,
        symbols=SYMBOLS,
        atr_period=10,
        multiplier=3.0,
        timeframes=TIMEFRAMES,
    )
    out_dir = Path("reports") / f"adaptive_hedge_gate_refinement_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for scenario in _scenarios():
        row = _run_scenario(scenario, price_data, green_scores)
        rows.append(row)
        print(
            f"{row['name']}\t"
            f"usd=${float(row['final_portfolio_value_usd']):,.2f}\t"
            f"sol={float(row['final_sol_equiv']):.3f}\t"
            f"dd={float(row['max_drawdown_pct']):.3f}\t"
            f"avg_short={float(row['avg_target_short_fraction']):.3f}\t"
            f"gate={float(row['hedge_gate_active_share']):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values(
        ["final_portfolio_value_usd", "max_drawdown_pct"],
        ascending=[False, True],
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    _write_report(out_dir, rows)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
