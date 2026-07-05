from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arblab.backtest.traffic_lights import asset_green_counts_by_bar
from temp.run_drawdown_tier_refinement import _custom_tiers
from temp.run_realized_vol_governor_sweep import (
    REFERENCE,
    SYMBOLS,
    TIMEFRAMES,
    _base_scenario,
    _price_data,
    _run_scenario,
    _rv_best_scenario,
)
from temp.run_return_recovery_sweep import _with_recovery_boost


def _scenario_with_tiers(name: str, tiers: list[dict[str, float]]) -> dict[str, object]:
    scenario = _rv_best_scenario(name)
    scenario.update(
        {
            "enable_drawdown_governor": True,
            "drawdown_exposure_tiers": tiers,
        }
    )
    return scenario


def _tier_bases() -> list[dict[str, object]]:
    return [
        _base_scenario(REFERENCE["name"]),
        _scenario_with_tiers(
            "rv336_dd_soft",
            [
                {"drawdown": 0.00, "target_long_fraction": 1.075},
                {"drawdown": 0.35, "target_long_fraction": 1.000},
                {"drawdown": 0.45, "target_long_fraction": 0.900},
                {"drawdown": 0.52, "target_long_fraction": 0.800},
            ],
        ),
        _scenario_with_tiers(
            "rv336_dd_medium_plus_a",
            _custom_tiers(0.28, 1.000, 0.38, 0.825, 0.48, 0.675),
        ),
        _scenario_with_tiers(
            "rv336_dd_medium_plus_b",
            _custom_tiers(0.30, 1.000, 0.40, 0.800, 0.48, 0.650),
        ),
        _scenario_with_tiers(
            "rv336_dd_medium_plus_c",
            _custom_tiers(0.32, 1.000, 0.42, 0.825, 0.50, 0.675),
        ),
        _scenario_with_tiers(
            "rv336_dd_medium_plus_d",
            _custom_tiers(0.30, 1.025, 0.40, 0.850, 0.50, 0.700),
        ),
        _scenario_with_tiers(
            "rv336_dd_medium_plus_e",
            _custom_tiers(0.32, 1.025, 0.42, 0.850, 0.50, 0.725),
        ),
    ]


def _scenarios() -> list[dict[str, object]]:
    scenarios = _tier_bases()
    for base in scenarios[2:]:
        for target, min_drawdown, max_worsening in (
            (1.075, 0.18, -0.005),
            (1.150, 0.18, -0.005),
            (1.200, 0.18, -0.005),
            (1.250, 0.18, -0.005),
            (1.150, 0.20, -0.010),
            (1.200, 0.20, -0.010),
            (1.250, 0.20, -0.010),
            (1.150, 0.25, -0.015),
            (1.200, 0.25, -0.015),
        ):
            scenarios.append(
                _with_recovery_boost(
                    base,
                    (
                        f"rec_t{target:.3f}_dd{min_drawdown:.2f}"
                        f"_w{abs(max_worsening):.3f}"
                    ),
                    target=target,
                    min_drawdown=min_drawdown,
                    max_worsening=max_worsening,
                    max_realized_vol=None,
                )
            )
    return scenarios


def _write_report(out_dir: Path, rows: pd.DataFrame) -> None:
    feasible = rows[
        (rows["final_sol_equiv"] >= 500.0)
        & (rows["max_drawdown_pct"] <= 55.0)
    ].sort_values(["final_portfolio_value_usd", "max_drawdown_pct"], ascending=[False, True])
    strict = rows[
        (rows["final_portfolio_value_usd"] >= REFERENCE["final_usd"])
        & (rows["max_drawdown_pct"] <= REFERENCE["max_drawdown_pct"])
    ].sort_values(["final_portfolio_value_usd", "max_drawdown_pct"], ascending=[False, True])
    sorted_rows = rows.sort_values(
        ["final_portfolio_value_usd", "max_drawdown_pct"],
        ascending=[False, True],
    )
    lines = [
        "# Return Recovery Refinement",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Refine the medium-plus drawdown tier family to recover more return while keeping drawdown below 55%.",
        "",
        "## Headline",
        "",
    ]
    if not strict.empty:
        row = strict.iloc[0]
        lines.append(
            f"- Best strict improvement versus checkpoint: `{row['name']}` with ${float(row['final_portfolio_value_usd']):,.2f}, {float(row['final_sol_equiv']):.3f} SOL, {float(row['max_drawdown_pct']):.3f}% DD, Sortino {float(row['sortino_ratio']):.3f}."
        )
    if not feasible.empty:
        row = feasible.iloc[0]
        lines.append(
            f"- Best >=500 SOL and <=55% DD: `{row['name']}` with ${float(row['final_portfolio_value_usd']):,.2f}, {float(row['final_sol_equiv']):.3f} SOL, {float(row['max_drawdown_pct']):.3f}% DD, Sortino {float(row['sortino_ratio']):.3f}."
        )
    lines.extend(
        [
            "",
            "## Top Results",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | Sortino | USD Gap | DD Gap | Boost Share |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in sorted_rows.head(20).iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['sortino_ratio']):.3f} | "
            f"${float(row['usd_gap_vs_reference']):,.2f} | "
            f"{float(row['dd_gap_vs_reference']):.3f}pp | "
            f"{float(row['recovery_boost_active_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Feasible Frontier",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | Sortino |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in feasible.head(20).iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['sortino_ratio']):.3f} |"
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
    out_dir = Path("reports") / f"return_recovery_refinement_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in _scenarios():
        row = _run_scenario(scenario, price_data, green_scores)
        rows.append(row)
        print(
            f"{row['name']}\t"
            f"usd=${float(row['final_portfolio_value_usd']):,.2f}\t"
            f"sol={float(row['final_sol_equiv']):.3f}\t"
            f"dd={float(row['max_drawdown_pct']):.3f}\t"
            f"sortino={float(row['sortino_ratio']):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, default=str)
    )
    _write_report(out_dir, summary)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
