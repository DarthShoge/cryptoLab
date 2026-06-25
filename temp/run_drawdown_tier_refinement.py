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
from temp.run_realized_vol_governor_sweep import (
    REFERENCE,
    SYMBOLS,
    TIMEFRAMES,
    _base_scenario,
    _drawdown_tiers,
    _price_data,
    _run_scenario,
    _rv_best_scenario,
)


def _custom_tiers(
    first_dd: float,
    first_target: float,
    second_dd: float,
    second_target: float,
    third_dd: float,
    third_target: float,
) -> list[dict[str, float]]:
    return [
        {"drawdown": 0.00, "target_long_fraction": 1.075},
        {"drawdown": first_dd, "target_long_fraction": first_target},
        {"drawdown": second_dd, "target_long_fraction": second_target},
        {"drawdown": third_dd, "target_long_fraction": third_target},
    ]


def _scenario_with_tiers(name: str, tiers: list[dict[str, float]]) -> dict[str, object]:
    scenario = _rv_best_scenario(name)
    scenario.update(
        {
            "enable_drawdown_governor": True,
            "drawdown_exposure_tiers": tiers,
        }
    )
    return scenario


def _scenarios() -> list[dict[str, object]]:
    scenarios = [
        _base_scenario(REFERENCE["name"]),
        _rv_best_scenario("rv336_tv0.018_min0.70"),
    ]
    for profile in ("soft", "medium", "hard"):
        scenarios.append(
            _scenario_with_tiers(
                f"rv336_tv0.018_min0.70_dd_{profile}",
                _drawdown_tiers(profile),
            )
        )
    for name, tiers in {
        "rv336_dd_hard_lite_a": _custom_tiers(0.25, 0.975, 0.35, 0.800, 0.45, 0.600),
        "rv336_dd_hard_lite_b": _custom_tiers(0.25, 1.000, 0.35, 0.825, 0.45, 0.625),
        "rv336_dd_hard_lite_c": _custom_tiers(0.28, 0.975, 0.38, 0.800, 0.48, 0.600),
        "rv336_dd_hard_lite_d": _custom_tiers(0.30, 0.950, 0.40, 0.775, 0.50, 0.600),
        "rv336_dd_medium_plus_a": _custom_tiers(0.28, 1.000, 0.38, 0.825, 0.48, 0.675),
        "rv336_dd_medium_plus_b": _custom_tiers(0.30, 1.000, 0.40, 0.800, 0.48, 0.650),
        "rv336_dd_medium_plus_c": _custom_tiers(0.32, 1.000, 0.42, 0.825, 0.50, 0.675),
    }.items():
        scenarios.append(_scenario_with_tiers(name, tiers))
    return scenarios


def _write_report(out_dir: Path, rows: pd.DataFrame) -> None:
    best_strict = rows[
        (rows["final_portfolio_value_usd"] >= REFERENCE["final_usd"])
        & (rows["max_drawdown_pct"] <= REFERENCE["max_drawdown_pct"])
    ].sort_values(["final_portfolio_value_usd", "max_drawdown_pct"], ascending=[False, True])
    above_420 = rows[rows["final_sol_equiv"] >= 420.0].sort_values(
        ["max_drawdown_pct", "final_portfolio_value_usd"],
        ascending=[True, False],
    )
    lines = [
        "# Drawdown Tier Refinement",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Refine the high-return 336-hour realized-volatility profile with drawdown tiers, looking for the lowest DD while preserving at least 420 SOL and checking for strict improvement over the checkpoint.",
        "",
        "## Headline",
        "",
    ]
    if not best_strict.empty:
        row = best_strict.iloc[0]
        lines.append(
            f"- Best strict improvement: `{row['name']}` with ${float(row['final_portfolio_value_usd']):,.2f}, {float(row['final_sol_equiv']):.3f} SOL, and {float(row['max_drawdown_pct']):.3f}% DD."
        )
    if not above_420.empty:
        row = above_420.iloc[0]
        lines.append(
            f"- Lowest DD above 420 SOL: `{row['name']}` with ${float(row['final_portfolio_value_usd']):,.2f}, {float(row['final_sol_equiv']):.3f} SOL, and {float(row['max_drawdown_pct']):.3f}% DD."
        )
    lines.extend(
        [
            "",
            "## Ranked Results",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | USD Gap | SOL Gap | DD Gap | Avg Long | Capped Share |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    ranked = rows.sort_values(["max_drawdown_pct", "final_portfolio_value_usd"], ascending=[True, False])
    for _, row in ranked.iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"${float(row['usd_gap_vs_reference']):,.2f} | "
            f"{float(row['sol_gap_vs_reference']):.3f} | "
            f"{float(row['dd_gap_vs_reference']):.3f}pp | "
            f"{float(row['avg_target_long_fraction']):.3f} | "
            f"{float(row['vol_capped_share']):.3f} |"
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
    out_dir = Path("reports") / f"drawdown_tier_refinement_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in _scenarios():
        row = _run_scenario(scenario, price_data, green_scores)
        rows.append(row)
        print(
            f"{row['name']}\t"
            f"usd=${float(row['final_portfolio_value_usd']):,.2f}\t"
            f"sol={float(row['final_sol_equiv']):.3f}\t"
            f"dd={float(row['max_drawdown_pct']):.3f}",
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
