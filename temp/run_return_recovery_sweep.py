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


def _scenario_with_tiers(name: str, tiers: list[dict[str, float]]) -> dict[str, object]:
    scenario = _rv_best_scenario(name)
    scenario.update(
        {
            "enable_drawdown_governor": True,
            "drawdown_exposure_tiers": tiers,
        }
    )
    return scenario


def _with_recovery_boost(
    scenario: dict[str, object],
    suffix: str,
    target: float,
    min_drawdown: float,
    max_worsening: float,
    max_realized_vol: float | None,
) -> dict[str, object]:
    out = dict(scenario)
    out["name"] = f"{scenario['name']}_{suffix}"
    out.update(
        {
            "enable_recovery_boost": True,
            "recovery_boost_target_long_fraction": target,
            "recovery_boost_min_drawdown": min_drawdown,
            "recovery_boost_max_worsening": max_worsening,
            "recovery_boost_min_green": 4,
        }
    )
    if max_realized_vol is not None:
        out["recovery_boost_max_realized_vol"] = max_realized_vol
    return out


def _anchors() -> dict[str, dict[str, object]]:
    hard_lite = _scenario_with_tiers(
        "rv336_dd_hard_lite_a",
        _custom_tiers(0.25, 0.975, 0.35, 0.800, 0.45, 0.600),
    )
    medium_plus = _scenario_with_tiers(
        "rv336_dd_medium_plus_b",
        _custom_tiers(0.30, 1.000, 0.40, 0.800, 0.48, 0.650),
    )
    strict = _scenario_with_tiers(
        "rv336_dd_soft",
        [
            {"drawdown": 0.00, "target_long_fraction": 1.075},
            {"drawdown": 0.35, "target_long_fraction": 1.000},
            {"drawdown": 0.45, "target_long_fraction": 0.900},
            {"drawdown": 0.52, "target_long_fraction": 0.800},
        ],
    )
    return {
        "checkpoint": _base_scenario(REFERENCE["name"]),
        "rv_best": _rv_best_scenario("rv336_tv0.018_min0.70"),
        "strict": strict,
        "medium_plus": medium_plus,
        "hard_lite": hard_lite,
    }


def _scenarios() -> list[dict[str, object]]:
    anchors = _anchors()
    scenarios = list(anchors.values())
    bases = [anchors["hard_lite"], anchors["medium_plus"]]
    for base in bases:
        for target, min_drawdown, max_worsening, max_vol in (
            (0.90, 0.20, -0.010, None),
            (1.00, 0.20, -0.010, None),
            (1.075, 0.20, -0.010, None),
            (1.15, 0.20, -0.010, None),
            (1.00, 0.30, -0.020, None),
            (1.075, 0.30, -0.020, None),
            (1.15, 0.30, -0.020, None),
            (1.075, 0.25, -0.030, 0.018),
            (1.15, 0.25, -0.030, 0.018),
        ):
            scenarios.append(
                _with_recovery_boost(
                    base,
                    (
                        f"rec_t{target:.3f}_dd{min_drawdown:.2f}"
                        f"_w{abs(max_worsening):.3f}"
                        f"{'_vol' if max_vol is not None else ''}"
                    ),
                    target=target,
                    min_drawdown=min_drawdown,
                    max_worsening=max_worsening,
                    max_realized_vol=max_vol,
                )
            )
    return scenarios


def _write_report(out_dir: Path, rows: pd.DataFrame) -> None:
    strict = rows[
        (rows["final_portfolio_value_usd"] >= REFERENCE["final_usd"])
        & (rows["max_drawdown_pct"] <= REFERENCE["max_drawdown_pct"])
    ].sort_values(["final_portfolio_value_usd", "sortino_ratio"], ascending=[False, False])
    above_500_low_dd = rows[
        (rows["final_sol_equiv"] >= 500.0)
        & (rows["max_drawdown_pct"] <= 55.0)
    ].sort_values(["final_portfolio_value_usd", "max_drawdown_pct"], ascending=[False, True])
    above_420 = rows[rows["final_sol_equiv"] >= 420.0].sort_values(
        ["max_drawdown_pct", "final_portfolio_value_usd"],
        ascending=[True, False],
    )
    top_usd = rows.sort_values(
        ["final_portfolio_value_usd", "max_drawdown_pct"],
        ascending=[False, True],
    ).head(12)
    lines = [
        "# Return Recovery Sweep",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Improve returns from the lower-drawdown tier candidates while keeping the original RS-gated checkpoint as the benchmark for return and Sortino.",
        "",
        "## Reference",
        "",
        "| Reference | Final USD | Final SOL | Max DD |",
        "| --- | ---: | ---: | ---: |",
        f"| `{REFERENCE['name']}` | ${REFERENCE['final_usd']:,.2f} | {REFERENCE['final_sol']:.3f} | {REFERENCE['max_drawdown_pct']:.3f}% |",
        "",
        "## Headline",
        "",
    ]
    if not strict.empty:
        row = strict.iloc[0]
        lines.append(
            f"- Best strict return/DD improvement: `{row['name']}` with ${float(row['final_portfolio_value_usd']):,.2f}, {float(row['final_sol_equiv']):.3f} SOL, {float(row['max_drawdown_pct']):.3f}% DD, Sortino {float(row['sortino_ratio']):.3f}."
        )
    if not above_500_low_dd.empty:
        row = above_500_low_dd.iloc[0]
        lines.append(
            f"- Best >=500 SOL and <=55% DD candidate: `{row['name']}` with ${float(row['final_portfolio_value_usd']):,.2f}, {float(row['final_sol_equiv']):.3f} SOL, {float(row['max_drawdown_pct']):.3f}% DD."
        )
    if not above_420.empty:
        row = above_420.iloc[0]
        lines.append(
            f"- Lowest DD above 420 SOL: `{row['name']}` with ${float(row['final_portfolio_value_usd']):,.2f}, {float(row['final_sol_equiv']):.3f} SOL, {float(row['max_drawdown_pct']):.3f}% DD."
        )
    lines.extend(
        [
            "",
            "## Top By USD",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | Sortino | USD Gap | DD Gap | Boost Share |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in top_usd.iterrows():
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
            "## Ranked By DD Above 420 SOL",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | Sortino | Boost Share |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in above_420.head(12).iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['sortino_ratio']):.3f} | "
            f"{float(row['recovery_boost_active_share']):.3f} |"
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
    out_dir = Path("reports") / f"return_recovery_sweep_{datetime.now():%Y%m%d_%H%M%S}"
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
            f"sortino={float(row['sortino_ratio']):.3f}\t"
            f"boost={float(row['recovery_boost_active_share']):.3f}",
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
