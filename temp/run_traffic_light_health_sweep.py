from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from arblab.backtest.app_helpers import (
    SOL_SUPERTREND_SHORT_STRATEGY,
    build_price_configs,
    run_selected_backtest,
)
from arblab.backtest.data import fetch_ohlcv
from arblab.backtest.market import MarketParams
from temp.run_drawdown_containment_sweep import END, START, _base_config, _fmt, _table
from temp.run_profit_lock_reserve_sweep import CURRENT_FAST_BREAK
from temp.run_traffic_light_governor_sweep import (
    MIN_HF_GATE,
    NEAR_SOL_GATE,
    SOL_GATE,
    _format_rows,
    _scenario_row,
    _traffic_light_overrides,
)


PROMISING_SHAPES = (
    (0.75, 1.25, 1.25, 3),
    (0.50, 1.00, 1.25, 3),
    (0.50, 1.25, 1.25, 3),
    (0.35, 1.25, 1.25, 3),
    (0.50, 0.75, 1.25, 4),
    (0.75, 0.75, 1.25, 4),
    (0.75, 0.75, 1.00, 4),
)


def _scenarios() -> list[tuple[str, str, dict[str, object]]]:
    scenarios: list[tuple[str, str, dict[str, object]]] = [
        ("v2_best_no_overlays", "control", CURRENT_FAST_BREAK),
    ]
    for yellow, orange, red, reinvest_green in PROMISING_SHAPES:
        for add_min_hf in (2.50, 3.00, 3.50):
            overrides = _traffic_light_overrides(
                yellow,
                orange,
                red,
                reinvest_green,
                4,
            )
            overrides["traffic_light_add_min_hf"] = add_min_hf
            scenarios.append(
                (
                    (
                        f"tlg_y{yellow:.2f}_o{orange:.2f}_r{red:.2f}"
                        f"_reinv{reinvest_green}_addhf{add_min_hf:.2f}"
                    ),
                    "traffic_light_health",
                    overrides,
                )
            )
    return scenarios


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/sol_supertrend_3y") / (
        f"traffic_light_health_{timestamp}"
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
    scenarios = _scenarios()

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "window": {"start": START, "end": END, "timeframe": "1h"},
        "constraints": {
            "final_sol_equiv_min_exclusive": SOL_GATE,
            "near_final_sol_equiv_min": NEAR_SOL_GATE,
            "min_health_factor_min": MIN_HF_GATE,
        },
        "scenarios": {},
    }

    for idx, (name, family, overrides) in enumerate(scenarios, start=1):
        print(f"[{idx}/{len(scenarios)}] {name}", flush=True)
        result = run_selected_backtest(
            SOL_SUPERTREND_SHORT_STRATEGY,
            price_data,
            {**base_config, **overrides},
            market_params,
        )
        events = pd.DataFrame(result.strategy_events)
        scenario_dir = out_dir / name
        scenario_dir.mkdir()
        result.history.to_csv(scenario_dir / "history.csv")
        events.to_csv(scenario_dir / "strategy_events.csv", index=False)

        row = _scenario_row(name, family, overrides, result, events, final_sol_price)
        row["traffic_light_add_min_hf"] = overrides.get("traffic_light_add_min_hf", "")
        rows.append(row)
        summary["scenarios"][name] = {
            "family": family,
            "config_overrides": overrides,
            "metrics": row,
            "artifact_dir": str(scenario_dir),
        }

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    valid = comparison[comparison["valid"]].sort_values(
        ["max_drawdown_pct", "post_2024_peak_to_trough_pct", "final_sol_equiv"],
        ascending=[True, True, False],
    )
    near_valid = comparison[comparison["near_valid"]].sort_values(
        ["max_drawdown_pct", "post_2024_peak_to_trough_pct", "final_sol_equiv"],
        ascending=[True, True, False],
    )
    all_by_drawdown = comparison.sort_values(
        ["max_drawdown_pct", "final_sol_equiv"],
        ascending=[True, False],
    )
    all_by_sol = comparison.sort_values(
        ["final_sol_equiv", "max_drawdown_pct"],
        ascending=[False, True],
    )
    valid.to_csv(out_dir / "valid_configs.csv", index=False)
    near_valid.to_csv(out_dir / "near_valid_configs.csv", index=False)
    all_by_drawdown.to_csv(out_dir / "all_by_drawdown.csv", index=False)
    all_by_sol.to_csv(out_dir / "all_by_sol.csv", index=False)

    family_best_rows = []
    for family, group in comparison.groupby("family"):
        family_best = group.sort_values(
            ["valid", "near_valid", "max_drawdown_pct", "final_sol_equiv"],
            ascending=[False, False, True, False],
        ).iloc[0]
        family_best_rows.append(family_best.to_dict())
    family_best = pd.DataFrame(family_best_rows).sort_values(
        ["valid", "near_valid", "max_drawdown_pct", "final_sol_equiv"],
        ascending=[False, False, True, False],
    )
    family_best.to_csv(out_dir / "family_best.csv", index=False)

    failure_counts = defaultdict(int)
    for _, row in comparison.iterrows():
        if not bool(row["sol_gate_pass"]):
            failure_counts["missed_420_sol_gate"] += 1
        if not bool(row["hf_gate_pass"]):
            failure_counts["missed_min_hf_gate"] += 1
        if bool(row["valid"]):
            failure_counts["valid"] += 1
    summary["result_counts"] = {
        "total": len(comparison),
        "valid": int(comparison["valid"].sum()),
        "near_valid": int(comparison["near_valid"].sum()),
        **dict(failure_counts),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )

    columns = [
        "scenario",
        "family",
        "SOL-eq",
        "USD",
        "maxDD",
        "2024DD",
        "halfRec",
        "Sortino",
        "minHF",
        "valid",
        "near",
    ]
    valid_section = (
        _table(_format_rows(valid, 20), columns)
        if not valid.empty
        else "No configs passed the hard constraints."
    )
    near_section = (
        _table(_format_rows(near_valid, 20), columns)
        if not near_valid.empty
        else "No configs passed the near-valid constraints."
    )

    report = f"""# Traffic-Light Health-Constrained Sweep

Window: `{START}` to `{END}` at 1h bars from cached Binance SOL/ETH data.

## Objective

Retest the most promising raw traffic-light shapes with stricter projected-health-factor floors for traffic-light hedge additions.

Hard valid constraints:

- Final SOL-equivalent > `{SOL_GATE:.1f} SOL`
- Minimum health factor >= `{MIN_HF_GATE:.2f}`

## Search Space

- Seven promising traffic-light floor shapes from the first governor sweep.
- Traffic-light hedge-add projected min HF: `2.50`, `3.00`, `3.50`.
- Existing fast-break v2 overlay kept constant.

Total scenarios: `{len(comparison)}`.

## Valid Configs Ranked By Drawdown

{valid_section}

## Near-Valid Configs Ranked By Drawdown

{near_section}

## Best Per Family

{_table(_format_rows(family_best, 20), columns)}

## Top By Drawdown Regardless Of Constraints

{_table(_format_rows(all_by_drawdown, 15), columns)}

## Top By SOL-Equivalent

{_table(_format_rows(all_by_sol, 15), columns)}

## Readout

TODO: Fill after inspecting the frontier.

## Artifacts

- Comparison CSV: `{out_dir / "comparison.csv"}`
- Valid configs CSV: `{out_dir / "valid_configs.csv"}`
- Near-valid configs CSV: `{out_dir / "near_valid_configs.csv"}`
- All by drawdown CSV: `{out_dir / "all_by_drawdown.csv"}`
- All by SOL CSV: `{out_dir / "all_by_sol.csv"}`
- Family best CSV: `{out_dir / "family_best.csv"}`
- Summary JSON: `{out_dir / "summary.json"}`
- Scenario folders: `{out_dir}`
"""
    (out_dir / "report.md").write_text(report)
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
