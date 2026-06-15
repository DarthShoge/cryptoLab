from __future__ import annotations

import json
from collections import defaultdict
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
from temp.run_drawdown_containment_sweep import END, START, _base_config, _fmt, _table
from temp.run_fast_break_crisis_gated_partial_fill_sweep import _drawdown_window
from temp.run_traffic_light_governor_sweep import (
    MIN_HF_GATE,
    NEAR_SOL_GATE,
    SOL_GATE,
    _format_rows,
    _traffic_light_overrides,
)


RAW_BEST_TRAFFIC_LIGHT = _traffic_light_overrides(
    yellow_floor=0.50,
    orange_floor=1.00,
    red_floor=1.25,
    min_reinvestment_green=3,
    min_releverage_green=4,
)

HEALTH_GUARDED_TRAFFIC_LIGHT = {
    **RAW_BEST_TRAFFIC_LIGHT,
    "traffic_light_add_min_hf": 2.50,
}

PROTECTION_SHAPES = {
    "mild": {4: 0.0, 3: 0.0, 2: 0.10, 1: 0.25, 0: 0.50},
    "medium": {4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.75},
    "strong": {4: 0.0, 3: 0.10, 2: 0.50, 1: 0.75, 0: 1.00},
}


def _event_count(events: pd.DataFrame, reason: str) -> int:
    if events.empty or "reason" not in events:
        return 0
    return int((events["reason"] == reason).sum())


def _scenarios() -> list[tuple[str, str, dict[str, object]]]:
    base_shapes = {
        "raw": RAW_BEST_TRAFFIC_LIGHT,
        "health": HEALTH_GUARDED_TRAFFIC_LIGHT,
    }
    scenarios: list[tuple[str, str, dict[str, object]]] = []
    for base_name, base_overrides in base_shapes.items():
        scenarios.append(
            (
                f"tlg_{base_name}_control",
                "control",
                base_overrides,
            )
        )
        for protection_name, fractions in PROTECTION_SHAPES.items():
            for rebuy_min_green in (3, 4):
                for rebuy_fraction in (0.25, 0.50):
                    for rebuy_cap in (0.025, 0.05):
                        name = (
                            f"insurance_{base_name}_{protection_name}"
                            f"_recovery{rebuy_min_green}"
                            f"_frac{rebuy_fraction:.2f}"
                            f"_cap{rebuy_cap:.3f}"
                        )
                        scenarios.append(
                            (
                                name,
                                "traffic_light_insurance",
                                {
                                    **base_overrides,
                                    "enable_traffic_light_protected_book": True,
                                    "traffic_light_protected_book_fractions": fractions,
                                    "traffic_light_protected_rebuy_min_green": (
                                        rebuy_min_green
                                    ),
                                    "traffic_light_protected_rebuy_fraction": (
                                        rebuy_fraction
                                    ),
                                    "traffic_light_protected_rebuy_max_pct_of_sol_collateral": (
                                        rebuy_cap
                                    ),
                                    "traffic_light_protected_rebuy_min_hf": 2.0,
                                },
                            )
                        )
    return scenarios


def _scenario_row(
    name: str,
    family: str,
    overrides: dict[str, object],
    result,
    events: pd.DataFrame,
    final_sol_price: float,
) -> dict[str, object]:
    final_positions = final_position_summary(result.history, final_sol_price)
    recent_dd = _drawdown_window(result.history, "2024-01-01")
    row = {
        "scenario": name,
        "family": family,
        "final_sol_equiv": final_positions["net"]["sol_equivalent"],
        "final_portfolio_value_usd": final_positions["net"]["portfolio_value_usd"],
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_peak_to_trough_pct": recent_dd["drawdown_pct"],
        "avg_half_recovery_bars": None,
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "final_sol_collateral": final_positions["collateral"][0]["Amount"],
        "final_usdc_collateral_value_usd": final_positions["collateral"][1][
            "Value USD"
        ],
        "final_eth_debt_value_usd": final_positions["debt"][0]["Value USD"],
        "protected_book_final_usdc": (
            float(result.history["protected_book_usdc"].iloc[-1])
            if "protected_book_usdc" in result.history
            else 0.0
        ),
        "protected_rebuy_events": _event_count(
            events,
            "traffic_light_protected_rebuy",
        ),
        "protection_shape": name.split("_")[2] if name.startswith("insurance_") else "",
        "rebuy_min_green": overrides.get("traffic_light_protected_rebuy_min_green", ""),
        "rebuy_fraction": overrides.get("traffic_light_protected_rebuy_fraction", ""),
        "rebuy_cap": overrides.get(
            "traffic_light_protected_rebuy_max_pct_of_sol_collateral",
            "",
        ),
    }
    sol = float(row["final_sol_equiv"])
    min_hf = float(row["min_health_factor"])
    row.update(
        {
            "sol_gate_pass": sol > SOL_GATE,
            "near_sol_gate_pass": sol >= NEAR_SOL_GATE,
            "hf_gate_pass": min_hf >= MIN_HF_GATE,
            "valid": sol > SOL_GATE and min_hf >= MIN_HF_GATE,
            "near_valid": sol >= NEAR_SOL_GATE and min_hf >= MIN_HF_GATE,
        }
    )
    return row


def _insurance_format_rows(df: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    rows = _format_rows(df, limit)
    for idx, (_, row) in enumerate(df.head(limit).iterrows()):
        rows[idx]["book"] = _fmt(float(row["protected_book_final_usdc"]))
        rows[idx]["rebuy"] = int(row["protected_rebuy_events"])
    return rows


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/sol_supertrend_3y") / (
        f"traffic_light_insurance_{timestamp}"
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
        "book",
        "rebuy",
        "valid",
        "near",
    ]
    valid_section = (
        _table(_insurance_format_rows(valid, 20), columns)
        if not valid.empty
        else "No configs passed the hard constraints."
    )
    near_section = (
        _table(_insurance_format_rows(near_valid, 20), columns)
        if not near_valid.empty
        else "No configs passed the near-valid constraints."
    )

    report = f"""# Traffic-Light Insurance Sweep

Window: `{START}` to `{END}` at 1h bars from cached Binance SOL/ETH data.

## Objective

Test whether traffic-light-dependent hedge-profit routing plus recovery-light rebuy improves the current raw traffic-light frontier.

Hard valid constraints:

- Final SOL-equivalent > `{SOL_GATE:.1f} SOL`
- Minimum health factor >= `{MIN_HF_GATE:.2f}`

## Search Space

- Controls: raw best traffic-light shape `yellow=0.50`, `orange=1.00`, `red=1.25`, reinvest allowed from 3 green votes, both with and without `traffic_light_add_min_hf=2.50`.
- Protection routing: mild, medium, strong realized hedge PnL fractions by traffic-light state.
- Recovery rebuy: min green votes `3` or `4`, rebuy fraction `0.25` or `0.50`, SOL-collateral cap `0.025` or `0.050`.

Total scenarios: `{len(comparison)}`.

## Valid Configs Ranked By Drawdown

{valid_section}

## Near-Valid Configs Ranked By Drawdown

{near_section}

## Top By Drawdown Regardless Of Constraints

{_table(_insurance_format_rows(all_by_drawdown, 15), columns)}

## Top By SOL-Equivalent

{_table(_insurance_format_rows(all_by_sol, 15), columns)}

## Readout

Best drawdown candidate: `{all_by_drawdown.iloc[0]["scenario"]}` with {_fmt(float(all_by_drawdown.iloc[0]["max_drawdown_pct"]))}% max drawdown and {_fmt(float(all_by_drawdown.iloc[0]["final_sol_equiv"]))} SOL-equivalent.

Best SOL-equivalent candidate: `{all_by_sol.iloc[0]["scenario"]}` with {_fmt(float(all_by_sol.iloc[0]["final_sol_equiv"]))} SOL-equivalent and {_fmt(float(all_by_sol.iloc[0]["max_drawdown_pct"]))}% max drawdown.

Valid configs: `{int(comparison["valid"].sum())}` of `{len(comparison)}`.

## Artifacts

- Comparison CSV: `{out_dir / "comparison.csv"}`
- Valid configs: `{out_dir / "valid_configs.csv"}`
- Near-valid configs: `{out_dir / "near_valid_configs.csv"}`
- Summary JSON: `{out_dir / "summary.json"}`
- Per-scenario history/events: `{out_dir}/<scenario>/`
"""
    (out_dir / "report.md").write_text(report)
    print(out_dir)
    print(report)


if __name__ == "__main__":
    main()
