from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import pandas as pd

from arblab.backtest.app_helpers import (
    SOL_SUPERTREND_SHORT_STRATEGY,
    build_price_configs,
    final_position_summary,
    run_selected_backtest,
)
from arblab.backtest.data import fetch_ohlcv
from arblab.backtest.market import MarketParams
from arblab.backtest.recovery import drawdown_recovery_events
from temp.run_drawdown_containment_sweep import END, START, _base_config, _fmt, _table
from temp.run_fast_break_crisis_gated_partial_fill_sweep import _drawdown_window
from temp.run_profit_lock_reserve_sweep import CURRENT_FAST_BREAK


SOL_GATE = 420.0
MIN_HF_GATE = 1.50
NEAR_SOL_GATE = 400.0


def _traffic_light_overrides(
    yellow_floor: float,
    orange_floor: float,
    red_floor: float,
    min_reinvestment_green: int,
    min_releverage_green: int,
) -> dict[str, object]:
    return {
        **CURRENT_FAST_BREAK,
        "enable_traffic_light_governor": True,
        "traffic_light_hedge_floors": {
            4: 0.0,
            3: 0.10,
            2: yellow_floor,
            1: orange_floor,
            0: red_floor,
        },
        "traffic_light_min_reinvestment_green": min_reinvestment_green,
        "traffic_light_min_releverage_green": min_releverage_green,
    }


def _scenarios() -> list[tuple[str, str, dict[str, object]]]:
    scenarios: list[tuple[str, str, dict[str, object]]] = [
        ("v2_best_no_overlays", "control", CURRENT_FAST_BREAK),
    ]

    for yellow_floor in (0.35, 0.50, 0.75):
        for orange_floor in (0.75, 1.00, 1.25):
            for red_floor in (1.00, 1.25, 1.50):
                if red_floor < orange_floor:
                    continue
                for min_reinvestment_green in (3, 4):
                    name = (
                        f"tlg_y{yellow_floor:.2f}"
                        f"_o{orange_floor:.2f}"
                        f"_r{red_floor:.2f}"
                        f"_reinv{min_reinvestment_green}"
                    )
                    scenarios.append(
                        (
                            name,
                            "traffic_light_exposure",
                            _traffic_light_overrides(
                                yellow_floor,
                                orange_floor,
                                red_floor,
                                min_reinvestment_green,
                                4,
                            ),
                        )
                    )
    return scenarios


def _recovery_summary(history: pd.DataFrame) -> dict[str, object]:
    events = drawdown_recovery_events(history["portfolio_value"], min_drawdown_pct=30.0)
    recovered = [event for event in events if event["half_recovery_bars"] is not None]
    new_highs = [event for event in events if event["new_high_bars"] is not None]
    return {
        "major_drawdown_events": len(events),
        "avg_half_recovery_bars": (
            mean(float(event["half_recovery_bars"]) for event in recovered)
            if recovered
            else None
        ),
        "avg_new_high_bars": (
            mean(float(event["new_high_bars"]) for event in new_highs)
            if new_highs
            else None
        ),
    }


def _traffic_light_state_pct(history: pd.DataFrame, state: str) -> float:
    if "traffic_light_state" not in history:
        return 0.0
    return float((history["traffic_light_state"] == state).mean() * 100.0)


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
    recovery = _recovery_summary(result.history)
    row = {
        "scenario": name,
        "family": family,
        "yellow_floor": overrides.get("traffic_light_hedge_floors", {}).get(2, ""),
        "orange_floor": overrides.get("traffic_light_hedge_floors", {}).get(1, ""),
        "red_floor": overrides.get("traffic_light_hedge_floors", {}).get(0, ""),
        "min_reinvestment_green": overrides.get(
            "traffic_light_min_reinvestment_green",
            "",
        ),
        "final_sol_equiv": final_positions["net"]["sol_equivalent"],
        "final_portfolio_value_usd": final_positions["net"]["portfolio_value_usd"],
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_peak_to_trough_pct": recent_dd["drawdown_pct"],
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "final_sol_collateral": final_positions["collateral"][0]["Amount"],
        "final_usdc_collateral_value_usd": final_positions["collateral"][1][
            "Value USD"
        ],
        "final_eth_debt_value_usd": final_positions["debt"][0]["Value USD"],
        "traffic_light_hedge_events": _event_count_prefix(events, "traffic_light"),
        "green_pct": _traffic_light_state_pct(result.history, "green"),
        "lime_pct": _traffic_light_state_pct(result.history, "lime"),
        "yellow_pct": _traffic_light_state_pct(result.history, "yellow"),
        "orange_pct": _traffic_light_state_pct(result.history, "orange"),
        "red_pct": _traffic_light_state_pct(result.history, "red"),
        **recovery,
    }
    row.update(_validity(row))
    return row


def _event_count_prefix(events: pd.DataFrame, prefix: str) -> int:
    if events.empty or "reason" not in events:
        return 0
    return int(events["reason"].astype(str).str.startswith(prefix).sum())


def _validity(row: dict[str, object]) -> dict[str, object]:
    sol = float(row["final_sol_equiv"])
    min_hf = float(row["min_health_factor"])
    return {
        "sol_gate_pass": sol > SOL_GATE,
        "near_sol_gate_pass": sol >= NEAR_SOL_GATE,
        "hf_gate_pass": min_hf >= MIN_HF_GATE,
        "valid": sol > SOL_GATE and min_hf >= MIN_HF_GATE,
        "near_valid": sol >= NEAR_SOL_GATE and min_hf >= MIN_HF_GATE,
    }


def _format_rows(df: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    rows = []
    for _, row in df.head(limit).iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "family": row["family"],
                "SOL-eq": _fmt(float(row["final_sol_equiv"])),
                "USD": _fmt(float(row["final_portfolio_value_usd"])),
                "maxDD": _fmt(float(row["max_drawdown_pct"])),
                "2024DD": _fmt(float(row["post_2024_peak_to_trough_pct"])),
                "halfRec": (
                    ""
                    if pd.isna(row["avg_half_recovery_bars"])
                    else _fmt(float(row["avg_half_recovery_bars"]), 1)
                ),
                "Sortino": _fmt(float(row["sortino_ratio"])),
                "minHF": _fmt(float(row["min_health_factor"])),
                "valid": bool(row["valid"]),
                "near": bool(row["near_valid"]),
            }
        )
    return rows


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/sol_supertrend_3y") / (
        f"traffic_light_governor_{timestamp}"
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
        [
            "max_drawdown_pct",
            "post_2024_peak_to_trough_pct",
            "avg_half_recovery_bars",
            "final_sol_equiv",
        ],
        ascending=[True, True, True, False],
        na_position="last",
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

    report = f"""# Traffic-Light Governor Sweep

Window: `{START}` to `{END}` at 1h bars from cached Binance SOL/ETH data.

## Objective

Test the first traffic-light exposure-governor slice.

Hard valid constraints:

- Final SOL-equivalent > `{SOL_GATE:.1f} SOL`
- Minimum health factor >= `{MIN_HF_GATE:.2f}`

Investor target:

- Max drawdown <= `50%`

## Search Space

- Yellow hedge floor: `0.35`, `0.50`, `0.75`
- Orange hedge floor: `0.75`, `1.00`, `1.25`
- Red hedge floor: `1.00`, `1.25`, `1.50`
- Minimum green votes for surplus reinvestment: `3`, `4`
- Minimum green votes for USDC releverage: `4`
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
