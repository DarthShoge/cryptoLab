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
from temp.run_profit_lock_reserve_sweep import CURRENT_FAST_BREAK


SOL_GATE = 420.0
NEAR_SOL_GATE = 400.0
MIN_HF_GATE = 1.55


def _scenario_row(
    name: str,
    family: str,
    overrides: dict[str, object],
    result,
    final_sol_price: float,
) -> dict[str, object]:
    final_positions = final_position_summary(result.history, final_sol_price)
    recent_dd = _drawdown_window(result.history, "2024-01-01")
    history = result.history
    protected_book_usdc = (
        float(history["protected_book_usdc"].max())
        if "protected_book_usdc" in history
        else 0.0
    )
    hedge_failure_active_pct = (
        float(history["in_hedge_failure_circuit_breaker"].astype(bool).mean() * 100.0)
        if "in_hedge_failure_circuit_breaker" in history
        else 0.0
    )
    return {
        "scenario": name,
        "family": family,
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
        "protected_book_fraction": overrides.get(
            "protected_book_realized_pnl_fraction",
            "",
        ),
        "hedge_failure_sell_fraction": overrides.get(
            "hedge_failure_sell_fraction",
            "",
        ),
        "hedge_failure_threshold": overrides.get(
            "hedge_failure_underperformance_threshold",
            "",
        ),
        "fast_break_return_threshold": overrides.get(
            "fast_break_return_threshold",
            "",
        ),
        "fast_break_vol_multiplier": overrides.get(
            "fast_break_vol_multiplier",
            "",
        ),
        "fast_break_hedge_floor": overrides.get("fast_break_hedge_floor", ""),
        "fast_break_hold_bars": overrides.get("fast_break_hold_bars", ""),
        "profit_lock_hedge_floor": overrides.get("profit_lock_hedge_floor", ""),
        "profit_lock_stateful_exit_gap": overrides.get(
            "profit_lock_stateful_exit_gap",
            "",
        ),
        "max_protected_book_usdc": protected_book_usdc,
        "hedge_failure_active_pct": hedge_failure_active_pct,
    }


def _validity(row: dict[str, object]) -> dict[str, object]:
    sol = float(row["final_sol_equiv"])
    min_hf = float(row["min_health_factor"])
    return {
        "sol_gate_pass": sol >= SOL_GATE,
        "near_sol_gate_pass": sol >= NEAR_SOL_GATE,
        "hf_gate_pass": min_hf >= MIN_HF_GATE,
        "valid": sol >= SOL_GATE and min_hf >= MIN_HF_GATE,
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
                "Sortino": _fmt(float(row["sortino_ratio"])),
                "minHF": _fmt(float(row["min_health_factor"])),
                "valid": bool(row["valid"]),
                "near": bool(row["near_valid"]),
            }
        )
    return rows


def _with_protected_book(fraction: float) -> dict[str, object]:
    return {
        **CURRENT_FAST_BREAK,
        "enable_protected_book": True,
        "protected_book_realized_pnl_fraction": fraction,
    }


def _with_hedge_failure(
    sell_fraction: float,
    threshold: float = 0.10,
) -> dict[str, object]:
    return {
        **CURRENT_FAST_BREAK,
        "enable_hedge_failure_circuit_breaker": True,
        "hedge_failure_lookback_bars": 72,
        "hedge_failure_underperformance_threshold": threshold,
        "hedge_failure_hold_bars": 168,
        "hedge_failure_sell_fraction": sell_fraction,
        "hedge_failure_min_sol_collateral": 100.0,
    }


def _combined_micro(
    protected_fraction: float,
    hedge_failure_sell_fraction: float,
    threshold: float = 0.10,
) -> dict[str, object]:
    return {
        **_with_hedge_failure(hedge_failure_sell_fraction, threshold),
        "enable_protected_book": True,
        "protected_book_realized_pnl_fraction": protected_fraction,
    }


def _fast_break_variant(
    return_threshold: float,
    vol_multiplier: float,
    hedge_floor: float,
    hold_bars: int,
) -> dict[str, object]:
    return {
        **CURRENT_FAST_BREAK,
        "fast_break_return_threshold": return_threshold,
        "fast_break_vol_multiplier": vol_multiplier,
        "fast_break_hedge_floor": hedge_floor,
        "fast_break_hold_bars": hold_bars,
    }


def _profit_lock_variant(
    hedge_floor: float,
    exit_gap: float,
    drawdown_threshold: float,
) -> dict[str, object]:
    return {
        **CURRENT_FAST_BREAK,
        "enable_profit_lock": True,
        "profit_lock_metric": "portfolio",
        "profit_lock_min_gain_pct": 0.25,
        "profit_lock_drawdown_threshold": drawdown_threshold,
        "profit_lock_near_high_threshold": 0.03,
        "profit_lock_stateful": True,
        "profit_lock_stateful_exit_gap": exit_gap,
        "profit_lock_hedge_floor": hedge_floor,
        "profit_lock_max_green": 3,
    }


def _scenarios() -> list[tuple[str, str, dict[str, object]]]:
    scenarios: list[tuple[str, str, dict[str, object]]] = [
        ("v2_best_no_overlays", "control", CURRENT_FAST_BREAK),
    ]

    for fraction in (0.01, 0.025, 0.05, 0.075, 0.10, 0.15):
        scenarios.append(
            (
                f"protected_book_pnl{fraction:.3f}",
                "protected_book_micro",
                _with_protected_book(fraction),
            )
        )

    for threshold in (0.08, 0.10, 0.12):
        for sell_fraction in (0.0, 0.01, 0.02, 0.03, 0.05):
            scenarios.append(
                (
                    f"hedge_failure_thr{threshold:.2f}_sell{sell_fraction:.2f}",
                    "hedge_failure_micro",
                    _with_hedge_failure(sell_fraction, threshold),
                )
            )

    for protected_fraction in (0.01, 0.025, 0.05, 0.075, 0.10):
        for sell_fraction in (0.0, 0.01, 0.02, 0.03):
            scenarios.append(
                (
                    (
                        f"combo_pb{protected_fraction:.3f}"
                        f"_hf{sell_fraction:.2f}"
                    ),
                    "combo_micro",
                    _combined_micro(protected_fraction, sell_fraction),
                )
            )

    for return_threshold in (-0.06, -0.08, -0.10):
        for vol_multiplier in (2.0, 2.5):
            for hold_bars in (48, 72, 120):
                scenarios.append(
                    (
                        (
                            f"fb_ret{return_threshold:.2f}"
                            f"_vol{vol_multiplier:.1f}"
                            f"_hold{hold_bars}"
                        ),
                        "fast_break_shape",
                        _fast_break_variant(
                            return_threshold,
                            vol_multiplier,
                            1.0,
                            hold_bars,
                        ),
                    )
                )

    for hedge_floor in (0.25, 0.35, 0.50):
        for exit_gap in (0.03, 0.05, 0.07):
            scenarios.append(
                (
                    f"pl_floor{hedge_floor:.2f}_exit{exit_gap:.2f}",
                    "profit_lock_hedge_only",
                    _profit_lock_variant(hedge_floor, exit_gap, 0.10),
                )
            )

    return scenarios


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/sol_supertrend_3y") / (
        f"constrained_drawdown_search_{timestamp}"
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
            "final_sol_equiv_min": SOL_GATE,
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
        row = _scenario_row(name, family, overrides, result, final_sol_price)
        row.update(_validity(row))
        rows.append(row)
        scenario_dir = out_dir / name
        scenario_dir.mkdir()
        result.history.to_csv(scenario_dir / "history.csv")
        events.to_csv(scenario_dir / "strategy_events.csv", index=False)
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

    report = f"""# Constrained Drawdown Search

Window: `{START}` to `{END}` at 1h bars from cached Binance SOL/ETH data.

## Objective

Find configurations that reduce drawdown while still making money in SOL terms.

Hard valid constraints:

- Final SOL-equivalent >= `{SOL_GATE:.1f} SOL`
- Minimum health factor >= `{MIN_HF_GATE:.2f}`

Near-valid constraints:

- Final SOL-equivalent >= `{NEAR_SOL_GATE:.1f} SOL`
- Minimum health factor >= `{MIN_HF_GATE:.2f}`

## Search Space

- Tiny protected-book allocations: `1%`, `2.5%`, `5%`, `7.5%`, `10%`, `15%`
- Hedge-failure overlay thresholds: `8%`, `10%`, `12%`; sell fractions: `0%`, `1%`, `2%`, `3%`, `5%`
- Combined micro-overlays: protected book `1%-10%` with hedge-failure sells `0%-3%`
- Fast-break shape variants around return threshold, volatility multiplier, and hold bars
- Profit-lock hedge-only variants around hedge floor and stateful exit gap

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

TODO: Fill after inspecting the constrained and near-valid frontier.

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
