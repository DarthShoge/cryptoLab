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


def _scenario_row(
    name: str,
    group: str,
    overrides: dict[str, object],
    result,
    events: pd.DataFrame,
    final_sol_price: float,
) -> dict[str, object]:
    final_positions = final_position_summary(result.history, final_sol_price)
    history = result.history
    full_dd = _drawdown_window(history)
    recent_dd = _drawdown_window(history, "2024-01-01")
    reserve_usdc = (
        float(history["profit_lock_reserve_usdc"].max())
        if "profit_lock_reserve_usdc" in history
        else 0.0
    )
    reserve_active_pct = (
        float(history["in_profit_lock_reserve"].astype(bool).mean() * 100.0)
        if "in_profit_lock_reserve" in history
        else 0.0
    )
    return {
        "scenario": name,
        "group": group,
        "sell_fraction": overrides.get("profit_lock_reserve_sell_fraction", ""),
        "escalation_fraction": overrides.get(
            "profit_lock_reserve_escalation_sell_fraction",
            "",
        ),
        "max_fraction": overrides.get("profit_lock_reserve_max_fraction", ""),
        "rebuy_fraction": overrides.get("profit_lock_reserve_rebuy_fraction", ""),
        "cooldown_bars": overrides.get("profit_lock_reserve_rebuy_cooldown_bars", ""),
        "reset_gap": overrides.get("profit_lock_reserve_new_high_reset_gap", ""),
        "final_sol_equiv": final_positions["net"]["sol_equivalent"],
        "final_portfolio_value_usd": final_positions["net"]["portfolio_value_usd"],
        "final_sol_collateral": final_positions["collateral"][0]["Amount"],
        "final_usdc_collateral_value_usd": final_positions["collateral"][1][
            "Value USD"
        ],
        "final_eth_debt_value_usd": final_positions["debt"][0]["Value USD"],
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "sortino_ratio": result.metrics.sortino_ratio,
        "sharpe_ratio": result.metrics.sharpe_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "post_2024_peak_to_trough_pct": recent_dd["drawdown_pct"],
        "reserve_active_pct": reserve_active_pct,
        "reserve_sell_events": _event_count(events, "profit_lock_reserve_sell"),
        "reserve_escalate_events": _event_count(
            events,
            "profit_lock_reserve_escalate",
        ),
        "reserve_rebuy_events": _event_count(events, "profit_lock_reserve_rebuy"),
        "max_reserve_usdc": reserve_usdc,
        "peak_timestamp": full_dd["peak_timestamp"],
        "trough_timestamp": full_dd["trough_timestamp"],
        "post_2024_peak_timestamp": recent_dd["peak_timestamp"],
        "post_2024_trough_timestamp": recent_dd["trough_timestamp"],
    }


def _report_rows(df: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    rows = []
    for _, row in df.head(limit).iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "sell": _fmt(row["sell_fraction"]),
                "esc": _fmt(row["escalation_fraction"]),
                "max": _fmt(row["max_fraction"]),
                "rebuy": _fmt(row["rebuy_fraction"]),
                "cool": _fmt(row["cooldown_bars"]),
                "reset": _fmt(row["reset_gap"]),
                "SOL-eq": _fmt(float(row["final_sol_equiv"])),
                "USD": _fmt(float(row["final_portfolio_value_usd"])),
                "maxDD": _fmt(float(row["max_drawdown_pct"])),
                "2024DD": _fmt(float(row["post_2024_peak_to_trough_pct"])),
                "Sortino": _fmt(float(row["sortino_ratio"])),
                "minHF": _fmt(float(row["min_health_factor"])),
                "active%": _fmt(float(row["reserve_active_pct"])),
                "sell/esc/rebuy": (
                    f"{int(row['reserve_sell_events'])}/"
                    f"{int(row['reserve_escalate_events'])}/"
                    f"{int(row['reserve_rebuy_events'])}"
                ),
            }
        )
    return rows


def _tipp_overrides(
    sell_fraction: float,
    escalation_fraction: float,
    max_fraction: float,
    rebuy_fraction: float,
    cooldown_bars: int,
    reset_gap: float,
) -> dict[str, object]:
    return {
        **CURRENT_FAST_BREAK,
        "enable_profit_lock_reserve": True,
        "profit_lock_reserve_episode_mode": True,
        "profit_lock_reserve_sell_fraction": sell_fraction,
        "profit_lock_reserve_escalation_sell_fraction": escalation_fraction,
        "profit_lock_reserve_max_fraction": max_fraction,
        "profit_lock_reserve_min_sol_collateral": 100.0,
        "profit_lock_reserve_min_gain_pct": 1.00,
        "profit_lock_reserve_near_high_threshold": 0.10,
        "profit_lock_reserve_escalation_drawdown": 0.15,
        "profit_lock_reserve_rebuy_fraction": rebuy_fraction,
        "profit_lock_reserve_rebuy_cooldown_bars": cooldown_bars,
        "profit_lock_reserve_new_high_reset_gap": reset_gap,
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/sol_supertrend_3y") / (
        f"tipp_profit_lock_reserve_sweep_{timestamp}"
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

    scenarios: list[tuple[str, str, dict[str, object]]] = [
        ("incumbent_stateful_best", "control", {}),
        ("v2_best_no_reserve", "control", CURRENT_FAST_BREAK),
    ]
    for sell_fraction in (0.05, 0.10):
        for escalation_fraction in (0.00, 0.05, 0.10):
            for max_fraction in (0.15, 0.25):
                for rebuy_fraction in (0.25, 0.50):
                    cooldown_bars = 168
                    reset_gap = 0.02
                    if sell_fraction > max_fraction:
                        continue
                    scenarios.append(
                        (
                            (
                                f"tipp_sell{sell_fraction:.2f}"
                                f"_esc{escalation_fraction:.2f}"
                                f"_max{max_fraction:.2f}"
                                f"_rebuy{rebuy_fraction:.2f}"
                                f"_cool{cooldown_bars}"
                                f"_reset{reset_gap:.2f}"
                            ),
                            "tipp_profit_lock_reserve",
                            _tipp_overrides(
                                sell_fraction,
                                escalation_fraction,
                                max_fraction,
                                rebuy_fraction,
                                cooldown_bars,
                                reset_gap,
                            ),
                        )
                    )

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "window": {"start": START, "end": END, "timeframe": "1h"},
        "scenarios": {},
    }

    for idx, (name, group, overrides) in enumerate(scenarios, start=1):
        print(f"[{idx}/{len(scenarios)}] {name}", flush=True)
        config = {**base_config, **overrides}
        result = run_selected_backtest(
            SOL_SUPERTREND_SHORT_STRATEGY,
            price_data,
            config,
            market_params,
        )
        scenario_dir = out_dir / name
        scenario_dir.mkdir()
        result.history.to_csv(scenario_dir / "history.csv")
        events = pd.DataFrame(result.strategy_events)
        events.to_csv(scenario_dir / "strategy_events.csv", index=False)
        row = _scenario_row(name, group, overrides, result, events, final_sol_price)
        rows.append(row)
        summary["scenarios"][name] = {
            "group": group,
            "config_overrides": overrides,
            "metrics": row,
            "artifact_dir": str(scenario_dir),
        }

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison_unsorted.csv", index=False)
    by_sortino = comparison.sort_values(
        ["sortino_ratio", "final_sol_equiv", "max_drawdown_pct"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    by_drawdown = comparison.sort_values(
        ["max_drawdown_pct", "sortino_ratio", "final_sol_equiv"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    by_2024_drawdown = comparison.sort_values(
        ["post_2024_peak_to_trough_pct", "sortino_ratio", "final_sol_equiv"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    by_sol = comparison.sort_values(
        ["final_sol_equiv", "sortino_ratio", "max_drawdown_pct"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    by_sortino.to_csv(out_dir / "comparison.csv", index=False)
    by_drawdown.to_csv(out_dir / "top_by_drawdown.csv", index=False)
    by_2024_drawdown.to_csv(out_dir / "top_by_2024_drawdown.csv", index=False)
    by_sol.to_csv(out_dir / "top_by_sol_equiv.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )

    control = comparison[comparison["scenario"] == "v2_best_no_reserve"].iloc[0]
    best_sortino = by_sortino.iloc[0]
    best_dd = by_drawdown.iloc[0]
    best_2024_dd = by_2024_drawdown.iloc[0]
    best_sol = by_sol.iloc[0]
    columns = [
        "scenario",
        "sell",
        "esc",
        "max",
        "rebuy",
        "cool",
        "reset",
        "SOL-eq",
        "USD",
        "maxDD",
        "2024DD",
        "Sortino",
        "minHF",
        "active%",
        "sell/esc/rebuy",
    ]
    report = f"""# TIPP Profit-Lock Reserve Sweep

Window: `{START}` to `{END}` at 1h bars from cached Binance SOL/ETH data.

## Question

Can a stateful, high-watermark-gated SOL reserve reduce late-cycle drawdown without the reserve churn that damaged the earlier profit-lock reserve experiment?

## Scenario Set

The sweep compares the current v2 fast-break no-reserve control against episode-bounded profit-lock reserve variants.

- Entry: profit lock active, portfolio up at least `100%` from initial, and within `10%` of the trailing high.
- Episode rule: one initial slice and at most one escalation slice per completed high-watermark episode.
- Re-entry rule: after a full reserve rebuy, a new episode requires portfolio value above the completed episode high by the reset gap.
- Rebuy rule: 1w/1d/3d recovery plus a cooldown from the last reserve sale.
- Minimum SOL collateral: `100 SOL`.
- First sell fraction grid: `5%`, `10%`.
- Escalation sell fraction grid: `0%`, `5%`, `10%`.
- Max reserve fraction grid: `15%`, `25%`.
- Rebuy fraction grid: `25%`, `50%`.
- Cooldown: `168` bars.
- Reset gap: `2%`.

## Ranking By Sortino

{_table(_report_rows(by_sortino, 12), columns)}

## Ranking By Max Drawdown

{_table(_report_rows(by_drawdown, 12), columns)}

## Ranking By 2024+ Peak-To-Trough

{_table(_report_rows(by_2024_drawdown, 12), columns)}

## Ranking By Final SOL-Equivalent

{_table(_report_rows(by_sol, 12), columns)}

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | Reserve sell/esc/rebuy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 no reserve | {control["scenario"]} | {_fmt(float(control["final_sol_equiv"]))} | {_fmt(float(control["final_portfolio_value_usd"]))} | {_fmt(float(control["max_drawdown_pct"]))} | {_fmt(float(control["post_2024_peak_to_trough_pct"]))} | {_fmt(float(control["sortino_ratio"]))} | {_fmt(float(control["min_health_factor"]))} | {int(control["reserve_sell_events"])}/{int(control["reserve_escalate_events"])}/{int(control["reserve_rebuy_events"])} |
| Best Sortino | {best_sortino["scenario"]} | {_fmt(float(best_sortino["final_sol_equiv"]))} | {_fmt(float(best_sortino["final_portfolio_value_usd"]))} | {_fmt(float(best_sortino["max_drawdown_pct"]))} | {_fmt(float(best_sortino["post_2024_peak_to_trough_pct"]))} | {_fmt(float(best_sortino["sortino_ratio"]))} | {_fmt(float(best_sortino["min_health_factor"]))} | {int(best_sortino["reserve_sell_events"])}/{int(best_sortino["reserve_escalate_events"])}/{int(best_sortino["reserve_rebuy_events"])} |
| Best Max DD | {best_dd["scenario"]} | {_fmt(float(best_dd["final_sol_equiv"]))} | {_fmt(float(best_dd["final_portfolio_value_usd"]))} | {_fmt(float(best_dd["max_drawdown_pct"]))} | {_fmt(float(best_dd["post_2024_peak_to_trough_pct"]))} | {_fmt(float(best_dd["sortino_ratio"]))} | {_fmt(float(best_dd["min_health_factor"]))} | {int(best_dd["reserve_sell_events"])}/{int(best_dd["reserve_escalate_events"])}/{int(best_dd["reserve_rebuy_events"])} |
| Best 2024 DD | {best_2024_dd["scenario"]} | {_fmt(float(best_2024_dd["final_sol_equiv"]))} | {_fmt(float(best_2024_dd["final_portfolio_value_usd"]))} | {_fmt(float(best_2024_dd["max_drawdown_pct"]))} | {_fmt(float(best_2024_dd["post_2024_peak_to_trough_pct"]))} | {_fmt(float(best_2024_dd["sortino_ratio"]))} | {_fmt(float(best_2024_dd["min_health_factor"]))} | {int(best_2024_dd["reserve_sell_events"])}/{int(best_2024_dd["reserve_escalate_events"])}/{int(best_2024_dd["reserve_rebuy_events"])} |
| Best SOL-eq | {best_sol["scenario"]} | {_fmt(float(best_sol["final_sol_equiv"]))} | {_fmt(float(best_sol["final_portfolio_value_usd"]))} | {_fmt(float(best_sol["max_drawdown_pct"]))} | {_fmt(float(best_sol["post_2024_peak_to_trough_pct"]))} | {_fmt(float(best_sol["sortino_ratio"]))} | {_fmt(float(best_sol["min_health_factor"]))} | {int(best_sol["reserve_sell_events"])}/{int(best_sol["reserve_escalate_events"])}/{int(best_sol["reserve_rebuy_events"])} |

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
