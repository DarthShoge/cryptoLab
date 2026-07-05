from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arblab.backtest.data import OHLCVConfig, fetch_ohlcv
from arblab.backtest.engine import BacktestEngine, EngineConfig
from arblab.backtest.traffic_lights import asset_green_counts_by_bar
from arblab.strategies.multi_asset_traffic_light import MultiAssetTrafficLightStrategy


START = "2021-01-01"
END = "2025-12-31"
INITIAL_SOL = 100.0
SYMBOLS = ["SOL", "ETH"]
TIMEFRAMES = ("1h", "4h", "8h", "1d")
REFERENCE = {
    "name": "static_light_hedge",
    "final_usd": 68_291.651846,
    "final_sol": 546.551836,
    "max_drawdown_pct": 57.439767,
}
LIGHT_FLOORS = {4: 0.00, 3: 0.050, 2: 0.025, 1: 0.00, 0: 0.00}


def _max_drawdown_pct(values: pd.Series) -> float:
    running_peak = values.cummax()
    drawdown = (values - running_peak) / running_peak
    return abs(float(drawdown.min())) * 100.0


def _drawdown_window(history: pd.DataFrame, start: str) -> float:
    window = history.loc[history.index >= pd.Timestamp(start, tz="UTC")]
    if window.empty:
        return 0.0
    return _max_drawdown_pct(window["portfolio_value"])


def _price_data() -> pd.DataFrame:
    return fetch_ohlcv(
        symbols=[
            OHLCVConfig(symbol="SOL/USDT", display_name="SOL"),
            OHLCVConfig(symbol="ETH/USDT", display_name="ETH"),
        ],
        timeframe="1h",
        start=START,
        end=END,
        cache_dir=Path("notebooks/.price_cache"),
        use_cache=True,
    )


def _signal_tiers() -> list[dict[str, float]]:
    return [
        {"green": 3, "target_long_fraction": 1.025},
        {"green": 4, "target_long_fraction": 1.075},
    ]


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
        (0.00, 0.000),
        (0.02, 0.000),
        (0.05, 0.000),
        (0.10, 0.000),
        (0.05, 0.0025),
        (0.10, 0.0025),
        (0.05, 0.0050),
    ):
        scenario = _base_scenario(
            f"dd_slope_dd{min_drawdown:.3f}_w{min_worsening:.4f}"
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
        (24, 0.00),
        (72, 0.00),
        (168, 0.00),
        (72, 0.01),
        (168, 0.01),
    ):
        scenario = _base_scenario(
            f"rs_lb{lookback}_u{underperformance:.3f}"
        )
        scenario.update(
            {
                "enable_relative_strength_hedge_gate": True,
                "relative_strength_lookback_bars": lookback,
                "relative_strength_min_underperformance": underperformance,
            }
        )
        scenarios.append(scenario)

    for min_drawdown, min_worsening, lookback, underperformance in (
        (0.05, 0.0000, 72, 0.00),
        (0.05, 0.0025, 72, 0.00),
        (0.10, 0.0000, 72, 0.00),
        (0.05, 0.0000, 168, 0.00),
    ):
        scenario = _base_scenario(
            (
                f"combo_dd{min_drawdown:.3f}_w{min_worsening:.4f}"
                f"_rs{lookback}_u{underperformance:.3f}"
            )
        )
        scenario.update(
            {
                "enable_drawdown_slope_hedge_gate": True,
                "drawdown_slope_min_drawdown": min_drawdown,
                "drawdown_slope_min_worsening": min_worsening,
                "enable_relative_strength_hedge_gate": True,
                "relative_strength_lookback_bars": lookback,
                "relative_strength_min_underperformance": underperformance,
            }
        )
        scenarios.append(scenario)
    return scenarios


def _run_scenario(
    scenario: dict[str, object],
    price_data: pd.DataFrame,
    green_scores: pd.DataFrame,
) -> dict[str, object]:
    first = price_data.iloc[0]
    final_sol_price = float(price_data.iloc[-1][("SOL", "close")])
    config = {
        "initial_collateral_symbol": "SOL",
        "initial_collateral_amount": INITIAL_SOL,
        "directional_symbols": SYMBOLS,
        "initial_prices": {
            "SOL": float(first[("SOL", "close")]),
            "ETH": float(first[("ETH", "close")]),
        },
        "green_scores": green_scores,
        "atr_period": 10,
        "supertrend_multiplier": 3.0,
        "timeframes": TIMEFRAMES,
        "min_long_green": 3,
        "max_short_green": -1,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.00,
        "enable_signal_governor": True,
        "signal_exposure_tiers": _signal_tiers(),
        "rebalance_threshold": 0.05,
        "swap_fee_bps": 10.0,
        **scenario,
    }
    result = BacktestEngine(
        MultiAssetTrafficLightStrategy(),
        EngineConfig(lookback_bars=24 * 14),
    ).run(price_data, config)
    history = result.history
    active_gate = history.get("hedge_gate_active", pd.Series(True, index=history.index))
    row = {
        "name": scenario["name"],
        "final_portfolio_value_usd": result.metrics.final_value,
        "final_sol_equiv": result.metrics.final_value / final_sol_price,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_drawdown_pct": _drawdown_window(history, "2024-01-01"),
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "bars_below_hf_1_5": result.metrics.bars_below_hf_1_5,
        "total_actions": result.metrics.total_actions,
        "total_interest_paid": result.metrics.total_interest_paid,
        "total_liquidations": result.metrics.total_liquidations,
        "liquidated": result.liquidated,
        "avg_target_short_fraction": float(history["target_short_fraction"].mean()),
        "max_target_short_fraction": float(history["target_short_fraction"].max()),
        "hedge_gate_active_share": float(active_gate.astype(bool).mean()),
        "bars_with_short_target": int((history["target_short_fraction"] > 0.0).sum()),
        "max_debt_ETH_value": float(history["debt_ETH_value"].max()),
        "final_debt_ETH_value": float(history.iloc[-1].get("debt_ETH_value", 0.0)),
        "usd_gap_vs_reference": result.metrics.final_value - REFERENCE["final_usd"],
        "sol_gap_vs_reference": result.metrics.final_value / final_sol_price - REFERENCE["final_sol"],
        "dd_gap_vs_reference": result.metrics.max_drawdown_pct - REFERENCE["max_drawdown_pct"],
    }
    return row


def _selected_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    highest_usd = max(rows, key=lambda row: float(row["final_portfolio_value_usd"]))
    lowest_dd = min(rows, key=lambda row: float(row["max_drawdown_pct"]))
    strict = [
        row
        for row in rows
        if float(row["final_portfolio_value_usd"]) >= REFERENCE["final_usd"]
        and float(row["max_drawdown_pct"]) <= REFERENCE["max_drawdown_pct"]
    ]
    selected = {
        "Highest USD": highest_usd,
        "Lowest DD": lowest_dd,
    }
    if strict:
        selected["Strict reference improvement"] = max(
            strict,
            key=lambda row: float(row["final_portfolio_value_usd"]),
        )
    return selected


def _write_report(out_dir: Path, rows: list[dict[str, object]]) -> None:
    selected = _selected_rows(rows)
    strict = selected.get("Strict reference improvement")
    lines = [
        "# Adaptive Protective Hedge Gate Sweep",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Test path-aware gates for the current SOL/ETH light protective hedge.",
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
                "- No adaptive gate variant beat the static light hedge on both final USD and max DD."
                if strict is None
                else (
                    f"- Strict reference improvement: `{strict['name']}` with "
                    f"${float(strict['final_portfolio_value_usd']):,.2f}, "
                    f"{float(strict['final_sol_equiv']):.3f} SOL, and "
                    f"{float(strict['max_drawdown_pct']):.3f}% max DD."
                )
            ),
            "- Drawdown-slope gates test whether the hedge should activate only while portfolio drawdown is worsening.",
            "- Relative-strength gates test whether ETH should only be shorted when it is underperforming the SOL basis.",
            "",
            "## Top Variants By Final USD",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | Avg Hedge | Gate Active |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: float(item["final_portfolio_value_usd"]), reverse=True)[:8]:
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
    out_dir = Path("reports") / f"adaptive_hedge_gate_sweep_{datetime.now():%Y%m%d_%H%M%S}"
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
