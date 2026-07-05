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
CHECKPOINTS = {
    "best_no_hf_dd_final_sol": 496.353122,
    "best_no_hf_dd_final_usd": 62_003.62,
    "best_no_hf_dd_max_drawdown_pct": 78.990833,
    "multi_asset_long_1_5_final_sol": 1978.739546,
    "multi_asset_long_1_5_final_usd": 247_243.51,
    "multi_asset_long_1_5_max_drawdown_pct": 73.662851,
}


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
        use_cache=True,
    )


def _tiers(name: str) -> list[dict[str, float]]:
    if name == "mild":
        return [
            {"drawdown": 0.00, "target_long_fraction": 1.50},
            {"drawdown": 0.25, "target_long_fraction": 1.25},
            {"drawdown": 0.40, "target_long_fraction": 1.00},
        ]
    if name == "balanced":
        return [
            {"drawdown": 0.00, "target_long_fraction": 1.50},
            {"drawdown": 0.15, "target_long_fraction": 1.25},
            {"drawdown": 0.30, "target_long_fraction": 1.00},
            {"drawdown": 0.45, "target_long_fraction": 0.75},
        ]
    if name == "strict":
        return [
            {"drawdown": 0.00, "target_long_fraction": 1.50},
            {"drawdown": 0.10, "target_long_fraction": 1.25},
            {"drawdown": 0.20, "target_long_fraction": 1.00},
            {"drawdown": 0.35, "target_long_fraction": 0.50},
        ]
    raise ValueError(f"Unknown tier preset: {name}")


def _scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for exposure in (
        1.00,
        1.025,
        1.05,
        1.075,
        1.10,
        1.20,
        1.25,
        1.30,
        1.40,
        1.50,
    ):
        scenarios.append(
            {
                "name": f"static_long{exposure:.2f}",
                "target_long_fraction": exposure,
                "enable_drawdown_governor": False,
            }
        )
    for preset in ("mild", "balanced", "strict"):
        scenarios.append(
            {
                "name": f"governor_{preset}",
                "target_long_fraction": 1.50,
                "enable_drawdown_governor": True,
                "drawdown_exposure_tiers": _tiers(preset),
            }
        )
    return scenarios


def _run_scenario(
    scenario: dict[str, object],
    price_data: pd.DataFrame,
    green_scores: pd.DataFrame,
) -> tuple[dict[str, object], object]:
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
        "target_short_fraction": 0.00,
        "rebalance_threshold": 0.05,
        "swap_fee_bps": 10.0,
        **scenario,
    }
    result = BacktestEngine(
        MultiAssetTrafficLightStrategy(),
        EngineConfig(lookback_bars=24 * 14),
    ).run(price_data, config)
    history = result.history
    final_value = result.metrics.final_value
    row = {
        "name": scenario["name"],
        "target_long_fraction": scenario["target_long_fraction"],
        "enable_drawdown_governor": scenario.get("enable_drawdown_governor", False),
        "final_portfolio_value_usd": final_value,
        "final_sol_equiv": final_value / final_sol_price,
        "total_return_pct": result.metrics.total_return_pct,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_drawdown_pct": _drawdown_window(history, "2024-01-01"),
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "total_liquidations": result.metrics.total_liquidations,
        "liquidated": result.liquidated,
        "max_target_long_fraction": float(history["target_long_fraction"].max()),
        "avg_target_long_fraction": float(history["target_long_fraction"].mean()),
        "min_target_long_fraction": float(history["target_long_fraction"].min()),
        "max_equity_drawdown_signal_pct": float(history["equity_drawdown_pct"].max()),
        "final_collateral_SOL": float(history.iloc[-1].get("collateral_SOL", 0.0)),
        "final_collateral_ETH": float(history.iloc[-1].get("collateral_ETH", 0.0)),
        "final_collateral_USDC": float(history.iloc[-1].get("collateral_USDC", 0.0)),
        "final_debt_USDC_value": float(history.iloc[-1].get("debt_USDC_value", 0.0)),
        "usd_gap_vs_static_1_5": final_value
        - CHECKPOINTS["multi_asset_long_1_5_final_usd"],
        "sol_gap_vs_static_1_5": final_value / final_sol_price
        - CHECKPOINTS["multi_asset_long_1_5_final_sol"],
        "dd_gap_vs_static_1_5": result.metrics.max_drawdown_pct
        - CHECKPOINTS["multi_asset_long_1_5_max_drawdown_pct"],
    }
    return row, result


def _selected_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    lowest_dd = min(rows, key=lambda row: float(row["max_drawdown_pct"]))
    highest_usd = max(rows, key=lambda row: float(row["final_portfolio_value_usd"]))
    highest_sol = max(rows, key=lambda row: float(row["final_sol_equiv"]))
    best_usd_under_60 = max(
        (row for row in rows if float(row["max_drawdown_pct"]) <= 60.0),
        key=lambda row: float(row["final_portfolio_value_usd"]),
        default=lowest_dd,
    )
    return {
        "Lowest DD": lowest_dd,
        "Highest USD": highest_usd,
        "Highest SOL": highest_sol,
        "Best USD <=60% DD": best_usd_under_60,
    }


def _write_report(
    out_dir: Path,
    rows: list[dict[str, object]],
    selected_results: dict[str, object],
) -> None:
    selected = _selected_rows(rows)
    highest_usd = selected["Highest USD"]
    highest_sol = selected["Highest SOL"]
    best_under_60 = selected["Best USD <=60% DD"]
    usd_also_sol = highest_usd["name"] == highest_sol["name"]
    lines = [
        "# USD-First Governed Multi-Asset Sweep",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Primary objective is final USD equity. SOL-equivalent is tracked as a secondary outcome to test whether maximizing USD also maximizes SOL by effect.",
        "",
        "## Setup",
        "",
        f"- Window: `{START}` to `{END}` hourly.",
        "- Directional assets with local cached data: `SOL`, `ETH`.",
        "- Starting capital: 100 SOL collateral.",
        "- Signal: three-green multi-timeframe traffic-light entry.",
        "- Short overlay disabled because the prior sweep showed it hurt both USD/SOL outcome and drawdown.",
        "",
        "## Checkpoint Comparison",
        "",
        "| Reference | Final USD | Final SOL equiv | Max DD |",
        "| --- | ---: | ---: | ---: |",
        (
            "| Prior static 1.5x multi-asset | "
            f"${CHECKPOINTS['multi_asset_long_1_5_final_usd']:,.2f} | "
            f"{CHECKPOINTS['multi_asset_long_1_5_final_sol']:.3f} | "
            f"{CHECKPOINTS['multi_asset_long_1_5_max_drawdown_pct']:.3f}% |"
        ),
        (
            "| Prior SOL checkpoint | "
            f"${CHECKPOINTS['best_no_hf_dd_final_usd']:,.2f} | "
            f"{CHECKPOINTS['best_no_hf_dd_final_sol']:.3f} | "
            f"{CHECKPOINTS['best_no_hf_dd_max_drawdown_pct']:.3f}% |"
        ),
        "",
        "## Winners",
        "",
        "| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Min HF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in selected.items():
        lines.append(
            f"| {label} | `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['post_2024_drawdown_pct']):.3f}% | "
            f"{float(row['min_health_factor']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Does USD Maximization Also Maximize SOL?",
            "",
            (
                "Yes for this sweep."
                if usd_also_sol
                else "No for this sweep."
            ),
            (
                f"The highest-USD scenario is `{highest_usd['name']}` and the "
                f"highest-SOL scenario is `{highest_sol['name']}`."
            ),
            "",
            "## Findings",
            "",
            "- USD and SOL-equivalent rank together in this limited SOL/ETH universe because final SOL price is common across candidates.",
            (
                f"- The best sub-60% drawdown row is `{best_under_60['name']}`: "
                f"${float(best_under_60['final_portfolio_value_usd']):,.2f}, "
                f"{float(best_under_60['final_sol_equiv']):.3f} SOL, "
                f"{float(best_under_60['max_drawdown_pct']):.3f}% max DD, "
                f"and {int(best_under_60['total_liquidations'])} liquidations."
            ),
            "- The drawdown governor needs to improve the USD/DD frontier, not merely cut exposure after losses. If it exits too late, it can reduce compounding without materially reducing the historical max drawdown.",
            "- The next governor should be proactive rather than reactive: reduce exposure before equity drawdown compounds, using traffic-light deterioration, volatility, or trend-break conditions rather than portfolio drawdown alone.",
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{out_dir / 'summary.csv'}`",
            f"- Summary JSON: `{out_dir / 'summary.json'}`",
            f"- Winner history/event files generated locally: `{out_dir}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    for label, result in selected_results.items():
        safe_label = label.lower().replace(" ", "_").replace("<=", "lte").replace("%", "pct")
        result.history.to_csv(out_dir / f"{safe_label}_history.csv")
        (out_dir / f"{safe_label}_events.json").write_text(
            json.dumps(result.strategy_events, indent=2, default=str)
        )


def main() -> None:
    price_data = _price_data()
    green_scores = asset_green_counts_by_bar(
        price_data,
        symbols=SYMBOLS,
        atr_period=10,
        multiplier=3.0,
        timeframes=TIMEFRAMES,
    )
    out_dir = Path("reports") / f"usd_first_multi_asset_governor_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    results: dict[str, object] = {}
    for scenario in _scenarios():
        row, result = _run_scenario(scenario, price_data, green_scores)
        rows.append(row)
        results[str(row["name"])] = result
        print(
            f"{row['name']}\t"
            f"usd=${float(row['final_portfolio_value_usd']):,.2f}\t"
            f"sol={float(row['final_sol_equiv']):.3f}\t"
            f"dd={float(row['max_drawdown_pct']):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values(
        ["final_portfolio_value_usd", "max_drawdown_pct"],
        ascending=[False, True],
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str))

    selections = _selected_rows(rows)
    selected_results = {
        label: results[str(row["name"])]
        for label, row in selections.items()
    }
    _write_report(out_dir, rows, selected_results)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
