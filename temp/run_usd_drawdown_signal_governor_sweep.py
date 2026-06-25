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
ANCHORS = {
    "static_1_05_final_usd": 65_410.664091,
    "static_1_05_final_sol": 523.494711,
    "static_1_05_max_drawdown_pct": 59.354412,
    "static_1_075_final_usd": 70_853.338101,
    "static_1_075_final_sol": 567.053526,
    "static_1_075_max_drawdown_pct": 60.246312,
    "prior_sol_checkpoint_final_usd": 62_003.62,
    "prior_sol_checkpoint_final_sol": 496.353122,
    "prior_sol_checkpoint_max_drawdown_pct": 78.990833,
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


def _signal_tiers(green3: float, green4: float) -> list[dict[str, float]]:
    return [
        {"green": 3, "target_long_fraction": green3},
        {"green": 4, "target_long_fraction": green4},
    ]


def _scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = [
        {
            "name": "static_1.05",
            "target_long_fraction": 1.05,
            "enable_signal_governor": False,
        },
        {
            "name": "static_1.075",
            "target_long_fraction": 1.075,
            "enable_signal_governor": False,
        },
        {
            "name": "static_1.10",
            "target_long_fraction": 1.10,
            "enable_signal_governor": False,
        },
    ]
    for green4 in (1.075, 1.10, 1.125, 1.15):
        for green3 in (0.75, 0.90, 1.00, 1.025, 1.05):
            if green3 > green4:
                continue
            scenarios.append(
                {
                    "name": f"signal_g3_{green3:.3f}_g4_{green4:.3f}",
                    "target_long_fraction": green4,
                    "enable_signal_governor": True,
                    "signal_exposure_tiers": _signal_tiers(green3, green4),
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
        "enable_signal_governor": scenario.get("enable_signal_governor", False),
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
        "green3_pct": float((history["long_green"] == 3).mean() * 100.0),
        "green4_pct": float((history["long_green"] == 4).mean() * 100.0),
        "risk_off_pct": float((history["target_long_fraction"] < 1.0).mean() * 100.0),
        "usd_gap_vs_static_1_05": final_value - ANCHORS["static_1_05_final_usd"],
        "dd_gap_vs_static_1_05": result.metrics.max_drawdown_pct
        - ANCHORS["static_1_05_max_drawdown_pct"],
        "usd_gap_vs_prior_sol_checkpoint": final_value
        - ANCHORS["prior_sol_checkpoint_final_usd"],
        "sol_gap_vs_prior_sol_checkpoint": final_value / final_sol_price
        - ANCHORS["prior_sol_checkpoint_final_sol"],
    }
    return row, result


def _selected_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    highest_usd = max(rows, key=lambda row: float(row["final_portfolio_value_usd"]))
    lowest_dd = min(rows, key=lambda row: float(row["max_drawdown_pct"]))
    best_under_60 = max(
        (row for row in rows if float(row["max_drawdown_pct"]) <= 60.0),
        key=lambda row: float(row["final_portfolio_value_usd"]),
        default=lowest_dd,
    )
    beats_anchor = [
        row
        for row in rows
        if float(row["final_portfolio_value_usd"]) >= ANCHORS["static_1_05_final_usd"]
        and float(row["max_drawdown_pct"]) <= ANCHORS["static_1_05_max_drawdown_pct"]
    ]
    best_anchor_improvement = (
        max(beats_anchor, key=lambda row: float(row["final_portfolio_value_usd"]))
        if beats_anchor
        else best_under_60
    )
    return {
        "Highest USD": highest_usd,
        "Lowest DD": lowest_dd,
        "Best USD <=60% DD": best_under_60,
        "Best anchor improvement": best_anchor_improvement,
    }


def _write_report(
    out_dir: Path,
    rows: list[dict[str, object]],
    selected_results: dict[str, object],
) -> None:
    selected = _selected_rows(rows)
    anchor_improvement = selected["Best anchor improvement"]
    lines = [
        "# USD Drawdown Signal-Governor Sweep",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Minimize USD drawdown while preserving the USD-first advantage from the static 1.05x row.",
        "",
        "## Anchors",
        "",
        "| Anchor | Final USD | Final SOL | Max DD |",
        "| --- | ---: | ---: | ---: |",
        f"| Static 1.05x | ${ANCHORS['static_1_05_final_usd']:,.2f} | {ANCHORS['static_1_05_final_sol']:.3f} | {ANCHORS['static_1_05_max_drawdown_pct']:.3f}% |",
        f"| Static 1.075x | ${ANCHORS['static_1_075_final_usd']:,.2f} | {ANCHORS['static_1_075_final_sol']:.3f} | {ANCHORS['static_1_075_max_drawdown_pct']:.3f}% |",
        f"| Prior SOL checkpoint | ${ANCHORS['prior_sol_checkpoint_final_usd']:,.2f} | {ANCHORS['prior_sol_checkpoint_final_sol']:.3f} | {ANCHORS['prior_sol_checkpoint_max_drawdown_pct']:.3f}% |",
        "",
        "## Winners",
        "",
        "| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Exposure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in selected.items():
        lines.append(
            f"| {label} | `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['post_2024_drawdown_pct']):.3f}% | "
            f"{float(row['avg_target_long_fraction']):.3f}x |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            (
                f"- Best anchor-improvement candidate: `{anchor_improvement['name']}` "
                f"with ${float(anchor_improvement['final_portfolio_value_usd']):,.2f}, "
                f"{float(anchor_improvement['final_sol_equiv']):.3f} SOL, "
                f"and {float(anchor_improvement['max_drawdown_pct']):.3f}% max DD."
            ),
            "- A valid improvement must beat static 1.05x on USD without increasing max DD.",
            "- If no signal-governed row qualifies, the static 1.05x anchor remains the best USD/DD checkpoint.",
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{out_dir / 'summary.csv'}`",
            f"- Summary JSON: `{out_dir / 'summary.json'}`",
            f"- Winner event files generated locally: `{out_dir}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    for label, result in selected_results.items():
        safe_label = label.lower().replace(" ", "_").replace("<=", "lte").replace("%", "pct")
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
    out_dir = Path("reports") / f"usd_drawdown_signal_governor_{datetime.now():%Y%m%d_%H%M%S}"
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
            f"dd={float(row['max_drawdown_pct']):.3f}\t"
            f"avg_exp={float(row['avg_target_long_fraction']):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values(
        ["max_drawdown_pct", "final_portfolio_value_usd"],
        ascending=[True, False],
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
