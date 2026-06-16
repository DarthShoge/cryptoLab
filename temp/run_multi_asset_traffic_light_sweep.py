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
    "best_no_hf_dd_max_drawdown_pct": 78.990833,
    "best_no_hf_pareto_final_sol": 506.680104,
    "best_no_hf_pareto_max_drawdown_pct": 79.348742,
}


def _max_drawdown_pct(values: pd.Series) -> float:
    running_peak = values.cummax()
    drawdown = (values - running_peak) / running_peak
    return abs(float(drawdown.min())) * 100.0


def _post_2024_drawdown_pct(history: pd.DataFrame) -> float:
    window = history.loc[history.index >= pd.Timestamp("2024-01-01", tz="UTC")]
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


def _scenarios() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for min_long_green in (3, 4):
        for max_short_green in (0,):
            for target_long_fraction in (1.00, 1.50):
                for target_short_fraction in (0.00, 0.30):
                    rows.append(
                        {
                            "name": (
                                f"mlg{min_long_green}_msg{max_short_green}"
                                f"_long{target_long_fraction:.2f}"
                                f"_short{target_short_fraction:.2f}"
                            ),
                            "min_long_green": min_long_green,
                            "max_short_green": max_short_green,
                            "target_long_fraction": target_long_fraction,
                            "target_short_fraction": target_short_fraction,
                        }
                    )
    return rows


def _run_scenario(
    scenario: dict[str, object],
    price_data: pd.DataFrame,
    green_scores: pd.DataFrame,
) -> tuple[dict[str, object], object]:
    first = price_data.iloc[0]
    initial_sol_price = float(first[("SOL", "close")])
    final_sol_price = float(price_data.iloc[-1][("SOL", "close")])
    config = {
        "initial_collateral_symbol": "SOL",
        "initial_collateral_amount": INITIAL_SOL,
        "directional_symbols": SYMBOLS,
        "initial_prices": {
            "SOL": initial_sol_price,
            "ETH": float(first[("ETH", "close")]),
        },
        "green_scores": green_scores,
        "atr_period": 10,
        "supertrend_multiplier": 3.0,
        "timeframes": TIMEFRAMES,
        "rebalance_threshold": 0.05,
        "swap_fee_bps": 10.0,
        **scenario,
    }
    result = BacktestEngine(
        MultiAssetTrafficLightStrategy(),
        EngineConfig(lookback_bars=24 * 14),
    ).run(price_data, config)
    final_value = result.metrics.final_value
    row = {
        **scenario,
        "initial_sol_collateral": INITIAL_SOL,
        "final_portfolio_value_usd": final_value,
        "final_sol_equiv": final_value / final_sol_price,
        "total_return_pct": result.metrics.total_return_pct,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_drawdown_pct": _post_2024_drawdown_pct(result.history),
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "total_liquidations": result.metrics.total_liquidations,
        "liquidated": result.liquidated,
        "final_collateral_SOL": float(result.history.iloc[-1].get("collateral_SOL", 0.0)),
        "final_collateral_ETH": float(result.history.iloc[-1].get("collateral_ETH", 0.0)),
        "final_collateral_USDC": float(result.history.iloc[-1].get("collateral_USDC", 0.0)),
        "final_debt_SOL_value": float(result.history.iloc[-1].get("debt_SOL_value", 0.0)),
        "final_debt_ETH_value": float(result.history.iloc[-1].get("debt_ETH_value", 0.0)),
        "sol_checkpoint_gap": final_value / final_sol_price
        - CHECKPOINTS["best_no_hf_dd_final_sol"],
        "drawdown_checkpoint_gap": result.metrics.max_drawdown_pct
        - CHECKPOINTS["best_no_hf_dd_max_drawdown_pct"],
    }
    return row, result


def _write_report(
    out_dir: Path,
    rows: list[dict[str, object]],
    best_name: str,
    best_result,
) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["max_drawdown_pct"]),
            -float(row["final_sol_equiv"]),
        ),
    )
    best_dd = ranked[0]
    best_sol = max(rows, key=lambda row: float(row["final_sol_equiv"]))
    selected = next(row for row in rows if row["name"] == best_name)
    lines = [
        "# Multi-Asset Traffic-Light Sweep",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        f"- Window: `{START}` to `{END}` hourly.",
        "- Directional assets with local cached data: `SOL`, `ETH`.",
        "- Starting capital: 100 SOL collateral, matching the SOL-equivalent benchmark frame.",
        "- Strategy: rotate long collateral to strongest traffic-light asset and short weakest asset.",
        "",
        "## Checkpoint Comparison",
        "",
        "| Reference | Final SOL | Max DD |",
        "| --- | ---: | ---: |",
        (
            "| Current best DD checkpoint | "
            f"{CHECKPOINTS['best_no_hf_dd_final_sol']:.3f} | "
            f"{CHECKPOINTS['best_no_hf_dd_max_drawdown_pct']:.3f}% |"
        ),
        (
            "| Current Pareto checkpoint | "
            f"{CHECKPOINTS['best_no_hf_pareto_final_sol']:.3f} | "
            f"{CHECKPOINTS['best_no_hf_pareto_max_drawdown_pct']:.3f}% |"
        ),
        "",
        "## Best Candidates",
        "",
        "| Selection | Scenario | Final SOL | Max DD | Post-2024 DD | Liquidations |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, row in (
        ("Lowest DD", best_dd),
        ("Highest SOL", best_sol),
        ("Selected viable", selected),
    ):
        lines.append(
            f"| {label} | `{row['name']}` | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['post_2024_drawdown_pct']):.3f}% | "
            f"{int(row['total_liquidations'])} |"
        )
    lines.extend(
        [
            "",
            "## Initial Read",
            "",
            "This is a first mechanics test, not a tuned result. The important questions are whether cross-asset selection reduces drawdown, whether it preserves enough SOL-equivalent upside, and whether the chosen short leg creates unacceptable liquidation or debt-path risk.",
            "",
            "The selected row is the lowest-drawdown candidate that clears the current best-DD checkpoint final SOL value. If no row clears that gate, selection falls back to the lowest-drawdown row.",
            "",
            "## Findings",
            "",
            (
                f"- The selected viable candidate is `{selected['name']}`: "
                f"{float(selected['final_sol_equiv']):.3f} SOL, "
                f"{float(selected['max_drawdown_pct']):.3f}% max drawdown, "
                f"{float(selected['post_2024_drawdown_pct']):.3f}% post-2024 drawdown, "
                f"and {int(selected['total_liquidations'])} liquidations."
            ),
            (
                "- This still fails the investor-quality drawdown target "
                "if max drawdown remains above 50%."
            ),
            "- In this pass, the weak-asset short overlay should be treated skeptically unless it improves both drawdown and final SOL.",
            "- The next frontier is controlled long exposure between the best low-DD row and the best viable row.",
            "",
            "## Selected Final Exposure",
            "",
            f"- Final SOL collateral: `{float(selected['final_collateral_SOL']):.6f}`",
            f"- Final ETH collateral: `{float(selected['final_collateral_ETH']):.6f}`",
            f"- Final USDC collateral: `{float(selected['final_collateral_USDC']):.2f}`",
            f"- Final SOL debt value: `${float(selected['final_debt_SOL_value']):.2f}`",
            f"- Final ETH debt value: `${float(selected['final_debt_ETH_value']):.2f}`",
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{out_dir / 'summary.csv'}`",
            f"- Summary JSON: `{out_dir / 'summary.json'}`",
            f"- Selected history CSV: `{out_dir / 'selected_history.csv'}`",
            f"- Selected events JSON: `{out_dir / 'selected_events.json'}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    best_result.history.to_csv(out_dir / "selected_history.csv")
    (out_dir / "selected_events.json").write_text(
        json.dumps(best_result.strategy_events, indent=2, default=str)
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
    out_dir = Path("reports") / f"multi_asset_traffic_light_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    green_scores.to_csv(out_dir / "green_scores.csv")

    rows: list[dict[str, object]] = []
    results: dict[str, object] = {}
    for scenario in _scenarios():
        row, result = _run_scenario(scenario, price_data, green_scores)
        rows.append(row)
        results[str(scenario["name"])] = result
        print(
            f"{row['name']}\t"
            f"final_sol={float(row['final_sol_equiv']):.3f}\t"
            f"dd={float(row['max_drawdown_pct']):.3f}\t"
            f"post2024={float(row['post_2024_drawdown_pct']):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values(
        ["max_drawdown_pct", "final_sol_equiv"],
        ascending=[True, False],
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str))

    viable = summary[
        summary["final_sol_equiv"] >= CHECKPOINTS["best_no_hf_dd_final_sol"]
    ].sort_values(["max_drawdown_pct", "final_sol_equiv"], ascending=[True, False])
    selected_name = str(
        viable.iloc[0]["name"] if not viable.empty else summary.iloc[0]["name"]
    )
    _write_report(out_dir, rows, selected_name, results[selected_name])
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
