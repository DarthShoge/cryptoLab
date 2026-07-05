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
ANCHOR = {
    "name": "signal_g3_1.025_g4_1.075",
    "final_usd": 66_475.117990,
    "final_sol": 532.013749,
    "max_drawdown_pct": 59.293681,
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


def _base_signal_tiers() -> list[dict[str, float]]:
    return [
        {"green": 3, "target_long_fraction": 1.025},
        {"green": 4, "target_long_fraction": 1.075},
    ]


def _scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = [
        {
            "name": "anchor_no_hedge",
            "enable_protective_short_hedge": False,
            "protective_hedge_floors": {},
        }
    ]
    for green3_floor in (0.00, 0.03, 0.05):
        for green2_floor in (0.05, 0.10, 0.15, 0.20):
            for red_floor in (0.20, 0.35):
                floors = {
                    4: 0.00,
                    3: green3_floor,
                    2: green2_floor,
                    1: red_floor,
                    0: red_floor,
                }
                scenarios.append(
                    {
                        "name": (
                            f"hedge_g3_{green3_floor:.2f}"
                            f"_g2_{green2_floor:.2f}"
                            f"_r_{red_floor:.2f}"
                        ),
                        "enable_protective_short_hedge": True,
                        "protective_short_symbol": "ETH",
                        "protective_hedge_floors": floors,
                    }
                )
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
        "signal_exposure_tiers": _base_signal_tiers(),
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
        "final_portfolio_value_usd": final_value,
        "final_sol_equiv": final_value / final_sol_price,
        "total_return_pct": result.metrics.total_return_pct,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_drawdown_pct": _drawdown_window(history, "2024-01-01"),
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "total_liquidations": result.metrics.total_liquidations,
        "liquidated": result.liquidated,
        "avg_target_long_fraction": float(history["target_long_fraction"].mean()),
        "avg_target_short_fraction": float(history["target_short_fraction"].mean()),
        "max_target_short_fraction": float(history["target_short_fraction"].max()),
        "final_debt_ETH_value": float(history.iloc[-1].get("debt_ETH_value", 0.0)),
        "final_collateral_USDC": float(history.iloc[-1].get("collateral_USDC", 0.0)),
        "usd_gap_vs_anchor": final_value - ANCHOR["final_usd"],
        "sol_gap_vs_anchor": final_value / final_sol_price - ANCHOR["final_sol"],
        "dd_gap_vs_anchor": result.metrics.max_drawdown_pct - ANCHOR["max_drawdown_pct"],
    }
    return row


def _selected_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    highest_usd = max(rows, key=lambda row: float(row["final_portfolio_value_usd"]))
    lowest_dd = min(rows, key=lambda row: float(row["max_drawdown_pct"]))
    improving = [
        row
        for row in rows
        if float(row["final_portfolio_value_usd"]) >= ANCHOR["final_usd"]
        and float(row["max_drawdown_pct"]) <= ANCHOR["max_drawdown_pct"]
    ]
    best_under_58 = max(
        (row for row in rows if float(row["max_drawdown_pct"]) <= 58.0),
        key=lambda row: float(row["final_portfolio_value_usd"]),
        default=lowest_dd,
    )
    selected = {
        "Highest USD": highest_usd,
        "Lowest DD": lowest_dd,
        "Best USD <=58% DD": best_under_58,
    }
    if improving:
        selected["Strict anchor improvement"] = max(
            improving,
            key=lambda row: float(row["final_portfolio_value_usd"]),
        )
    return selected


def _write_report(
    out_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    selected = _selected_rows(rows)
    strict_improvement = selected.get("Strict anchor improvement")
    lowest_dd = selected["Lowest DD"]
    highest_usd = selected["Highest USD"]
    lines = [
        "# USD Protective Hedge Sweep",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Adapt the SOL strategy's hedge-floor shorting mechanism to the USD-first multi-asset strategy.",
        "",
        "## Anchor",
        "",
        "| Anchor | Final USD | Final SOL | Max DD |",
        "| --- | ---: | ---: | ---: |",
        f"| `{ANCHOR['name']}` | ${ANCHOR['final_usd']:,.2f} | {ANCHOR['final_sol']:.3f} | {ANCHOR['max_drawdown_pct']:.3f}% |",
        "",
        "## Winners",
        "",
        "| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Hedge |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in selected.items():
        lines.append(
            f"| {label} | `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['post_2024_drawdown_pct']):.3f}% | "
            f"{float(row['avg_target_short_fraction']):.3f}x |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            (
                f"- Best drawdown candidate: `{lowest_dd['name']}` "
                f"with ${float(lowest_dd['final_portfolio_value_usd']):,.2f}, "
                f"{float(lowest_dd['final_sol_equiv']):.3f} SOL, "
                f"and {float(lowest_dd['max_drawdown_pct']):.3f}% max DD."
            ),
            (
                f"- Highest final-value candidate remains `{highest_usd['name']}` "
                f"with ${float(highest_usd['final_portfolio_value_usd']):,.2f} "
                f"and {float(highest_usd['max_drawdown_pct']):.3f}% max DD."
            ),
            (
                "- No protective hedge variant in this grid improved max DD while also "
                "preserving final USD at or above the signal-governor anchor."
                if strict_improvement is None
                else (
                    f"- Strict anchor improvement: `{strict_improvement['name']}` "
                    f"with ${float(strict_improvement['final_portfolio_value_usd']):,.2f}, "
                    f"{float(strict_improvement['final_sol_equiv']):.3f} SOL, "
                    f"and {float(strict_improvement['max_drawdown_pct']):.3f}% max DD."
                )
            ),
            "- This sweep uses ETH debt as the protective short instrument and sizes it against selected or existing long collateral.",
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
    out_dir = Path("reports") / f"usd_protective_hedge_{datetime.now():%Y%m%d_%H%M%S}"
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
            f"avg_short={float(row['avg_target_short_fraction']):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values(
        ["max_drawdown_pct", "final_portfolio_value_usd"],
        ascending=[True, False],
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str))

    _write_report(out_dir, rows)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
