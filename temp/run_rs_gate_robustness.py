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
LOOKBACK_BARS = 24 * 14
WINNING_FLOORS = {4: 0.00, 3: 0.050, 2: 0.025, 1: 0.00, 0: 0.00}


def _max_drawdown_pct(values: pd.Series) -> float:
    running_peak = values.cummax()
    drawdown = (values - running_peak) / running_peak
    return abs(float(drawdown.min())) * 100.0


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


def _base_scenario(name: str, floors: dict[int, float] | None = None) -> dict[str, object]:
    return {
        "name": name,
        "enable_protective_short_hedge": True,
        "protective_short_symbol": "ETH",
        "protective_hedge_floors": floors or WINNING_FLOORS,
    }


def _scenarios() -> list[dict[str, object]]:
    scenarios = [_base_scenario("static_light_hedge")]

    for lookback, underperformance in (
        (18, 0.0000),
        (18, 0.0025),
        (18, 0.0050),
        (18, 0.0075),
        (24, 0.0000),
        (24, 0.0025),
        (24, 0.0050),
        (24, 0.0075),
        (36, 0.0000),
        (36, 0.0025),
        (36, 0.0050),
    ):
        scenario = _base_scenario(f"rs_lb{lookback}_u{underperformance:.4f}")
        scenario.update(
            {
                "enable_relative_strength_hedge_gate": True,
                "relative_strength_lookback_bars": lookback,
                "relative_strength_min_underperformance": underperformance,
            }
        )
        scenarios.append(scenario)

    for green3, green2 in (
        (0.040, 0.000),
        (0.040, 0.025),
        (0.050, 0.000),
        (0.050, 0.025),
        (0.050, 0.050),
        (0.060, 0.025),
    ):
        floors = {4: 0.00, 3: green3, 2: green2, 1: 0.00, 0: 0.00}
        scenario = _base_scenario(
            f"rs_lb24_u0.0050_g3_{green3:.3f}_g2_{green2:.3f}",
            floors=floors,
        )
        scenario.update(
            {
                "enable_relative_strength_hedge_gate": True,
                "relative_strength_lookback_bars": 24,
                "relative_strength_min_underperformance": 0.0050,
            }
        )
        scenarios.append(scenario)

    return scenarios


def _windows() -> dict[str, tuple[str, str]]:
    return {
        "full_2021_2025": ("2021-01-01", "2025-12-31"),
        "early_2021_2023": ("2021-01-01", "2023-12-31"),
        "late_2024_2025": ("2024-01-01", "2025-12-31"),
        "single_2025": ("2025-01-01", "2025-12-31"),
    }


def _slice_window(price_data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return price_data.loc[(price_data.index >= start_ts) & (price_data.index <= end_ts)].copy()


def _run_scenario(
    scenario: dict[str, object],
    price_data: pd.DataFrame,
    green_scores: pd.DataFrame,
    window_name: str,
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
        EngineConfig(lookback_bars=LOOKBACK_BARS),
    ).run(price_data, config)
    history = result.history
    active_gate = history.get("hedge_gate_active", pd.Series(True, index=history.index))
    return {
        "window": window_name,
        "name": scenario["name"],
        "final_portfolio_value_usd": result.metrics.final_value,
        "final_sol_equiv": result.metrics.final_value / final_sol_price,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
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
    }


def _add_static_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    static = (
        out[out["name"] == "static_light_hedge"]
        .set_index("window")
        [["final_portfolio_value_usd", "final_sol_equiv", "max_drawdown_pct"]]
    )
    out["usd_gap_vs_static"] = out.apply(
        lambda row: row["final_portfolio_value_usd"]
        - float(static.loc[row["window"], "final_portfolio_value_usd"]),
        axis=1,
    )
    out["sol_gap_vs_static"] = out.apply(
        lambda row: row["final_sol_equiv"]
        - float(static.loc[row["window"], "final_sol_equiv"]),
        axis=1,
    )
    out["dd_gap_vs_static"] = out.apply(
        lambda row: row["max_drawdown_pct"]
        - float(static.loc[row["window"], "max_drawdown_pct"]),
        axis=1,
    )
    out["strict_improvement_vs_static"] = (
        (out["usd_gap_vs_static"] >= 0.0) & (out["dd_gap_vs_static"] <= 0.0)
    )
    return out


def _stability_table(rows: pd.DataFrame) -> pd.DataFrame:
    candidates = rows[rows["name"] != "static_light_hedge"].copy()
    return (
        candidates.groupby("name")
        .agg(
            windows=("window", "count"),
            strict_wins=("strict_improvement_vs_static", "sum"),
            avg_usd_gap=("usd_gap_vs_static", "mean"),
            min_usd_gap=("usd_gap_vs_static", "min"),
            avg_dd_gap=("dd_gap_vs_static", "mean"),
            max_dd_gap=("dd_gap_vs_static", "max"),
            avg_final_usd=("final_portfolio_value_usd", "mean"),
            avg_final_sol=("final_sol_equiv", "mean"),
        )
        .sort_values(["strict_wins", "avg_usd_gap", "avg_dd_gap"], ascending=[False, False, True])
        .reset_index()
    )


def _write_report(out_dir: Path, rows: pd.DataFrame, stability: pd.DataFrame) -> None:
    best_stability = stability.iloc[0]
    full = rows[rows["window"] == "full_2021_2025"].sort_values(
        ["final_portfolio_value_usd", "max_drawdown_pct"],
        ascending=[False, True],
    )
    best_full = full.iloc[0]
    lines = [
        "# RS Hedge Gate Robustness",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Check whether the relative-strength hedge gate remains useful across adjacent parameters, hedge floors, and subperiods.",
        "",
        "## Headline",
        "",
        (
            f"- Best full-sample variant: `{best_full['name']}` with "
            f"${float(best_full['final_portfolio_value_usd']):,.2f}, "
            f"{float(best_full['final_sol_equiv']):.3f} SOL, "
            f"and {float(best_full['max_drawdown_pct']):.3f}% max DD."
        ),
        (
            f"- Most stable variant by strict wins: `{best_stability['name']}` "
            f"with {int(best_stability['strict_wins'])}/{int(best_stability['windows'])} "
            "subperiod wins versus the static hedge."
        ),
        "",
        "## Stability Ranking",
        "",
        "| Scenario | Strict Wins | Avg USD Gap | Min USD Gap | Avg DD Gap | Max DD Gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in stability.head(10).iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"{int(row['strict_wins'])}/{int(row['windows'])} | "
            f"${float(row['avg_usd_gap']):,.2f} | "
            f"${float(row['min_usd_gap']):,.2f} | "
            f"{float(row['avg_dd_gap']):.3f}pp | "
            f"{float(row['max_dd_gap']):.3f}pp |"
        )
    lines.extend(
        [
            "",
            "## Full-Sample Top Variants",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | USD Gap | DD Gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in full.head(8).iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"${float(row['usd_gap_vs_static']):,.2f} | "
            f"{float(row['dd_gap_vs_static']):.3f}pp |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{out_dir / 'summary.csv'}`",
            f"- Stability CSV: `{out_dir / 'stability.csv'}`",
            f"- Summary JSON: `{out_dir / 'summary.json'}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    all_price_data = _price_data()
    scenarios = _scenarios()
    out_dir = Path("reports") / f"rs_gate_robustness_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for window_name, (start, end) in _windows().items():
        price_data = _slice_window(all_price_data, start, end)
        green_scores = asset_green_counts_by_bar(
            price_data,
            symbols=SYMBOLS,
            atr_period=10,
            multiplier=3.0,
            timeframes=TIMEFRAMES,
        )
        for scenario in scenarios:
            row = _run_scenario(scenario, price_data, green_scores, window_name)
            rows.append(row)
            print(
                f"{window_name}\t{row['name']}\t"
                f"usd=${float(row['final_portfolio_value_usd']):,.2f}\t"
                f"sol={float(row['final_sol_equiv']):.3f}\t"
                f"dd={float(row['max_drawdown_pct']):.3f}",
                flush=True,
            )

    summary = _add_static_deltas(pd.DataFrame(rows))
    stability = _stability_table(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    stability.to_csv(out_dir / "stability.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary.to_dict("records"), indent=2, default=str))
    _write_report(out_dir, summary, stability)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
