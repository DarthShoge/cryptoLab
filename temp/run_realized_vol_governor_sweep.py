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
REFERENCE = {
    "name": "rs_lb24_u0.0050_g3_0.050_g2_0.050",
    "final_usd": 69_809.83,
    "final_sol": 558.702,
    "max_drawdown_pct": 57.431,
}
ROBUST_FLOORS = {4: 0.00, 3: 0.050, 2: 0.050, 1: 0.00, 0: 0.00}


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
        "protective_hedge_floors": ROBUST_FLOORS,
        "enable_relative_strength_hedge_gate": True,
        "relative_strength_lookback_bars": 24,
        "relative_strength_min_underperformance": 0.005,
    }


def _rv_best_scenario(name: str) -> dict[str, object]:
    scenario = _base_scenario(name)
    scenario.update(
        {
            "enable_realized_vol_governor": True,
            "realized_vol_lookback_bars": 336,
            "realized_vol_target": 0.018,
            "realized_vol_min_long_fraction": 0.70,
        }
    )
    return scenario


def _drawdown_tiers(profile: str) -> list[dict[str, float]]:
    if profile == "soft":
        return [
            {"drawdown": 0.00, "target_long_fraction": 1.075},
            {"drawdown": 0.35, "target_long_fraction": 1.000},
            {"drawdown": 0.45, "target_long_fraction": 0.900},
            {"drawdown": 0.52, "target_long_fraction": 0.800},
        ]
    if profile == "medium":
        return [
            {"drawdown": 0.00, "target_long_fraction": 1.075},
            {"drawdown": 0.30, "target_long_fraction": 1.000},
            {"drawdown": 0.40, "target_long_fraction": 0.850},
            {"drawdown": 0.50, "target_long_fraction": 0.700},
        ]
    if profile == "hard":
        return [
            {"drawdown": 0.00, "target_long_fraction": 1.075},
            {"drawdown": 0.25, "target_long_fraction": 0.950},
            {"drawdown": 0.35, "target_long_fraction": 0.750},
            {"drawdown": 0.45, "target_long_fraction": 0.550},
        ]
    raise ValueError(profile)


def _scenarios() -> list[dict[str, object]]:
    scenarios = [_base_scenario(REFERENCE["name"])]
    for lookback, target_vol, min_long in (
        (24, 0.020, 0.60),
        (24, 0.025, 0.70),
        (24, 0.030, 0.80),
        (72, 0.018, 0.60),
        (72, 0.022, 0.70),
        (72, 0.026, 0.80),
        (72, 0.030, 0.90),
        (168, 0.015, 0.60),
        (168, 0.020, 0.70),
        (168, 0.025, 0.80),
        (168, 0.030, 0.90),
        (336, 0.018, 0.70),
        (336, 0.024, 0.85),
    ):
        scenario = _base_scenario(
            f"rv_lb{lookback}_tv{target_vol:.3f}_min{min_long:.2f}"
        )
        scenario.update(
            {
                "enable_realized_vol_governor": True,
                "realized_vol_lookback_bars": lookback,
                "realized_vol_target": target_vol,
                "realized_vol_min_long_fraction": min_long,
            }
        )
        scenarios.append(scenario)
    for profile in ("soft", "medium", "hard"):
        scenario = _base_scenario(f"dd_{profile}")
        scenario.update(
            {
                "enable_drawdown_governor": True,
                "drawdown_exposure_tiers": _drawdown_tiers(profile),
            }
        )
        scenarios.append(scenario)
        combo = _rv_best_scenario(f"rv336_tv0.018_min0.70_dd_{profile}")
        combo.update(
            {
                "enable_drawdown_governor": True,
                "drawdown_exposure_tiers": _drawdown_tiers(profile),
            }
        )
        scenarios.append(combo)
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
        EngineConfig(lookback_bars=LOOKBACK_BARS),
    ).run(price_data, config)
    history = result.history
    final_sol = result.metrics.final_value / final_sol_price
    realized_vol = history.get("realized_vol_pct", pd.Series(0.0, index=history.index))
    return {
        "name": scenario["name"],
        "final_portfolio_value_usd": result.metrics.final_value,
        "final_sol_equiv": final_sol,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_drawdown_pct": _drawdown_window(history, "2024-01-01"),
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "bars_below_hf_1_5": result.metrics.bars_below_hf_1_5,
        "total_actions": result.metrics.total_actions,
        "total_interest_paid": result.metrics.total_interest_paid,
        "total_liquidations": result.metrics.total_liquidations,
        "liquidated": result.liquidated,
        "avg_target_long_fraction": float(history["target_long_fraction"].mean()),
        "min_target_long_fraction": float(history["target_long_fraction"].min()),
        "avg_target_short_fraction": float(history["target_short_fraction"].mean()),
        "avg_realized_vol_pct": float(realized_vol.mean()),
        "p95_realized_vol_pct": float(realized_vol.quantile(0.95)),
        "vol_capped_share": float(
            (
                history["target_long_fraction"]
                < history["long_green"].map({3: 1.025, 4: 1.075}).fillna(0.0)
            ).mean()
        ),
        "usd_gap_vs_reference": result.metrics.final_value - REFERENCE["final_usd"],
        "sol_gap_vs_reference": final_sol - REFERENCE["final_sol"],
        "dd_gap_vs_reference": result.metrics.max_drawdown_pct
        - REFERENCE["max_drawdown_pct"],
    }


def _selected_rows(rows: pd.DataFrame) -> dict[str, pd.Series]:
    selected = {
        "Highest USD": rows.sort_values(
            ["final_portfolio_value_usd", "max_drawdown_pct"],
            ascending=[False, True],
        ).iloc[0],
        "Lowest DD": rows.sort_values(
            ["max_drawdown_pct", "final_portfolio_value_usd"],
            ascending=[True, False],
        ).iloc[0],
    }
    strict = rows[
        (rows["final_portfolio_value_usd"] >= REFERENCE["final_usd"])
        & (rows["max_drawdown_pct"] <= REFERENCE["max_drawdown_pct"])
    ]
    if not strict.empty:
        selected["Strict reference improvement"] = strict.sort_values(
            ["final_portfolio_value_usd", "max_drawdown_pct"],
            ascending=[False, True],
        ).iloc[0]
    below_55 = rows[rows["max_drawdown_pct"] <= 55.0]
    if not below_55.empty:
        selected["Best DD <=55%"] = below_55.sort_values(
            ["final_portfolio_value_usd", "max_drawdown_pct"],
            ascending=[False, True],
        ).iloc[0]
    return selected


def _write_report(out_dir: Path, rows: pd.DataFrame) -> None:
    selected = _selected_rows(rows)
    top_usd = rows.sort_values(
        ["final_portfolio_value_usd", "max_drawdown_pct"],
        ascending=[False, True],
    ).head(10)
    top_dd = rows.sort_values(
        ["max_drawdown_pct", "final_portfolio_value_usd"],
        ascending=[True, False],
    ).head(10)
    lines = [
        "# Realized-Volatility Governor Sweep",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Test whether scaling long exposure before high-volatility damage can reduce drawdown while retaining the RS-gated SOL/ETH hedge checkpoint.",
        "",
        "## Reference",
        "",
        "| Reference | Final USD | Final SOL | Max DD |",
        "| --- | ---: | ---: | ---: |",
        f"| `{REFERENCE['name']}` | ${REFERENCE['final_usd']:,.2f} | {REFERENCE['final_sol']:.3f} | {REFERENCE['max_drawdown_pct']:.3f}% |",
        "",
        "## Winners",
        "",
        "| Selection | Scenario | Final USD | Final SOL | Max DD | USD Gap | DD Gap | Avg Long | Capped Share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in selected.items():
        lines.append(
            f"| {label} | `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"${float(row['usd_gap_vs_reference']):,.2f} | "
            f"{float(row['dd_gap_vs_reference']):.3f}pp | "
            f"{float(row['avg_target_long_fraction']):.3f} | "
            f"{float(row['vol_capped_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top By USD",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Long | Min Long | Capped Share |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in top_usd.iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['post_2024_drawdown_pct']):.3f}% | "
            f"{float(row['avg_target_long_fraction']):.3f} | "
            f"{float(row['min_target_long_fraction']):.3f} | "
            f"{float(row['vol_capped_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top By Drawdown",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | USD Gap | DD Gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in top_dd.iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"${float(row['usd_gap_vs_reference']):,.2f} | "
            f"{float(row['dd_gap_vs_reference']):.3f}pp |"
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
    out_dir = Path("reports") / f"realized_vol_governor_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in _scenarios():
        row = _run_scenario(scenario, price_data, green_scores)
        rows.append(row)
        print(
            f"{row['name']}\t"
            f"usd=${float(row['final_portfolio_value_usd']):,.2f}\t"
            f"sol={float(row['final_sol_equiv']):.3f}\t"
            f"dd={float(row['max_drawdown_pct']):.3f}\t"
            f"avg_long={float(row['avg_target_long_fraction']):.3f}\t"
            f"cap={float(row['vol_capped_share']):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["final_portfolio_value_usd", "max_drawdown_pct"],
        ascending=[False, True],
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, default=str)
    )
    _write_report(out_dir, summary)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
