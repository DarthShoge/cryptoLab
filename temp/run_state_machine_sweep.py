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


def _state_machine_scenario(
    name: str,
    orange_long: float,
    orange_short: float,
    red_long: float,
    recovery_long: float,
    orange_min_drawdown: float = 0.10,
    orange_min_worsening: float = 0.03,
    recovery_max_worsening: float = -0.03,
) -> dict[str, object]:
    scenario = _base_scenario(name)
    scenario.update(
        {
            "enable_traffic_light_state_machine": True,
            "state_machine_orange_min_drawdown": orange_min_drawdown,
            "state_machine_orange_min_worsening": orange_min_worsening,
            "state_machine_recovery_min_drawdown": 0.10,
            "state_machine_recovery_max_worsening": recovery_max_worsening,
            "state_machine_recovery_min_green": 4,
            "state_machine_targets": {
                "red": {
                    "target_long_fraction": red_long,
                    "target_short_fraction": 0.00,
                },
                "orange": {
                    "target_long_fraction": orange_long,
                    "target_short_fraction": orange_short,
                },
                "recovery": {
                    "target_long_fraction": recovery_long,
                    "target_short_fraction": 0.00,
                },
            },
        }
    )
    return scenario


def _hedge_only_state_machine_scenario(
    name: str,
    yellow_short: float,
    orange_short: float,
    green_short: float = 0.0,
    orange_min_drawdown: float = 0.05,
    orange_min_worsening: float = 0.02,
) -> dict[str, object]:
    scenario = _base_scenario(name)
    scenario.update(
        {
            "enable_traffic_light_state_machine": True,
            "state_machine_orange_min_drawdown": orange_min_drawdown,
            "state_machine_orange_min_worsening": orange_min_worsening,
            "state_machine_recovery_min_drawdown": 0.10,
            "state_machine_recovery_max_worsening": -0.03,
            "state_machine_recovery_min_green": 4,
            "state_machine_targets": {
                "green": {"target_short_fraction": green_short},
                "yellow": {"target_short_fraction": yellow_short},
                "orange": {"target_short_fraction": orange_short},
                "recovery": {"target_short_fraction": 0.00},
            },
        }
    )
    return scenario


def _scenarios() -> list[dict[str, object]]:
    scenarios = [_base_scenario(REFERENCE["name"])]
    for recovery_long in (1.10, 1.15, 1.20):
        for orange_long, orange_short in (
            (0.80, 0.05),
            (0.65, 0.10),
            (0.50, 0.15),
        ):
            scenarios.append(
                _state_machine_scenario(
                    (
                        f"sm_o{orange_long:.2f}_s{orange_short:.2f}"
                        f"_r0.25_rec{recovery_long:.2f}"
                    ),
                    orange_long=orange_long,
                    orange_short=orange_short,
                    red_long=0.25,
                    recovery_long=recovery_long,
                )
            )
    for red_long in (0.00, 0.50):
        scenarios.append(
            _state_machine_scenario(
                f"sm_o0.65_s0.10_r{red_long:.2f}_rec1.15",
                orange_long=0.65,
                orange_short=0.10,
                red_long=red_long,
                recovery_long=1.15,
            )
        )
    for orange_min_drawdown, orange_min_worsening in ((0.05, 0.02), (0.15, 0.05)):
        scenarios.append(
            _state_machine_scenario(
                (
                    f"sm_o0.65_s0.10_r0.25_rec1.15"
                    f"_dd{orange_min_drawdown:.2f}_w{orange_min_worsening:.2f}"
                ),
                orange_long=0.65,
                orange_short=0.10,
                red_long=0.25,
                recovery_long=1.15,
                orange_min_drawdown=orange_min_drawdown,
                orange_min_worsening=orange_min_worsening,
            )
        )
    for yellow_short, orange_short in (
        (0.025, 0.075),
        (0.050, 0.100),
        (0.075, 0.150),
        (0.100, 0.200),
    ):
        scenarios.append(
            _hedge_only_state_machine_scenario(
                f"sm_hedge_only_y{yellow_short:.3f}_o{orange_short:.3f}",
                yellow_short=yellow_short,
                orange_short=orange_short,
            )
        )
    for green_short, yellow_short, orange_short in (
        (0.025, 0.050, 0.100),
        (0.025, 0.075, 0.150),
        (0.050, 0.075, 0.150),
    ):
        scenarios.append(
            _hedge_only_state_machine_scenario(
                (
                    f"sm_hedge_only_g{green_short:.3f}"
                    f"_y{yellow_short:.3f}_o{orange_short:.3f}"
                ),
                green_short=green_short,
                yellow_short=yellow_short,
                orange_short=orange_short,
            )
        )
    for orange_min_drawdown, orange_min_worsening in ((0.00, 0.00), (0.10, 0.03)):
        scenarios.append(
            _hedge_only_state_machine_scenario(
                (
                    "sm_hedge_only_y0.050_o0.100"
                    f"_dd{orange_min_drawdown:.2f}_w{orange_min_worsening:.2f}"
                ),
                yellow_short=0.050,
                orange_short=0.100,
                orange_min_drawdown=orange_min_drawdown,
                orange_min_worsening=orange_min_worsening,
            )
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
    state_share = (
        history.get("traffic_light_state", pd.Series("base", index=history.index))
        .value_counts(normalize=True)
        .to_dict()
    )
    active_gate = history.get("hedge_gate_active", pd.Series(True, index=history.index))
    final_sol = result.metrics.final_value / final_sol_price
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
        "avg_target_short_fraction": float(history["target_short_fraction"].mean()),
        "max_target_short_fraction": float(history["target_short_fraction"].max()),
        "hedge_gate_active_share": float(active_gate.astype(bool).mean()),
        "bars_with_short_target": int((history["target_short_fraction"] > 0.0).sum()),
        "base_share": float(state_share.get("base", 0.0)),
        "green_share": float(state_share.get("green", 0.0)),
        "yellow_share": float(state_share.get("yellow", 0.0)),
        "orange_share": float(state_share.get("orange", 0.0)),
        "red_share": float(state_share.get("red", 0.0)),
        "recovery_share": float(state_share.get("recovery", 0.0)),
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
        "# Traffic-Light State Machine Sweep",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Test whether a more explicit green/yellow/orange/red/recovery process can reduce USD drawdown without giving up the current best SOL/ETH RS-gated hedge result.",
        "",
        "## Reference",
        "",
        "| Reference | Final USD | Final SOL | Max DD |",
        "| --- | ---: | ---: | ---: |",
        f"| `{REFERENCE['name']}` | ${REFERENCE['final_usd']:,.2f} | {REFERENCE['final_sol']:.3f} | {REFERENCE['max_drawdown_pct']:.3f}% |",
        "",
        "## Winners",
        "",
        "| Selection | Scenario | Final USD | Final SOL | Max DD | USD Gap | DD Gap | Orange | Recovery |",
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
            f"{float(row['orange_share']):.3f} | "
            f"{float(row['recovery_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top By USD",
            "",
            "| Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Long | Avg Short | Mode Shares G/Y/O/R/Rec |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
            f"{float(row['avg_target_short_fraction']):.3f} | "
            f"{float(row['green_share']):.2f}/"
            f"{float(row['yellow_share']):.2f}/"
            f"{float(row['orange_share']):.2f}/"
            f"{float(row['red_share']):.2f}/"
            f"{float(row['recovery_share']):.2f} |"
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
    out_dir = Path("reports") / f"state_machine_sweep_{datetime.now():%Y%m%d_%H%M%S}"
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
            f"orange={float(row['orange_share']):.3f}\t"
            f"recovery={float(row['recovery_share']):.3f}",
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
