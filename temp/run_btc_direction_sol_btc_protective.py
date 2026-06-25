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
TIMEFRAMES = ("1h", "4h", "8h", "1d")
REFERENCE = {
    "name": "current_best_sol_eth_eth_hedge",
    "final_usd": 68_291.651846,
    "final_sol": 546.551836,
    "max_drawdown_pct": 57.439767,
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
            OHLCVConfig(symbol="BTC/USDT", display_name="BTC"),
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


def _scenarios() -> list[dict[str, object]]:
    light_floors = {4: 0.00, 3: 0.050, 2: 0.025, 1: 0.00, 0: 0.00}
    return [
        {
            "name": "current_best_sol_eth_eth_hedge",
            "directional_symbols": ["SOL", "ETH"],
            "enable_protective_short_hedge": True,
            "protective_short_symbol": "ETH",
            "protective_hedge_floors": light_floors,
        },
        {
            "name": "btc_direction_no_protective_hedge",
            "directional_symbols": ["SOL", "ETH", "BTC"],
            "enable_protective_short_hedge": False,
            "protective_hedge_floors": {},
        },
        {
            "name": "btc_direction_eth_hedge_light",
            "directional_symbols": ["SOL", "ETH", "BTC"],
            "enable_protective_short_hedge": True,
            "protective_short_symbol": "ETH",
            "protective_hedge_floors": light_floors,
        },
        {
            "name": "btc_direction_sol_btc_candidates_light",
            "directional_symbols": ["SOL", "ETH", "BTC"],
            "enable_protective_short_hedge": True,
            "protective_short_symbols": ["SOL", "BTC"],
            "protective_hedge_floors": light_floors,
        },
        {
            "name": "btc_direction_sol_btc_candidates_red10",
            "directional_symbols": ["SOL", "ETH", "BTC"],
            "enable_protective_short_hedge": True,
            "protective_short_symbols": ["SOL", "BTC"],
            "protective_hedge_floors": {4: 0.00, 3: 0.050, 2: 0.025, 1: 0.10, 0: 0.10},
        },
        {
            "name": "btc_direction_sol_btc_candidates_red20",
            "directional_symbols": ["SOL", "ETH", "BTC"],
            "enable_protective_short_hedge": True,
            "protective_short_symbols": ["SOL", "BTC"],
            "protective_hedge_floors": {4: 0.00, 3: 0.050, 2: 0.025, 1: 0.20, 0: 0.20},
        },
        {
            "name": "sol_btc_direction_no_protective_hedge",
            "directional_symbols": ["SOL", "BTC"],
            "enable_protective_short_hedge": False,
            "protective_hedge_floors": {},
        },
        {
            "name": "sol_btc_direction_sol_btc_candidates_light",
            "directional_symbols": ["SOL", "BTC"],
            "enable_protective_short_hedge": True,
            "protective_short_symbols": ["SOL", "BTC"],
            "protective_hedge_floors": light_floors,
        },
    ]


def _run_scenario(
    scenario: dict[str, object],
    price_data: pd.DataFrame,
    green_scores: pd.DataFrame,
) -> dict[str, object]:
    symbols = list(scenario["directional_symbols"])
    first = price_data.iloc[0]
    final_sol_price = float(price_data.iloc[-1][("SOL", "close")])
    config = {
        "initial_collateral_symbol": "SOL",
        "initial_collateral_amount": INITIAL_SOL,
        "initial_prices": {
            symbol: float(first[(symbol, "close")])
            for symbol in symbols
        },
        "green_scores": green_scores[symbols],
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
    selected_counts = history["selected_long"].value_counts(normalize=True)
    short_debt_cols = [
        col for col in history.columns if col.startswith("debt_") and col.endswith("_value")
    ]
    row = {
        "name": scenario["name"],
        "directional_symbols": ",".join(symbols),
        "protective_short": ",".join(scenario.get("protective_short_symbols", []))
        or str(scenario.get("protective_short_symbol", "")),
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
        "max_debt_SOL_value": float(history.get("debt_SOL_value", pd.Series(0.0, index=history.index)).max()),
        "max_debt_BTC_value": float(history.get("debt_BTC_value", pd.Series(0.0, index=history.index)).max()),
        "max_debt_ETH_value": float(history.get("debt_ETH_value", pd.Series(0.0, index=history.index)).max()),
        "selected_SOL_share": float(selected_counts.get("SOL", 0.0)),
        "selected_ETH_share": float(selected_counts.get("ETH", 0.0)),
        "selected_BTC_share": float(selected_counts.get("BTC", 0.0)),
        "selected_cash_share": float(selected_counts.get("", 0.0)),
        "usd_gap_vs_reference": result.metrics.final_value - REFERENCE["final_usd"],
        "sol_gap_vs_reference": result.metrics.final_value / final_sol_price - REFERENCE["final_sol"],
        "dd_gap_vs_reference": result.metrics.max_drawdown_pct - REFERENCE["max_drawdown_pct"],
    }
    row["final_short_debt_value"] = float(sum(history.iloc[-1].get(col, 0.0) for col in short_debt_cols))
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
        "# BTC Directional + SOL/BTC Protective Hedge Test",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Objective",
        "",
        "Test BTC in the directional long universe and SOL/BTC as dynamic protective short candidates.",
        "",
        "## Reference",
        "",
        "| Reference | Final USD | Final SOL | Max DD |",
        "| --- | ---: | ---: | ---: |",
        f"| `{REFERENCE['name']}` | ${REFERENCE['final_usd']:,.2f} | {REFERENCE['final_sol']:.3f} | {REFERENCE['max_drawdown_pct']:.3f}% |",
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
                "- No BTC/SOL protective candidate variant beat the current reference on both final USD and max DD."
                if strict is None
                else (
                    f"- Strict reference improvement: `{strict['name']}` with "
                    f"${float(strict['final_portfolio_value_usd']):,.2f}, "
                    f"{float(strict['final_sol_equiv']):.3f} SOL, and "
                    f"{float(strict['max_drawdown_pct']):.3f}% max DD."
                )
            ),
            "- BTC was added to the modeled research market with conservative simple lending parameters; this is a research assumption, not a live Kamino availability claim.",
            "- `protective_short_symbols` chooses the weakest eligible hedge candidate from SOL/BTC and excludes the selected long asset.",
            "",
            "## Tested Variants",
            "",
            "| Scenario | Direction | Protective | Final USD | Final SOL | Max DD | BTC Long Share |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: float(item["final_portfolio_value_usd"]), reverse=True):
        lines.append(
            f"| `{row['name']}` | "
            f"{str(row['directional_symbols']).replace(',', ', ')} | "
            f"{row['protective_short'] or 'none'} | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['selected_BTC_share']):.3f} |"
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
        symbols=["SOL", "ETH", "BTC"],
        atr_period=10,
        multiplier=3.0,
        timeframes=TIMEFRAMES,
    )
    out_dir = Path("reports") / f"btc_direction_sol_btc_protective_{datetime.now():%Y%m%d_%H%M%S}"
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
        ["final_portfolio_value_usd", "max_drawdown_pct"],
        ascending=[False, True],
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    _write_report(out_dir, rows)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
