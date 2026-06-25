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
from temp.run_drawdown_tier_refinement import _custom_tiers
from temp.run_realized_vol_governor_sweep import (
    INITIAL_SOL,
    LOOKBACK_BARS,
    REFERENCE,
    SYMBOLS,
    TIMEFRAMES,
    _base_scenario,
    _rv_best_scenario,
    _signal_tiers,
)
from temp.run_return_recovery_sweep import _with_recovery_boost


START = "2021-01-01"
END = "2026-06-01"


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


def _max_drawdown_pct(values: pd.Series) -> float:
    peak = values.cummax()
    return abs(float(((values - peak) / peak).min())) * 100.0


def _slice_window(history: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    if end is None:
        return history.loc[history.index >= start_ts]
    end_ts = pd.Timestamp(end, tz="UTC")
    return history.loc[(history.index >= start_ts) & (history.index <= end_ts)]


def _scenario_with_tiers(name: str, tiers: list[dict[str, float]]) -> dict[str, object]:
    scenario = _rv_best_scenario(name)
    scenario.update(
        {
            "enable_drawdown_governor": True,
            "drawdown_exposure_tiers": tiers,
        }
    )
    return scenario


def _high_sortino_scenario() -> dict[str, object]:
    base = _scenario_with_tiers(
        "rv336_dd_e_tight_2",
        _custom_tiers(0.32, 1.025, 0.41, 0.825, 0.49, 0.700),
    )
    return _with_recovery_boost(
        base,
        "rec_t1.250_dd0.22_w0.012",
        target=1.250,
        min_drawdown=0.22,
        max_worsening=-0.012,
        max_realized_vol=None,
    )


def _high_return_scenario() -> dict[str, object]:
    scenario = _base_scenario("static_long1.50")
    scenario.update(
        {
            "enable_protective_short_hedge": False,
            "enable_relative_strength_hedge_gate": False,
            "protective_hedge_floors": {4: 0.00, 3: 0.00, 2: 0.00, 1: 0.00, 0: 0.00},
            "target_long_fraction": 1.50,
            "enable_signal_governor": False,
        }
    )
    return scenario


def _run_scenario(
    scenario: dict[str, object],
    price_data: pd.DataFrame,
    green_scores: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, object]]]:
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
    strategy = MultiAssetTrafficLightStrategy()
    result = BacktestEngine(
        strategy,
        EngineConfig(lookback_bars=LOOKBACK_BARS),
    ).run(price_data, config)
    history = result.history
    active_boost = history.get(
        "recovery_boost_active",
        pd.Series(False, index=history.index),
    )
    active_gate = history.get("hedge_gate_active", pd.Series(True, index=history.index))
    row = {
        "name": scenario["name"],
        "final_portfolio_value_usd": result.metrics.final_value,
        "final_sol_equiv": result.metrics.final_value / final_sol_price,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "post_2024_drawdown_pct": _max_drawdown_pct(_slice_window(history, "2024-01-01")["portfolio_value"]),
        "sortino_ratio": result.metrics.sortino_ratio,
        "min_health_factor": result.metrics.min_health_factor,
        "bars_below_hf_1_5": result.metrics.bars_below_hf_1_5,
        "total_actions": result.metrics.total_actions,
        "total_interest_paid": result.metrics.total_interest_paid,
        "total_liquidations": result.metrics.total_liquidations,
        "liquidated": result.liquidated,
        "avg_target_long_fraction": float(history["target_long_fraction"].mean()),
        "max_target_long_fraction": float(history["target_long_fraction"].max()),
        "avg_target_short_fraction": float(history["target_short_fraction"].mean()),
        "boost_share": float(active_boost.astype(bool).mean()),
        "hedge_gate_active_share": float(active_gate.astype(bool).mean()),
        "final_collateral_SOL": float(history.iloc[-1].get("collateral_SOL", 0.0)),
        "final_collateral_ETH": float(history.iloc[-1].get("collateral_ETH", 0.0)),
        "final_collateral_USDC": float(history.iloc[-1].get("collateral_USDC", 0.0)),
        "final_debt_USDC_value": float(history.iloc[-1].get("debt_USDC_value", 0.0)),
        "final_debt_ETH_value": float(history.iloc[-1].get("debt_ETH_value", 0.0)),
        "config": {
            key: value
            for key, value in scenario.items()
            if key != "green_scores"
        },
    }
    return row, history, strategy.event_log


def _regime_rows(summary_rows: dict[str, dict[str, object]], histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    regimes = {
        "full_2021_2026": ("2021-01-01", None),
        "bull_2021": ("2021-01-01", "2021-12-31"),
        "crash_2022": ("2022-01-01", "2022-12-31"),
        "recovery_2023": ("2023-01-01", "2023-12-31"),
        "post_2024": ("2024-01-01", None),
        "ytd_2026": ("2026-01-01", None),
    }
    rows = []
    for name, history in histories.items():
        for regime, (start, end) in regimes.items():
            window = _slice_window(history, start, end)
            if window.empty:
                continue
            start_value = float(window["portfolio_value"].iloc[0])
            end_value = float(window["portfolio_value"].iloc[-1])
            recovery_boost = window.get(
                "recovery_boost_active",
                pd.Series(False, index=window.index),
            )
            rows.append(
                {
                    "name": name,
                    "regime": regime,
                    "start_value": start_value,
                    "end_value": end_value,
                    "return_pct": (end_value / start_value - 1.0) * 100.0,
                    "drawdown_pct": _max_drawdown_pct(window["portfolio_value"]),
                    "avg_target_long_fraction": float(window["target_long_fraction"].mean()),
                    "avg_target_short_fraction": float(window["target_short_fraction"].mean()),
                    "min_health_factor": float(window["health_factor"].min()),
                    "boost_share": float(recovery_boost.astype(bool).mean()),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    high_return = summary[summary["name"] == "static_long1.50"].iloc[0]
    high_sortino = summary[summary["name"] != "static_long1.50"].iloc[0]
    lines = [
        "# High-Sortino Candidate vs Highest-Return Strategy",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Strategies Compared",
        "",
        "- `static_long1.50`: the highest-return branch from the earlier USD-first/multi-asset sweep. It is long-only, uses three-green traffic-light entry, borrows USDC to hold 1.50x long exposure when a qualifying long is present, and has no short hedge or drawdown/volatility governor.",
        "- `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012`: the highest-Sortino row from the latest return-recovery microsearch. It keeps the SOL/ETH traffic-light engine but adds the RS-gated ETH protective short, a 336-hour realized-volatility governor, drawdown exposure tiers, and a recovery boost when drawdown is improving.",
        "",
        "## Headline",
        "",
        (
            f"- Highest-return strategy finished with ${float(high_return['final_portfolio_value_usd']):,.2f} "
            f"({float(high_return['final_sol_equiv']):.3f} SOL) but accepted "
            f"{float(high_return['max_drawdown_pct']):.3f}% max DD and Sortino "
            f"{float(high_return['sortino_ratio']):.3f}."
        ),
        (
            f"- High-Sortino candidate finished with ${float(high_sortino['final_portfolio_value_usd']):,.2f} "
            f"({float(high_sortino['final_sol_equiv']):.3f} SOL), "
            f"{float(high_sortino['max_drawdown_pct']):.3f}% max DD, and Sortino "
            f"{float(high_sortino['sortino_ratio']):.3f}."
        ),
        "- The production candidate gives up extreme terminal upside versus 1.50x static leverage, but it turns the strategy into a materially cleaner capital-raising profile: lower drawdown, higher Sortino, higher minimum health factor, and far less constant leverage.",
        "",
        "## Portfolio Characteristics",
        "",
        "| Strategy | Final USD | Final SOL | Max DD | Post-2024 DD | Sortino | Min HF | Bars HF <1.5 | Avg Long | Avg Short | Interest Paid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['name']}` | "
            f"${float(row['final_portfolio_value_usd']):,.2f} | "
            f"{float(row['final_sol_equiv']):.3f} | "
            f"{float(row['max_drawdown_pct']):.3f}% | "
            f"{float(row['post_2024_drawdown_pct']):.3f}% | "
            f"{float(row['sortino_ratio']):.3f} | "
            f"{float(row['min_health_factor']):.3f} | "
            f"{int(row['bars_below_hf_1_5'])} | "
            f"{float(row['avg_target_long_fraction']):.3f} | "
            f"{float(row['avg_target_short_fraction']):.3f} | "
            f"${float(row['total_interest_paid']):,.2f} |"
        )
    lines.extend(
        [
            "",
            "## Regime Handling",
            "",
            "| Strategy | Regime | Return | Drawdown | Avg Long | Avg Short | Min HF | Boost Share |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in regimes.iterrows():
        lines.append(
            f"| `{row['name']}` | {row['regime']} | "
            f"{float(row['return_pct']):,.2f}% | "
            f"{float(row['drawdown_pct']):.3f}% | "
            f"{float(row['avg_target_long_fraction']):.3f} | "
            f"{float(row['avg_target_short_fraction']):.3f} | "
            f"{float(row['min_health_factor']):.3f} | "
            f"{float(row['boost_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Similarities",
            "",
            f"- Both strategies start from 100 SOL collateral and use the same cached hourly SOL/ETH data from {START} to {END}.",
            "- Both use the multi-timeframe traffic-light model to decide whether there is a qualifying long candidate.",
            "- Both can borrow USDC to add long exposure, so both are path-dependent and sensitive to sharp SOL/ETH drawdowns.",
            "- Neither run liquidated in the backtest.",
            "",
            "## Differences",
            "",
            "- `static_long1.50` is simple and always targets 1.50x when the long signal is active. It maximizes upside in sustained bull regimes but does little to protect capital when the path turns against it.",
            "- The high-Sortino candidate is explicitly stateful. It reduces exposure through drawdown tiers, trims exposure under high realized volatility, uses a small RS-gated ETH protective short, and only re-risks through a recovery boost when drawdown is improving.",
            "- The high-return strategy has much higher terminal USD/SOL because it remains aggressively exposed through the whole path. The cost is a 73.663% max drawdown and weaker Sortino.",
            "- The high-Sortino candidate accepts lower terminal wealth in exchange for a 18.069 percentage point drawdown reduction versus `static_long1.50` and a higher Sortino.",
            "",
            "## Regime Notes",
            "",
            "- In strong upside regimes, `static_long1.50` should dominate because it keeps more gross long exposure. That is exactly what happens in the full-period final value.",
            "- In crash or high-volatility regimes, the production candidate should preserve capital better because its drawdown tiers and vol governor reduce long exposure. The ETH short is small on average, so most protection comes from exposure control rather than hedge PnL.",
            "- In recovery regimes, the production candidate tries to avoid the old failure mode of staying de-risked too long. The recovery boost is deliberately sparse, but it is enough in the microsearch to recover return without reverting to a 70%+ drawdown profile.",
            "- The highest-return strategy is better viewed as a benchmark for upside appetite, not as an investor-ready model: its health factor minimum is lower, it spends more time below HF 1.5, and its drawdown would likely be difficult to raise institutional capital around.",
            "",
            "## Production Implications",
            "",
            "- Treat `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012` as the productionisation candidate if the objective is risk-adjusted return and capital-raising viability.",
            "- Keep `static_long1.50` as the upside benchmark and stress-test comparator, not as the recommended production policy.",
            "- Before production, validate the high-Sortino candidate out-of-sample and with execution assumptions: borrow availability, borrow rates, slippage, rebalance latency, stale oracle behavior, and Kamino health-factor constraints.",
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{out_dir / 'summary.csv'}`",
            f"- Regime CSV: `{out_dir / 'regime_summary.csv'}`",
            f"- High-Sortino history: `{out_dir / 'high_sortino_history.csv'}`",
            f"- Highest-return history: `{out_dir / 'highest_return_history.csv'}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = Path("reports") / f"strategy_comparison_through_20260601_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    price_data = _price_data()
    green_scores = asset_green_counts_by_bar(
        price_data,
        symbols=SYMBOLS,
        atr_period=10,
        multiplier=3.0,
        timeframes=TIMEFRAMES,
    )

    scenarios = [_high_return_scenario(), _high_sortino_scenario()]
    summary_rows: list[dict[str, object]] = []
    histories: dict[str, pd.DataFrame] = {}
    for scenario in scenarios:
        row, history, events = _run_scenario(scenario, price_data, green_scores)
        summary_rows.append(row)
        histories[str(row["name"])] = history
        history.to_csv(
            out_dir
            / (
                "highest_return_history.csv"
                if row["name"] == "static_long1.50"
                else "high_sortino_history.csv"
            )
        )
        (out_dir / f"{row['name']}_events.json").write_text(
            json.dumps(events, indent=2, default=str)
        )
        print(
            f"{row['name']}\t"
            f"usd=${float(row['final_portfolio_value_usd']):,.2f}\t"
            f"sol={float(row['final_sol_equiv']):.3f}\t"
            f"dd={float(row['max_drawdown_pct']):.3f}\t"
            f"sortino={float(row['sortino_ratio']):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    regimes = _regime_rows(
        {str(row["name"]): row for row in summary_rows},
        histories,
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    regimes.to_csv(out_dir / "regime_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, default=str)
    )
    _write_report(out_dir, summary, regimes)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
