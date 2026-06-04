"""Run initial candidate strategy tests on saved Hyperliquid OHLCV data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arblab.research.data import load_hyperliquid_ohlcv
from arblab.research.candidates import (
    backtest_weight_strategy,
    cross_sectional_momentum_weights,
    moving_average_trend_weights,
)


def run_initial_candidates(
    data_dir: Path,
    interval: str,
    markets: list[str],
    out_dir: Path,
    fee_bps: float,
) -> pd.DataFrame:
    price_data = load_hyperliquid_ohlcv(data_dir, interval, markets)
    periods_per_year = {
        "1h": 8760.0,
        "4h": 2190.0,
        "1d": 365.0,
    }.get(interval, 8760.0)

    results = []

    if "BTC" in markets:
        for fast, slow in [(24, 96), (50, 200)]:
            if len(price_data) < slow + 2:
                continue
            weights = moving_average_trend_weights(price_data, "BTC", fast=fast, slow=slow)
            result = backtest_weight_strategy(
                price_data,
                weights,
                fee_bps=fee_bps,
                periods_per_year=periods_per_year,
            )
            results.append(
                {
                    "candidate": f"btc_sma_{fast}_{slow}",
                    "markets": "BTC",
                    **asdict(result.metrics),
                }
            )

    if len(markets) >= 2:
        xsec_configs = [
            (24, 1, 24),
            (72, 1, 24),
            (168, 3, 168),
            (336, 3, 168),
            (336, 5, 336),
            (504, 5, 168),
        ]
        for lookback, top_n, rebalance_every in xsec_configs:
            if len(price_data) < lookback + 2:
                continue
            weights = cross_sectional_momentum_weights(
                price_data,
                lookback=lookback,
                top_n=top_n,
                rebalance_every=rebalance_every,
            )
            result = backtest_weight_strategy(
                price_data,
                weights,
                fee_bps=fee_bps,
                periods_per_year=periods_per_year,
            )
            results.append(
                {
                    "candidate": f"xsec_mom_{lookback}h_top{top_n}_reb{rebalance_every}h",
                    "markets": ",".join(markets),
                    **asdict(result.metrics),
                }
            )

    summary = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_markdown_report(summary, out_dir / "report.md", data_dir, interval, fee_bps)
    return summary


def write_markdown_report(
    summary: pd.DataFrame,
    path: Path,
    data_dir: Path,
    interval: str,
    fee_bps: float,
) -> None:
    lines = [
        "# Initial Candidate Test Results",
        "",
        f"Data directory: `{data_dir}`",
        f"Interval: `{interval}`",
        f"Fee assumption: `{fee_bps:.2f}` bps per unit turnover",
        "",
    ]
    if summary.empty:
        lines.append("No candidates had enough data to run.")
    else:
        display = summary[
            [
                "candidate",
                "markets",
                "total_return_pct",
                "annualized_return_pct",
                "annualized_volatility_pct",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown_pct",
                "calmar_ratio",
                "avg_turnover",
                "total_trades",
                "exposure_pct",
            ]
        ].copy()
        for col in display.columns:
            if col not in {"candidate", "markets", "total_trades"}:
                display[col] = display[col].map(lambda x: f"{x:.4f}")
        lines.extend(_markdown_table(display))
        lines.append("")
        lines.append("These are smoke-test results on currently available Hyperliquid history, not production research conclusions.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> list[str]:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/hyperliquid/ohlcv_3y_test_chunked"))
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--markets", nargs="+", default=["BTC", "xyz:AAPL"])
    parser.add_argument("--out-dir", type=Path, default=Path("reports/candidate_tests/initial"))
    parser.add_argument("--fee-bps", type=float, default=5.0)
    args = parser.parse_args()

    summary = run_initial_candidates(
        data_dir=args.data_dir,
        interval=args.interval,
        markets=args.markets,
        out_dir=args.out_dir,
        fee_bps=args.fee_bps,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
