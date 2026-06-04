"""Fetch recent Hyperliquid candles for a category universe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import sleep

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arblab.backtest.hyperliquid_data import HyperliquidDataClient
from arblab.research.data import select_liquid_markets, summarize_ohlcv_coverage


def fetch_category_universe(
    category: str,
    interval: str,
    lookback_days: int,
    out_dir: Path,
    max_markets: int | None,
    sleep_seconds: float,
) -> pd.DataFrame:
    client = HyperliquidDataClient(rate_limit_sleep=sleep_seconds)
    if category == "core":
        markets = [
            asset["coin"]
            for asset in client.fetch_perp_universe()
            if not asset.get("is_delisted", False)
        ]
    else:
        markets = client.fetch_assets_by_category(category)
    if max_markets is not None:
        markets = markets[:max_markets]

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=lookback_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    rows = []
    for idx, market in enumerate(markets, start=1):
        try:
            df = client.backfill_candles(market, interval, start_ms, end_ms, out_dir)
            rows.append(
                {
                    "market": market,
                    "rows": len(df),
                    "start": df["timestamp"].min() if len(df) else pd.NaT,
                    "end": df["timestamp"].max() if len(df) else pd.NaT,
                    "status": "ok",
                }
            )
            print(f"[{idx}/{len(markets)}] {market}: {len(df)} rows")
        except Exception as exc:  # pragma: no cover - live data guardrail
            rows.append(
                {
                    "market": market,
                    "rows": 0,
                    "start": pd.NaT,
                    "end": pd.NaT,
                    "status": f"error: {exc}",
                }
            )
            print(f"[{idx}/{len(markets)}] {market}: ERROR {exc}")
        if sleep_seconds:
            sleep(sleep_seconds)

    report = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / f"{category}_{interval}_fetch_report.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="crypto")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--out-dir", type=Path, default=Path("data/hyperliquid/universe"))
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--dedupe-underlying", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()

    fetch_category_universe(
        category=args.category,
        interval=args.interval,
        lookback_days=args.lookback_days,
        out_dir=args.out_dir,
        max_markets=args.max_markets,
        sleep_seconds=args.sleep_seconds,
    )

    summary = summarize_ohlcv_coverage(args.out_dir, args.interval)
    selected = select_liquid_markets(
        summary,
        min_rows=args.min_rows,
        top_n=args.top_n,
        dedupe_underlying=args.dedupe_underlying,
    )
    selected_path = args.out_dir / f"{args.category}_{args.interval}_selected_top{args.top_n}.txt"
    selected_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    summary.to_csv(args.out_dir / f"{args.category}_{args.interval}_coverage.csv", index=False)
    print(f"selected {len(selected)} markets -> {selected_path}")
    print(selected)


if __name__ == "__main__":
    main()
