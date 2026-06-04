"""Data loading and universe selection helpers for research backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arblab.backtest.hyperliquid_data import safe_market_filename


def base_market_name(market: str) -> str:
    """Return the underlying name without a HIP-3 builder namespace."""
    return market.split(":", 1)[-1]


def load_hyperliquid_ohlcv(
    data_dir: Path,
    interval: str,
    markets: list[str],
) -> pd.DataFrame:
    """Load normalized Hyperliquid OHLCV CSVs into a multi-index frame."""
    frames = {}
    for market in markets:
        path = data_dir / interval / f"{safe_market_filename(market)}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        frames[market] = df[["open", "high", "low", "close", "volume"]].astype(float)

    combined = pd.concat(frames, axis=1).ffill().dropna()
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)
    return combined


def summarize_ohlcv_coverage(data_dir: Path, interval: str) -> pd.DataFrame:
    """Summarize stored OHLCV file coverage and approximate dollar volume."""
    rows = []
    interval_dir = data_dir / interval
    if not interval_dir.exists():
        return pd.DataFrame(
            columns=["market", "rows", "start", "end", "avg_dollar_volume"]
        )

    for path in sorted(interval_dir.glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df.empty:
            continue
        market = str(df["coin"].iloc[-1]) if "coin" in df else path.stem
        dollar_volume = df["close"].astype(float) * df["volume"].astype(float)
        rows.append(
            {
                "market": market,
                "rows": int(len(df)),
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
                "avg_dollar_volume": float(dollar_volume.mean()),
            }
        )

    return pd.DataFrame(rows)


def select_liquid_markets(
    summary: pd.DataFrame,
    min_rows: int,
    top_n: int,
    dedupe_underlying: bool = False,
) -> list[str]:
    """Select top markets by average dollar volume after coverage filters."""
    if summary.empty:
        return []

    eligible = summary[summary["rows"] >= min_rows].copy()
    if "is_delisted" in eligible:
        eligible = eligible[~eligible["is_delisted"].fillna(False).astype(bool)]

    eligible = eligible.sort_values("avg_dollar_volume", ascending=False)
    if dedupe_underlying:
        eligible = eligible.assign(_underlying=eligible["market"].map(base_market_name))
        eligible = eligible.drop_duplicates(subset=["_underlying"], keep="first")
    return eligible["market"].head(top_n).tolist()
