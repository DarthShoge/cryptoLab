"""OHLCV data fetching via ccxt with CSV caching and multi-asset alignment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_CACHE_DIR = Path(".price_cache")


@dataclass(frozen=True)
class OHLCVConfig:
    """Map a ccxt market symbol to a display name used in the backtest."""

    symbol: str  # e.g. "SOL/USDT"
    display_name: str  # e.g. "SOL"


def _cache_path(
    cache_dir: Path, symbol: str, timeframe: str,
    since_ms: int, until_ms: Optional[int] = None,
) -> Path:
    safe = symbol.replace("/", "_")
    parts = [safe, timeframe, str(since_ms)]
    if until_ms is not None:
        parts.append(str(until_ms))
    return cache_dir / f"{'_'.join(parts)}.csv"


def _fetch_single(
    exchange: "ccxt.Exchange",
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int],
    rate_limit_sleep: float,
) -> pd.DataFrame:
    """Paginated OHLCV fetch for one symbol."""
    all_candles: list = []
    cursor = since_ms

    while True:
        candles = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=cursor, limit=1000
        )
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if until_ms and last_ts >= until_ms:
            break
        if len(candles) < 1000:
            break
        cursor = last_ts + 1
        time.sleep(rate_limit_sleep)

    if not all_candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]

    if until_ms:
        cutoff = pd.Timestamp(until_ms, unit="ms", tz="UTC")
        df = df[df.index <= cutoff]

    return df


def fetch_ohlcv(
    symbols: List[OHLCVConfig],
    timeframe: str = "1h",
    start: str | datetime = "2024-01-01",
    end: Optional[str | datetime] = None,
    exchange_id: str = "binance",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rate_limit_sleep: float = 0.5,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data for multiple assets and return an aligned DataFrame.

    Returns a DataFrame with a ``DatetimeIndex`` and multi-level columns
    ``(display_name, field)`` where *field* is one of
    ``open, high, low, close, volume``.
    """
    import ccxt as _ccxt  # lazy import

    if isinstance(start, str):
        start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    since_ms = int(start.timestamp() * 1000)

    until_ms: Optional[int] = None
    if end is not None:
        if isinstance(end, str):
            end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        until_ms = int(end.timestamp() * 1000)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    exchange = getattr(_ccxt, exchange_id)({"enableRateLimit": True})

    frames: dict[str, pd.DataFrame] = {}

    for cfg in symbols:
        cp = _cache_path(cache_dir, cfg.symbol, timeframe, since_ms, until_ms)

        if use_cache and cp.exists():
            df = pd.read_csv(cp, index_col="timestamp", parse_dates=True)
        else:
            df = _fetch_single(
                exchange, cfg.symbol, timeframe, since_ms, until_ms,
                rate_limit_sleep,
            )
            if use_cache and not df.empty:
                df.to_csv(cp)

        # Filter date range
        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df[df.index >= start]
            if until_ms:
                cutoff = pd.Timestamp(until_ms, unit="ms", tz="UTC")
                df = df[df.index <= cutoff]

        frames[cfg.display_name] = df

    # Align: outer join + forward-fill
    combined = pd.concat(
        {name: df for name, df in frames.items()},
        axis=1,
    )
    combined = combined.ffill()
    combined = combined.dropna()

    return combined
