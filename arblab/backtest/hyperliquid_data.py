"""Read-only Hyperliquid market data ingestion."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

CANONICAL_CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "num_trades",
    "coin",
    "interval",
    "source",
]

_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


def interval_to_milliseconds(interval: str) -> int:
    """Return the Hyperliquid candle interval length in milliseconds."""
    try:
        return _INTERVAL_MS[interval]
    except KeyError as exc:
        supported = ", ".join(sorted(_INTERVAL_MS))
        raise ValueError(f"Unsupported Hyperliquid interval {interval!r}. Supported: {supported}") from exc


def safe_market_filename(coin: str) -> str:
    """Return a filesystem-safe stem for Hyperliquid market names."""
    unsafe_chars = '<>:"/\\|?*'
    safe = coin
    for char in unsafe_chars:
        safe = safe.replace(char, "_")
    return safe


def _empty_candles() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_CANDLE_COLUMNS)


def _default_post_info(payload: dict[str, Any]) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        HYPERLIQUID_INFO_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "cryptoLab/0.1",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hyperliquid info request failed: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Hyperliquid info request failed: {exc.reason}") from exc


@dataclass
class HyperliquidDataClient:
    """Small client for Hyperliquid public Info API market data."""

    post_info: Callable[[dict[str, Any]], Any] = field(default=_default_post_info)
    rate_limit_sleep: float = 0.1
    max_candles_per_request: int = 5_000

    def request_info(self, payload: dict[str, Any]) -> Any:
        return self.post_info(payload)

    def fetch_perp_universe(self) -> list[dict[str, Any]]:
        """Fetch and normalize Hyperliquid perp universe metadata."""
        meta = self.request_info({"type": "meta"})
        universe = meta.get("universe", [])
        normalized = []
        for asset in universe:
            normalized.append(
                {
                    "coin": asset["name"],
                    "sz_decimals": asset.get("szDecimals"),
                    "max_leverage": asset.get("maxLeverage"),
                    "is_delisted": bool(asset.get("isDelisted", False)),
                }
            )
        return normalized

    def fetch_all_mids(self) -> dict[str, float]:
        """Fetch current midpoint prices for the Hyperliquid perp universe."""
        mids = self.request_info({"type": "allMids"})
        return {coin: float(price) for coin, price in mids.items()}

    def fetch_perp_categories(self) -> dict[str, str]:
        """Fetch category labels for core and HIP-3 perpetual markets."""
        rows = self.request_info({"type": "perpCategories"})
        return {coin: category for coin, category in rows}

    def fetch_assets_by_category(self, category: str) -> list[str]:
        """Return sorted perp market names matching a category label."""
        categories = self.fetch_perp_categories()
        return sorted(
            coin
            for coin, asset_category in categories.items()
            if asset_category.lower() == category.lower()
        )

    def fetch_all_perp_metas(self) -> list[dict[str, Any]]:
        """Fetch normalized metadata for core and builder-deployed perp markets."""
        metas = self.request_info({"type": "allPerpMetas"})
        normalized = []

        for meta in metas:
            dex = meta.get("dex", "")
            for asset in meta.get("universe", []):
                raw_name = asset["name"]
                coin = f"{dex}:{raw_name}" if dex else raw_name
                normalized.append(
                    {
                        "dex": dex,
                        "coin": coin,
                        "raw_name": raw_name,
                        "max_leverage": asset.get("maxLeverage"),
                    }
                )

        return normalized

    def fetch_candles(
        self,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch one Hyperliquid candle snapshot and return canonical OHLCV rows."""
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": int(start_ms),
                "endTime": int(end_ms),
            },
        }
        rows = self.request_info(payload)
        return normalize_candles(rows, coin=coin, interval=interval)

    def backfill_candles(
        self,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        out_dir: str | Path,
    ) -> pd.DataFrame:
        """Fetch candles across a time range, handling Hyperliquid pagination."""
        interval_ms = interval_to_milliseconds(interval)
        cursor = int(start_ms)
        end_ms = int(end_ms)
        frames: list[pd.DataFrame] = []

        while cursor <= end_ms:
            chunk_end = min(
                end_ms,
                cursor + (self.max_candles_per_request - 1) * interval_ms,
            )
            df = self.fetch_candles(coin, interval, cursor, chunk_end)
            if df.empty:
                next_cursor = chunk_end + interval_ms
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                if self.rate_limit_sleep:
                    time.sleep(self.rate_limit_sleep)
                continue
            frames.append(df)

            last_ts = int(df["timestamp"].max().timestamp() * 1000)
            next_cursor = last_ts + interval_ms
            if next_cursor <= cursor or last_ts >= end_ms:
                break
            cursor = next_cursor
            if self.rate_limit_sleep:
                time.sleep(self.rate_limit_sleep)

        combined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else _empty_candles()
        )
        if not combined.empty:
            combined = combined[combined["timestamp"] <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)

        out_path = Path(out_dir) / interval / f"{safe_market_filename(coin)}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        return combined


def normalize_candles(
    rows: list[dict[str, Any]],
    coin: str,
    interval: str,
) -> pd.DataFrame:
    """Normalize raw Hyperliquid candle rows to the project OHLCV schema."""
    if not rows:
        return _empty_candles()

    normalized = []
    for row in rows:
        normalized.append(
            {
                "timestamp": pd.Timestamp(int(row["t"]), unit="ms", tz="UTC"),
                "open": float(row["o"]),
                "high": float(row["h"]),
                "low": float(row["l"]),
                "close": float(row["c"]),
                "volume": float(row["v"]),
                "num_trades": int(row.get("n", 0)),
                "coin": row.get("s", coin),
                "interval": row.get("i", interval),
                "source": "hyperliquid",
            }
        )

    df = pd.DataFrame(normalized, columns=CANONICAL_CANDLE_COLUMNS)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
