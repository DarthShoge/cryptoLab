"""Pure path composition; source dates and symbols cannot escape the root."""
from datetime import date
from pathlib import Path

from .contracts import symbol


def fill_partition(root: Path, day: str) -> Path:
    return root / "fills" / "hyperliquid" / f"date={date.fromisoformat(day).isoformat()}" / "fills.parquet"


def market_partition(root: Path, coin: str, day: str) -> Path:
    return root / "l2book" / "hyperliquid" / f"{symbol(coin).lower()}_perp" / f"date={date.fromisoformat(day).strftime('%Y%m%d')}" / "l2book.parquet"
