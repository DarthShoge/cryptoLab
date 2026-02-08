"""Strategy interface for the backtesting engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from arblab.backtest.market import LiquidationEvent, MarketParams
from arblab.kamino_risk import AccountSnapshot


@dataclass(frozen=True)
class PriceBar:
    """OHLCV data for a single asset at one time step."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class BarData:
    """All data available to a strategy at one time step."""

    timestamp: datetime
    prices: Dict[str, PriceBar]
    history: pd.DataFrame
    bar_index: int
    market_params: MarketParams


class Strategy(ABC):
    """Base class for DeFi lending strategies."""

    @abstractmethod
    def setup(
        self, snapshot: AccountSnapshot, config: Dict[str, Any]
    ) -> AccountSnapshot:
        """Build the initial position. Called once before the backtest loop.

        *snapshot* is typically empty; the strategy should add collateral and
        debt positions as needed and return the resulting snapshot.
        """

    @abstractmethod
    def on_bar(
        self, snapshot: AccountSnapshot, bar: BarData
    ) -> List[Dict[str, Any]]:
        """Return a list of actions for this bar.

        Prices and interest have already been updated before this is called.
        The action format is the same as ``apply_actions`` expects.
        """

    def on_liquidation(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
        event: LiquidationEvent,
    ) -> List[Dict[str, Any]]:
        """Called after a liquidation event. Return recovery actions or ``[]``."""
        return []
