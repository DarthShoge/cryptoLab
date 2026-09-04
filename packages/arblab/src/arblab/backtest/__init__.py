"""Backtesting framework for DeFi lending strategies."""

from arblab.backtest.market import AssetConfig, LiquidationEvent, MarketParams
from arblab.backtest.strategy import BarData, PriceBar, Strategy
from arblab.backtest.metrics import PerformanceMetrics, compute_metrics
from arblab.backtest.results import BacktestResult
from arblab.backtest.engine import BacktestEngine, EngineConfig
from arblab.backtest.data import OHLCVConfig, fetch_ohlcv
from arblab.backtest.optimizer import OptimizationResult, grid_search

__all__ = [
    "AssetConfig",
    "BacktestEngine",
    "BacktestResult",
    "BarData",
    "EngineConfig",
    "LiquidationEvent",
    "MarketParams",
    "OHLCVConfig",
    "OptimizationResult",
    "PerformanceMetrics",
    "PriceBar",
    "Strategy",
    "compute_metrics",
    "fetch_ohlcv",
    "grid_search",
]
