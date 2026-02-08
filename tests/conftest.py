"""Shared test helpers for Kamino Risk Simulator tests."""

import numpy as np
import pandas as pd

from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition


def make_collateral(
    symbol: str = "SOL",
    amount: float = 10.0,
    price: float = 150.0,
    ltv: float = 0.65,
    liq_threshold: float = 0.75,
) -> CollateralPosition:
    return CollateralPosition(
        symbol=symbol, amount=amount, price=price,
        ltv=ltv, liquidation_threshold=liq_threshold,
    )


def make_debt(
    symbol: str = "USDC",
    amount: float = 500.0,
    price: float = 1.0,
) -> DebtPosition:
    return DebtPosition(symbol=symbol, amount=amount, price=price)


def make_snapshot(
    collateral: list[CollateralPosition] | None = None,
    debt: list[DebtPosition] | None = None,
) -> AccountSnapshot:
    return AccountSnapshot(
        collateral=collateral or [],
        debt=debt or [],
    )


def make_price_data(
    symbols: list[str] | None = None,
    n_bars: int = 100,
    start_prices: dict[str, float] | None = None,
    daily_returns: dict[str, float] | None = None,
    freq: str = "h",
) -> pd.DataFrame:
    """Create synthetic multi-level OHLCV price data for backtesting tests."""
    symbols = symbols or ["SOL"]
    start_prices = start_prices or {s: 150.0 for s in symbols}
    daily_returns = daily_returns or {s: 0.0 for s in symbols}

    dates = pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="UTC")
    frames = {}

    for sym in symbols:
        sp = start_prices.get(sym, 150.0)
        hourly_ret = daily_returns.get(sym, 0.0) / 24.0
        prices = sp * np.cumprod(1 + np.full(n_bars, hourly_ret))
        frames[(sym, "open")] = prices
        frames[(sym, "high")] = prices * 1.01
        frames[(sym, "low")] = prices * 0.99
        frames[(sym, "close")] = prices
        frames[(sym, "volume")] = np.full(n_bars, 1000.0)

    df = pd.DataFrame(frames, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def make_market_params(**overrides):
    """Create MarketParams with kamino_defaults, optionally overriding fields."""
    from arblab.backtest.market import MarketParams
    params = MarketParams.kamino_defaults()
    if overrides:
        from dataclasses import replace
        params = replace(params, **overrides)
    return params
