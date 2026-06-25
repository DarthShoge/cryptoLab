"""Curated asset universe for multi-asset Kamino backtest research."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from arblab.backtest.data import OHLCVConfig
from arblab.backtest.market import MarketParams


class PriceKind(StrEnum):
    """How the backtester should source an asset's USD price."""

    EXCHANGE = "exchange"
    PEGGED = "pegged"
    DERIVED = "derived"


@dataclass(frozen=True)
class AssetUniverseEntry:
    """Research metadata for one supported Kamino asset."""

    symbol: str
    price_kind: PriceKind
    can_be_collateral: bool
    can_be_borrowed: bool
    include_in_long_short_research: bool
    exchange_symbol: str | None = None
    peg_target: str | None = None
    derived_from: str | None = None
    notes: str = ""


def curated_kamino_universe() -> dict[str, AssetUniverseEntry]:
    """Return the v1 curated universe supported by current market defaults.

    This deliberately starts from assets already modeled in ``MarketParams``.
    Broader Kamino coverage should be added only after validating both market
    availability and a durable price source.
    """
    entries = [
        AssetUniverseEntry(
            symbol="SOL",
            price_kind=PriceKind.EXCHANGE,
            exchange_symbol="SOL/USDT",
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=True,
            notes="Base SOL market; primary benchmark and collateral asset.",
        ),
        AssetUniverseEntry(
            symbol="JitoSOL",
            price_kind=PriceKind.EXCHANGE,
            exchange_symbol="JITOSOL/USDT",
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=True,
            notes="SOL liquid staking token with direct exchange data in v1.",
        ),
        AssetUniverseEntry(
            symbol="mSOL",
            price_kind=PriceKind.EXCHANGE,
            exchange_symbol="MSOL/USDT",
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=True,
            notes="SOL liquid staking token with direct exchange data in v1.",
        ),
        AssetUniverseEntry(
            symbol="USDC",
            price_kind=PriceKind.PEGGED,
            peg_target="USD",
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=False,
            notes="Stablecoin used for quote, collateral, debt, and reserve legs.",
        ),
        AssetUniverseEntry(
            symbol="USDT",
            price_kind=PriceKind.PEGGED,
            peg_target="USD",
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=False,
            notes="Stablecoin modeled as USD-pegged debt/collateral capacity.",
        ),
        AssetUniverseEntry(
            symbol="ETH",
            price_kind=PriceKind.EXCHANGE,
            exchange_symbol="ETH/USDT",
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=True,
            notes="Major exchange asset currently used as the SOL hedge leg.",
        ),
        AssetUniverseEntry(
            symbol="BTC",
            price_kind=PriceKind.EXCHANGE,
            exchange_symbol="BTC/USDT",
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=True,
            notes="Major exchange asset added for BTC directional and protective hedge research.",
        ),
    ]
    return {entry.symbol: entry for entry in entries}


def validate_universe_for_market(
    universe: Mapping[str, AssetUniverseEntry],
    market_params: MarketParams,
) -> None:
    """Raise if the universe cannot be safely used with market params."""
    for symbol, entry in universe.items():
        if symbol != entry.symbol:
            raise ValueError(f"{symbol} entry symbol mismatch: {entry.symbol}")
        if symbol not in market_params.assets:
            raise ValueError(f"{symbol} is not configured in market params")
        if entry.price_kind == PriceKind.EXCHANGE and not entry.exchange_symbol:
            raise ValueError(f"{symbol} is missing exchange_symbol")
        if entry.price_kind == PriceKind.PEGGED and not entry.peg_target:
            raise ValueError(f"{symbol} is missing peg_target")
        if entry.price_kind == PriceKind.DERIVED and not entry.derived_from:
            raise ValueError(f"{symbol} is missing derived_from")
        if entry.include_in_long_short_research and not (
            entry.can_be_collateral and entry.can_be_borrowed
        ):
            raise ValueError(
                f"{symbol} cannot be long/short researched without both "
                "collateral and borrow support"
            )


def exchange_price_configs(
    universe: Mapping[str, AssetUniverseEntry] | None = None,
) -> list[OHLCVConfig]:
    """Build OHLCV configs for assets that need external exchange data."""
    if universe is None:
        universe = curated_kamino_universe()
    return [
        OHLCVConfig(symbol=entry.exchange_symbol or "", display_name=entry.symbol)
        for entry in universe.values()
        if entry.price_kind == PriceKind.EXCHANGE
    ]


def research_symbols(
    universe: Mapping[str, AssetUniverseEntry] | None = None,
    long_short_only: bool = True,
) -> list[str]:
    """Return curated symbols in deterministic research order."""
    if universe is None:
        universe = curated_kamino_universe()
    return [
        entry.symbol
        for entry in universe.values()
        if not long_short_only or entry.include_in_long_short_research
    ]
