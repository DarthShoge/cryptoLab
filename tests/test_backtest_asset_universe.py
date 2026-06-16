"""Tests for the curated Kamino research universe."""

from __future__ import annotations

import pytest

from arblab.backtest.asset_universe import (
    AssetUniverseEntry,
    PriceKind,
    curated_kamino_universe,
    exchange_price_configs,
    research_symbols,
    validate_universe_for_market,
)
from arblab.backtest.market import MarketParams


def test_curated_universe_matches_supported_kamino_market_assets():
    market_params = MarketParams.kamino_defaults()

    universe = curated_kamino_universe()

    assert set(universe) == set(market_params.assets)
    validate_universe_for_market(universe, market_params)


def test_curated_universe_separates_exchange_prices_from_pegged_assets():
    universe = curated_kamino_universe()

    exchange_symbols = {
        symbol
        for symbol, entry in universe.items()
        if entry.price_kind == PriceKind.EXCHANGE
    }
    pegged_symbols = {
        symbol
        for symbol, entry in universe.items()
        if entry.price_kind == PriceKind.PEGGED
    }

    assert exchange_symbols == {"SOL", "JitoSOL", "mSOL", "ETH"}
    assert pegged_symbols == {"USDC", "USDT"}
    assert universe["USDC"].peg_target == "USD"
    assert universe["USDT"].peg_target == "USD"


def test_exchange_price_configs_are_deterministic_and_skip_pegged_assets():
    universe = curated_kamino_universe()

    configs = exchange_price_configs(universe)

    assert [(cfg.display_name, cfg.symbol) for cfg in configs] == [
        ("SOL", "SOL/USDT"),
        ("JitoSOL", "JITOSOL/USDT"),
        ("mSOL", "MSOL/USDT"),
        ("ETH", "ETH/USDT"),
    ]


def test_research_symbols_can_filter_for_long_short_candidates():
    universe = curated_kamino_universe()

    assert research_symbols(universe, long_short_only=True) == [
        "SOL",
        "JitoSOL",
        "mSOL",
        "ETH",
    ]
    assert research_symbols(universe, long_short_only=False) == [
        "SOL",
        "JitoSOL",
        "mSOL",
        "USDC",
        "USDT",
        "ETH",
    ]


def test_validation_rejects_assets_missing_from_market_params():
    universe = {
        "BTC": AssetUniverseEntry(
            symbol="BTC",
            price_kind=PriceKind.EXCHANGE,
            exchange_symbol="BTC/USDT",
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=True,
        )
    }

    with pytest.raises(ValueError, match="not configured in market params"):
        validate_universe_for_market(universe, MarketParams.kamino_defaults())


def test_validation_rejects_exchange_assets_without_exchange_symbol():
    universe = {
        "SOL": AssetUniverseEntry(
            symbol="SOL",
            price_kind=PriceKind.EXCHANGE,
            can_be_collateral=True,
            can_be_borrowed=True,
            include_in_long_short_research=True,
        )
    }

    with pytest.raises(ValueError, match="missing exchange_symbol"):
        validate_universe_for_market(universe, MarketParams.kamino_defaults())
