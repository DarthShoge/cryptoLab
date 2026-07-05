from __future__ import annotations

import pytest

from arblab.perps.venue import (
    FeeSchedule,
    MarginTier,
    OrderConstraints,
    PerpContractSpec,
    PerpVenueConfig,
    binance_usdm_btcusdt_like,
    generic_linear_usdt_perp,
)


def test_margin_tier_selects_by_absolute_notional():
    venue = PerpVenueConfig(
        name="test",
        contract=PerpContractSpec(symbol="BTC-PERP", settlement_asset="USDT"),
        fees=FeeSchedule(taker=0.001),
        margin_tiers=(
            MarginTier(notional_floor=0.0, notional_cap=50_000.0, initial_margin_rate=0.02, maintenance_margin_rate=0.01),
            MarginTier(notional_floor=50_000.0, notional_cap=None, initial_margin_rate=0.05, maintenance_margin_rate=0.025),
        ),
    )

    assert venue.margin_tier(10_000.0).maintenance_margin_rate == pytest.approx(0.01)
    assert venue.margin_tier(-75_000.0).maintenance_margin_rate == pytest.approx(0.025)


def test_order_constraints_round_quantity_toward_zero_and_skip_dust():
    constraints = OrderConstraints(min_notional=10.0, quantity_step=0.01)

    assert constraints.adjust_notional(123.456, price=100.0) == pytest.approx(123.0)
    assert constraints.adjust_notional(-123.456, price=100.0) == pytest.approx(-123.0)
    assert constraints.adjust_notional(9.99, price=100.0) == 0.0


def test_generic_and_binance_like_presets_are_plain_configs():
    generic = generic_linear_usdt_perp()
    binance_like = binance_usdm_btcusdt_like()

    assert generic.contract.margin_type == "linear"
    assert generic.contract.settlement_asset == "USDT"
    assert binance_like.name == "binance_usdm_like"
    assert binance_like.contract.symbol == "BTCUSDT"
    assert binance_like.order_constraints.min_notional > 0.0
