"""Comprehensive unit tests for arblab.kamino_risk module."""

import math

import pytest

from arblab.kamino_risk import (
    apply_actions,
    liquidation_price_for_collateral,
    liquidation_price_for_debt,
    liquidation_prices,
    scenario_report,
)
from tests.conftest import make_collateral, make_debt, make_snapshot


# ---------------------------------------------------------------------------
# 1. Position values (6 tests)
# ---------------------------------------------------------------------------


class TestCollateralPositionValue:
    def test_value_basic(self):
        pos = make_collateral(amount=10.0, price=150.0)
        assert pos.value() == pytest.approx(1500.0)

    def test_liquidation_value(self):
        pos = make_collateral(amount=10.0, price=150.0, liq_threshold=0.75)
        assert pos.liquidation_value() == pytest.approx(1125.0)

    def test_borrow_limit_value(self):
        pos = make_collateral(amount=10.0, price=150.0, ltv=0.65)
        assert pos.borrow_limit_value() == pytest.approx(975.0)

    def test_zero_amount(self):
        pos = make_collateral(amount=0.0, price=150.0)
        assert pos.value() == pytest.approx(0.0)
        assert pos.liquidation_value() == pytest.approx(0.0)
        assert pos.borrow_limit_value() == pytest.approx(0.0)

    def test_zero_price(self):
        pos = make_collateral(amount=10.0, price=0.0)
        assert pos.value() == pytest.approx(0.0)

    def test_extreme_values(self):
        pos = make_collateral(amount=1e12, price=1e6, ltv=0.99, liq_threshold=0.99)
        assert pos.value() == pytest.approx(1e18)
        assert pos.liquidation_value() == pytest.approx(1e18 * 0.99)
        assert pos.borrow_limit_value() == pytest.approx(1e18 * 0.99)


class TestDebtPositionValue:
    def test_value_basic(self):
        pos = make_debt(amount=500.0, price=1.0)
        assert pos.value() == pytest.approx(500.0)

    def test_zero_amount(self):
        pos = make_debt(amount=0.0, price=1.0)
        assert pos.value() == pytest.approx(0.0)

    def test_zero_price(self):
        pos = make_debt(amount=500.0, price=0.0)
        assert pos.value() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. AccountSnapshot metrics (12 tests)
# ---------------------------------------------------------------------------


class TestAccountSnapshotMetrics:
    def test_total_collateral_value_single(self):
        snap = make_snapshot(collateral=[make_collateral(amount=10.0, price=150.0)])
        assert snap.total_collateral_value() == pytest.approx(1500.0)

    def test_total_collateral_value_multi(self):
        snap = make_snapshot(
            collateral=[
                make_collateral(symbol="SOL", amount=10.0, price=150.0),
                make_collateral(symbol="ETH", amount=2.0, price=3000.0),
            ]
        )
        assert snap.total_collateral_value() == pytest.approx(7500.0)

    def test_total_debt_value_single(self):
        snap = make_snapshot(debt=[make_debt(amount=500.0, price=1.0)])
        assert snap.total_debt_value() == pytest.approx(500.0)

    def test_total_debt_value_multi(self):
        snap = make_snapshot(
            debt=[
                make_debt(symbol="USDC", amount=500.0, price=1.0),
                make_debt(symbol="USDT", amount=300.0, price=1.0),
            ]
        )
        assert snap.total_debt_value() == pytest.approx(800.0)

    def test_liquidation_value_single(self):
        snap = make_snapshot(
            collateral=[make_collateral(amount=10.0, price=150.0, liq_threshold=0.75)]
        )
        assert snap.liquidation_value() == pytest.approx(1125.0)

    def test_liquidation_value_multi(self):
        snap = make_snapshot(
            collateral=[
                make_collateral(symbol="SOL", amount=10.0, price=150.0, liq_threshold=0.75),
                make_collateral(symbol="ETH", amount=2.0, price=3000.0, liq_threshold=0.80),
            ]
        )
        # SOL: 1500*0.75=1125, ETH: 6000*0.80=4800 => 5925
        assert snap.liquidation_value() == pytest.approx(5925.0)

    def test_borrow_limit_multi(self):
        snap = make_snapshot(
            collateral=[
                make_collateral(symbol="SOL", amount=10.0, price=150.0, ltv=0.65),
                make_collateral(symbol="ETH", amount=2.0, price=3000.0, ltv=0.70),
            ]
        )
        # SOL: 1500*0.65=975, ETH: 6000*0.70=4200 => 5175
        assert snap.borrow_limit() == pytest.approx(5175.0)

    def test_health_factor_normal(self):
        snap = make_snapshot(
            collateral=[make_collateral(amount=10.0, price=150.0, liq_threshold=0.75)],
            debt=[make_debt(amount=500.0, price=1.0)],
        )
        # liq_value=1125, debt=500 => hf=2.25
        assert snap.health_factor() == pytest.approx(2.25)

    def test_health_factor_no_debt_is_inf(self):
        snap = make_snapshot(
            collateral=[make_collateral(amount=10.0, price=150.0)],
            debt=[],
        )
        assert snap.health_factor() == float("inf")

    def test_health_factor_below_one(self):
        snap = make_snapshot(
            collateral=[make_collateral(amount=10.0, price=150.0, liq_threshold=0.75)],
            debt=[make_debt(amount=1200.0, price=1.0)],
        )
        # liq_value=1125, debt=1200 => hf=0.9375
        assert snap.health_factor() == pytest.approx(0.9375)

    def test_current_ltv_normal(self):
        snap = make_snapshot(
            collateral=[make_collateral(amount=10.0, price=150.0)],
            debt=[make_debt(amount=500.0, price=1.0)],
        )
        # debt/collateral = 500/1500
        assert snap.current_ltv() == pytest.approx(500.0 / 1500.0)

    def test_current_ltv_no_collateral_is_inf(self):
        snap = make_snapshot(
            collateral=[],
            debt=[make_debt(amount=500.0, price=1.0)],
        )
        assert snap.current_ltv() == float("inf")

    def test_liquidation_buffer_positive(self):
        snap = make_snapshot(
            collateral=[make_collateral(amount=10.0, price=150.0, liq_threshold=0.75)],
            debt=[make_debt(amount=500.0, price=1.0)],
        )
        # 1125 - 500 = 625
        assert snap.liquidation_buffer() == pytest.approx(625.0)

    def test_liquidation_buffer_negative(self):
        snap = make_snapshot(
            collateral=[make_collateral(amount=10.0, price=150.0, liq_threshold=0.75)],
            debt=[make_debt(amount=1200.0, price=1.0)],
        )
        # 1125 - 1200 = -75
        assert snap.liquidation_buffer() == pytest.approx(-75.0)

    def test_liquidation_buffer_zero(self):
        snap = make_snapshot(
            collateral=[make_collateral(amount=10.0, price=150.0, liq_threshold=0.75)],
            debt=[make_debt(amount=1125.0, price=1.0)],
        )
        assert snap.liquidation_buffer() == pytest.approx(0.0)

    def test_empty_snapshot(self):
        snap = make_snapshot()
        assert snap.total_collateral_value() == pytest.approx(0.0)
        assert snap.total_debt_value() == pytest.approx(0.0)
        assert snap.liquidation_value() == pytest.approx(0.0)
        assert snap.borrow_limit() == pytest.approx(0.0)
        assert snap.liquidation_buffer() == pytest.approx(0.0)
        assert snap.current_ltv() == float("inf")
        assert snap.health_factor() == float("inf")


# ---------------------------------------------------------------------------
# 3. Liquidation price calculators (12 tests)
# ---------------------------------------------------------------------------


class TestLiquidationPriceForCollateral:
    def test_basic(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0, liq_threshold=0.75)],
            debt=[make_debt(amount=500.0, price=1.0)],
        )
        # liq_price = total_debt / (amount * liq_threshold) = 500 / (10*0.75) = 66.666...
        result = liquidation_price_for_collateral(snap, "SOL")
        assert result == pytest.approx(500.0 / 7.5)

    def test_multi_collateral(self):
        snap = make_snapshot(
            collateral=[
                make_collateral(symbol="SOL", amount=10.0, price=150.0, liq_threshold=0.75),
                make_collateral(symbol="ETH", amount=2.0, price=3000.0, liq_threshold=0.80),
            ],
            debt=[make_debt(amount=5000.0, price=1.0)],
        )
        # other_liq_value for SOL = ETH: 6000*0.80=4800
        # required = 5000 - 4800 = 200
        # liq_price = 200 / (10*0.75) = 26.666...
        result = liquidation_price_for_collateral(snap, "SOL")
        assert result == pytest.approx(200.0 / 7.5)

    def test_zero_threshold_returns_none(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0, liq_threshold=0.0)],
            debt=[make_debt(amount=500.0, price=1.0)],
        )
        assert liquidation_price_for_collateral(snap, "SOL") is None

    def test_zero_amount_returns_none(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=0.0, price=150.0, liq_threshold=0.75)],
            debt=[make_debt(amount=500.0, price=1.0)],
        )
        assert liquidation_price_for_collateral(snap, "SOL") is None

    def test_missing_symbol_returns_none(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
            debt=[make_debt(amount=500.0, price=1.0)],
        )
        assert liquidation_price_for_collateral(snap, "BTC") is None

    def test_floor_at_zero(self):
        # When other collateral covers the debt, liquidation price floors at 0
        snap = make_snapshot(
            collateral=[
                make_collateral(symbol="SOL", amount=10.0, price=150.0, liq_threshold=0.75),
                make_collateral(symbol="ETH", amount=10.0, price=3000.0, liq_threshold=0.80),
            ],
            debt=[make_debt(amount=100.0, price=1.0)],
        )
        # other_liq_value = 30000*0.80 = 24000 >> 100
        # required = 100 - 24000 = -23900, max(0, negative) = 0
        result = liquidation_price_for_collateral(snap, "SOL")
        assert result == pytest.approx(0.0)


class TestLiquidationPriceForDebt:
    def test_basic(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0, liq_threshold=0.75)],
            debt=[make_debt(symbol="USDC", amount=500.0, price=1.0)],
        )
        # liq_price = liq_value / amount = 1125 / 500 = 2.25
        result = liquidation_price_for_debt(snap, "USDC")
        assert result == pytest.approx(2.25)

    def test_multi_debt(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0, liq_threshold=0.75)],
            debt=[
                make_debt(symbol="USDC", amount=500.0, price=1.0),
                make_debt(symbol="USDT", amount=300.0, price=1.0),
            ],
        )
        # other_debt for USDC = 300
        # required = 1125 - 300 = 825
        # liq_price = 825 / 500 = 1.65
        result = liquidation_price_for_debt(snap, "USDC")
        assert result == pytest.approx(1.65)

    def test_zero_amount_returns_none(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
            debt=[make_debt(symbol="USDC", amount=0.0, price=1.0)],
        )
        assert liquidation_price_for_debt(snap, "USDC") is None

    def test_missing_symbol_returns_none(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
            debt=[make_debt(symbol="USDC", amount=500.0, price=1.0)],
        )
        assert liquidation_price_for_debt(snap, "DAI") is None

    def test_floor_at_zero(self):
        # When other debt already exceeds the liquidation value
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=1.0, price=10.0, liq_threshold=0.5)],
            debt=[
                make_debt(symbol="USDC", amount=100.0, price=1.0),
                make_debt(symbol="USDT", amount=1000.0, price=1.0),
            ],
        )
        # liq_value = 10*0.5 = 5, other_debt for USDC = 1000
        # required = 5 - 1000 = -995, max(0, negative) = 0
        result = liquidation_price_for_debt(snap, "USDC")
        assert result == pytest.approx(0.0)


class TestLiquidationPricesDict:
    def test_structure(self):
        snap = make_snapshot(
            collateral=[
                make_collateral(symbol="SOL", amount=10.0, price=150.0, liq_threshold=0.75),
                make_collateral(symbol="ETH", amount=2.0, price=3000.0, liq_threshold=0.80),
            ],
            debt=[
                make_debt(symbol="USDC", amount=500.0, price=1.0),
                make_debt(symbol="USDT", amount=300.0, price=1.0),
            ],
        )
        result = liquidation_prices(snap)
        assert "collateral" in result
        assert "debt" in result
        assert set(result["collateral"].keys()) == {"SOL", "ETH"}
        assert set(result["debt"].keys()) == {"USDC", "USDT"}
        # Verify values match the individual function calls
        assert result["collateral"]["SOL"] == pytest.approx(
            liquidation_price_for_collateral(snap, "SOL")
        )
        assert result["debt"]["USDC"] == pytest.approx(
            liquidation_price_for_debt(snap, "USDC")
        )


# ---------------------------------------------------------------------------
# 4. apply_actions (11 tests)
# ---------------------------------------------------------------------------


class TestApplyActions:
    def test_deposit_collateral(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
        )
        result = apply_actions(snap, [{"type": "deposit_collateral", "symbol": "SOL", "amount": 5.0}])
        sol = next(p for p in result.collateral if p.symbol == "SOL")
        assert sol.amount == pytest.approx(15.0)

    def test_withdraw_collateral(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
        )
        result = apply_actions(snap, [{"type": "withdraw_collateral", "symbol": "SOL", "amount": 3.0}])
        sol = next(p for p in result.collateral if p.symbol == "SOL")
        assert sol.amount == pytest.approx(7.0)

    def test_borrow(self):
        snap = make_snapshot(
            debt=[make_debt(symbol="USDC", amount=500.0, price=1.0)],
        )
        result = apply_actions(snap, [{"type": "borrow", "symbol": "USDC", "amount": 200.0}])
        usdc = next(p for p in result.debt if p.symbol == "USDC")
        assert usdc.amount == pytest.approx(700.0)

    def test_repay(self):
        snap = make_snapshot(
            debt=[make_debt(symbol="USDC", amount=500.0, price=1.0)],
        )
        result = apply_actions(snap, [{"type": "repay", "symbol": "USDC", "amount": 200.0}])
        usdc = next(p for p in result.debt if p.symbol == "USDC")
        assert usdc.amount == pytest.approx(300.0)

    def test_set_price_updates_collateral_and_debt(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
            debt=[make_debt(symbol="SOL", amount=5.0, price=150.0)],
        )
        result = apply_actions(snap, [{"type": "set_price", "symbol": "SOL", "price": 200.0}])
        col_sol = next(p for p in result.collateral if p.symbol == "SOL")
        debt_sol = next(p for p in result.debt if p.symbol == "SOL")
        assert col_sol.price == pytest.approx(200.0)
        assert debt_sol.price == pytest.approx(200.0)

    def test_update_price_updates_collateral_and_debt(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
            debt=[make_debt(symbol="SOL", amount=5.0, price=150.0)],
        )
        result = apply_actions(snap, [{"type": "update_price", "symbol": "SOL", "price": 180.0}])
        col_sol = next(p for p in result.collateral if p.symbol == "SOL")
        debt_sol = next(p for p in result.debt if p.symbol == "SOL")
        assert col_sol.price == pytest.approx(180.0)
        assert debt_sol.price == pytest.approx(180.0)

    def test_missing_collateral_symbol_raises(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
        )
        with pytest.raises(ValueError, match="Collateral asset BTC not found"):
            apply_actions(snap, [{"type": "deposit_collateral", "symbol": "BTC", "amount": 1.0}])

    def test_missing_debt_symbol_raises(self):
        snap = make_snapshot(
            debt=[make_debt(symbol="USDC", amount=500.0, price=1.0)],
        )
        with pytest.raises(ValueError, match="Debt asset DAI not found"):
            apply_actions(snap, [{"type": "borrow", "symbol": "DAI", "amount": 100.0}])

    def test_unknown_action_raises(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
        )
        with pytest.raises(ValueError, match="Unknown action type"):
            apply_actions(snap, [{"type": "liquidate", "symbol": "SOL", "amount": 1.0}])

    def test_set_price_missing_price_raises(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0)],
        )
        with pytest.raises(ValueError, match="Price is required"):
            apply_actions(snap, [{"type": "set_price", "symbol": "SOL"}])

    def test_multiple_sequential_actions(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0, ltv=0.65, liq_threshold=0.75)],
            debt=[make_debt(symbol="USDC", amount=500.0, price=1.0)],
        )
        result = apply_actions(snap, [
            {"type": "deposit_collateral", "symbol": "SOL", "amount": 5.0},
            {"type": "repay", "symbol": "USDC", "amount": 100.0},
            {"type": "set_price", "symbol": "SOL", "price": 200.0},
        ])
        sol = next(p for p in result.collateral if p.symbol == "SOL")
        usdc = next(p for p in result.debt if p.symbol == "USDC")
        assert sol.amount == pytest.approx(15.0)
        assert sol.price == pytest.approx(200.0)
        assert usdc.amount == pytest.approx(400.0)

    def test_immutability(self):
        original_collateral = make_collateral(symbol="SOL", amount=10.0, price=150.0)
        original_debt = make_debt(symbol="USDC", amount=500.0, price=1.0)
        snap = make_snapshot(
            collateral=[original_collateral],
            debt=[original_debt],
        )
        apply_actions(snap, [
            {"type": "deposit_collateral", "symbol": "SOL", "amount": 5.0},
            {"type": "repay", "symbol": "USDC", "amount": 100.0},
        ])
        # Original snapshot should be unchanged
        assert snap.collateral[0].amount == pytest.approx(10.0)
        assert snap.debt[0].amount == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 5. scenario_report (2 tests)
# ---------------------------------------------------------------------------


class TestScenarioReport:
    def test_has_all_keys(self):
        snap = make_snapshot(
            collateral=[make_collateral(symbol="SOL", amount=10.0, price=150.0, ltv=0.65, liq_threshold=0.75)],
            debt=[make_debt(symbol="USDC", amount=500.0, price=1.0)],
        )
        report = scenario_report(snap)
        expected_keys = {
            "total_collateral_value",
            "total_debt_value",
            "borrow_limit",
            "current_ltv",
            "health_factor",
            "liquidation_buffer",
        }
        assert set(report.keys()) == expected_keys

    def test_values_match_direct_calls(self):
        snap = make_snapshot(
            collateral=[
                make_collateral(symbol="SOL", amount=10.0, price=150.0, ltv=0.65, liq_threshold=0.75),
                make_collateral(symbol="ETH", amount=2.0, price=3000.0, ltv=0.70, liq_threshold=0.80),
            ],
            debt=[
                make_debt(symbol="USDC", amount=500.0, price=1.0),
                make_debt(symbol="USDT", amount=300.0, price=1.0),
            ],
        )
        report = scenario_report(snap)
        assert report["total_collateral_value"] == pytest.approx(snap.total_collateral_value())
        assert report["total_debt_value"] == pytest.approx(snap.total_debt_value())
        assert report["borrow_limit"] == pytest.approx(snap.borrow_limit())
        assert report["current_ltv"] == pytest.approx(snap.current_ltv())
        assert report["health_factor"] == pytest.approx(snap.health_factor())
        assert report["liquidation_buffer"] == pytest.approx(snap.liquidation_buffer())
