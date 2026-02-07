"""Unit tests for arblab.kamino_recovery pure functions."""

import pytest
from pytest import approx

from arblab.kamino_recovery import (
    auto_loop_deposit,
    collateral_rebalance,
    compute_deficit,
    recovery_deposit_tokens,
    recovery_repay_usd,
    recovery_swap_withdraw,
)
from arblab.kamino_risk import apply_actions, AccountSnapshot
from tests.conftest import make_collateral, make_debt, make_snapshot


# ---------------------------------------------------------------------------
# compute_deficit
# ---------------------------------------------------------------------------

class TestComputeDeficit:
    def test_positive_deficit_needs_recovery(self):
        """Debt outweighs liq_value at target HF => positive deficit."""
        # target_hf * debt - liq_value = 1.25 * 1000 - 750 = 500
        assert compute_deficit(1000, 750, 1.25) == approx(500.0)

    def test_zero_deficit_exactly_at_target(self):
        """Position is exactly at target HF => zero deficit."""
        # 1.5 * 1000 - 1500 = 0
        assert compute_deficit(1000, 1500, 1.5) == approx(0.0)

    def test_negative_deficit_already_safe(self):
        """liq_value exceeds target => negative deficit (already safe)."""
        # 1.0 * 500 - 800 = -300
        assert compute_deficit(500, 800, 1.0) == approx(-300.0)

    def test_zero_debt(self):
        """No debt => deficit is negative (surplus)."""
        assert compute_deficit(0, 500, 1.5) == approx(-500.0)

    def test_zero_liq_value(self):
        """No collateral liq_value => deficit equals target_hf * debt."""
        assert compute_deficit(1000, 0, 1.25) == approx(1250.0)


# ---------------------------------------------------------------------------
# recovery_repay_usd
# ---------------------------------------------------------------------------

class TestRecoveryRepayUsd:
    def test_basic_calculation(self):
        """Straightforward repayment to reach target HF."""
        # repay = debt - liq_value / target_hf = 1000 - 750/1.25 = 1000 - 600 = 400
        assert recovery_repay_usd(1000, 750, 1.25) == approx(400.0)

    def test_target_hf_zero_returns_zero(self):
        """target_hf=0 guard returns 0."""
        assert recovery_repay_usd(1000, 750, 0) == approx(0.0)

    def test_target_hf_negative_returns_zero(self):
        """Negative target_hf returns 0."""
        assert recovery_repay_usd(1000, 750, -1.0) == approx(0.0)

    def test_round_trip_repay_achieves_target_hf(self):
        """Apply the computed repayment to a snapshot and verify HF reaches target."""
        target_hf = 1.25
        # 10 SOL @ $150, liq_threshold=0.75 => liq_value = 1125
        # 1000 USDC debt => HF = 1125/1000 = 1.125 (below 1.25)
        snap = make_snapshot(
            collateral=[make_collateral("SOL", 10, 150, 0.65, 0.75)],
            debt=[make_debt("USDC", 1000, 1.0)],
        )
        total_debt = snap.total_debt_value()
        liq_value = snap.liquidation_value()

        repay_amount = recovery_repay_usd(total_debt, liq_value, target_hf)
        assert repay_amount > 0

        recovered = apply_actions(snap, [
            {"type": "repay", "symbol": "USDC", "amount": repay_amount},
        ])
        assert recovered.health_factor() == approx(target_hf, rel=1e-9)


# ---------------------------------------------------------------------------
# recovery_swap_withdraw
# ---------------------------------------------------------------------------

class TestRecoverySwapWithdraw:
    def test_basic_calculation(self):
        """Basic swap-withdraw amount."""
        # deficit=500, price=150, target_hf=1.25, liq_threshold=0.75
        # raw = 500 / (150 * (1.25 - 0.75)) = 500/75 = 6.667
        result = recovery_swap_withdraw(500, 150, 1.25, 0.75, 100)
        assert result == approx(500 / 75)

    def test_target_hf_equals_liq_threshold_returns_zero(self):
        """When target_hf == liq_threshold, swap can't help."""
        assert recovery_swap_withdraw(500, 150, 0.75, 0.75, 100) == approx(0.0)

    def test_target_hf_below_liq_threshold_returns_zero(self):
        """When target_hf < liq_threshold, returns 0."""
        assert recovery_swap_withdraw(500, 150, 0.5, 0.75, 100) == approx(0.0)

    def test_zero_price_returns_zero(self):
        """Zero collateral price returns 0."""
        assert recovery_swap_withdraw(500, 0, 1.25, 0.75, 100) == approx(0.0)

    def test_capped_at_max_amount(self):
        """Result is capped at max_amount when raw exceeds it."""
        # raw = 500 / (150 * 0.5) = 6.667, max_amount = 3
        result = recovery_swap_withdraw(500, 150, 1.25, 0.75, 3)
        assert result == approx(3.0)

    def test_round_trip_swap_withdraw_achieves_target_hf(self):
        """Withdraw collateral + use proceeds to repay debt => HF reaches target.

        The swap-withdraw formula assumes that withdrawing W tokens of collateral
        (worth W*price) reduces liq_value by W*price*liq_threshold and simultaneously
        reduces debt by W*price. Net HF change should hit the target.
        """
        target_hf = 1.25
        sol_price = 150.0
        liq_threshold = 0.75

        snap = make_snapshot(
            collateral=[make_collateral("SOL", 20, sol_price, 0.65, liq_threshold)],
            debt=[make_debt("USDC", 2000, 1.0)],
        )
        total_debt = snap.total_debt_value()
        liq_value = snap.liquidation_value()
        deficit = compute_deficit(total_debt, liq_value, target_hf)

        withdraw_tokens = recovery_swap_withdraw(
            deficit, sol_price, target_hf, liq_threshold, 20
        )
        withdraw_usd = withdraw_tokens * sol_price

        recovered = apply_actions(snap, [
            {"type": "withdraw_collateral", "symbol": "SOL", "amount": withdraw_tokens},
            {"type": "repay", "symbol": "USDC", "amount": withdraw_usd},
        ])
        assert recovered.health_factor() == approx(target_hf, rel=1e-6)


# ---------------------------------------------------------------------------
# recovery_deposit_tokens
# ---------------------------------------------------------------------------

class TestRecoveryDepositTokens:
    def test_basic_calculation(self):
        """Basic deposit token calculation."""
        # deficit=500, price=150, liq_threshold=0.75
        # deposit = 500 / (150*0.75) = 4.444
        result = recovery_deposit_tokens(500, 150, 0.75)
        assert result == approx(500 / (150 * 0.75))

    def test_zero_price_returns_zero(self):
        assert recovery_deposit_tokens(500, 0, 0.75) == approx(0.0)

    def test_zero_threshold_returns_zero(self):
        assert recovery_deposit_tokens(500, 150, 0) == approx(0.0)

    def test_round_trip_deposit_achieves_target_hf(self):
        """Deposit tokens to reach target HF."""
        target_hf = 1.25
        sol_price = 150.0
        liq_threshold = 0.75

        snap = make_snapshot(
            collateral=[make_collateral("SOL", 10, sol_price, 0.65, liq_threshold)],
            debt=[make_debt("USDC", 1000, 1.0)],
        )
        total_debt = snap.total_debt_value()
        liq_value = snap.liquidation_value()
        deficit = compute_deficit(total_debt, liq_value, target_hf)

        deposit_tokens = recovery_deposit_tokens(deficit, sol_price, liq_threshold)

        recovered = apply_actions(snap, [
            {"type": "deposit_collateral", "symbol": "SOL", "amount": deposit_tokens},
        ])
        assert recovered.health_factor() == approx(target_hf, rel=1e-9)


# ---------------------------------------------------------------------------
# auto_loop_deposit
# ---------------------------------------------------------------------------

class TestAutoLoopDeposit:
    def test_basic_usdc_to_sol(self):
        """Borrow USDC, convert to SOL."""
        # 1000 USDC @ $1, target SOL @ $150 => 1000/150 = 6.667 SOL
        result = auto_loop_deposit(1000, 1.0, 150.0)
        assert result == approx(1000 / 150)

    def test_same_asset_prices_equal(self):
        """Same-price swap => amount unchanged."""
        result = auto_loop_deposit(100, 1.0, 1.0)
        assert result == approx(100.0)

    def test_sol_to_sol_same_price(self):
        """Same-asset loop (SOL@150 -> SOL@150) => 1:1."""
        result = auto_loop_deposit(5, 150, 150)
        assert result == approx(5.0)

    def test_zero_target_price_returns_zero(self):
        assert auto_loop_deposit(1000, 1.0, 0) == approx(0.0)

    def test_negative_target_price_returns_zero(self):
        assert auto_loop_deposit(1000, 1.0, -10) == approx(0.0)


# ---------------------------------------------------------------------------
# collateral_rebalance
# ---------------------------------------------------------------------------

class TestCollateralRebalance:
    def test_basic_calculation(self):
        """Rebalance SOL->USDC at known prices."""
        # withdraw 5 SOL @ $150, deposit into USDC @ $1 => 750 USDC
        result = collateral_rebalance(5, 150, 1.0)
        assert result == approx(750.0)

    def test_same_price_one_to_one(self):
        """Same price => 1:1 token swap."""
        result = collateral_rebalance(10, 100, 100)
        assert result == approx(10.0)

    def test_usd_value_preserved(self):
        """USD value of withdrawn tokens equals USD value of deposited tokens."""
        source_price = 150.0
        target_price = 42.0
        delta = 7.0
        deposit_amount = collateral_rebalance(delta, source_price, target_price)
        assert delta * source_price == approx(deposit_amount * target_price)

    def test_negative_delta_handled(self):
        """Negative delta treated as absolute value."""
        result = collateral_rebalance(-5, 150, 1.0)
        assert result == approx(750.0)

    def test_zero_target_price_returns_zero(self):
        assert collateral_rebalance(5, 150, 0) == approx(0.0)
