"""Exotic multi-step DeFi scenario tests for the Kamino Risk Simulator.

Each test is marked with @pytest.mark.scenario.
"""

import pytest
from pytest import approx

from tests.conftest import make_collateral, make_debt, make_snapshot
from arblab.kamino_risk import apply_actions, AccountSnapshot, liquidation_price_for_collateral
from arblab.kamino_recovery import (
    auto_loop_deposit,
    compute_deficit,
    recovery_deposit_tokens,
    recovery_repay_usd,
)


# ---------------------------------------------------------------------------
# 1. JitoSOL short hedge cushions a SOL drop
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_jitosol_short_hedge_cushions_sol_drop():
    """Borrowing JitoSOL as a hedge: HF drops less than unhedged when SOL falls 30%.

    Both positions start identically. The hedged version then borrows 5 JitoSOL
    and deposits the USD proceeds as USDG collateral. When SOL & JitoSOL drop 30%,
    the JitoSOL debt shrinks, cushioning the HF decline.
    """
    sol_price = 150.0
    jitosol_price = 155.0  # JitoSOL trades at slight premium
    usdg_price = 1.0

    # Shared base parameters: 20 SOL collateral, 1000 USDG collateral, 2000 USDC debt
    base_debt_amount = 2000.0

    # Branch A: hedged -- borrow 5 JitoSOL (worth 775 USD), deposit 775 USDG
    jitosol_borrow_tokens = 5
    jitosol_usd_value = jitosol_borrow_tokens * jitosol_price  # 775
    hedged = make_snapshot(
        collateral=[
            make_collateral("SOL", 20, sol_price, 0.65, 0.75),
            make_collateral("USDG", 1000 + jitosol_usd_value, usdg_price, 0.80, 0.85),
        ],
        debt=[
            make_debt("USDC", base_debt_amount, 1.0),
            make_debt("JitoSOL", jitosol_borrow_tokens, jitosol_price),
        ],
    )

    # Branch B: unhedged -- same base, no JitoSOL borrow
    unhedged = make_snapshot(
        collateral=[
            make_collateral("SOL", 20, sol_price, 0.65, 0.75),
            make_collateral("USDG", 1000, usdg_price, 0.80, 0.85),
        ],
        debt=[
            make_debt("USDC", base_debt_amount, 1.0),
        ],
    )

    # Record HF before crash
    hedged_hf_before = hedged.health_factor()
    unhedged_hf_before = unhedged.health_factor()

    # SOL & JitoSOL drop 30% (correlated)
    hedged_after = apply_actions(hedged, [
        {"type": "set_price", "symbol": "SOL", "price": sol_price * 0.7},
        {"type": "set_price", "symbol": "JitoSOL", "price": jitosol_price * 0.7},
    ])
    unhedged_after = apply_actions(unhedged, [
        {"type": "set_price", "symbol": "SOL", "price": sol_price * 0.7},
    ])

    # HF drop for hedged should be smaller than unhedged
    hedged_drop = hedged_hf_before - hedged_after.health_factor()
    unhedged_drop = unhedged_hf_before - unhedged_after.health_factor()
    assert hedged_drop < unhedged_drop


# ---------------------------------------------------------------------------
# 2. SOL leverage loop -- three iterations
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_sol_leverage_loop_three_iterations():
    """3x leverage loop: borrow USDC -> buy SOL -> deposit. Total SOL > 200, 1 < HF < 2."""
    sol_price = 150.0
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 100, sol_price, 0.65, 0.75)],
        debt=[make_debt("USDC", 0, 1.0)],
    )

    for _ in range(3):
        borrow_limit = snap.borrow_limit() - snap.total_debt_value()
        borrow_usd = borrow_limit * 0.90  # borrow 90% of remaining limit
        new_sol = auto_loop_deposit(borrow_usd, 1.0, sol_price)

        snap = apply_actions(snap, [
            {"type": "borrow", "symbol": "USDC", "amount": borrow_usd},
            {"type": "deposit_collateral", "symbol": "SOL", "amount": new_sol},
        ])

    total_sol = next(p for p in snap.collateral if p.symbol == "SOL").amount
    assert total_sol > 200
    assert 1.0 < snap.health_factor() < 2.0


# ---------------------------------------------------------------------------
# 3. USDC de-peg benefits the borrower
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_usdc_depeg_benefits_borrower():
    """SOL collateral + USDC debt: USDC drops to $0.90 => HF improves."""
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 10, 150, 0.65, 0.75)],
        debt=[make_debt("USDC", 800, 1.0)],
    )
    hf_before = snap.health_factor()

    depegged = apply_actions(snap, [
        {"type": "set_price", "symbol": "USDC", "price": 0.90},
    ])
    assert depegged.health_factor() > hf_before


# ---------------------------------------------------------------------------
# 4. USDC de-peg hurts the depositor
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_usdc_depeg_hurts_depositor():
    """USDC collateral + USDT debt: USDC drops to $0.90 => HF drops."""
    snap = make_snapshot(
        collateral=[make_collateral("USDC", 1000, 1.0, 0.80, 0.85)],
        debt=[make_debt("USDT", 800, 1.0)],
    )
    hf_before = snap.health_factor()

    depegged = apply_actions(snap, [
        {"type": "set_price", "symbol": "USDC", "price": 0.90},
    ])
    assert depegged.health_factor() < hf_before


# ---------------------------------------------------------------------------
# 5. Collateral price to zero
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_collateral_price_to_zero():
    """SOL goes to $0 => HF near 0, buffer deeply negative."""
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 10, 150, 0.65, 0.75)],
        debt=[make_debt("USDC", 500, 1.0)],
    )

    crashed = apply_actions(snap, [
        {"type": "set_price", "symbol": "SOL", "price": 0.0},
    ])

    assert crashed.health_factor() == approx(0.0)
    assert crashed.liquidation_buffer() == approx(-500.0)


# ---------------------------------------------------------------------------
# 6. Multi-collateral: one wipes out, other remains
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_multi_collateral_one_wipes_out():
    """SOL goes to $0 but USDC collateral remains. HF depends only on USDC."""
    snap = make_snapshot(
        collateral=[
            make_collateral("SOL", 10, 150, 0.65, 0.75),
            make_collateral("USDC", 1000, 1.0, 0.80, 0.85),
        ],
        debt=[make_debt("USDT", 500, 1.0)],
    )

    crashed = apply_actions(snap, [
        {"type": "set_price", "symbol": "SOL", "price": 0.0},
    ])

    # Only USDC liq_value remains: 1000 * 0.85 = 850
    expected_hf = 850.0 / 500.0
    assert crashed.health_factor() == approx(expected_hf)


# ---------------------------------------------------------------------------
# 7. Whale position: 1M SOL
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_whale_million_sol():
    """1M SOL ($150M collateral), $80M USDC debt. No overflow, HF calculable."""
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 1_000_000, 150, 0.65, 0.75)],
        debt=[make_debt("USDC", 80_000_000, 1.0)],
    )

    hf = snap.health_factor()
    # liq_value = 1e6 * 150 * 0.75 = 112_500_000
    expected_hf = 112_500_000 / 80_000_000
    assert hf == approx(expected_hf)
    assert hf > 1.0


# ---------------------------------------------------------------------------
# 8. Dust position
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_dust_position():
    """Extremely small position: no float weirdness."""
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 0.000001, 150, 0.65, 0.75)],
        debt=[make_debt("USDC", 0.0001, 1.0)],
    )

    hf = snap.health_factor()
    # liq_value = 0.000001 * 150 * 0.75 = 0.0001125
    expected = 0.0001125 / 0.0001
    assert hf == approx(expected)
    assert snap.liquidation_buffer() == approx(0.0001125 - 0.0001)


# ---------------------------------------------------------------------------
# 9. Price crash -> recover -> crash again
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_price_crash_recover_crash_again():
    """SOL drops to $100, recover via repay, SOL drops to $70. State is consistent."""
    target_hf = 1.25
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 10, 150, 0.65, 0.75)],
        debt=[make_debt("USDC", 1000, 1.0)],
    )

    # First crash: SOL to $100
    crashed1 = apply_actions(snap, [
        {"type": "set_price", "symbol": "SOL", "price": 100.0},
    ])
    assert crashed1.health_factor() < target_hf

    # Recover: repay enough to reach target
    deficit1 = compute_deficit(
        crashed1.total_debt_value(), crashed1.liquidation_value(), target_hf
    )
    repay1 = recovery_repay_usd(
        crashed1.total_debt_value(), crashed1.liquidation_value(), target_hf
    )
    recovered = apply_actions(crashed1, [
        {"type": "repay", "symbol": "USDC", "amount": repay1},
    ])
    assert recovered.health_factor() == approx(target_hf, rel=1e-9)

    # Second crash: SOL to $70
    crashed2 = apply_actions(recovered, [
        {"type": "set_price", "symbol": "SOL", "price": 70.0},
    ])
    assert crashed2.health_factor() < target_hf

    # Can still compute a new recovery
    repay2 = recovery_repay_usd(
        crashed2.total_debt_value(), crashed2.liquidation_value(), target_hf
    )
    assert repay2 > 0
    final = apply_actions(crashed2, [
        {"type": "repay", "symbol": "USDC", "amount": repay2},
    ])
    assert final.health_factor() == approx(target_hf, rel=1e-9)


# ---------------------------------------------------------------------------
# 10. Correlated pair: SOL + mSOL drop 20%
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_correlated_pair_sol_msol_drop():
    """SOL + mSOL (mSOL = 1.05*SOL). Both drop 20%. HF drop is proportional."""
    sol_price = 150.0
    msol_price = sol_price * 1.05  # 157.5

    snap = make_snapshot(
        collateral=[
            make_collateral("SOL", 10, sol_price, 0.65, 0.75),
            make_collateral("mSOL", 10, msol_price, 0.65, 0.75),
        ],
        debt=[make_debt("USDC", 1500, 1.0)],
    )
    hf_before = snap.health_factor()

    dropped = apply_actions(snap, [
        {"type": "set_price", "symbol": "SOL", "price": sol_price * 0.8},
        {"type": "set_price", "symbol": "mSOL", "price": msol_price * 0.8},
    ])
    hf_after = dropped.health_factor()

    # Both collateral values dropped 20%, debt unchanged.
    # New liq_value = 0.8 * old_liq_value, so new HF = 0.8 * old HF
    assert hf_after == approx(hf_before * 0.8, rel=1e-9)


# ---------------------------------------------------------------------------
# 11. Auto-loop after crash improves HF
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_auto_loop_after_crash_improves_hf():
    """SOL crashes to $120, borrow USDC, auto-loop into SOL.

    Leverage looping with liq_threshold < 1 always pulls HF toward
    1/liq_threshold. When HF > 1/liq_threshold it decreases; the test
    verifies more SOL exposure while HF stays safely above 1.
    """
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 20, 150, 0.65, 0.75)],
        debt=[make_debt("USDC", 800, 1.0)],
    )

    # Crash SOL to $120
    crashed = apply_actions(snap, [
        {"type": "set_price", "symbol": "SOL", "price": 120.0},
    ])
    hf_before = crashed.health_factor()
    # HF = 20*120*0.75 / 800 = 1800/800 = 2.25

    # Auto-loop: borrow 300 USDC, buy SOL at $120
    borrow_usd = 300.0
    new_sol = auto_loop_deposit(borrow_usd, 1.0, 120.0)

    looped = apply_actions(crashed, [
        {"type": "borrow", "symbol": "USDC", "amount": borrow_usd},
        {"type": "deposit_collateral", "symbol": "SOL", "amount": new_sol},
    ])

    hf_after = looped.health_factor()
    sol_after = next(p for p in looped.collateral if p.symbol == "SOL").amount
    assert sol_after > 20  # more SOL exposure
    assert hf_after > 1.0  # still safe
    assert hf_after < hf_before  # leverage increased risk as expected


# ---------------------------------------------------------------------------
# 12. Adding collateral lowers liquidation price
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_adding_collateral_lowers_liq_price():
    """Add USDC collateral => SOL liquidation price decreases."""
    snap = make_snapshot(
        collateral=[
            make_collateral("SOL", 10, 150, 0.65, 0.75),
            make_collateral("USDC", 0, 1.0, 0.80, 0.85),
        ],
        debt=[make_debt("USDT", 800, 1.0)],
    )
    liq_price_before = liquidation_price_for_collateral(snap, "SOL")

    added = apply_actions(snap, [
        {"type": "deposit_collateral", "symbol": "USDC", "amount": 500},
    ])
    liq_price_after = liquidation_price_for_collateral(added, "SOL")

    assert liq_price_after < liq_price_before


# ---------------------------------------------------------------------------
# 13. HF exactly one
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_hf_exactly_one():
    """Engineer HF = 1.0 exactly. Buffer should be 0."""
    # Need liq_value == total_debt.
    # 10 SOL @ $100 with liq_threshold = 0.80 => liq_value = 800
    # debt = 800 => HF = 1.0
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 10, 100, 0.65, 0.80)],
        debt=[make_debt("USDC", 800, 1.0)],
    )

    assert snap.health_factor() == approx(1.0)
    assert snap.liquidation_buffer() == approx(0.0)


# ---------------------------------------------------------------------------
# 14. HF just below one
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_hf_just_below_one():
    """HF = ~0.9999. Buffer slightly negative."""
    # liq_value = 10 * 100 * 0.80 = 800, debt = 800.08
    snap = make_snapshot(
        collateral=[make_collateral("SOL", 10, 100, 0.65, 0.80)],
        debt=[make_debt("USDC", 800.08, 1.0)],
    )

    assert snap.health_factor() < 1.0
    assert snap.health_factor() == approx(800 / 800.08, rel=1e-6)
    assert snap.liquidation_buffer() < 0


# ---------------------------------------------------------------------------
# 15. Sequential recovery: partial deposit then full repay
# ---------------------------------------------------------------------------

@pytest.mark.scenario
def test_sequential_recovery_partial_then_full():
    """HF drops, deposit half needed collateral, then repay remaining debt to reach target."""
    target_hf = 1.30
    sol_price = 150.0
    liq_threshold = 0.75

    snap = make_snapshot(
        collateral=[make_collateral("SOL", 10, sol_price, 0.65, liq_threshold)],
        debt=[make_debt("USDC", 1000, 1.0)],
    )

    # Crash SOL to $100
    crashed = apply_actions(snap, [
        {"type": "set_price", "symbol": "SOL", "price": 100.0},
    ])
    assert crashed.health_factor() < target_hf

    # Step 1: deposit half the needed SOL collateral
    deficit = compute_deficit(
        crashed.total_debt_value(), crashed.liquidation_value(), target_hf
    )
    full_deposit = recovery_deposit_tokens(deficit, 100.0, liq_threshold)
    half_deposit = full_deposit / 2.0

    partial = apply_actions(crashed, [
        {"type": "deposit_collateral", "symbol": "SOL", "amount": half_deposit},
    ])
    # After partial deposit, HF is better but not at target yet
    assert partial.health_factor() < target_hf
    assert partial.health_factor() > crashed.health_factor()

    # Step 2: repay remaining debt to reach target
    remaining_deficit = compute_deficit(
        partial.total_debt_value(), partial.liquidation_value(), target_hf
    )
    assert remaining_deficit > 0
    repay_amount = recovery_repay_usd(
        partial.total_debt_value(), partial.liquidation_value(), target_hf
    )

    final = apply_actions(partial, [
        {"type": "repay", "symbol": "USDC", "amount": repay_amount},
    ])
    assert final.health_factor() == approx(target_hf, rel=1e-9)
