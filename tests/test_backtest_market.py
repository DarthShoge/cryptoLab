"""Tests for arblab.backtest.market module.

Covers interest accrual, LST yield, liquidation mechanics, bonus scaling,
cascading liquidation, close factor capping, and priority rules.
"""

import pytest

from tests.conftest import make_collateral, make_debt, make_snapshot

from arblab.backtest.market import (
    AssetConfig,
    LiquidationEvent,
    MarketParams,
    _liquidation_bonus_pct,
    accrue_interest,
    accrue_lst_yield,
    simulate_liquidation,
    simulate_liquidation_cascade,
)
from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_market(
    borrow_rate_apy: float = 0.10,
    lst_base_apy: float = 0.07,
    close_factor_pct: float = 0.20,
) -> MarketParams:
    """Build a minimal MarketParams with SOL collateral and USDC debt."""
    return MarketParams(
        assets={
            "SOL": AssetConfig(
                symbol="SOL",
                ltv=0.75,
                liquidation_threshold=0.80,
                min_liquidation_bonus_bps=200,
                max_liquidation_bonus_bps=1000,
                borrow_rate_apy=0.0,
                protocol_take_rate_pct=0.15,
                borrow_factor=1.0,
            ),
            "JitoSOL": AssetConfig(
                symbol="JitoSOL",
                ltv=0.75,
                liquidation_threshold=0.80,
                min_liquidation_bonus_bps=200,
                max_liquidation_bonus_bps=1000,
                borrow_rate_apy=0.0,
                protocol_take_rate_pct=0.15,
                borrow_factor=1.0,
                is_lst=True,
                lst_base_apy=lst_base_apy,
            ),
            "USDC": AssetConfig(
                symbol="USDC",
                ltv=0.80,
                liquidation_threshold=0.85,
                min_liquidation_bonus_bps=200,
                max_liquidation_bonus_bps=500,
                borrow_rate_apy=borrow_rate_apy,
                protocol_take_rate_pct=0.15,
                borrow_factor=1.0,
            ),
            "USDT": AssetConfig(
                symbol="USDT",
                ltv=0.80,
                liquidation_threshold=0.85,
                min_liquidation_bonus_bps=200,
                max_liquidation_bonus_bps=500,
                borrow_rate_apy=borrow_rate_apy,
                protocol_take_rate_pct=0.15,
                borrow_factor=1.5,
            ),
        },
        close_factor_pct=close_factor_pct,
    )


# ===================================================================
# accrue_interest
# ===================================================================

def test_accrue_interest_single_debt():
    """Single USDC debt at 10% APY for 1 hour accrues expected interest."""
    market = _simple_market(borrow_rate_apy=0.10)
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10, price=150.0)],
        debt=[make_debt("USDC", amount=1000.0, price=1.0)],
    )
    new_snap, interest_usd = accrue_interest(snap, market, hours_elapsed=1.0)

    expected_rate_per_hour = 0.10 / 8760.0
    expected_interest = 1000.0 * expected_rate_per_hour * 1.0
    assert interest_usd == pytest.approx(expected_interest, rel=1e-9)
    # Debt should have increased
    usdc_debt = next(d for d in new_snap.debt if d.symbol == "USDC")
    assert usdc_debt.amount == pytest.approx(1000.0 + expected_interest, rel=1e-9)


def test_accrue_interest_multi_debt():
    """Interest accrues independently on multiple debt positions."""
    market = _simple_market(borrow_rate_apy=0.10)
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=100, price=150.0)],
        debt=[
            make_debt("USDC", amount=1000.0, price=1.0),
            make_debt("USDT", amount=500.0, price=1.0),
        ],
    )
    new_snap, interest_usd = accrue_interest(snap, market, hours_elapsed=1.0)

    rate = 0.10 / 8760.0
    expected = (1000.0 * rate + 500.0 * rate) * 1.0  # both at price=1
    assert interest_usd == pytest.approx(expected, rel=1e-9)


def test_accrue_interest_zero_rate():
    """Zero borrow rate produces no interest."""
    market = _simple_market(borrow_rate_apy=0.0)
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10, price=150.0)],
        debt=[make_debt("USDC", amount=1000.0, price=1.0)],
    )
    new_snap, interest_usd = accrue_interest(snap, market, hours_elapsed=1.0)
    assert interest_usd == 0.0
    usdc_debt = next(d for d in new_snap.debt if d.symbol == "USDC")
    assert usdc_debt.amount == 1000.0


def test_accrue_interest_24h():
    """24 hours of accrual equals 24x the hourly rate."""
    market = _simple_market(borrow_rate_apy=0.10)
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10, price=150.0)],
        debt=[make_debt("USDC", amount=1000.0, price=1.0)],
    )
    _, interest_1h = accrue_interest(snap, market, hours_elapsed=1.0)
    _, interest_24h = accrue_interest(snap, market, hours_elapsed=24.0)
    assert interest_24h == pytest.approx(interest_1h * 24.0, rel=1e-9)


def test_accrue_interest_no_debt():
    """No debt positions returns zero interest and unchanged snapshot."""
    market = _simple_market()
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10, price=150.0)],
        debt=[],
    )
    new_snap, interest_usd = accrue_interest(snap, market, hours_elapsed=1.0)
    assert interest_usd == 0.0
    assert new_snap is snap  # same object, no changes


# ===================================================================
# accrue_lst_yield
# ===================================================================

def test_accrue_lst_yield_jitosol_appreciates():
    """JitoSOL price increases by the LST yield over the elapsed period."""
    market = _simple_market(lst_base_apy=0.07)
    snap = make_snapshot(
        collateral=[make_collateral("JitoSOL", amount=100.0, price=160.0, ltv=0.75, liq_threshold=0.80)],
        debt=[make_debt("USDC", amount=100.0, price=1.0)],
    )
    new_snap, yield_usd = accrue_lst_yield(snap, market, hours_elapsed=1.0)

    drift = 0.07 / 8760.0
    expected_new_price = 160.0 * (1.0 + drift)
    expected_yield = 100.0 * (expected_new_price - 160.0)

    assert yield_usd == pytest.approx(expected_yield, rel=1e-9)
    jitosol = next(c for c in new_snap.collateral if c.symbol == "JitoSOL")
    assert jitosol.price == pytest.approx(expected_new_price, rel=1e-9)


def test_accrue_lst_yield_sol_unaffected():
    """SOL is not an LST, so its price should not change."""
    market = _simple_market()
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10.0, price=150.0)],
        debt=[make_debt("USDC", amount=100.0, price=1.0)],
    )
    new_snap, yield_usd = accrue_lst_yield(snap, market, hours_elapsed=1.0)

    assert yield_usd == 0.0
    sol = next(c for c in new_snap.collateral if c.symbol == "SOL")
    assert sol.price == 150.0


def test_accrue_lst_yield_zero_apy():
    """LST with zero APY produces no yield."""
    market = _simple_market(lst_base_apy=0.0)
    snap = make_snapshot(
        collateral=[make_collateral("JitoSOL", amount=100.0, price=160.0, ltv=0.75, liq_threshold=0.80)],
        debt=[make_debt("USDC", amount=100.0, price=1.0)],
    )
    new_snap, yield_usd = accrue_lst_yield(snap, market, hours_elapsed=1.0)
    assert yield_usd == 0.0


# ===================================================================
# _liquidation_bonus_pct
# ===================================================================

def test_liquidation_bonus_at_threshold():
    """At exactly the liquidation threshold, bonus equals min."""
    bonus = _liquidation_bonus_pct(
        current_ltv=0.80, liq_threshold=0.80,
        min_bonus_bps=200, max_bonus_bps=1000,
    )
    # LTV <= threshold, so returns min
    assert bonus == pytest.approx(200 / 10_000, rel=1e-9)


def test_liquidation_bonus_below_threshold():
    """Below the liquidation threshold, bonus equals min."""
    bonus = _liquidation_bonus_pct(
        current_ltv=0.60, liq_threshold=0.80,
        min_bonus_bps=200, max_bonus_bps=1000,
    )
    assert bonus == pytest.approx(200 / 10_000, rel=1e-9)


def test_liquidation_bonus_at_100_ltv():
    """At 100% LTV, bonus equals max."""
    bonus = _liquidation_bonus_pct(
        current_ltv=1.0, liq_threshold=0.80,
        min_bonus_bps=200, max_bonus_bps=1000,
    )
    assert bonus == pytest.approx(1000 / 10_000, rel=1e-9)


def test_liquidation_bonus_midpoint():
    """Midway between threshold and 100% LTV gives midpoint bonus."""
    # threshold=0.80, max_excess=0.20, midpoint LTV=0.90, t=0.5
    bonus = _liquidation_bonus_pct(
        current_ltv=0.90, liq_threshold=0.80,
        min_bonus_bps=200, max_bonus_bps=1000,
    )
    expected = (200 + 0.5 * (1000 - 200)) / 10_000
    assert bonus == pytest.approx(expected, rel=1e-9)


# ===================================================================
# simulate_liquidation
# ===================================================================

def test_simulate_liquidation_partial():
    """Partial liquidation reduces debt and collateral by expected amounts."""
    market = _simple_market(close_factor_pct=0.20)
    # Create an underwater position: HF < 1.0
    # collateral: 10 SOL at $100 = $1000, liq_value = $1000 * 0.80 = $800
    # debt: 900 USDC at $1 = $900
    # HF = 800 / 900 = 0.889 < 1.0
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10.0, price=100.0, ltv=0.75, liq_threshold=0.80)],
        debt=[make_debt("USDC", amount=900.0, price=1.0)],
    )
    assert snap.health_factor() < 1.0

    new_snap, event = simulate_liquidation(snap, market, timestamp="t0")

    # Close factor: 20% of $900 debt = $180 repaid
    assert event.debt_repaid_usd == pytest.approx(180.0, rel=1e-6)
    assert event.debt_symbol == "USDC"
    assert event.collateral_symbol == "SOL"
    assert event.timestamp == "t0"

    # Debt should decrease
    usdc_debt = next(d for d in new_snap.debt if d.symbol == "USDC")
    assert usdc_debt.amount < 900.0
    assert usdc_debt.amount == pytest.approx(900.0 - event.debt_repaid, rel=1e-6)

    # Collateral should decrease
    sol_coll = next(c for c in new_snap.collateral if c.symbol == "SOL")
    assert sol_coll.amount < 10.0
    assert sol_coll.amount == pytest.approx(10.0 - event.collateral_seized, rel=1e-6)


def test_simulate_liquidation_raises_when_healthy():
    """Cannot liquidate when health factor >= 1.0."""
    market = _simple_market()
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10.0, price=150.0, ltv=0.75, liq_threshold=0.80)],
        debt=[make_debt("USDC", amount=100.0, price=1.0)],
    )
    assert snap.health_factor() >= 1.0
    with pytest.raises(ValueError, match="health factor >= 1.0"):
        simulate_liquidation(snap, market)


def test_close_factor_caps_repayment():
    """Close factor limits how much debt is repaid in a single pass."""
    # Use a 50% close factor
    market = _simple_market(close_factor_pct=0.50)
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10.0, price=100.0, ltv=0.75, liq_threshold=0.80)],
        debt=[make_debt("USDC", amount=900.0, price=1.0)],
    )
    _, event = simulate_liquidation(snap, market)
    # 50% of $900 = $450
    assert event.debt_repaid_usd == pytest.approx(450.0, rel=1e-6)


# ===================================================================
# Priority rules
# ===================================================================

def test_priority_seize_lowest_liq_threshold_collateral():
    """Liquidation seizes collateral with the lowest liquidation threshold first."""
    market = MarketParams(
        assets={
            "SOL": AssetConfig(
                symbol="SOL", ltv=0.75, liquidation_threshold=0.80,
                min_liquidation_bonus_bps=200, max_liquidation_bonus_bps=1000,
                borrow_rate_apy=0.0, protocol_take_rate_pct=0.15, borrow_factor=1.0,
            ),
            "LOWTHRESH": AssetConfig(
                symbol="LOWTHRESH", ltv=0.50, liquidation_threshold=0.60,
                min_liquidation_bonus_bps=200, max_liquidation_bonus_bps=1000,
                borrow_rate_apy=0.0, protocol_take_rate_pct=0.15, borrow_factor=1.0,
            ),
            "USDC": AssetConfig(
                symbol="USDC", ltv=0.80, liquidation_threshold=0.85,
                min_liquidation_bonus_bps=200, max_liquidation_bonus_bps=500,
                borrow_rate_apy=0.08, protocol_take_rate_pct=0.15, borrow_factor=1.0,
            ),
        },
    )
    # Both collaterals present; LOWTHRESH has liq_threshold=0.60, SOL has 0.80
    # Total collateral: 5*100 + 5*100 = $1000
    # Liq value: 500*0.60 + 500*0.80 = 300+400 = 700
    # Debt: $800 => HF = 700/800 = 0.875 < 1.0
    snap = make_snapshot(
        collateral=[
            make_collateral("SOL", amount=5.0, price=100.0, ltv=0.75, liq_threshold=0.80),
            make_collateral("LOWTHRESH", amount=5.0, price=100.0, ltv=0.50, liq_threshold=0.60),
        ],
        debt=[make_debt("USDC", amount=800.0, price=1.0)],
    )
    assert snap.health_factor() < 1.0

    _, event = simulate_liquidation(snap, market)
    assert event.collateral_symbol == "LOWTHRESH"


def test_priority_repay_highest_borrow_factor_debt():
    """Liquidation repays debt with the highest borrow factor first."""
    market = _simple_market()  # USDT has borrow_factor=1.5, USDC has 1.0
    # Both debts present; USDT borrow_factor=1.5 > USDC borrow_factor=1.0
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10.0, price=100.0, ltv=0.75, liq_threshold=0.80)],
        debt=[
            make_debt("USDC", amount=400.0, price=1.0),
            make_debt("USDT", amount=400.0, price=1.0),
        ],
    )
    # Liq value = 10*100*0.80 = 800; debt = 800; HF = 1.0 exactly
    # Need HF < 1.0, bump debt slightly
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10.0, price=100.0, ltv=0.75, liq_threshold=0.80)],
        debt=[
            make_debt("USDC", amount=450.0, price=1.0),
            make_debt("USDT", amount=450.0, price=1.0),
        ],
    )
    assert snap.health_factor() < 1.0

    _, event = simulate_liquidation(snap, market)
    assert event.debt_symbol == "USDT"


# ===================================================================
# simulate_liquidation_cascade
# ===================================================================

def test_simulate_liquidation_cascade_restores_health():
    """Cascading liquidations should bring health factor back above 1.0."""
    market = _simple_market(close_factor_pct=0.20)
    # Mildly underwater: HF just below 1.0
    # collateral: 10 SOL at $100 = $1000, liq_value = $800
    # debt: 850 USDC => HF = 800/850 = 0.941
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=10.0, price=100.0, ltv=0.75, liq_threshold=0.80)],
        debt=[make_debt("USDC", amount=850.0, price=1.0)],
    )
    assert snap.health_factor() < 1.0

    new_snap, events = simulate_liquidation_cascade(snap, market, max_cascades=10)
    assert len(events) >= 1
    # Either health factor restored or position wiped
    if new_snap.total_collateral_value() > 0 and any(d.amount > 0 for d in new_snap.debt):
        assert new_snap.health_factor() >= 1.0


def test_simulate_liquidation_cascade_position_wiped():
    """Deeply underwater position gets fully wiped out."""
    market = _simple_market(close_factor_pct=0.50)
    # Deeply underwater: collateral barely covers debt
    # collateral: 1 SOL at $100 = $100, liq_value = $80
    # debt: 200 USDC => HF = 80/200 = 0.40
    snap = make_snapshot(
        collateral=[make_collateral("SOL", amount=1.0, price=100.0, ltv=0.75, liq_threshold=0.80)],
        debt=[make_debt("USDC", amount=200.0, price=1.0)],
    )
    assert snap.health_factor() < 1.0

    new_snap, events = simulate_liquidation_cascade(snap, market, max_cascades=20)
    assert len(events) >= 1
    # Collateral should be mostly or entirely consumed
    sol_remaining = next((c for c in new_snap.collateral if c.symbol == "SOL"), None)
    assert sol_remaining is None or sol_remaining.amount == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# MarketParams.kamino_defaults
# ===================================================================

def test_kamino_defaults_has_expected_assets():
    """kamino_defaults() produces a MarketParams with the expected asset set."""
    params = MarketParams.kamino_defaults()
    expected_symbols = {"SOL", "JitoSOL", "mSOL", "USDC", "USDT"}
    assert set(params.assets.keys()) == expected_symbols


def test_kamino_defaults_lst_flags():
    """LST assets in defaults have is_lst=True and positive lst_base_apy."""
    params = MarketParams.kamino_defaults()
    for symbol in ("JitoSOL", "mSOL"):
        cfg = params.assets[symbol]
        assert cfg.is_lst is True
        assert cfg.lst_base_apy > 0
    for symbol in ("SOL", "USDC", "USDT"):
        cfg = params.assets[symbol]
        assert cfg.is_lst is False


def test_kamino_defaults_close_factor():
    """Default close factor is 20%."""
    params = MarketParams.kamino_defaults()
    assert params.close_factor_pct == pytest.approx(0.20)
