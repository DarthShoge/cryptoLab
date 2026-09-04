"""Unit tests for pure calculation functions in arblab.kamino_onchain."""

from decimal import Decimal
from types import SimpleNamespace

from arblab.kamino_onchain import (
    SF_SCALE,
    _compute_collateral_exchange_rate,
    _normalize_amount,
    _sf_to_decimal,
)


# ---------------------------------------------------------------------------
# _sf_to_decimal
# ---------------------------------------------------------------------------

class TestSfToDecimal:
    def test_zero(self):
        assert _sf_to_decimal(0) == Decimal(0)

    def test_one_scale_factor(self):
        """2**60 in Sf representation equals 1."""
        assert _sf_to_decimal(2**60) == Decimal(1)

    def test_two(self):
        """2**61 in Sf representation equals 2."""
        assert _sf_to_decimal(2**61) == Decimal(2)

    def test_half(self):
        """2**59 in Sf representation equals 0.5."""
        assert _sf_to_decimal(2**59) == Decimal("0.5")

    def test_large_128_bit_number(self):
        """Verify correct handling of a large 128-bit Sf value.

        A value near the upper end of 128-bit range:
        2**127 / 2**60 = 2**67 = 147573952589676412928
        """
        sf_value = 2**127
        expected = Decimal(2**67)
        assert _sf_to_decimal(sf_value) == expected


# ---------------------------------------------------------------------------
# _normalize_amount
# ---------------------------------------------------------------------------

class TestNormalizeAmount:
    def test_usdc_one_token(self):
        """1_000_000 raw units with 6 decimals (USDC) equals 1."""
        assert _normalize_amount(1_000_000, 6) == Decimal(1)

    def test_sol_one_token(self):
        """1_000_000_000 raw units with 9 decimals (SOL) equals 1."""
        assert _normalize_amount(1_000_000_000, 9) == Decimal(1)

    def test_zero_amount(self):
        assert _normalize_amount(0, 6) == Decimal(0)

    def test_fractional_usdc(self):
        """500_000 raw units with 6 decimals equals 0.5."""
        assert _normalize_amount(500_000, 6) == Decimal("0.5")


# ---------------------------------------------------------------------------
# _compute_collateral_exchange_rate
# ---------------------------------------------------------------------------

def _make_reserve(
    available_amount: int = 0,
    borrowed_amount_sf: int = 0,
    accumulated_protocol_fees_sf: int = 0,
    accumulated_referrer_fees_sf: int = 0,
    pending_referrer_fees_sf: int = 0,
    mint_total_supply: int = 0,
) -> SimpleNamespace:
    """Build a minimal reserve-like object using SimpleNamespace."""
    return SimpleNamespace(
        liquidity=SimpleNamespace(
            available_amount=available_amount,
            borrowed_amount_sf=borrowed_amount_sf,
            accumulated_protocol_fees_sf=accumulated_protocol_fees_sf,
            accumulated_referrer_fees_sf=accumulated_referrer_fees_sf,
            pending_referrer_fees_sf=pending_referrer_fees_sf,
        ),
        collateral=SimpleNamespace(mint_total_supply=mint_total_supply),
    )


class TestComputeCollateralExchangeRate:
    def test_one_to_one(self):
        """When available equals mint supply and no fees/borrows, rate is 1."""
        reserve = _make_reserve(
            available_amount=1_000_000,
            mint_total_supply=1_000_000,
        )
        assert _compute_collateral_exchange_rate(reserve) == Decimal(1)

    def test_zero_mint_supply_returns_one(self):
        """When mint_total_supply is 0 the function returns Decimal(1) as a fallback."""
        reserve = _make_reserve(
            available_amount=500_000,
            borrowed_amount_sf=int(SF_SCALE) * 200,
            mint_total_supply=0,
        )
        assert _compute_collateral_exchange_rate(reserve) == Decimal(1)

    def test_with_fees(self):
        """Exchange rate accounts for borrowed amounts and protocol fees.

        available=1000, borrowed_sf=SF_SCALE*500 (=500), protocol_fees_sf=SF_SCALE*100 (=100),
        referrer_fees=0, pending_fees=0, mint_supply=700.
        Result: (1000 + 500 - 100 - 0 - 0) / 700 = 1400 / 700 = 2
        """
        reserve = _make_reserve(
            available_amount=1000,
            borrowed_amount_sf=int(SF_SCALE) * 500,
            accumulated_protocol_fees_sf=int(SF_SCALE) * 100,
            accumulated_referrer_fees_sf=0,
            pending_referrer_fees_sf=0,
            mint_total_supply=700,
        )
        assert _compute_collateral_exchange_rate(reserve) == Decimal(2)
