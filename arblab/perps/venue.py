"""Generic perpetual futures venue primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class PerpContractSpec:
    """Exchange-agnostic contract metadata for a linear perp."""

    symbol: str
    settlement_asset: str
    margin_type: str = "linear"
    contract_multiplier: float = 1.0
    max_leverage: float | None = None
    max_notional: float | None = None


@dataclass(frozen=True)
class FeeSchedule:
    """Trading fee rates as fractions of notional."""

    taker: float
    maker: float | None = None

    def rate(self, liquidity: str = "taker") -> float:
        if liquidity == "maker" and self.maker is not None:
            return self.maker
        return self.taker


@dataclass(frozen=True)
class MarginTier:
    """Notional tier for initial and maintenance margin."""

    notional_floor: float
    notional_cap: float | None
    initial_margin_rate: float
    maintenance_margin_rate: float

    def contains(self, notional: float) -> bool:
        absolute = abs(notional)
        if absolute < self.notional_floor:
            return False
        return self.notional_cap is None or absolute <= self.notional_cap


@dataclass(frozen=True)
class OrderConstraints:
    """Portable order filters shared by most perp venues."""

    min_notional: float = 0.0
    quantity_step: float = 0.0
    price_tick: float = 0.0
    max_notional: float | None = None

    def adjust_notional(self, notional: float, price: float) -> float:
        """Round a trade notional to executable quantity and dust constraints."""
        if notional == 0.0 or price <= 0.0:
            return 0.0

        capped = notional
        if self.max_notional is not None:
            capped = max(-self.max_notional, min(self.max_notional, capped))

        quantity = capped / price
        if self.quantity_step > 0.0:
            steps = math.floor(abs(quantity) / self.quantity_step)
            quantity = math.copysign(steps * self.quantity_step, quantity)

        adjusted = quantity * price
        if abs(adjusted) < self.min_notional:
            return 0.0
        return adjusted


@dataclass(frozen=True)
class PerpVenueConfig:
    """Venue-specific perp rules expressed as generic risk primitives."""

    name: str
    contract: PerpContractSpec
    fees: FeeSchedule
    margin_tiers: tuple[MarginTier, ...]
    order_constraints: OrderConstraints = field(default_factory=OrderConstraints)
    liquidation_fee_rate: float = 0.0
    default_funding_interval_hours: int = 8

    def margin_tier(self, notional: float) -> MarginTier:
        for tier in sorted(self.margin_tiers, key=lambda x: x.notional_floor):
            if tier.contains(notional):
                return tier
        if not self.margin_tiers:
            raise ValueError("at least one margin tier is required")
        return max(self.margin_tiers, key=lambda x: x.notional_floor)

    def maintenance_margin(self, notional: float) -> float:
        tier = self.margin_tier(notional)
        return abs(notional) * tier.maintenance_margin_rate

    def initial_margin(self, notional: float) -> float:
        tier = self.margin_tier(notional)
        return abs(notional) * tier.initial_margin_rate

    def max_allowed_notional(self, equity: float) -> float | None:
        candidates: list[float] = []
        if self.contract.max_leverage is not None:
            candidates.append(max(0.0, equity) * self.contract.max_leverage)
        if self.contract.max_notional is not None:
            candidates.append(self.contract.max_notional)
        if self.order_constraints.max_notional is not None:
            candidates.append(self.order_constraints.max_notional)
        return min(candidates) if candidates else None


def generic_linear_usdt_perp(
    *,
    symbol: str = "BTC-PERP",
    taker_fee: float = 0.0006,
    initial_margin_rate: float = 0.10,
    maintenance_margin_rate: float = 0.05,
    max_leverage: float | None = 10.0,
) -> PerpVenueConfig:
    """Return a portable linear USDT perp config for research."""
    return PerpVenueConfig(
        name="generic_linear_usdt",
        contract=PerpContractSpec(
            symbol=symbol,
            settlement_asset="USDT",
            max_leverage=max_leverage,
        ),
        fees=FeeSchedule(taker=taker_fee),
        margin_tiers=(
            MarginTier(
                notional_floor=0.0,
                notional_cap=None,
                initial_margin_rate=initial_margin_rate,
                maintenance_margin_rate=maintenance_margin_rate,
            ),
        ),
    )


def binance_usdm_btcusdt_like() -> PerpVenueConfig:
    """Research preset using Binance USD-M BTCUSDT-like public rule primitives."""
    return PerpVenueConfig(
        name="binance_usdm_like",
        contract=PerpContractSpec(
            symbol="BTCUSDT",
            settlement_asset="USDT",
            max_leverage=125.0,
        ),
        fees=FeeSchedule(taker=0.0006, maker=0.0002),
        order_constraints=OrderConstraints(
            min_notional=5.0,
            quantity_step=0.001,
            price_tick=0.10,
        ),
        liquidation_fee_rate=0.01,
        margin_tiers=(
            MarginTier(
                notional_floor=0.0,
                notional_cap=50_000.0,
                initial_margin_rate=0.008,
                maintenance_margin_rate=0.004,
            ),
            MarginTier(
                notional_floor=50_000.0,
                notional_cap=250_000.0,
                initial_margin_rate=0.01,
                maintenance_margin_rate=0.005,
            ),
            MarginTier(
                notional_floor=250_000.0,
                notional_cap=None,
                initial_margin_rate=0.02,
                maintenance_margin_rate=0.01,
            ),
        ),
    )
