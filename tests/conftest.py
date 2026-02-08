"""Shared test helpers for Kamino Risk Simulator tests."""

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
