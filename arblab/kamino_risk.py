from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class CollateralPosition:
    symbol: str
    amount: float
    price: float
    ltv: float
    liquidation_threshold: float

    def value(self) -> float:
        return self.amount * self.price

    def liquidation_value(self) -> float:
        return self.value() * self.liquidation_threshold

    def borrow_limit_value(self) -> float:
        return self.value() * self.ltv


@dataclass(frozen=True)
class DebtPosition:
    symbol: str
    amount: float
    price: float

    def value(self) -> float:
        return self.amount * self.price


@dataclass(frozen=True)
class AccountSnapshot:
    collateral: List[CollateralPosition]
    debt: List[DebtPosition]

    def total_collateral_value(self) -> float:
        return sum(position.value() for position in self.collateral)

    def total_debt_value(self) -> float:
        return sum(position.value() for position in self.debt)

    def liquidation_value(self) -> float:
        return sum(position.liquidation_value() for position in self.collateral)

    def borrow_limit(self) -> float:
        return sum(position.borrow_limit_value() for position in self.collateral)

    def current_ltv(self) -> float:
        total_collateral = self.total_collateral_value()
        if total_collateral == 0:
            return float("inf")
        return self.total_debt_value() / total_collateral

    def health_factor(self) -> float:
        total_debt = self.total_debt_value()
        if total_debt == 0:
            return float("inf")
        return self.liquidation_value() / total_debt

    def liquidation_buffer(self) -> float:
        return self.liquidation_value() - self.total_debt_value()


def _find_collateral(snapshot: AccountSnapshot, symbol: str) -> Optional[CollateralPosition]:
    return next((position for position in snapshot.collateral if position.symbol == symbol), None)


def _find_debt(snapshot: AccountSnapshot, symbol: str) -> Optional[DebtPosition]:
    return next((position for position in snapshot.debt if position.symbol == symbol), None)


def liquidation_price_for_collateral(snapshot: AccountSnapshot, symbol: str) -> Optional[float]:
    target = _find_collateral(snapshot, symbol)
    if target is None or target.liquidation_threshold == 0 or target.amount == 0:
        return None
    other_liquidation_value = sum(
        position.liquidation_value()
        for position in snapshot.collateral
        if position.symbol != symbol
    )
    total_debt = snapshot.total_debt_value()
    required_value = total_debt - other_liquidation_value
    return max(0.0, required_value / (target.amount * target.liquidation_threshold))


def liquidation_price_for_debt(snapshot: AccountSnapshot, symbol: str) -> Optional[float]:
    target = _find_debt(snapshot, symbol)
    if target is None or target.amount == 0:
        return None
    other_debt_value = sum(
        position.value()
        for position in snapshot.debt
        if position.symbol != symbol
    )
    liquidation_value = snapshot.liquidation_value()
    required_value = liquidation_value - other_debt_value
    return max(0.0, required_value / target.amount)


def liquidation_prices(snapshot: AccountSnapshot) -> Dict[str, Dict[str, Optional[float]]]:
    collateral_prices = {
        position.symbol: liquidation_price_for_collateral(snapshot, position.symbol)
        for position in snapshot.collateral
    }
    debt_prices = {
        position.symbol: liquidation_price_for_debt(snapshot, position.symbol)
        for position in snapshot.debt
    }
    return {"collateral": collateral_prices, "debt": debt_prices}


def apply_actions(snapshot: AccountSnapshot, actions: Iterable[Dict[str, float]]) -> AccountSnapshot:
    collateral = list(snapshot.collateral)
    debt = list(snapshot.debt)

    for action in actions:
        action_type = action["type"]
        symbol = action["symbol"]
        amount = float(action.get("amount", 0))
        price = action.get("price")

        if action_type in {"deposit_collateral", "withdraw_collateral"}:
            position = _find_collateral(AccountSnapshot(collateral, debt), symbol)
            if position is None:
                raise ValueError(f"Collateral asset {symbol} not found for action {action_type}.")
            delta = amount if action_type == "deposit_collateral" else -amount
            updated = replace(position, amount=position.amount + delta)
            collateral = [updated if p.symbol == symbol else p for p in collateral]
        elif action_type in {"borrow", "repay"}:
            position = _find_debt(AccountSnapshot(collateral, debt), symbol)
            if position is None:
                raise ValueError(f"Debt asset {symbol} not found for action {action_type}.")
            delta = amount if action_type == "borrow" else -amount
            updated = replace(position, amount=position.amount + delta)
            debt = [updated if p.symbol == symbol else p for p in debt]
        elif action_type in {"update_price", "set_price"}:
            if price is None:
                raise ValueError(f"Price is required for action {action_type}.")
            collateral_position = _find_collateral(AccountSnapshot(collateral, debt), symbol)
            debt_position = _find_debt(AccountSnapshot(collateral, debt), symbol)
            if collateral_position is None and debt_position is None:
                raise ValueError(f"Asset {symbol} not found for action {action_type}.")
            if collateral_position is not None:
                updated = replace(collateral_position, price=float(price))
                collateral = [updated if p.symbol == symbol else p for p in collateral]
            if debt_position is not None:
                updated = replace(debt_position, price=float(price))
                debt = [updated if p.symbol == symbol else p for p in debt]
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    return AccountSnapshot(collateral=collateral, debt=debt)


def scenario_report(snapshot: AccountSnapshot) -> Dict[str, float]:
    return {
        "total_collateral_value": snapshot.total_collateral_value(),
        "total_debt_value": snapshot.total_debt_value(),
        "borrow_limit": snapshot.borrow_limit(),
        "current_ltv": snapshot.current_ltv(),
        "health_factor": snapshot.health_factor(),
        "liquidation_buffer": snapshot.liquidation_buffer(),
    }
