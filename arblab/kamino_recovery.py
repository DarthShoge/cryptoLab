"""Pure calculation functions for recovery, auto-loop, and collateral rebalance.

Extracted from kamino_app.py so they can be unit tested independently.
"""


def compute_deficit(total_debt: float, liq_value: float, target_hf: float) -> float:
    """How much additional liquidation value is needed to reach *target_hf*.

    deficit = target_hf * total_debt - liq_value
    Positive means recovery action needed; negative means already safe.
    """
    return target_hf * total_debt - liq_value


def recovery_repay_usd(total_debt: float, liq_value: float, target_hf: float) -> float:
    """Option A: USD amount of debt to repay to reach *target_hf*.

    repay_usd = total_debt - liq_value / target_hf
    """
    if target_hf <= 0:
        return 0.0
    return total_debt - liq_value / target_hf


def recovery_swap_withdraw(
    deficit: float,
    collateral_price: float,
    target_hf: float,
    liq_threshold: float,
    max_amount: float,
) -> float:
    """Option B: tokens of collateral to withdraw and swap into debt repayment.

    withdraw = min(deficit / (price * (target_hf - liq_threshold)), max_amount)
    Returns 0 if target_hf <= liq_threshold (swap can't improve HF).
    """
    if target_hf <= liq_threshold or collateral_price <= 0:
        return 0.0
    raw = deficit / (collateral_price * (target_hf - liq_threshold))
    return min(raw, max_amount)


def recovery_deposit_tokens(
    deficit: float,
    collateral_price: float,
    liq_threshold: float,
) -> float:
    """Option C: tokens of collateral to deposit to reach *target_hf*.

    deposit = deficit / (price * liq_threshold)
    """
    if collateral_price <= 0 or liq_threshold <= 0:
        return 0.0
    return deficit / (collateral_price * liq_threshold)


def auto_loop_deposit(
    borrow_amount: float,
    borrow_price: float,
    target_price: float,
) -> float:
    """Auto-loop: convert borrowed tokens to collateral tokens (USD-neutral swap).

    deposit = (borrow_amount * borrow_price) / target_price
    """
    if target_price <= 0:
        return 0.0
    return (borrow_amount * borrow_price) / target_price


def collateral_rebalance(
    withdraw_delta: float,
    source_price: float,
    target_price: float,
) -> float:
    """Collateral rebalance: convert withdrawn collateral to another asset.

    deposit = (abs(withdraw_delta) * source_price) / target_price
    """
    if target_price <= 0:
        return 0.0
    return (abs(withdraw_delta) * source_price) / target_price
