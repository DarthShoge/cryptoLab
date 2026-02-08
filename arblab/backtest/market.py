"""Kamino-specific protocol mechanics for the backtesting engine.

Encapsulates liquidation, interest accrual, and LST staking yield so
the engine loop stays generic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from arblab.kamino_risk import (
    AccountSnapshot,
    CollateralPosition,
    DebtPosition,
    apply_actions,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssetConfig:
    """Per-asset Kamino lending parameters."""

    symbol: str
    ltv: float
    liquidation_threshold: float
    min_liquidation_bonus_bps: int
    max_liquidation_bonus_bps: int
    borrow_rate_apy: float  # simplified: fixed annual borrow rate
    protocol_take_rate_pct: float
    borrow_factor: float
    is_lst: bool = False
    lst_base_apy: float = 0.0


@dataclass(frozen=True)
class MarketParams:
    """Protocol-level parameters for a backtest run."""

    assets: Dict[str, AssetConfig]
    close_factor_pct: float = 0.20
    target_hf_after_liquidation: float = 1.05
    swap_fee_bps: float = 10.0

    @classmethod
    def kamino_defaults(cls) -> MarketParams:
        assets = {
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
                lst_base_apy=0.07,
            ),
            "mSOL": AssetConfig(
                symbol="mSOL",
                ltv=0.75,
                liquidation_threshold=0.80,
                min_liquidation_bonus_bps=200,
                max_liquidation_bonus_bps=1000,
                borrow_rate_apy=0.0,
                protocol_take_rate_pct=0.15,
                borrow_factor=1.0,
                is_lst=True,
                lst_base_apy=0.07,
            ),
            "USDC": AssetConfig(
                symbol="USDC",
                ltv=0.80,
                liquidation_threshold=0.85,
                min_liquidation_bonus_bps=200,
                max_liquidation_bonus_bps=500,
                borrow_rate_apy=0.08,
                protocol_take_rate_pct=0.15,
                borrow_factor=1.0,
            ),
            "USDT": AssetConfig(
                symbol="USDT",
                ltv=0.80,
                liquidation_threshold=0.85,
                min_liquidation_bonus_bps=200,
                max_liquidation_bonus_bps=500,
                borrow_rate_apy=0.08,
                protocol_take_rate_pct=0.15,
                borrow_factor=1.0,
            ),
        }
        return cls(assets=assets)


@dataclass(frozen=True)
class LiquidationEvent:
    """Record of a single liquidation pass."""

    timestamp: Any  # datetime or bar index
    debt_symbol: str
    debt_repaid: float  # token amount
    debt_repaid_usd: float
    collateral_symbol: str
    collateral_seized: float  # token amount
    collateral_seized_usd: float
    bonus_pct: float
    resulting_hf: float


# ---------------------------------------------------------------------------
# Interest accrual
# ---------------------------------------------------------------------------

def accrue_interest(
    snapshot: AccountSnapshot,
    market_params: MarketParams,
    hours_elapsed: float = 1.0,
) -> Tuple[AccountSnapshot, float]:
    """Accrue borrow interest on all debt positions.

    Returns (updated_snapshot, interest_accrued_usd).
    """
    if not snapshot.debt:
        return snapshot, 0.0

    total_interest_usd = 0.0
    actions: List[Dict] = []

    for pos in snapshot.debt:
        cfg = market_params.assets.get(pos.symbol)
        if cfg is None or cfg.borrow_rate_apy <= 0:
            continue
        rate_per_hour = cfg.borrow_rate_apy / 8760.0
        interest_tokens = pos.amount * rate_per_hour * hours_elapsed
        interest_usd = interest_tokens * pos.price
        total_interest_usd += interest_usd
        actions.append({
            "type": "borrow",
            "symbol": pos.symbol,
            "amount": interest_tokens,
        })

    if actions:
        snapshot = apply_actions(snapshot, actions)

    return snapshot, total_interest_usd


# ---------------------------------------------------------------------------
# LST staking yield
# ---------------------------------------------------------------------------

def accrue_lst_yield(
    snapshot: AccountSnapshot,
    market_params: MarketParams,
    hours_elapsed: float = 1.0,
) -> Tuple[AccountSnapshot, float]:
    """Appreciate LST collateral prices to simulate staking yield.

    Returns (updated_snapshot, yield_earned_usd).
    """
    if not snapshot.collateral:
        return snapshot, 0.0

    total_yield_usd = 0.0
    actions: List[Dict] = []

    for pos in snapshot.collateral:
        cfg = market_params.assets.get(pos.symbol)
        if cfg is None or not cfg.is_lst or cfg.lst_base_apy <= 0:
            continue
        price_drift = cfg.lst_base_apy / 8760.0 * hours_elapsed
        new_price = pos.price * (1.0 + price_drift)
        yield_usd = pos.amount * (new_price - pos.price)
        total_yield_usd += yield_usd
        actions.append({
            "type": "set_price",
            "symbol": pos.symbol,
            "amount": 0,
            "price": new_price,
        })

    if actions:
        snapshot = apply_actions(snapshot, actions)

    return snapshot, total_yield_usd


# ---------------------------------------------------------------------------
# Liquidation simulation
# ---------------------------------------------------------------------------

def _liquidation_bonus_pct(
    current_ltv: float,
    liq_threshold: float,
    min_bonus_bps: int,
    max_bonus_bps: int,
) -> float:
    """Scale liquidation bonus linearly from min to max based on LTV excess.

    At threshold → min bonus; at 100% LTV → max bonus.
    """
    if current_ltv <= liq_threshold:
        return min_bonus_bps / 10_000.0
    excess = current_ltv - liq_threshold
    max_excess = 1.0 - liq_threshold
    if max_excess <= 0:
        return max_bonus_bps / 10_000.0
    t = min(excess / max_excess, 1.0)
    bonus_bps = min_bonus_bps + t * (max_bonus_bps - min_bonus_bps)
    return bonus_bps / 10_000.0


def simulate_liquidation(
    snapshot: AccountSnapshot,
    market_params: MarketParams,
    timestamp: Any = None,
) -> Tuple[AccountSnapshot, LiquidationEvent]:
    """Simulate a single partial Kamino liquidation pass.

    Priority rules:
    - Seize collateral with lowest liquidation_threshold first
    - Repay debt with highest borrow_factor first
    """
    if snapshot.health_factor() >= 1.0:
        raise ValueError("Cannot liquidate: health factor >= 1.0")

    # Pick debt: highest borrow_factor
    debt_candidates = [
        (pos, market_params.assets.get(pos.symbol))
        for pos in snapshot.debt
        if pos.amount > 0
    ]
    debt_candidates.sort(
        key=lambda x: x[1].borrow_factor if x[1] else 0, reverse=True
    )
    debt_pos, debt_cfg = debt_candidates[0]

    # Pick collateral: lowest liquidation_threshold
    coll_candidates = [
        (pos, market_params.assets.get(pos.symbol))
        for pos in snapshot.collateral
        if pos.amount > 0
    ]
    coll_candidates.sort(
        key=lambda x: x[1].liquidation_threshold if x[1] else 1.0
    )
    coll_pos, coll_cfg = coll_candidates[0]

    # Amount of debt to repay (close_factor_pct of total debt value)
    total_debt_usd = snapshot.total_debt_value()
    repay_usd = total_debt_usd * market_params.close_factor_pct
    repay_tokens = repay_usd / debt_pos.price if debt_pos.price > 0 else 0.0
    repay_tokens = min(repay_tokens, debt_pos.amount)
    repay_usd = repay_tokens * debt_pos.price

    # Liquidation bonus
    current_ltv = snapshot.current_ltv()
    bonus_pct = _liquidation_bonus_pct(
        current_ltv,
        coll_cfg.liquidation_threshold if coll_cfg else 0.80,
        coll_cfg.min_liquidation_bonus_bps if coll_cfg else 200,
        coll_cfg.max_liquidation_bonus_bps if coll_cfg else 1000,
    )

    # Collateral seized = repay_usd * (1 + bonus) / collateral_price
    seize_usd = repay_usd * (1.0 + bonus_pct)
    seize_tokens = seize_usd / coll_pos.price if coll_pos.price > 0 else 0.0
    seize_tokens = min(seize_tokens, coll_pos.amount)
    seize_usd = seize_tokens * coll_pos.price

    # Apply: repay debt + withdraw collateral
    actions = [
        {"type": "repay", "symbol": debt_pos.symbol, "amount": repay_tokens},
        {"type": "withdraw_collateral", "symbol": coll_pos.symbol, "amount": seize_tokens},
    ]
    new_snapshot = apply_actions(snapshot, actions)

    event = LiquidationEvent(
        timestamp=timestamp,
        debt_symbol=debt_pos.symbol,
        debt_repaid=repay_tokens,
        debt_repaid_usd=repay_usd,
        collateral_symbol=coll_pos.symbol,
        collateral_seized=seize_tokens,
        collateral_seized_usd=seize_usd,
        bonus_pct=bonus_pct,
        resulting_hf=new_snapshot.health_factor(),
    )

    return new_snapshot, event


def simulate_liquidation_cascade(
    snapshot: AccountSnapshot,
    market_params: MarketParams,
    max_cascades: int = 5,
    timestamp: Any = None,
) -> Tuple[AccountSnapshot, List[LiquidationEvent]]:
    """Simulate cascading liquidations until HF >= 1.0 or position wiped out."""
    events: List[LiquidationEvent] = []

    for _ in range(max_cascades):
        if snapshot.health_factor() >= 1.0:
            break
        # Check if position is effectively wiped out
        if snapshot.total_collateral_value() <= 0 or all(
            p.amount <= 0 for p in snapshot.collateral
        ):
            break

        snapshot, event = simulate_liquidation(snapshot, market_params, timestamp)
        events.append(event)

    return snapshot, events
