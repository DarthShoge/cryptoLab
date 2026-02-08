"""Leverage loop strategy for Kamino SOL/JitoSOL lending."""

from __future__ import annotations

from typing import Any, Dict, List

from arblab.backtest.market import LiquidationEvent
from arblab.backtest.strategy import BarData, Strategy
from arblab.kamino_recovery import auto_loop_deposit, recovery_repay_usd
from arblab.kamino_risk import (
    AccountSnapshot,
    CollateralPosition,
    DebtPosition,
)


class LeverageLoopStrategy(Strategy):
    """Leverage loop: deposit collateral, borrow stablecoins, swap to more
    collateral, repeat.

    Config keys:
        collateral_symbol: str       — "SOL" or "JitoSOL" (default "SOL")
        debt_symbol: str             — "USDC" (default)
        initial_collateral: float    — starting tokens (e.g. 100)
        num_loops: int               — leverage iterations (default 3)
        loop_utilization: float      — fraction of borrow limit per loop (default 0.85)
        target_hf: float             — target HF after rebalance (default 1.3)
        rebalance_hf_low: float      — delever trigger (default 1.15)
        rebalance_hf_high: float     — re-lever trigger (default 2.5)
    """

    def setup(
        self, snapshot: AccountSnapshot, config: Dict[str, Any]
    ) -> AccountSnapshot:
        self._config = config
        coll_sym = config.get("collateral_symbol", "SOL")
        debt_sym = config.get("debt_symbol", "USDC")
        initial = config.get("initial_collateral", 100.0)
        num_loops = config.get("num_loops", 3)
        utilization = config.get("loop_utilization", 0.85)
        coll_price = config.get("initial_collateral_price", 150.0)
        debt_price = config.get("initial_debt_price", 1.0)

        # Look up asset params from market config if available
        ltv = config.get("ltv", 0.75)
        liq_threshold = config.get("liquidation_threshold", 0.80)

        # Start with initial collateral and zero debt
        snapshot = AccountSnapshot(
            collateral=[
                CollateralPosition(
                    symbol=coll_sym,
                    amount=initial,
                    price=coll_price,
                    ltv=ltv,
                    liquidation_threshold=liq_threshold,
                )
            ],
            debt=[
                DebtPosition(symbol=debt_sym, amount=0.0, price=debt_price)
            ],
        )

        # Execute leverage loops
        for _ in range(num_loops):
            available_borrow = snapshot.borrow_limit() - snapshot.total_debt_value()
            if available_borrow <= 0:
                break
            borrow_usd = available_borrow * utilization
            borrow_tokens = borrow_usd / debt_price

            # Swap borrowed stablecoins to collateral
            deposit_tokens = auto_loop_deposit(
                borrow_tokens, debt_price, coll_price
            )

            from arblab.kamino_risk import apply_actions

            snapshot = apply_actions(snapshot, [
                {"type": "borrow", "symbol": debt_sym, "amount": borrow_tokens},
                {"type": "deposit_collateral", "symbol": coll_sym, "amount": deposit_tokens},
            ])

        return snapshot

    def on_bar(
        self, snapshot: AccountSnapshot, bar: BarData
    ) -> List[Dict[str, Any]]:
        config = {}
        # Access config through bar_data; we store it on the strategy instance
        if hasattr(self, "_config"):
            config = self._config

        target_hf = config.get("target_hf", 1.3)
        low = config.get("rebalance_hf_low", 1.15)
        high = config.get("rebalance_hf_high", 2.5)

        hf = snapshot.health_factor()
        if hf == float("inf") or snapshot.total_debt_value() == 0:
            return []

        coll_sym = config.get("collateral_symbol", "SOL")
        debt_sym = config.get("debt_symbol", "USDC")

        # Delever: HF too low → repay debt
        if hf < low:
            repay_usd = recovery_repay_usd(
                snapshot.total_debt_value(),
                snapshot.liquidation_value(),
                target_hf,
            )
            if repay_usd > 0:
                debt_pos = next(
                    (p for p in snapshot.debt if p.symbol == debt_sym), None
                )
                if debt_pos and debt_pos.price > 0:
                    repay_tokens = min(
                        repay_usd / debt_pos.price, debt_pos.amount
                    )
                    if repay_tokens > 0:
                        # Also withdraw equivalent collateral to fund repayment
                        coll_pos = next(
                            (p for p in snapshot.collateral if p.symbol == coll_sym), None
                        )
                        if coll_pos and coll_pos.price > 0:
                            withdraw_tokens = min(
                                repay_usd / coll_pos.price,
                                coll_pos.amount * 0.5,  # cap at 50% of collateral
                            )
                            return [
                                {"type": "withdraw_collateral", "symbol": coll_sym, "amount": withdraw_tokens},
                                {"type": "repay", "symbol": debt_sym, "amount": repay_tokens},
                            ]

        # Re-lever: HF too high → add more leverage
        if hf > high:
            utilization = config.get("loop_utilization", 0.85)
            available = snapshot.borrow_limit() - snapshot.total_debt_value()
            if available > 0:
                borrow_usd = available * utilization
                debt_pos = next(
                    (p for p in snapshot.debt if p.symbol == debt_sym), None
                )
                coll_pos = next(
                    (p for p in snapshot.collateral if p.symbol == coll_sym), None
                )
                if debt_pos and coll_pos and debt_pos.price > 0 and coll_pos.price > 0:
                    borrow_tokens = borrow_usd / debt_pos.price
                    deposit_tokens = auto_loop_deposit(
                        borrow_tokens, debt_pos.price, coll_pos.price
                    )
                    return [
                        {"type": "borrow", "symbol": debt_sym, "amount": borrow_tokens},
                        {"type": "deposit_collateral", "symbol": coll_sym, "amount": deposit_tokens},
                    ]

        return []

    def on_liquidation(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
        event: LiquidationEvent,
    ) -> List[Dict[str, Any]]:
        # After liquidation just continue with the reduced position
        return []
