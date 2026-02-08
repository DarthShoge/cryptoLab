"""Backtest engine: drives the bar-by-bar simulation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from arblab.backtest.market import (
    LiquidationEvent,
    MarketParams,
    accrue_interest,
    accrue_lst_yield,
    simulate_liquidation_cascade,
)
from arblab.backtest.metrics import PerformanceMetrics, compute_metrics
from arblab.backtest.results import BacktestResult
from arblab.backtest.strategy import BarData, PriceBar, Strategy
from arblab.kamino_risk import AccountSnapshot, apply_actions, scenario_report


@dataclass
class EngineConfig:
    lookback_bars: int = 200
    stop_on_full_liquidation: bool = True
    max_liquidation_cascades: int = 5


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        config: EngineConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        price_data: pd.DataFrame,
        strategy_config: Dict[str, Any] | None = None,
        market_params: MarketParams | None = None,
    ) -> BacktestResult:
        """Run the backtest over *price_data*.

        *price_data* must be a DataFrame with a datetime index and
        multi-level columns ``(symbol, field)`` where *field* is one of
        ``open, high, low, close, volume``.
        """
        strategy_config = strategy_config or {}
        market_params = market_params or MarketParams.kamino_defaults()

        symbols = list(price_data.columns.get_level_values(0).unique())

        # Build initial snapshot via strategy
        empty_snapshot = AccountSnapshot(collateral=[], debt=[])
        snapshot = self.strategy.setup(empty_snapshot, strategy_config)

        # Set initial prices from first bar
        snapshot = self._update_prices(snapshot, price_data.iloc[0], symbols)

        initial_value = (
            snapshot.total_collateral_value() - snapshot.total_debt_value()
        )

        records: List[Dict[str, Any]] = []
        liquidation_events: List[LiquidationEvent] = []
        fully_liquidated = False

        for idx in range(len(price_data)):
            row = price_data.iloc[idx]
            timestamp = price_data.index[idx]

            # 1. Accrue interest
            snapshot, interest_usd = accrue_interest(
                snapshot, market_params
            )

            # 2. Accrue LST yield
            snapshot, lst_yield_usd = accrue_lst_yield(
                snapshot, market_params
            )

            # 3. Update prices from bar close
            snapshot = self._update_prices(snapshot, row, symbols)

            # 4. Check liquidation
            liq_penalty_usd = 0.0
            if snapshot.health_factor() < 1.0:
                snapshot, events = simulate_liquidation_cascade(
                    snapshot,
                    market_params,
                    max_cascades=self.config.max_liquidation_cascades,
                    timestamp=timestamp,
                )
                liquidation_events.extend(events)
                liq_penalty_usd = sum(
                    e.collateral_seized_usd - e.debt_repaid_usd for e in events
                )

                # Call strategy for each liquidation event
                for event in events:
                    bar_data = self._make_bar_data(
                        timestamp, row, symbols, price_data, idx, market_params
                    )
                    recovery_actions = self.strategy.on_liquidation(
                        snapshot, bar_data, event
                    )
                    if recovery_actions:
                        snapshot = apply_actions(snapshot, recovery_actions)

                # Check full liquidation (use $0.01 threshold for dust)
                if snapshot.total_collateral_value() < 0.01:
                    fully_liquidated = True
                    records.append(self._record(
                        timestamp, snapshot, 0, interest_usd,
                        lst_yield_usd, liq_penalty_usd,
                    ))
                    if self.config.stop_on_full_liquidation:
                        break
                    continue

            # 5. Call strategy
            bar_data = self._make_bar_data(
                timestamp, row, symbols, price_data, idx, market_params
            )
            actions = self.strategy.on_bar(snapshot, bar_data)

            # 6. Apply actions
            if actions:
                snapshot = apply_actions(snapshot, actions)

            # 7. Record
            records.append(self._record(
                timestamp, snapshot, len(actions), interest_usd,
                lst_yield_usd, liq_penalty_usd,
            ))

        history = pd.DataFrame(records)
        if not history.empty:
            history = history.set_index("timestamp")

        metrics = compute_metrics(history, initial_value)

        return BacktestResult(
            history=history,
            liquidation_events=liquidation_events,
            metrics=metrics,
            liquidated=fully_liquidated,
            strategy_config=strategy_config,
            engine_config=self.config,
            market_params=market_params,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_prices(
        self,
        snapshot: AccountSnapshot,
        row: pd.Series,
        symbols: List[str],
    ) -> AccountSnapshot:
        """Update all position prices from bar close prices."""
        actions = []
        for sym in symbols:
            try:
                close = float(row[(sym, "close")])
            except (KeyError, TypeError):
                continue
            # Check if this symbol exists in snapshot
            has_position = any(
                p.symbol == sym for p in snapshot.collateral
            ) or any(p.symbol == sym for p in snapshot.debt)
            if has_position:
                actions.append({
                    "type": "set_price",
                    "symbol": sym,
                    "amount": 0,
                    "price": close,
                })
        if actions:
            snapshot = apply_actions(snapshot, actions)
        return snapshot

    def _make_bar_data(
        self,
        timestamp: Any,
        row: pd.Series,
        symbols: List[str],
        price_data: pd.DataFrame,
        idx: int,
        market_params: MarketParams,
    ) -> BarData:
        prices: Dict[str, PriceBar] = {}
        for sym in symbols:
            try:
                prices[sym] = PriceBar(
                    symbol=sym,
                    open=float(row[(sym, "open")]),
                    high=float(row[(sym, "high")]),
                    low=float(row[(sym, "low")]),
                    close=float(row[(sym, "close")]),
                    volume=float(row[(sym, "volume")]),
                )
            except (KeyError, TypeError):
                continue

        start = max(0, idx - self.config.lookback_bars)
        history = price_data.iloc[start : idx + 1]

        return BarData(
            timestamp=timestamp,
            prices=prices,
            history=history,
            bar_index=idx,
            market_params=market_params,
        )

    @staticmethod
    def _record(
        timestamp: Any,
        snapshot: AccountSnapshot,
        action_count: int,
        interest_accrued: float,
        lst_yield: float,
        liquidation_penalty: float,
    ) -> Dict[str, Any]:
        report = scenario_report(snapshot)
        return {
            "timestamp": timestamp,
            "collateral_value": report["total_collateral_value"],
            "debt_value": report["total_debt_value"],
            "portfolio_value": report["total_collateral_value"] - report["total_debt_value"],
            "health_factor": report["health_factor"],
            "current_ltv": report["current_ltv"],
            "borrow_limit": report["borrow_limit"],
            "liquidation_buffer": report["liquidation_buffer"],
            "action_count": action_count,
            "interest_accrued": interest_accrued,
            "lst_yield": lst_yield,
            "liquidation_penalty": liquidation_penalty,
        }
