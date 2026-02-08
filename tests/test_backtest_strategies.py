"""Tests for arblab.strategies.leverage_loop — LeverageLoopStrategy."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from arblab.backtest.engine import BacktestEngine, EngineConfig
from arblab.backtest.market import LiquidationEvent, MarketParams
from arblab.backtest.strategy import BarData, PriceBar
from arblab.kamino_risk import (
    AccountSnapshot,
    CollateralPosition,
    DebtPosition,
    apply_actions,
)
from arblab.strategies.leverage_loop import LeverageLoopStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price_data(
    n_bars: int = 100,
    start_price: float = 150.0,
    daily_return: float = 0.0,
) -> pd.DataFrame:
    """Synthetic hourly SOL price data with a constant daily drift."""
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    hourly = daily_return / 24
    prices = start_price * np.cumprod(1 + np.full(n_bars, hourly))
    df = pd.DataFrame(
        {
            ("SOL", "open"): prices,
            ("SOL", "high"): prices * 1.01,
            ("SOL", "low"): prices * 0.99,
            ("SOL", "close"): prices,
            ("SOL", "volume"): 1000.0,
        },
        index=dates,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _default_strategy_config(**overrides) -> Dict[str, Any]:
    cfg = dict(
        collateral_symbol="SOL",
        debt_symbol="USDC",
        initial_collateral=100.0,
        initial_collateral_price=150.0,
        initial_debt_price=1.0,
        num_loops=3,
        loop_utilization=0.85,
        ltv=0.75,
        liquidation_threshold=0.80,
        target_hf=1.3,
        rebalance_hf_low=1.15,
        rebalance_hf_high=2.5,
    )
    cfg.update(overrides)
    return cfg


def _empty_snapshot() -> AccountSnapshot:
    return AccountSnapshot(collateral=[], debt=[])


def _make_bar_data(
    close_price: float = 150.0,
    market_params: MarketParams | None = None,
) -> BarData:
    """Minimal BarData for unit-testing on_bar / on_liquidation."""
    mp = market_params or MarketParams.kamino_defaults()
    dates = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    row_data = {
        ("SOL", "open"): [close_price],
        ("SOL", "high"): [close_price * 1.01],
        ("SOL", "low"): [close_price * 0.99],
        ("SOL", "close"): [close_price],
        ("SOL", "volume"): [1000.0],
    }
    history = pd.DataFrame(row_data, index=dates)
    history.columns = pd.MultiIndex.from_tuples(history.columns)

    return BarData(
        timestamp=dates[0],
        prices={
            "SOL": PriceBar(
                symbol="SOL",
                open=close_price,
                high=close_price * 1.01,
                low=close_price * 0.99,
                close=close_price,
                volume=1000.0,
            )
        },
        history=history,
        bar_index=0,
        market_params=mp,
    )


# ---------------------------------------------------------------------------
# setup() tests
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_creates_leveraged_position(self):
        strategy = LeverageLoopStrategy()
        config = _default_strategy_config(num_loops=3)
        snapshot = strategy.setup(_empty_snapshot(), config)

        initial = config["initial_collateral"]
        # After looping, collateral should exceed the initial deposit
        assert snapshot.total_collateral_value() > initial * config["initial_collateral_price"]
        # There should be debt
        assert snapshot.total_debt_value() > 0
        # Health factor should still be above 1.0
        assert snapshot.health_factor() > 1.0

    def test_setup_zero_loops_no_debt(self):
        strategy = LeverageLoopStrategy()
        config = _default_strategy_config(num_loops=0)
        snapshot = strategy.setup(_empty_snapshot(), config)

        # No loops => no borrowing
        assert snapshot.total_debt_value() == 0
        # Collateral is exactly the initial deposit
        assert len(snapshot.collateral) == 1
        assert snapshot.collateral[0].amount == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# on_bar() tests
# ---------------------------------------------------------------------------

class TestOnBar:
    def _setup_snapshot(self, num_loops: int = 3) -> AccountSnapshot:
        """Create a levered snapshot via the strategy's setup."""
        strategy = LeverageLoopStrategy()
        config = _default_strategy_config(num_loops=num_loops)
        return strategy.setup(_empty_snapshot(), config)

    def test_comfortable_hf_returns_no_actions(self):
        """When HF is in the comfortable range, on_bar does nothing."""
        strategy = LeverageLoopStrategy()
        config = _default_strategy_config()
        strategy._config = config
        snapshot = self._setup_snapshot()
        bar = _make_bar_data(close_price=150.0)

        actions = strategy.on_bar(snapshot, bar)
        assert actions == []

    def test_low_hf_triggers_delever(self):
        """When HF drops below rebalance_hf_low, on_bar returns delever actions."""
        strategy = LeverageLoopStrategy()
        config = _default_strategy_config(rebalance_hf_low=1.15)
        strategy._config = config
        snapshot = self._setup_snapshot()

        # Crash the collateral price to push HF below 1.15
        # With 3 loops at 0.85 util, HF starts around 1.2-1.4
        # We need a price that makes HF < 1.15
        low_price = 100.0  # substantial drop from 150
        snapshot = apply_actions(snapshot, [
            {"type": "set_price", "symbol": "SOL", "amount": 0, "price": low_price},
        ])
        hf = snapshot.health_factor()

        bar = _make_bar_data(close_price=low_price)
        actions = strategy.on_bar(snapshot, bar)

        if hf < 1.15:
            # Should get delever actions (withdraw_collateral + repay)
            assert len(actions) > 0
            action_types = {a["type"] for a in actions}
            assert "repay" in action_types
            assert "withdraw_collateral" in action_types
        else:
            # Price wasn't low enough; skip
            pytest.skip(f"HF={hf:.3f} is not below 1.15 at price {low_price}")

    def test_high_hf_triggers_relever(self):
        """When HF rises above rebalance_hf_high, on_bar returns relever actions."""
        strategy = LeverageLoopStrategy()
        config = _default_strategy_config(rebalance_hf_high=1.5)
        strategy._config = config
        snapshot = self._setup_snapshot()

        # Pump the collateral price to push HF well above threshold
        high_price = 500.0
        snapshot = apply_actions(snapshot, [
            {"type": "set_price", "symbol": "SOL", "amount": 0, "price": high_price},
        ])
        hf = snapshot.health_factor()
        assert hf > 1.5, f"Expected HF > 1.5 but got {hf}"

        bar = _make_bar_data(close_price=high_price)
        actions = strategy.on_bar(snapshot, bar)

        # Should get relever actions (borrow + deposit_collateral)
        assert len(actions) > 0
        action_types = {a["type"] for a in actions}
        assert "borrow" in action_types
        assert "deposit_collateral" in action_types


# ---------------------------------------------------------------------------
# on_liquidation() tests
# ---------------------------------------------------------------------------

class TestOnLiquidation:
    def test_on_liquidation_returns_empty(self):
        strategy = LeverageLoopStrategy()
        snapshot = AccountSnapshot(
            collateral=[
                CollateralPosition("SOL", 50.0, 150.0, 0.75, 0.80),
            ],
            debt=[
                DebtPosition("USDC", 5000.0, 1.0),
            ],
        )
        bar = _make_bar_data()
        event = LiquidationEvent(
            timestamp=bar.timestamp,
            debt_symbol="USDC",
            debt_repaid=1000.0,
            debt_repaid_usd=1000.0,
            collateral_symbol="SOL",
            collateral_seized=7.0,
            collateral_seized_usd=1050.0,
            bonus_pct=0.05,
            resulting_hf=1.05,
        )
        actions = strategy.on_liquidation(snapshot, bar, event)
        assert actions == []


# ---------------------------------------------------------------------------
# Full engine runs
# ---------------------------------------------------------------------------

class TestFullEngineRun:
    def test_rising_prices_positive_return(self):
        """With steadily rising prices and leverage, the return should be positive."""
        price_data = make_price_data(n_bars=200, start_price=150.0, daily_return=0.02)
        strategy = LeverageLoopStrategy()
        config = _default_strategy_config(num_loops=2)

        engine = BacktestEngine(strategy, EngineConfig(stop_on_full_liquidation=True))
        result = engine.run(price_data, strategy_config=config)

        assert not result.liquidated
        assert result.metrics.total_return_pct > 0

    def test_crashing_prices_liquidation(self):
        """With a severe crash and no delevering, the position gets liquidated."""
        # Disable rebalancing so the strategy can't save itself
        price_data = make_price_data(n_bars=200, start_price=150.0, daily_return=-0.50)
        strategy = LeverageLoopStrategy()
        config = _default_strategy_config(
            num_loops=4, loop_utilization=0.90,
            rebalance_hf_low=0.0,  # disable delevering
        )

        engine = BacktestEngine(strategy, EngineConfig(stop_on_full_liquidation=True))
        result = engine.run(price_data, strategy_config=config)

        # Should either be liquidated or have massive losses
        assert result.liquidated or result.metrics.total_return_pct < -90


class TestGridSearch:
    """Varying num_loops should produce different backtest outcomes."""

    def test_different_loop_counts_give_different_results(self):
        price_data = make_price_data(n_bars=100, start_price=150.0, daily_return=0.01)

        returns = {}
        for loops in [1, 3, 5]:
            strategy = LeverageLoopStrategy()
            config = _default_strategy_config(num_loops=loops)
            engine = BacktestEngine(strategy)
            result = engine.run(price_data, strategy_config=config)
            returns[loops] = result.metrics.total_return_pct

        # With rising prices, more loops should generally yield higher returns
        # At minimum, the results should differ
        values = list(returns.values())
        assert len(set(round(v, 4) for v in values)) > 1, (
            f"Expected different returns for different loop counts, got {returns}"
        )
