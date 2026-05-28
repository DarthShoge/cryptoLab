"""Tests for SOL Supertrend strategy with ETH short hedge/full-short modes."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from arblab.backtest.market import MarketParams
from arblab.backtest.engine import BacktestEngine
from arblab.backtest.strategy import BarData, PriceBar
from arblab.kamino_risk import AccountSnapshot
from arblab.kamino_risk import CollateralPosition, DebtPosition
from arblab.strategies.sol_supertrend_short import (
    SolSupertrendShortStrategy,
    supertrend_direction,
)


def _bar(
    index: int,
    sol_price: float = 100.0,
    eth_price: float = 2_000.0,
) -> BarData:
    dates = pd.date_range("2024-01-01", periods=max(index + 1, 1), freq="h", tz="UTC")
    history = pd.DataFrame(
        {
            ("SOL", "open"): [sol_price] * len(dates),
            ("SOL", "high"): [sol_price * 1.01] * len(dates),
            ("SOL", "low"): [sol_price * 0.99] * len(dates),
            ("SOL", "close"): [sol_price] * len(dates),
            ("SOL", "volume"): [1_000.0] * len(dates),
            ("ETH", "open"): [eth_price] * len(dates),
            ("ETH", "high"): [eth_price * 1.01] * len(dates),
            ("ETH", "low"): [eth_price * 0.99] * len(dates),
            ("ETH", "close"): [eth_price] * len(dates),
            ("ETH", "volume"): [1_000.0] * len(dates),
        },
        index=dates,
    )
    history.columns = pd.MultiIndex.from_tuples(history.columns)
    return BarData(
        timestamp=dates[-1],
        prices={
            "SOL": PriceBar("SOL", sol_price, sol_price * 1.01, sol_price * 0.99, sol_price, 1_000.0),
            "ETH": PriceBar("ETH", eth_price, eth_price * 1.01, eth_price * 0.99, eth_price, 1_000.0),
            "USDC": PriceBar("USDC", 1.0, 1.0, 1.0, 1.0, 0.0),
        },
        history=history,
        bar_index=index,
        market_params=MarketParams.kamino_defaults(),
    )


def _setup_strategy(**overrides: Any) -> tuple[SolSupertrendShortStrategy, AccountSnapshot]:
    config = {
        "initial_sol_collateral": 100.0,
        "initial_sol_price": 100.0,
        "initial_eth_price": 2_000.0,
        "signal_by_bar": {},
        "rebalance_cooldown_bars": 0,
    }
    config.update(overrides)
    strategy = SolSupertrendShortStrategy()
    snapshot = strategy.setup(AccountSnapshot(collateral=[], debt=[]), config)
    return strategy, snapshot


def test_setup_starts_with_unlevered_sol_and_zero_debt_positions():
    strategy, snapshot = _setup_strategy()

    assert strategy.event_log == []
    assert snapshot.total_collateral_value() == pytest.approx(10_000.0)
    assert snapshot.total_debt_value() == pytest.approx(0.0)
    assert {p.symbol: p.amount for p in snapshot.collateral} == {
        "SOL": 100.0,
        "USDC": 0.0,
    }
    assert {p.symbol: p.amount for p in snapshot.debt} == {
        "ETH": 0.0,
        "USDC": 0.0,
    }


def test_supertrend_stays_mostly_bullish_on_steady_uptrend():
    dates = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    closes = pd.Series(range(100, 180), index=dates, dtype=float)
    ohlcv = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": 1_000.0,
        }
    )

    direction = supertrend_direction(ohlcv, atr_period=10, multiplier=3.0)

    assert direction.iloc[20:].mean() > 0.90


def test_eth_short_proceeds_are_posted_as_usdc_collateral():
    strategy, snapshot = _setup_strategy(
        signal_by_bar={0: {"green": 1, "bearish_3d": False, "bearish_1w": False}},
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    assert actions == [
        {"type": "borrow", "symbol": "ETH", "amount": pytest.approx(3.75)},
        {"type": "deposit_collateral", "symbol": "USDC", "amount": pytest.approx(7_492.5)},
    ]
    assert strategy.event_log[-1]["reason"] == "hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.75)


def test_eth_short_open_updates_aggregate_hedge_basis():
    strategy, snapshot = _setup_strategy(
        signal_by_bar={0: {"green": 1, "bearish_3d": False, "bearish_1w": False}},
    )

    strategy.on_bar(snapshot, _bar(0, eth_price=2_000.0))

    accounting = strategy.hedge_accounting_state()
    assert accounting["open_eth_short_amount"] == pytest.approx(3.75)
    assert accounting["average_eth_short_basis_usdc"] == pytest.approx(1_998.0)
    assert accounting["lifetime_realized_hedge_pnl_usdc"] == pytest.approx(0.0)
    assert accounting["spendable_hedge_profit_usdc"] == pytest.approx(0.0)
    assert strategy.event_log[-1]["open_eth_short_amount"] == pytest.approx(3.75)
    assert strategy.event_log[-1]["average_eth_short_basis_usdc"] == pytest.approx(1_998.0)


def test_eth_short_cover_realizes_pnl_and_spendable_profit():
    strategy, snapshot = _setup_strategy(
        signal_by_bar={
            0: {"green": 0, "bearish_3d": True, "bearish_1w": True},
            1: {"green": 4, "bearish_3d": True, "bearish_1w": True},
        },
    )

    from arblab.kamino_risk import apply_actions

    open_actions = strategy.on_bar(snapshot, _bar(0, eth_price=2_000.0))
    snapshot = apply_actions(snapshot, open_actions)
    close_actions = strategy.on_bar(snapshot, _bar(1, eth_price=1_500.0))

    eth_repay = next(a for a in close_actions if a["type"] == "repay" and a["symbol"] == "ETH")
    expected_pnl = eth_repay["amount"] * (1_998.0 - 1_501.5)
    accounting = strategy.hedge_accounting_state()
    assert accounting["lifetime_realized_hedge_pnl_usdc"] == pytest.approx(expected_pnl)
    assert accounting["spendable_hedge_profit_usdc"] == pytest.approx(expected_pnl)
    assert accounting["open_eth_short_amount"] == pytest.approx(7.5 - eth_repay["amount"])
    assert strategy.event_log[-1]["lifetime_realized_hedge_pnl_usdc"] == pytest.approx(expected_pnl)


def test_profitable_hedge_unwind_reinvests_surplus_usdc_into_sol_without_usdc_releverage():
    strategy, snapshot = _setup_strategy(
        enable_usdc_releverage=False,
        enable_surplus_usdc_reinvestment=True,
        realized_hedge_profit_gate_pct=0.10,
        surplus_reinvestment_ladder={3: 0.25, 4: 0.50},
        max_surplus_reinvestment_pct_of_sol_collateral=0.05,
        surplus_reinvestment_min_hf=2.0,
        signal_by_bar={
            0: {"green": 0, "bearish_3d": True, "bearish_1w": True},
            1: {"green": 4, "bearish_3d": True, "bearish_1w": True},
            2: {"green": 4, "bearish_3d": False, "bearish_1w": False},
        },
    )

    from arblab.kamino_risk import apply_actions

    open_actions = strategy.on_bar(snapshot, _bar(0, eth_price=2_000.0))
    snapshot = apply_actions(snapshot, open_actions)
    close_actions = strategy.on_bar(snapshot, _bar(1, sol_price=100.0, eth_price=1_500.0))
    snapshot = apply_actions(snapshot, close_actions)
    reinvest_actions = strategy.on_bar(snapshot, _bar(2, sol_price=100.0, eth_price=1_500.0))

    assert [a["type"] for a in reinvest_actions] == [
        "withdraw_collateral",
        "deposit_collateral",
    ]
    assert reinvest_actions[0]["symbol"] == "USDC"
    assert reinvest_actions[0]["amount"] == pytest.approx(500.0)
    assert reinvest_actions[1] == {
        "type": "deposit_collateral",
        "symbol": "SOL",
        "amount": pytest.approx(4.995),
    }
    expected_realized_pnl = 7.5 * (1_998.0 - 1_501.5)
    assert strategy.hedge_accounting_state()["spendable_hedge_profit_usdc"] == pytest.approx(
        expected_realized_pnl - 500.0
    )
    assert strategy.event_log[-1]["reason"] == "surplus_reinvestment"


def test_strong_bullish_vote_adds_usdc_debt_to_buy_sol():
    strategy, snapshot = _setup_strategy(
        enable_usdc_releverage=True,
        signal_by_bar={0: {"green": 4, "bearish_3d": False, "bearish_1w": False}},
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    action_types = [a["type"] for a in actions]
    assert action_types == ["borrow", "deposit_collateral"]
    assert actions[0]["symbol"] == "USDC"
    assert actions[1]["symbol"] == "SOL"
    assert actions[0]["amount"] > 0
    assert actions[1]["amount"] > 0
    assert strategy.event_log[-1]["reason"] == "bullish_relever"


def test_usdc_releverage_can_be_disabled():
    strategy, snapshot = _setup_strategy(
        enable_usdc_releverage=False,
        signal_by_bar={0: {"green": 4, "bearish_3d": False, "bearish_1w": False}},
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    assert actions == []
    assert strategy.event_log == []


def test_full_short_uses_higher_regime_confirmation_for_size():
    strategy, snapshot = _setup_strategy(
        signal_by_bar={0: {"green": 0, "bearish_3d": True, "bearish_1w": True}},
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    eth_borrow = next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    usdc_deposit = next(a for a in actions if a["type"] == "deposit_collateral" and a["symbol"] == "USDC")
    assert eth_borrow["amount"] == pytest.approx(7.5)
    assert usdc_deposit["amount"] == pytest.approx(14_985.0)
    assert strategy.in_full_short_mode is True
    assert strategy.event_log[-1]["reason"] == "full_short_up"


def test_full_short_can_be_disabled_without_disabling_core_hedge():
    strategy, snapshot = _setup_strategy(
        enable_full_short_mode=False,
        signal_by_bar={0: {"green": 0, "bearish_3d": True, "bearish_1w": True}},
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    eth_borrow = next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    assert eth_borrow["amount"] == pytest.approx(3.75)
    assert strategy.in_full_short_mode is False
    assert strategy.event_log[-1]["reason"] == "hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.75)


def test_full_short_bounds_are_configurable():
    strategy, snapshot = _setup_strategy(
        full_short_lower_bound=0.9,
        full_short_upper_bound=1.2,
        signal_by_bar={0: {"green": 0, "bearish_3d": True, "bearish_1w": True}},
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    eth_borrow = next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    assert eth_borrow["amount"] == pytest.approx(6.0)


def test_eth_short_is_scaled_down_to_preserve_minimum_health_factor():
    strategy, snapshot = _setup_strategy(
        min_rebalance_hf=3.0,
        signal_by_bar={0: {"green": 0, "bearish_3d": True, "bearish_1w": True}},
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    eth_borrow = next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    # Uncapped 150% full-short target would borrow 7.5 ETH at $2,000.
    assert eth_borrow["amount"] < 7.5
    from arblab.kamino_risk import apply_actions

    new_snapshot = apply_actions(snapshot, actions)
    assert new_snapshot.health_factor() >= 3.0


def test_defensive_mode_repays_usdc_debt_toward_target_budget():
    strategy, _ = _setup_strategy(
        signal_by_bar={0: {"green": 1, "bearish_3d": False, "bearish_1w": False}},
        defensive_usdc_debt_targets={1: 0.50, 0: 0.0},
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 10_000.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 9_000.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    usdc_repays = [
        a for a in actions if a["type"] == "repay" and a["symbol"] == "USDC"
    ]
    assert usdc_repays
    # Equity is 11k, dynamic budget is 11k, defensive target is 50%.
    assert usdc_repays[0]["amount"] == pytest.approx(3_500.0)


def test_four_green_in_full_short_mode_cuts_eth_short_to_zero():
    strategy, snapshot = _setup_strategy(
        signal_by_bar={
            0: {"green": 0, "bearish_3d": True, "bearish_1w": True},
            1: {"green": 4, "bearish_3d": True, "bearish_1w": True},
        },
    )

    open_actions = strategy.on_bar(snapshot, _bar(0))
    from arblab.kamino_risk import apply_actions

    snapshot = apply_actions(snapshot, open_actions)
    close_actions = strategy.on_bar(snapshot, _bar(1))

    assert close_actions == [
        {"type": "withdraw_collateral", "symbol": "USDC", "amount": pytest.approx(14_985.0)},
        {"type": "repay", "symbol": "ETH", "amount": pytest.approx(7.485014985014985)},
    ]
    assert strategy.in_full_short_mode is False
    assert strategy.event_log[-1]["reason"] == "full_short_cover"


def test_backtest_result_includes_strategy_event_log():
    price_data = _bar(0).history
    strategy = SolSupertrendShortStrategy()
    result = BacktestEngine(strategy).run(
        price_data,
        strategy_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "signal_by_bar": {0: {"green": 1}},
            "rebalance_cooldown_bars": 0,
        },
    )

    assert result.strategy_events
    assert result.strategy_events[0]["reason"] == "hedge_up"
