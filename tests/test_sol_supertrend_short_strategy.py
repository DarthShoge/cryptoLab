"""Tests for SOL Supertrend strategy with ETH short hedge/full-short modes."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from arblab.backtest.market import MarketParams
from arblab.backtest.engine import BacktestEngine, EngineConfig
from arblab.backtest.strategy import BarData, PriceBar
from arblab.kamino_risk import AccountSnapshot
from arblab.kamino_risk import CollateralPosition, DebtPosition
from arblab.strategies.sol_supertrend_short import (
    FastBreakState,
    HedgeTargetOverlay,
    PortfolioObservationWindow,
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


def _drawdown_bar(
    index: int = 2_200,
    peak_sol_price: float = 150.0,
    current_sol_price: float = 100.0,
    eth_price: float = 2_000.0,
) -> BarData:
    dates = pd.date_range("2024-01-01", periods=index + 1, freq="h", tz="UTC")
    sol_closes = [peak_sol_price] * index + [current_sol_price]
    history = pd.DataFrame(
        {
            ("SOL", "open"): sol_closes,
            ("SOL", "high"): [peak_sol_price] * len(dates),
            ("SOL", "low"): [min(peak_sol_price, current_sol_price)] * len(dates),
            ("SOL", "close"): sol_closes,
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
            "SOL": PriceBar("SOL", current_sol_price, current_sol_price * 1.01, current_sol_price * 0.99, current_sol_price, 1_000.0),
            "ETH": PriceBar("ETH", eth_price, eth_price * 1.01, eth_price * 0.99, eth_price, 1_000.0),
            "USDC": PriceBar("USDC", 1.0, 1.0, 1.0, 1.0, 0.0),
        },
        history=history,
        bar_index=index,
        market_params=MarketParams.kamino_defaults(),
    )


def _fast_break_bar(
    index: int = 800,
    eth_price: float = 2_000.0,
    recover: bool = False,
) -> BarData:
    periods = index + 1
    dates = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    closes = []
    for i in range(periods):
        if i < periods - 24:
            closes.append(100.0 + (0.05 if i % 2 else 0.0))
        else:
            step = i - (periods - 24)
            if recover:
                closes.append(91.0 + step * 0.4)
            else:
                closes.append(100.0 - step * 0.45 + (0.8 if step % 2 else -0.8))
    if not recover:
        closes[-1] = 90.0
    history = pd.DataFrame(
        {
            ("SOL", "open"): closes,
            ("SOL", "high"): [price * 1.01 for price in closes],
            ("SOL", "low"): [price * 0.99 for price in closes],
            ("SOL", "close"): closes,
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
    sol_price = float(closes[-1])
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


def test_portfolio_observation_window_tracks_bounded_drawdowns():
    window = PortfolioObservationWindow(max_bars=2)

    window.record(portfolio_value=100.0, sol_price=10.0)
    window.record(portfolio_value=80.0, sol_price=10.0)
    window.record(portfolio_value=72.0, sol_price=12.0)

    assert window.portfolio_values == [80.0, 72.0]
    assert window.sol_equivalent_values == [8.0, 6.0]
    assert window.portfolio_drawdown_from_high() == pytest.approx(0.10)
    assert window.sol_equivalent_drawdown_from_high() == pytest.approx(0.25)
    assert window.sol_equivalent_recovered(0.30) is True


def test_hedge_target_overlay_uses_down_reason_when_floor_reduces_existing_short():
    strategy, snapshot = _setup_strategy()
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 10_000.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 5.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    target, reason = strategy._resolve_target_overlay(
        snapshot,
        normal_target=0.25,
        normal_reason="normal",
        overlay=HedgeTargetOverlay(
            floor=0.75,
            up_reason="overlay_up",
            down_reason="overlay_down",
        ),
    )

    assert target == pytest.approx(0.75)
    assert reason == "overlay_down"


def test_drawdown_containment_rejects_unknown_block_action():
    strategy, _ = _setup_strategy(enable_drawdown_containment=True)
    strategy.drawdown_containment_state.active = True

    with pytest.raises(ValueError, match="Unknown drawdown containment action"):
        strategy._drawdown_containment_blocks("unknown")


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


def test_protected_book_allocates_realized_hedge_profit_before_reinvestment():
    strategy, snapshot = _setup_strategy(
        enable_usdc_releverage=False,
        enable_surplus_usdc_reinvestment=True,
        enable_protected_book=True,
        protected_book_realized_pnl_fraction=0.50,
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
    accounting = strategy.hedge_accounting_state()
    expected_realized_pnl = 7.5 * (1_998.0 - 1_501.5)
    assert accounting["protected_book_usdc"] == pytest.approx(expected_realized_pnl * 0.50)
    assert accounting["spendable_hedge_profit_usdc"] == pytest.approx(
        expected_realized_pnl * 0.50
    )

    reinvest_actions = strategy.on_bar(snapshot, _bar(2, sol_price=100.0, eth_price=1_500.0))

    assert reinvest_actions[0]["amount"] == pytest.approx(500.0)
    assert strategy.hedge_accounting_state()["protected_book_usdc"] == pytest.approx(
        expected_realized_pnl * 0.50
    )


def test_surplus_reinvestment_does_not_spend_cppi_protected_usdc():
    strategy, snapshot = _setup_strategy(
        enable_usdc_releverage=False,
        enable_surplus_usdc_reinvestment=True,
        realized_hedge_profit_gate_pct=0.10,
        surplus_reinvestment_ladder={3: 0.25, 4: 0.50},
        max_surplus_reinvestment_pct_of_sol_collateral=0.05,
        surplus_reinvestment_min_hf=2.0,
        enable_cppi_exposure_cap=True,
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
    usdc_amount = next(p.amount for p in snapshot.collateral if p.symbol == "USDC")
    strategy.cppi_exposure_cap_state.protected_usdc = usdc_amount - 250.0
    strategy.cppi_exposure_cap_state.active = True

    reinvest_actions = strategy.on_bar(snapshot, _bar(2, sol_price=100.0, eth_price=1_500.0))

    assert reinvest_actions[0] == {
        "type": "withdraw_collateral",
        "symbol": "USDC",
        "amount": pytest.approx(250.0),
    }
    assert reinvest_actions[1] == {
        "type": "deposit_collateral",
        "symbol": "SOL",
        "amount": pytest.approx(2.4975),
    }
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


def test_fast_break_overlay_raises_hedge_floor_on_sol_break_with_vol_expansion():
    strategy, snapshot = _setup_strategy(
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=0.75,
        signal_by_bar={
            800: {"green": 3, "bearish_1d": False, "bearish_3d": False, "bearish_1w": False},
        },
    )

    actions = strategy.on_bar(snapshot, _fast_break_bar())

    eth_borrow = next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    assert eth_borrow["amount"] == pytest.approx(3.75)
    assert strategy.event_log[-1]["reason"] == "fast_break_hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.75)
    assert strategy.history_fields()["in_fast_break_overlay"] is True


def test_fast_break_overlay_exits_on_strong_recovery_signal():
    strategy, snapshot = _setup_strategy(
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=0.75,
        signal_by_bar={
            800: {"green": 3, "bearish_1d": False, "bearish_3d": False, "bearish_1w": False},
            801: {"green": 4, "bearish_1d": False, "bearish_3d": False, "bearish_1w": False},
        },
    )

    from arblab.kamino_risk import apply_actions

    open_actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))
    snapshot = apply_actions(snapshot, open_actions)
    close_actions = strategy.on_bar(snapshot, _fast_break_bar(index=801, recover=True))

    assert any(action["type"] == "repay" and action["symbol"] == "ETH" for action in close_actions)
    assert strategy.event_log[-1]["reason"] == "hedge_down"
    assert strategy.history_fields()["in_fast_break_overlay"] is False


def test_fast_break_overlay_stages_down_before_exit_when_decay_is_enabled():
    strategy, snapshot = _setup_strategy(
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=1.0,
        fast_break_hold_bars=24,
        fast_break_decay_enabled=True,
        fast_break_decay_floors=[0.75, 0.35],
        signal_by_bar={
            800: {"green": 3, "bearish_1d": False, "bearish_3d": False, "bearish_1w": False},
            825: {"green": 3, "bearish_1d": False, "bearish_3d": False, "bearish_1w": False},
        },
    )

    from arblab.kamino_risk import apply_actions

    open_actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))
    snapshot = apply_actions(snapshot, open_actions)
    decay_actions = strategy.on_bar(snapshot, _fast_break_bar(index=825))

    assert any(action["type"] == "repay" and action["symbol"] == "ETH" for action in decay_actions)
    assert strategy.fast_break_state.active is True
    assert strategy.fast_break_state.hedge_floor == pytest.approx(0.75)
    assert strategy.event_log[-1]["reason"] == "fast_break_hedge_down"


def test_fast_break_overlay_caps_new_hedge_adds_with_own_min_hf():
    strategy, snapshot = _setup_strategy(
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=1.0,
        fast_break_add_min_hf=3.0,
        signal_by_bar={
            800: {"green": 3, "bearish_1d": False, "bearish_3d": False, "bearish_1w": False},
        },
    )

    actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))

    eth_borrow = next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    assert eth_borrow["amount"] < 5.0
    assert strategy.event_log[-1]["reason"] == "fast_break_hedge_up"


def test_fast_break_state_is_recorded_in_event_log():
    strategy, snapshot = _setup_strategy(
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=0.75,
        signal_by_bar={
            800: {"green": 3, "bearish_1d": False, "bearish_3d": False, "bearish_1w": False},
        },
    )

    strategy.on_bar(snapshot, _fast_break_bar(index=800))

    assert strategy.event_log[-1]["in_fast_break_overlay"] is True
    assert strategy.event_log[-1]["fast_break_hedge_floor"] == pytest.approx(0.75)


def test_under_hedged_fast_break_takes_max_safe_partial_eth_hedge():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=True,
        crisis_sol_drawdown_threshold=0.05,
        crisis_sol_drawdown_lookback_bars=720,
        crisis_portfolio_drawdown_lookback_bars=720,
        crisis_sol_equivalent_drawdown_lookback_bars=720,
        crisis_partial_fill_budget_pct=0.0,
        min_rebalance_hf=2.0,
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=1.0,
        enable_fast_break_partial_fill=True,
        fast_break_partial_fill_min_hf=2.5,
        fast_break_partial_fill_budget_pct=0.25,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 90.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))

    borrow = next(action for action in actions if action["type"] == "borrow")
    deposit = next(
        action for action in actions if action["type"] == "deposit_collateral"
    )
    budget_usd = 9_000.0 * 0.25
    assert borrow == {
        "type": "borrow",
        "symbol": "ETH",
        "amount": pytest.approx(budget_usd / 2_000.0),
    }
    assert deposit == {
        "type": "deposit_collateral",
        "symbol": "USDC",
        "amount": pytest.approx(budget_usd * 0.999),
    }
    assert strategy.fast_break_state.partial_fill_added_usd == pytest.approx(budget_usd)
    assert strategy.event_log[-1]["reason"] == "fast_break_partial_fill"
    assert strategy.event_log[-1]["fast_break_partial_fill_added_usd"] == pytest.approx(
        budget_usd
    )


def test_under_hedged_fast_break_partial_fill_respects_episode_budget():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=True,
        crisis_sol_drawdown_threshold=0.05,
        crisis_sol_drawdown_lookback_bars=720,
        crisis_portfolio_drawdown_lookback_bars=720,
        crisis_sol_equivalent_drawdown_lookback_bars=720,
        crisis_partial_fill_budget_pct=0.0,
        min_rebalance_hf=2.0,
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=1.0,
        enable_fast_break_partial_fill=True,
        fast_break_partial_fill_min_hf=2.5,
        fast_break_partial_fill_budget_pct=0.25,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
            801: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 90.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    first_actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))
    event_count_before_second_bar = len(strategy.event_log)
    second_actions = strategy.on_bar(snapshot, _fast_break_bar(index=801))

    first_borrow = next(action for action in first_actions if action["type"] == "borrow")
    assert first_borrow["amount"] == pytest.approx(2_250.0 / 2_000.0)
    assert second_actions == []
    assert strategy.fast_break_state.partial_fill_added_usd == pytest.approx(2_250.0)
    assert len(strategy.event_log) == event_count_before_second_bar


def test_fast_break_partial_fill_can_require_crisis_mode():
    strategy, _ = _setup_strategy(
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=1.0,
        enable_fast_break_partial_fill=True,
        fast_break_partial_fill_requires_crisis=True,
        fast_break_partial_fill_min_hf=2.5,
        fast_break_partial_fill_budget_pct=0.25,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 90.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))

    borrow = next(action for action in actions if action["type"] == "borrow")
    assert borrow["amount"] > 2_250.0 / 2_000.0
    assert strategy.event_log[-1]["reason"] == "fast_break_hedge_up"
    assert strategy.fast_break_state.partial_fill_added_usd == 0.0


def test_crisis_gated_fast_break_partial_fill_still_fills_when_crisis_is_active():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=True,
        crisis_sol_drawdown_threshold=0.05,
        crisis_sol_drawdown_lookback_bars=720,
        crisis_portfolio_drawdown_lookback_bars=720,
        crisis_sol_equivalent_drawdown_lookback_bars=720,
        crisis_partial_fill_budget_pct=0.0,
        min_rebalance_hf=2.0,
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=1.0,
        enable_fast_break_partial_fill=True,
        fast_break_partial_fill_requires_crisis=True,
        fast_break_partial_fill_min_hf=2.5,
        fast_break_partial_fill_budget_pct=0.25,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 90.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))

    borrow = next(action for action in actions if action["type"] == "borrow")
    assert borrow["amount"] == pytest.approx(2_250.0 / 2_000.0)
    assert strategy.event_log[-1]["reason"] == "fast_break_partial_fill"
    assert strategy.event_log[-1]["in_crisis_mode"] is True


def test_fast_break_partial_fill_max_green_blocks_unconfirmed_crash_fill():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=True,
        crisis_sol_drawdown_threshold=0.05,
        crisis_sol_drawdown_lookback_bars=720,
        crisis_portfolio_drawdown_lookback_bars=720,
        crisis_sol_equivalent_drawdown_lookback_bars=720,
        crisis_partial_fill_budget_pct=0.0,
        min_rebalance_hf=2.0,
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=1.0,
        enable_fast_break_partial_fill=True,
        fast_break_partial_fill_requires_crisis=True,
        fast_break_partial_fill_max_green=1,
        fast_break_partial_fill_min_hf=2.5,
        fast_break_partial_fill_budget_pct=0.25,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 90.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))

    assert actions == []
    assert strategy.event_log[-1]["reason"] == "crisis_enter"
    assert strategy.fast_break_state.partial_fill_added_usd == 0.0


def test_weekly_bearish_reserve_sells_sol_into_usdc_collateral():
    strategy, snapshot = _setup_strategy(
        initial_sol_collateral=200.0,
        enable_weekly_bearish_reserve=True,
        weekly_bearish_reserve_sell_fraction=0.10,
        weekly_bearish_reserve_max_fraction=0.30,
        weekly_bearish_reserve_min_sol_collateral=100.0,
        signal_by_bar={
            800: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": True,
            },
        },
    )

    actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=100.0))

    assert actions == [
        {"type": "withdraw_collateral", "symbol": "SOL", "amount": pytest.approx(20.0)},
        {
            "type": "deposit_collateral",
            "symbol": "USDC",
            "amount": pytest.approx(1_998.0),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "weekly_bearish_reserve_sell"
    assert strategy.history_fields()["weekly_bearish_reserve_usdc"] == pytest.approx(
        1_998.0
    )


def test_weekly_bearish_reserve_rebuys_after_weekly_recovery():
    from arblab.kamino_risk import apply_actions

    strategy, snapshot = _setup_strategy(
        initial_sol_collateral=200.0,
        enable_weekly_bearish_reserve=True,
        weekly_bearish_reserve_sell_fraction=0.10,
        weekly_bearish_reserve_max_fraction=0.30,
        weekly_bearish_reserve_min_sol_collateral=100.0,
        weekly_bearish_reserve_rebuy_fraction=0.50,
        signal_by_bar={
            800: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": True,
            },
            801: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )

    sell_actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=100.0))
    snapshot = apply_actions(snapshot, sell_actions)
    rebuy_actions = strategy.on_bar(snapshot, _bar(index=801, sol_price=100.0))

    assert rebuy_actions == [
        {
            "type": "withdraw_collateral",
            "symbol": "USDC",
            "amount": pytest.approx(999.0),
        },
        {
            "type": "deposit_collateral",
            "symbol": "SOL",
            "amount": pytest.approx(9.98001),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "weekly_bearish_reserve_rebuy"
    assert strategy.history_fields()["weekly_bearish_reserve_usdc"] == pytest.approx(
        999.0
    )


def test_profit_lock_reserve_sells_sol_near_high_when_trend_weakens():
    strategy, _ = _setup_strategy(
        initial_sol_collateral=200.0,
        initial_sol_price=50.0,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=1.00,
        profit_lock_near_high_threshold=0.10,
        profit_lock_drawdown_threshold=0.50,
        profit_lock_max_green=3,
        enable_profit_lock_reserve=True,
        profit_lock_reserve_sell_fraction=0.10,
        profit_lock_reserve_max_fraction=0.30,
        profit_lock_reserve_min_sol_collateral=100.0,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 200.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=200.0))

    assert actions == [
        {"type": "withdraw_collateral", "symbol": "SOL", "amount": pytest.approx(20.0)},
        {
            "type": "deposit_collateral",
            "symbol": "USDC",
            "amount": pytest.approx(3_996.0),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "profit_lock_reserve_sell"
    assert strategy.history_fields()["in_profit_lock_reserve"] is True
    assert strategy.history_fields()["profit_lock_reserve_usdc"] == pytest.approx(
        3_996.0
    )


def test_profit_lock_reserve_escalates_when_three_day_turns_bearish():
    from arblab.kamino_risk import apply_actions

    strategy, _ = _setup_strategy(
        initial_sol_collateral=200.0,
        initial_sol_price=50.0,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=1.00,
        profit_lock_near_high_threshold=0.10,
        profit_lock_drawdown_threshold=0.50,
        profit_lock_max_green=3,
        enable_profit_lock_reserve=True,
        profit_lock_reserve_sell_fraction=0.10,
        profit_lock_reserve_escalation_sell_fraction=0.10,
        profit_lock_reserve_max_fraction=0.30,
        profit_lock_reserve_min_sol_collateral=100.0,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            801: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 200.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    first_actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=200.0))
    snapshot = apply_actions(snapshot, first_actions)
    second_actions = strategy.on_bar(snapshot, _bar(index=801, sol_price=200.0))

    assert second_actions == [
        {"type": "withdraw_collateral", "symbol": "SOL", "amount": pytest.approx(18.0)},
        {
            "type": "deposit_collateral",
            "symbol": "USDC",
            "amount": pytest.approx(3_596.4),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "profit_lock_reserve_escalate"
    assert strategy.history_fields()["profit_lock_reserve_sold_sol"] == pytest.approx(
        38.0
    )


def test_profit_lock_reserve_rebuys_only_after_weekly_recovery():
    from arblab.kamino_risk import apply_actions

    strategy, _ = _setup_strategy(
        initial_sol_collateral=200.0,
        initial_sol_price=50.0,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=1.00,
        profit_lock_near_high_threshold=0.10,
        profit_lock_drawdown_threshold=0.50,
        profit_lock_max_green=3,
        enable_profit_lock_reserve=True,
        profit_lock_reserve_sell_fraction=0.10,
        profit_lock_reserve_max_fraction=0.30,
        profit_lock_reserve_min_sol_collateral=100.0,
        profit_lock_reserve_rebuy_fraction=0.50,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            801: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": True,
            },
            802: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 200.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    sell_actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=200.0))
    snapshot = apply_actions(snapshot, sell_actions)
    blocked_rebuy = strategy.on_bar(snapshot, _bar(index=801, sol_price=200.0))
    rebuy_actions = strategy.on_bar(snapshot, _bar(index=802, sol_price=200.0))

    assert blocked_rebuy == []
    assert rebuy_actions == [
        {
            "type": "withdraw_collateral",
            "symbol": "USDC",
            "amount": pytest.approx(1_998.0),
        },
        {
            "type": "deposit_collateral",
            "symbol": "SOL",
            "amount": pytest.approx(9.98001),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "profit_lock_reserve_rebuy"


def test_stateful_profit_lock_reserve_escalates_once_per_episode():
    from arblab.kamino_risk import apply_actions

    strategy, _ = _setup_strategy(
        initial_sol_collateral=200.0,
        initial_sol_price=50.0,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=1.00,
        profit_lock_near_high_threshold=0.10,
        profit_lock_drawdown_threshold=0.50,
        profit_lock_max_green=3,
        enable_profit_lock_reserve=True,
        profit_lock_reserve_episode_mode=True,
        profit_lock_reserve_sell_fraction=0.10,
        profit_lock_reserve_escalation_sell_fraction=0.10,
        profit_lock_reserve_max_fraction=0.30,
        profit_lock_reserve_min_sol_collateral=100.0,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            801: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
            802: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 200.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    first_actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=200.0))
    snapshot = apply_actions(snapshot, first_actions)
    second_actions = strategy.on_bar(snapshot, _bar(index=801, sol_price=200.0))
    snapshot = apply_actions(snapshot, second_actions)
    strategy.on_bar(snapshot, _bar(index=802, sol_price=200.0))

    assert second_actions
    assert strategy.profit_lock_reserve_state.escalation_slice_sold is True
    assert sum(
        1
        for event in strategy.event_log
        if event["reason"] == "profit_lock_reserve_escalate"
    ) == 1


def test_stateful_profit_lock_reserve_enforces_rebuy_cooldown():
    from arblab.kamino_risk import apply_actions

    strategy, _ = _setup_strategy(
        initial_sol_collateral=200.0,
        initial_sol_price=50.0,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=1.00,
        profit_lock_near_high_threshold=0.10,
        profit_lock_drawdown_threshold=0.50,
        profit_lock_max_green=3,
        enable_profit_lock_reserve=True,
        profit_lock_reserve_episode_mode=True,
        profit_lock_reserve_sell_fraction=0.10,
        profit_lock_reserve_max_fraction=0.30,
        profit_lock_reserve_min_sol_collateral=100.0,
        profit_lock_reserve_rebuy_fraction=0.50,
        profit_lock_reserve_rebuy_cooldown_bars=24,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            801: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            825: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 200.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    sell_actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=200.0))
    snapshot = apply_actions(snapshot, sell_actions)
    early_rebuy = strategy.on_bar(snapshot, _bar(index=801, sol_price=200.0))
    mature_rebuy = strategy.on_bar(snapshot, _bar(index=825, sol_price=200.0))

    assert early_rebuy == []
    assert mature_rebuy
    assert strategy.event_log[-1]["reason"] == "profit_lock_reserve_rebuy"


def test_stateful_profit_lock_reserve_requires_new_high_after_completed_episode():
    from arblab.kamino_risk import apply_actions

    strategy, _ = _setup_strategy(
        initial_sol_collateral=200.0,
        initial_sol_price=50.0,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=1.00,
        profit_lock_near_high_threshold=0.10,
        profit_lock_drawdown_threshold=0.50,
        profit_lock_max_green=3,
        enable_profit_lock_reserve=True,
        profit_lock_reserve_episode_mode=True,
        profit_lock_reserve_sell_fraction=0.10,
        profit_lock_reserve_max_fraction=0.30,
        profit_lock_reserve_min_sol_collateral=100.0,
        profit_lock_reserve_rebuy_fraction=1.00,
        profit_lock_reserve_new_high_reset_gap=0.02,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            801: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            802: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            803: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 200.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    sell_actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=200.0))
    snapshot = apply_actions(snapshot, sell_actions)
    rebuy_actions = strategy.on_bar(snapshot, _bar(index=801, sol_price=200.0))
    snapshot = apply_actions(snapshot, rebuy_actions)
    sell_events_before_repeat = sum(
        1
        for event in strategy.event_log
        if event["reason"] == "profit_lock_reserve_sell"
    )
    strategy.on_bar(snapshot, _bar(index=802, sol_price=200.0))
    sell_events_after_repeat = sum(
        1
        for event in strategy.event_log
        if event["reason"] == "profit_lock_reserve_sell"
    )
    new_high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 199.96002, 210.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    strategy.on_bar(new_high_snapshot, _bar(index=803, sol_price=210.0))
    sell_events_after_new_high = sum(
        1
        for event in strategy.event_log
        if event["reason"] == "profit_lock_reserve_sell"
    )

    assert sell_events_after_repeat == sell_events_before_repeat
    assert sell_events_after_new_high == sell_events_before_repeat + 1


def test_cppi_exposure_cap_does_not_sell_before_activation_gain():
    strategy, _ = _setup_strategy(
        initial_sol_collateral=100.0,
        initial_sol_price=100.0,
        enable_cppi_exposure_cap=True,
        cppi_activation_gain=5.0,
        signal_by_bar={
            800: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 200.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=200.0))

    assert actions == []
    assert strategy.history_fields()["in_cppi_exposure_cap"] is False


def test_cppi_exposure_cap_sells_sol_above_cushion_budget():
    strategy, _ = _setup_strategy(
        initial_sol_collateral=100.0,
        initial_sol_price=100.0,
        enable_cppi_exposure_cap=True,
        cppi_activation_gain=1.5,
        cppi_protect_pct=0.65,
        cppi_cushion_multiplier=1.0,
        cppi_core_min_sol_collateral=100.0,
        cppi_max_sell_fraction_per_bar=1.0,
        signal_by_bar={
            800: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 200.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _bar(index=800, sol_price=200.0))

    assert actions == [
        {
            "type": "withdraw_collateral",
            "symbol": "SOL",
            "amount": pytest.approx(100.0),
        },
        {
            "type": "deposit_collateral",
            "symbol": "USDC",
            "amount": pytest.approx(19_980.0),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "cppi_exposure_cap_sell"
    assert strategy.history_fields()["cppi_protected_usdc"] == pytest.approx(19_980.0)
    assert strategy.history_fields()["cppi_exposure_cap_usd"] == pytest.approx(14_000.0)
    assert strategy.history_fields()["cppi_protected_floor_usd"] == pytest.approx(
        26_000.0
    )


def test_cppi_exposure_cap_rebuys_when_cushion_expands_with_green_trend():
    strategy, _ = _setup_strategy(
        initial_sol_collateral=100.0,
        initial_sol_price=100.0,
        enable_cppi_exposure_cap=True,
        cppi_activation_gain=1.5,
        cppi_protect_pct=0.55,
        cppi_cushion_multiplier=3.0,
        cppi_core_min_sol_collateral=100.0,
        cppi_max_sell_fraction_per_bar=1.0,
        cppi_rebuy_fraction=0.50,
        cppi_rebuy_min_green=4,
        signal_by_bar={
            800: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            801: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    strategy.cppi_exposure_cap_state.active = True
    strategy.cppi_exposure_cap_state.protected_usdc = 19_980.0
    strategy.cppi_exposure_cap_state.sold_sol = 100.0
    strategy._portfolio_high_watermark_value = 40_000.0
    recovery_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 300.0, 0.75, 0.80),
            CollateralPosition("USDC", 19_980.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    rebuy_actions = strategy.on_bar(recovery_snapshot, _bar(index=801, sol_price=300.0))

    assert rebuy_actions == [
        {
            "type": "withdraw_collateral",
            "symbol": "USDC",
            "amount": pytest.approx(17_986.5),
        },
        {
            "type": "deposit_collateral",
            "symbol": "SOL",
            "amount": pytest.approx(59.895045),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "cppi_exposure_cap_rebuy"
    assert strategy.history_fields()["cppi_protected_usdc"] == pytest.approx(1_993.5)


def test_hedge_failure_circuit_breaker_requires_defensive_overlay():
    strategy, _ = _setup_strategy(
        enable_hedge_failure_circuit_breaker=True,
        hedge_failure_lookback_bars=24,
        hedge_failure_underperformance_threshold=0.05,
        hedge_failure_sell_fraction=0.10,
        signal_by_bar={
            800: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 90.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))

    assert actions == []
    assert strategy.history_fields()["in_hedge_failure_circuit_breaker"] is False


def test_hedge_failure_circuit_breaker_sells_sol_when_sol_underperforms_eth():
    strategy, _ = _setup_strategy(
        enable_fast_break_overlay=True,
        fast_break_return_threshold=-0.08,
        fast_break_vol_multiplier=1.5,
        fast_break_hedge_floor=0.0,
        enable_hedge_failure_circuit_breaker=True,
        hedge_failure_lookback_bars=24,
        hedge_failure_underperformance_threshold=0.05,
        hedge_failure_sell_fraction=0.10,
        hedge_failure_min_sol_collateral=100.0,
        signal_by_bar={
            800: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 90.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _fast_break_bar(index=800))

    assert actions == [
        {"type": "withdraw_collateral", "symbol": "SOL", "amount": pytest.approx(20.0)},
        {
            "type": "deposit_collateral",
            "symbol": "USDC",
            "amount": pytest.approx(1_798.2),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "hedge_failure_circuit_breaker_sell"
    assert strategy.history_fields()["in_hedge_failure_circuit_breaker"] is True
    assert strategy.history_fields()["hedge_failure_protected_usdc"] == pytest.approx(
        1_798.2
    )


def test_crisis_mode_raises_target_to_floor_before_full_short_confirmation():
    strategy, snapshot = _setup_strategy(
        enable_crisis_mode=True,
        signal_by_bar={
            2_200: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            }
        },
    )

    actions = strategy.on_bar(snapshot, _drawdown_bar())

    eth_borrow = next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    assert eth_borrow["amount"] == pytest.approx(5.0)
    assert strategy.crisis_state.active is True
    assert strategy.event_log[-1]["reason"] == "crisis_hedge_up"
    assert strategy.event_log[-1]["in_crisis_mode"] is True
    assert strategy.event_log[-1]["crisis_hedge_floor"] == pytest.approx(1.0)
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(1.0)


def test_crisis_mode_can_enter_from_portfolio_drawdown_without_bearish_3d():
    strategy, snapshot = _setup_strategy(
        enable_crisis_mode=True,
        initial_sol_price=150.0,
        signal_by_bar={
            2_200: {
                "green": 3,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            2_201: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": False,
                "bearish_1w": False,
            },
        },
    )

    strategy.on_bar(snapshot, _drawdown_bar(index=2_200, current_sol_price=150.0))
    damaged_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(damaged_snapshot, _drawdown_bar(index=2_201))

    assert next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    assert strategy.crisis_state.active is True
    assert strategy.crisis_state.entry_reason == "sol_drawdown_with_portfolio_damage"
    assert strategy.event_log[-1]["crisis_hedge_floor"] == pytest.approx(0.75)


def test_crisis_mode_can_enter_from_sol_equivalent_drawdown():
    strategy, snapshot = _setup_strategy(
        enable_crisis_mode=True,
        signal_by_bar={
            2_200: {
                "green": 3,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
            },
            2_201: {
                "green": 3,
                "bearish_1d": False,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    strategy.on_bar(snapshot, _drawdown_bar(index=2_200, peak_sol_price=100.0))
    damaged_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 3_000.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(
        damaged_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=100.0, current_sol_price=100.0),
    )

    assert next(a for a in actions if a["type"] == "borrow" and a["symbol"] == "ETH")
    assert strategy.crisis_state.active is True
    assert strategy.crisis_state.entry_reason == "sol_equivalent_drawdown"


def test_active_crisis_mode_blocks_four_green_cover_below_current_floor():
    strategy, snapshot = _setup_strategy(
        enable_crisis_mode=True,
        signal_by_bar={
            2_200: {
                "green": 0,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": True,
            },
            2_201: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": True,
            },
            2_202: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": True,
            },
        },
    )

    from arblab.kamino_risk import apply_actions

    open_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_200))
    snapshot = apply_actions(snapshot, open_actions)
    cover_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_201))

    eth_repay = next(a for a in cover_actions if a["type"] == "repay" and a["symbol"] == "ETH")
    assert eth_repay["amount"] == pytest.approx(3.75)
    assert strategy.crisis_state.active is True
    assert strategy.event_log[-1]["reason"] == "crisis_hedge_down"
    assert strategy.event_log[-1]["crisis_hedge_floor"] == pytest.approx(0.75)
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.75)


def test_profit_lock_raises_hedge_floor_after_large_gain_rolls_over():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=False,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=0.25,
        profit_lock_drawdown_threshold=0.10,
        profit_lock_hedge_floor=0.35,
        profit_lock_max_green=3,
        hedge_ladder={4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50},
        rebalance_threshold=0.01,
        signal_by_bar={
            2_200: {"green": 4, "bearish_1d": False},
            2_201: {"green": 3, "bearish_1d": False},
        },
    )
    high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 150.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    rollover_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 130.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    strategy.on_bar(high_snapshot, _drawdown_bar(index=2_200, current_sol_price=150.0))
    actions = strategy.on_bar(
        rollover_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=150.0, current_sol_price=130.0),
    )

    eth_borrow = next(
        action for action in actions if action["type"] == "borrow" and action["symbol"] == "ETH"
    )
    assert eth_borrow["amount"] == pytest.approx((13_000.0 * 0.35) / 2_000.0)
    assert strategy.event_log[-1]["reason"] == "profit_lock_hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.35)
    assert strategy.history_fields()["in_profit_lock_mode"] is True
    assert strategy.history_fields()["profit_lock_hedge_floor"] == pytest.approx(0.35)


def test_profit_lock_ignores_rollover_without_enough_prior_gain():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=False,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=0.25,
        profit_lock_drawdown_threshold=0.05,
        profit_lock_hedge_floor=0.35,
        profit_lock_max_green=3,
        hedge_ladder={4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50},
        rebalance_threshold=0.01,
        signal_by_bar={
            2_200: {"green": 4, "bearish_1d": False},
            2_201: {"green": 3, "bearish_1d": False},
        },
    )
    high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 110.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    rollover_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    strategy.on_bar(high_snapshot, _drawdown_bar(index=2_200, current_sol_price=110.0))
    actions = strategy.on_bar(
        rollover_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=110.0, current_sol_price=100.0),
    )

    eth_borrow = next(
        action for action in actions if action["type"] == "borrow" and action["symbol"] == "ETH"
    )
    assert eth_borrow["amount"] == pytest.approx((10_000.0 * 0.10) / 2_000.0)
    assert strategy.event_log[-1]["reason"] == "hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.10)
    assert strategy.history_fields()["in_profit_lock_mode"] is False


def test_profit_lock_can_trigger_near_high_before_trailing_drawdown_threshold():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=False,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=0.25,
        profit_lock_drawdown_threshold=0.10,
        profit_lock_near_high_threshold=0.03,
        profit_lock_hedge_floor=0.35,
        profit_lock_max_green=3,
        hedge_ladder={4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50},
        rebalance_threshold=0.01,
        signal_by_bar={
            2_200: {"green": 4, "bearish_1d": False},
            2_201: {"green": 3, "bearish_1d": False},
        },
    )
    high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 150.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    near_high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 147.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    strategy.on_bar(high_snapshot, _drawdown_bar(index=2_200, current_sol_price=150.0))
    actions = strategy.on_bar(
        near_high_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=150.0, current_sol_price=147.0),
    )

    eth_borrow = next(
        action for action in actions if action["type"] == "borrow" and action["symbol"] == "ETH"
    )
    assert eth_borrow["amount"] == pytest.approx((14_700.0 * 0.35) / 2_000.0)
    assert strategy.event_log[-1]["reason"] == "profit_lock_hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.35)
    assert strategy.history_fields()["in_profit_lock_mode"] is True


def test_profit_lock_near_high_trigger_ignores_mid_drawdown_before_normal_trigger():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=False,
        enable_profit_lock=True,
        profit_lock_min_gain_pct=0.25,
        profit_lock_drawdown_threshold=0.10,
        profit_lock_near_high_threshold=0.03,
        profit_lock_hedge_floor=0.35,
        profit_lock_max_green=3,
        hedge_ladder={4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50},
        rebalance_threshold=0.01,
        signal_by_bar={
            2_200: {"green": 4, "bearish_1d": False},
            2_201: {"green": 3, "bearish_1d": False},
        },
    )
    high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 150.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    mid_drawdown_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 141.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    strategy.on_bar(high_snapshot, _drawdown_bar(index=2_200, current_sol_price=150.0))
    actions = strategy.on_bar(
        mid_drawdown_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=150.0, current_sol_price=141.0),
    )

    eth_borrow = next(
        action for action in actions if action["type"] == "borrow" and action["symbol"] == "ETH"
    )
    assert eth_borrow["amount"] == pytest.approx((14_100.0 * 0.10) / 2_000.0)
    assert strategy.event_log[-1]["reason"] == "hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.10)
    assert strategy.history_fields()["in_profit_lock_mode"] is False


def test_stateful_profit_lock_persists_through_mid_drawdown_zone():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=False,
        enable_profit_lock=True,
        profit_lock_stateful=True,
        profit_lock_min_gain_pct=0.25,
        profit_lock_drawdown_threshold=0.10,
        profit_lock_near_high_threshold=0.03,
        profit_lock_hedge_floor=0.35,
        profit_lock_max_green=3,
        hedge_ladder={4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50},
        rebalance_threshold=0.01,
        signal_by_bar={
            2_200: {"green": 4, "bearish_1d": False},
            2_201: {"green": 3, "bearish_1d": False},
            2_202: {"green": 3, "bearish_1d": False},
        },
    )
    high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 150.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    near_high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 147.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    mid_drawdown_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 141.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    strategy.on_bar(high_snapshot, _drawdown_bar(index=2_200, current_sol_price=150.0))
    strategy.on_bar(
        near_high_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=150.0, current_sol_price=147.0),
    )
    actions = strategy.on_bar(
        mid_drawdown_snapshot,
        _drawdown_bar(index=2_202, peak_sol_price=150.0, current_sol_price=141.0),
    )

    eth_borrow = next(
        action for action in actions if action["type"] == "borrow" and action["symbol"] == "ETH"
    )
    assert eth_borrow["amount"] == pytest.approx((14_100.0 * 0.35) / 2_000.0)
    assert strategy.event_log[-1]["reason"] == "profit_lock_hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.35)
    assert strategy.history_fields()["in_profit_lock_mode"] is True


def test_froth_reserve_rotates_excess_sol_to_usdc_once_per_profit_tier():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=False,
        enable_profit_lock=True,
        enable_froth_reserve=True,
        froth_reserve_min_sol_collateral=100.0,
        froth_reserve_tiers={1.0: 0.05},
        profit_lock_min_gain_pct=0.25,
        profit_lock_drawdown_threshold=0.10,
        profit_lock_near_high_threshold=0.03,
        profit_lock_hedge_floor=0.35,
        profit_lock_max_green=3,
        hedge_ladder={4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50},
        rebalance_threshold=0.01,
        signal_by_bar={
            2_200: {"green": 3, "bearish_1d": False},
            2_201: {"green": 3, "bearish_1d": False},
        },
    )
    frothy_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 7_000.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 3.5, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(
        frothy_snapshot,
        _drawdown_bar(index=2_200, peak_sol_price=100.0, current_sol_price=100.0),
    )
    repeat_actions = strategy.on_bar(
        frothy_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=100.0, current_sol_price=100.0),
    )

    assert actions == [
        {"type": "withdraw_collateral", "symbol": "SOL", "amount": pytest.approx(10.0)},
        {"type": "deposit_collateral", "symbol": "USDC", "amount": pytest.approx(999.0)},
    ]
    assert strategy.event_log[-1]["reason"] == "froth_reserve_rotate"
    assert strategy.event_log[-1]["froth_reserve_usdc"] == pytest.approx(999.0)
    assert strategy.history_fields()["froth_reserve_usdc"] == pytest.approx(999.0)
    assert repeat_actions == []


def test_froth_reserve_rebuys_sol_on_deep_sol_drawdown():
    strategy, snapshot = _setup_strategy(
        enable_crisis_mode=False,
        enable_profit_lock=True,
        enable_froth_reserve=True,
        froth_reserve_min_sol_collateral=100.0,
        froth_reserve_tiers={1.0: 0.05},
        froth_reserve_rebuy_drawdown_threshold=0.35,
        froth_reserve_rebuy_fraction=0.25,
        profit_lock_min_gain_pct=0.25,
        profit_lock_drawdown_threshold=0.10,
        profit_lock_near_high_threshold=0.03,
        profit_lock_hedge_floor=0.35,
        profit_lock_max_green=3,
        hedge_ladder={4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50},
        rebalance_threshold=0.10,
        signal_by_bar={
            2_200: {"green": 3, "bearish_1d": False},
            2_201: {"green": 3, "bearish_1d": False},
        },
    )

    from arblab.kamino_risk import apply_actions

    frothy_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 200.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 7_000.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 3.5, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    rotate_actions = strategy.on_bar(
        frothy_snapshot,
        _drawdown_bar(index=2_200, peak_sol_price=100.0, current_sol_price=100.0),
    )
    snapshot = apply_actions(frothy_snapshot, rotate_actions)

    actions = strategy.on_bar(
        snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=100.0, current_sol_price=60.0),
    )

    assert actions == [
        {"type": "withdraw_collateral", "symbol": "USDC", "amount": pytest.approx(249.75)},
        {
            "type": "deposit_collateral",
            "symbol": "SOL",
            "amount": pytest.approx(249.75 * 0.999 / 60.0),
        },
    ]
    assert strategy.event_log[-1]["reason"] == "froth_reserve_rebuy"
    assert strategy.event_log[-1]["froth_reserve_usdc"] == pytest.approx(749.25)


def test_drawdown_containment_raises_hedge_floor_after_portfolio_drawdown():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=False,
        enable_drawdown_containment=True,
        drawdown_containment_trigger=0.20,
        drawdown_containment_exit_gap=0.10,
        drawdown_containment_hedge_floor=0.75,
        hedge_ladder={4: 0.0, 3: 0.10, 2: 0.25, 1: 0.50, 0: 0.50},
        rebalance_threshold=0.01,
        signal_by_bar={
            2_200: {"green": 4, "bearish_1d": False},
            2_201: {"green": 3, "bearish_1d": False},
        },
    )
    high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 150.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    damaged_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    assert strategy.on_bar(
        high_snapshot,
        _drawdown_bar(index=2_200, peak_sol_price=150.0, current_sol_price=150.0),
    ) == []
    actions = strategy.on_bar(
        damaged_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=150.0, current_sol_price=100.0),
    )

    eth_borrow = next(
        action
        for action in actions
        if action["type"] == "borrow" and action["symbol"] == "ETH"
    )
    assert eth_borrow["amount"] == pytest.approx((10_000.0 * 0.75) / 2_000.0)
    assert strategy.event_log[-1]["reason"] == "drawdown_containment_hedge_up"
    assert strategy.event_log[-1]["target_eth_short_ratio"] == pytest.approx(0.75)
    assert strategy.event_log[-1]["in_drawdown_containment"] is True
    assert strategy.history_fields()["drawdown_containment_hedge_floor"] == pytest.approx(
        0.75
    )


def test_drawdown_containment_blocks_froth_rebuy_while_active():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=False,
        enable_froth_reserve=True,
        enable_drawdown_containment=True,
        drawdown_containment_trigger=0.20,
        drawdown_containment_exit_gap=0.10,
        drawdown_containment_hedge_floor=0.0,
        drawdown_containment_block_rebuy=True,
        hedge_ladder={4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0, 0: 0.0},
        rebalance_threshold=0.10,
        signal_by_bar={
            2_200: {"green": 4, "bearish_1d": False},
            2_201: {"green": 4, "bearish_1d": False},
        },
    )
    strategy._froth_reserve_usdc = 1_000.0
    high_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 150.0, 0.75, 0.80),
            CollateralPosition("USDC", 1_000.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )
    damaged_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 1_000.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    assert strategy.on_bar(
        high_snapshot,
        _drawdown_bar(index=2_200, peak_sol_price=150.0, current_sol_price=150.0),
    ) == []
    actions = strategy.on_bar(
        damaged_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=150.0, current_sol_price=90.0),
    )

    assert actions == []
    assert strategy.history_fields()["in_drawdown_containment"] is True
    assert strategy.history_fields()["froth_reserve_usdc"] == pytest.approx(1_000.0)


def test_crisis_mode_exits_on_configured_weekly_recovery_signal():
    strategy, snapshot = _setup_strategy(
        enable_crisis_mode=True,
        crisis_exit_on_1w_green=False,
        signal_by_bar={
            2_200: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
            2_201: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
                "crisis_exit_1w_green": True,
            },
        },
    )

    from arblab.kamino_risk import apply_actions

    open_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_200))
    snapshot = apply_actions(snapshot, open_actions)
    cover_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_201))

    assert strategy.crisis_state.active is False
    assert strategy.crisis_state.exit_reason == "weekly_recovery"
    assert strategy.event_log[-1]["reason"] == "hedge_down"
    assert strategy.event_log[-1]["in_crisis_mode"] is False
    assert strategy.event_log[-1]["crisis_exit_reason"] == "weekly_recovery"
    assert next(a for a in cover_actions if a["type"] == "repay" and a["symbol"] == "ETH")


def test_crisis_mode_exits_when_sol_equivalent_recovers_near_trailing_high():
    strategy, snapshot = _setup_strategy(
        enable_crisis_mode=True,
        crisis_exit_on_1w_green=False,
        signal_by_bar={
            2_200: {
                "green": 0,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": True,
            },
            2_201: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": True,
            },
        },
    )

    from arblab.kamino_risk import apply_actions

    open_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_200))
    snapshot = apply_actions(snapshot, open_actions)
    damaged_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 3_000.0, 1.0, 1.053),
        ],
    )
    strategy.on_bar(
        damaged_snapshot,
        _drawdown_bar(index=2_201, peak_sol_price=100.0, current_sol_price=100.0),
    )
    recovered_snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 500.0, 1.0, 1.053),
        ],
    )

    strategy.on_bar(
        recovered_snapshot,
        _drawdown_bar(index=2_202, peak_sol_price=100.0, current_sol_price=100.0),
    )

    assert strategy.crisis_state.active is False
    assert strategy.crisis_state.exit_reason == "sol_equivalent_recovery"
    assert strategy.event_log[-1]["in_crisis_mode"] is False


def test_under_hedged_crisis_forces_simple_usdc_debt_cleanup():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=True,
        min_rebalance_hf=10.0,
        signal_by_bar={
            2_200: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            }
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 1_000.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 600.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_200))

    assert actions == [
        {"type": "withdraw_collateral", "symbol": "USDC", "amount": pytest.approx(600.0)},
        {"type": "repay", "symbol": "USDC", "amount": pytest.approx(600.0)},
    ]
    assert strategy.crisis_state.under_hedged is True
    assert strategy.event_log[-1]["reason"] == "under_hedged_crisis_cleanup"
    assert strategy.event_log[-1]["under_hedged_crisis"] is True


def test_under_hedged_crisis_takes_max_safe_partial_eth_hedge():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=True,
        min_rebalance_hf=2.0,
        partial_fill_min_hf=2.5,
        crisis_partial_fill_budget_pct=1.0,
        signal_by_bar={
            2_200: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            }
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_200))

    borrow = next(action for action in actions if action["type"] == "borrow")
    deposit = next(
        action for action in actions if action["type"] == "deposit_collateral"
    )
    max_short_usd = 7_500.0 / (2.5 - (0.90 * 0.999))
    assert borrow == {
        "type": "borrow",
        "symbol": "ETH",
        "amount": pytest.approx(max_short_usd / 2_000.0),
    }
    assert deposit == {
        "type": "deposit_collateral",
        "symbol": "USDC",
        "amount": pytest.approx(max_short_usd * 0.999),
    }
    assert max_short_usd < 10_000.0
    assert strategy.crisis_state.under_hedged is True
    assert strategy.event_log[-1]["reason"] == "under_hedged_crisis_partial_fill"
    assert strategy.event_log[-1]["under_hedged_crisis"] is True


def test_under_hedged_crisis_partial_fill_respects_episode_budget():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=True,
        min_rebalance_hf=2.0,
        partial_fill_min_hf=2.5,
        crisis_partial_fill_budget_pct=0.25,
        signal_by_bar={
            2_200: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
            2_201: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    first_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_200))
    event_count_before_second_bar = len(strategy.event_log)
    second_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_201))

    first_borrow = next(action for action in first_actions if action["type"] == "borrow")
    assert first_borrow == {
        "type": "borrow",
        "symbol": "ETH",
        "amount": pytest.approx(2_500.0 / 2_000.0),
    }
    assert second_actions == []
    assert strategy.crisis_state.partial_fill_added_usd == pytest.approx(2_500.0)
    assert len(strategy.event_log) == event_count_before_second_bar


def test_crisis_partial_fill_budget_resets_after_crisis_exit():
    strategy, _ = _setup_strategy(
        enable_crisis_mode=True,
        min_rebalance_hf=2.0,
        partial_fill_min_hf=2.5,
        crisis_partial_fill_budget_pct=0.25,
        crisis_exit_on_1w_green=False,
        signal_by_bar={
            2_200: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
            2_201: {
                "green": 4,
                "bearish_1d": False,
                "bearish_3d": False,
                "bearish_1w": False,
                "crisis_exit_1w_green": True,
            },
            2_202: {
                "green": 3,
                "bearish_1d": True,
                "bearish_3d": True,
                "bearish_1w": False,
            },
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 0.0, 1.0, 1.053),
        ],
    )

    first_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_200))
    strategy.on_bar(snapshot, _drawdown_bar(index=2_201))
    second_episode_actions = strategy.on_bar(snapshot, _drawdown_bar(index=2_202))

    first_borrow = next(action for action in first_actions if action["type"] == "borrow")
    second_borrow = next(
        action for action in second_episode_actions if action["type"] == "borrow"
    )
    assert first_borrow["amount"] == pytest.approx(2_500.0 / 2_000.0)
    assert second_borrow["amount"] == pytest.approx(2_500.0 / 2_000.0)
    assert strategy.crisis_state.partial_fill_added_usd == pytest.approx(2_500.0)


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


def test_green_weekly_regime_repays_usdc_debt_before_surplus_reinvestment():
    strategy, _ = _setup_strategy(
        enable_surplus_usdc_reinvestment=True,
        signal_by_bar={0: {"green": 4, "bearish_3d": False, "bearish_1w": False}},
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 2_500.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 1_500.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    assert actions == [
        {"type": "withdraw_collateral", "symbol": "USDC", "amount": pytest.approx(1_500.0)},
        {"type": "repay", "symbol": "USDC", "amount": pytest.approx(1_500.0)},
    ]
    assert strategy.event_log[-1]["reason"] == "green_regime_usdc_debt_cleanup"


def test_green_weekly_regime_does_not_sell_sol_for_routine_usdc_debt_cleanup():
    strategy, _ = _setup_strategy(
        signal_by_bar={0: {"green": 4, "bearish_3d": False, "bearish_1w": False}},
    )
    snapshot = AccountSnapshot(
        collateral=[
            CollateralPosition("SOL", 100.0, 100.0, 0.75, 0.80),
            CollateralPosition("USDC", 0.0, 1.0, 0.90, 0.93),
        ],
        debt=[
            DebtPosition("ETH", 0.0, 2_000.0, 1.0),
            DebtPosition("USDC", 1_500.0, 1.0, 1.053),
        ],
    )

    actions = strategy.on_bar(snapshot, _bar(0))

    assert actions == []
    assert strategy.event_log == []


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


def test_backtest_history_includes_crisis_state_columns():
    price_data = _drawdown_bar(index=2_200).history
    strategy = SolSupertrendShortStrategy()
    result = BacktestEngine(strategy, EngineConfig(lookback_bars=2_200)).run(
        price_data,
        strategy_config={
            "initial_sol_collateral": 100.0,
            "initial_sol_price": 100.0,
            "initial_eth_price": 2_000.0,
            "enable_crisis_mode": True,
            "signal_by_bar": {
                2_200: {
                    "green": 3,
                    "bearish_1d": True,
                    "bearish_3d": True,
                    "bearish_1w": False,
                }
            },
            "rebalance_cooldown_bars": 0,
        },
    )

    final = result.history.iloc[-1]
    assert final["in_crisis_mode"] == True
    assert final["crisis_hedge_floor"] == pytest.approx(1.0)
    assert final["effective_hedge_target"] == pytest.approx(1.0)
    assert final["under_hedged_crisis"] == False
    assert "crisis_partial_fill_added_usd" in final
