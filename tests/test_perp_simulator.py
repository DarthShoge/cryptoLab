from __future__ import annotations

import pytest
import pandas as pd

from arblab.perps.simulator import PerpSimulatorConfig, simulate_perp_account
from arblab.perps.venue import (
    FeeSchedule,
    MarginTier,
    OrderConstraints,
    PerpContractSpec,
    PerpVenueConfig,
)


def _price_frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({"btc_close": closes}, index=index, dtype=float)


def _signals(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.DataFrame({"signal": values}, index=index, dtype=float)


def test_simulator_reconciles_long_pnl_and_fees():
    result = simulate_perp_account(
        _price_frame([100.0, 110.0]),
        _signals([1.0, 1.0]),
        PerpSimulatorConfig(starting_equity=10_000.0, fee_rate=0.001),
    )

    history = result.history

    assert history["fees"].sum() == pytest.approx(10.01)
    assert history["equity"].iloc[-1] == pytest.approx(10_989.99)
    assert history["position_notional"].iloc[-1] == pytest.approx(10_990.0)
    assert history["exposure"].iloc[-1] == pytest.approx(10_990.0 / 10_989.99)


def test_rebalance_deadband_suppresses_tiny_target_changes():
    result = simulate_perp_account(
        _price_frame([100.0, 100.0, 100.0]),
        _signals([1.0, 0.95, 0.80]),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            fee_rate=0.0,
            rebalance_deadband=0.10,
        ),
    )

    history = result.history

    assert history["traded_notional"].iloc[1] == 0.0
    assert history["traded_notional"].iloc[2] == pytest.approx(2_000.0)
    assert history["exposure"].iloc[1] == pytest.approx(1.0)
    assert history["exposure"].iloc[2] == pytest.approx(0.80)


def test_simulator_uses_dynamic_target_exposure_when_provided():
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    result = simulate_perp_account(
        pd.DataFrame({"btc_close": [100.0, 100.0]}, index=index),
        pd.DataFrame(
            {
                "signal": [1.0, 1.0],
                "target_exposure": [2.0, 12.0],
            },
            index=index,
        ),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            max_gross_exposure=10.0,
            fee_rate=0.0,
            rebalance_deadband=0.0,
        ),
    )

    history = result.history

    assert history["exposure"].iloc[0] == pytest.approx(2.0)
    assert history["effective_signal"].iloc[0] == pytest.approx(2.0)
    assert history["exposure"].iloc[1] == pytest.approx(10.0)
    assert history["effective_signal"].iloc[1] == pytest.approx(10.0)


def test_long_stop_loss_flattens_position_and_blocks_same_direction_reentry():
    result = simulate_perp_account(
        _price_frame([100.0, 94.0, 95.0, 96.0, 97.0]),
        _signals([1.0, 1.0, 1.0, 0.0, 1.0]),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            fee_rate=0.0,
            stop_loss_pct=0.05,
        ),
    )

    history = result.history

    assert history["exit_reason"].iloc[1] == "stop_loss"
    assert history["position_notional"].iloc[1] == 0.0
    assert history["blocked_direction"].iloc[2] == 1
    assert history["position_notional"].iloc[2] == 0.0
    assert history["blocked_direction"].iloc[3] == 0
    assert history["position_notional"].iloc[4] > 0.0
    assert result.summary["stop_count"] == 1


def test_short_stop_loss_and_take_profit_thresholds():
    stop_result = simulate_perp_account(
        _price_frame([100.0, 106.0]),
        _signals([-1.0, -1.0]),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            fee_rate=0.0,
            stop_loss_pct=0.05,
        ),
    )

    take_profit_result = simulate_perp_account(
        _price_frame([100.0, 90.0]),
        _signals([-1.0, -1.0]),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            fee_rate=0.0,
            take_profit_pct=0.10,
        ),
    )

    assert stop_result.history["exit_reason"].iloc[1] == "stop_loss"
    assert take_profit_result.history["exit_reason"].iloc[1] == "take_profit"
    assert stop_result.summary["stop_count"] == 1
    assert take_profit_result.summary["take_profit_count"] == 1


def test_long_take_profit_flattens_position():
    result = simulate_perp_account(
        _price_frame([100.0, 111.0]),
        _signals([1.0, 1.0]),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            fee_rate=0.0,
            take_profit_pct=0.10,
        ),
    )

    assert result.history["exit_reason"].iloc[1] == "take_profit"
    assert result.history["position_notional"].iloc[1] == 0.0
    assert result.summary["take_profit_count"] == 1


def test_venue_funding_uses_position_direction():
    venue = PerpVenueConfig(
        name="test",
        contract=PerpContractSpec(symbol="BTC-PERP", settlement_asset="USDT"),
        fees=FeeSchedule(taker=0.0),
        margin_tiers=(MarginTier(0.0, None, 0.01, 0.005),),
    )

    long_result = simulate_perp_account(
        _price_frame([100.0, 100.0]),
        _signals([1.0, 1.0]),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            fee_rate=0.0,
            funding_rate_per_bar=0.001,
            venue=venue,
        ),
    )
    short_result = simulate_perp_account(
        _price_frame([100.0, 100.0]),
        _signals([-1.0, -1.0]),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            fee_rate=0.0,
            funding_rate_per_bar=0.001,
            venue=venue,
        ),
    )

    assert long_result.history["funding"].iloc[1] == pytest.approx(10.0)
    assert long_result.history["equity"].iloc[-1] == pytest.approx(9_990.0)
    assert short_result.history["funding"].iloc[1] == pytest.approx(-10.0)
    assert short_result.history["equity"].iloc[-1] == pytest.approx(10_010.0)


def test_venue_order_constraints_round_trade_and_skip_min_notional():
    venue = PerpVenueConfig(
        name="test",
        contract=PerpContractSpec(symbol="BTC-PERP", settlement_asset="USDT"),
        fees=FeeSchedule(taker=0.0),
        order_constraints=OrderConstraints(min_notional=50.0, quantity_step=0.1),
        margin_tiers=(MarginTier(0.0, None, 0.01, 0.005),),
    )
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")

    result = simulate_perp_account(
        pd.DataFrame({"btc_close": [100.0, 100.0, 100.0]}, index=index),
        pd.DataFrame(
            {"signal": [1.0, 1.0, 1.0], "target_exposure": [0.1234, 0.1270, 0.20]},
            index=index,
        ),
        PerpSimulatorConfig(starting_equity=10_000.0, fee_rate=0.0, venue=venue),
    )

    history = result.history

    assert history["traded_notional"].iloc[0] == pytest.approx(1_230.0)
    assert history["position_notional"].iloc[0] == pytest.approx(1_230.0)
    assert history["traded_notional"].iloc[1] == 0.0
    assert history["traded_notional"].iloc[2] == pytest.approx(770.0)


def test_venue_liquidates_when_equity_breaches_maintenance_margin():
    venue = PerpVenueConfig(
        name="test",
        contract=PerpContractSpec(symbol="BTC-PERP", settlement_asset="USDT"),
        fees=FeeSchedule(taker=0.0),
        liquidation_fee_rate=0.01,
        margin_tiers=(MarginTier(0.0, None, 0.10, 0.05),),
    )
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")

    result = simulate_perp_account(
        pd.DataFrame({"btc_close": [100.0, 50.0]}, index=index),
        pd.DataFrame(
            {"signal": [1.0, 1.0], "target_exposure": [10.0, 10.0]},
            index=index,
        ),
        PerpSimulatorConfig(
            starting_equity=10_000.0,
            max_gross_exposure=10.0,
            fee_rate=0.0,
            venue=venue,
        ),
    )

    history = result.history

    assert history["exit_reason"].iloc[1] == "liquidation"
    assert history["position_notional"].iloc[1] == 0.0
    assert history["liquidation_fee"].iloc[1] == pytest.approx(500.0)
    assert history["maintenance_margin"].iloc[1] == pytest.approx(2_500.0)
    assert result.summary["liquidation_count"] == 1
    assert result.summary["liquidation_fee_drag"] == pytest.approx(500.0)
