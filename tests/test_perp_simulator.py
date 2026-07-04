from __future__ import annotations

import pytest
import pandas as pd

from arblab.perps.simulator import PerpSimulatorConfig, simulate_perp_account


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
