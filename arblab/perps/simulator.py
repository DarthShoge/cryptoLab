"""Single-market perpetual futures account simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PerpSimulatorConfig:
    """Configuration for a simple BTC perp account simulation."""

    starting_equity: float = 10_000.0
    max_gross_exposure: float = 1.0
    fee_rate: float = 0.0006
    funding_rate_per_bar: float = 0.0
    rebalance_deadband: float = 0.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass(frozen=True)
class PerpBacktestResult:
    """Perp simulation output."""

    history: pd.DataFrame
    summary: dict[str, Any]


def simulate_perp_account(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    config: PerpSimulatorConfig | None = None,
) -> PerpBacktestResult:
    """Simulate a one-asset perp account from a pure target exposure signal."""
    config = config or PerpSimulatorConfig()
    signal_columns = ["signal"]
    if "target_exposure" in signals.columns:
        signal_columns.append("target_exposure")
    aligned = price_data[["btc_close"]].join(signals[signal_columns], how="inner")
    if aligned.empty:
        raise ValueError("price_data and signals must overlap")

    equity = float(config.starting_equity)
    quantity = 0.0
    average_entry: float | None = None
    previous_price: float | None = None
    blocked_direction = 0

    records: list[dict[str, Any]] = []

    for timestamp, row in aligned.iterrows():
        price = float(row["btc_close"])
        raw_signal = _target_exposure_for_row(row, config)
        realized_pnl = 0.0
        fees = 0.0
        funding = 0.0
        traded_notional = 0.0
        exit_reason = ""

        if previous_price is not None and quantity != 0.0:
            realized_pnl = quantity * (price - previous_price)
            equity += realized_pnl

        current_notional = quantity * price
        if current_notional != 0.0 and config.funding_rate_per_bar != 0.0:
            funding = abs(current_notional) * config.funding_rate_per_bar
            equity -= funding

        stop_reason = _exit_reason(
            price=price,
            quantity=quantity,
            average_entry=average_entry,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
        )
        if stop_reason:
            traded_notional += abs(current_notional)
            fees += abs(current_notional) * config.fee_rate
            equity -= fees
            blocked_direction = _direction(quantity)
            quantity = 0.0
            average_entry = None
            current_notional = 0.0
            exit_reason = stop_reason

        signal_direction = _direction(raw_signal)
        if blocked_direction != 0:
            if signal_direction == 0 or signal_direction == -blocked_direction:
                blocked_direction = 0
            elif signal_direction == blocked_direction:
                raw_signal = 0.0

        target_notional = equity * raw_signal
        current_notional = quantity * price
        current_exposure = _exposure(current_notional, equity)
        target_exposure = _exposure(target_notional, equity)
        should_trade = (
            abs(target_exposure - current_exposure) >= config.rebalance_deadband
            or current_notional == 0.0
            and target_notional != 0.0
            or target_notional == 0.0
            and current_notional != 0.0
        )

        if should_trade:
            delta_notional = target_notional - current_notional
            if delta_notional != 0.0:
                trade_fee = abs(delta_notional) * config.fee_rate
                fees += trade_fee
                traded_notional += abs(delta_notional)
                equity -= trade_fee
                quantity, average_entry = _apply_trade(
                    quantity=quantity,
                    average_entry=average_entry,
                    delta_notional=delta_notional,
                    price=price,
                )

        position_notional = quantity * price
        records.append(
            {
                "timestamp": timestamp,
                "btc_close": price,
                "signal": float(row["signal"]),
                "effective_signal": raw_signal,
                "equity": equity,
                "position_notional": position_notional,
                "exposure": _exposure(position_notional, equity),
                "quantity": quantity,
                "average_entry": average_entry if average_entry is not None else 0.0,
                "traded_notional": traded_notional,
                "fees": fees,
                "funding": funding,
                "pnl": realized_pnl,
                "exit_reason": exit_reason,
                "blocked_direction": blocked_direction,
            }
        )
        previous_price = price

    history = pd.DataFrame(records).set_index("timestamp")
    return PerpBacktestResult(history=history, summary=_summary(history, config))


def _exit_reason(
    price: float,
    quantity: float,
    average_entry: float | None,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> str:
    if quantity == 0.0 or average_entry is None:
        return ""
    if quantity > 0.0:
        if stop_loss_pct is not None and price <= average_entry * (1.0 - stop_loss_pct):
            return "stop_loss"
        if take_profit_pct is not None and price >= average_entry * (1.0 + take_profit_pct):
            return "take_profit"
    if quantity < 0.0:
        if stop_loss_pct is not None and price >= average_entry * (1.0 + stop_loss_pct):
            return "stop_loss"
        if take_profit_pct is not None and price <= average_entry * (1.0 - take_profit_pct):
            return "take_profit"
    return ""


def _target_exposure_for_row(
    row: pd.Series,
    config: PerpSimulatorConfig,
) -> float:
    if "target_exposure" in row and pd.notna(row["target_exposure"]):
        target = float(row["target_exposure"])
    else:
        signal = float(max(-1.0, min(1.0, row["signal"])))
        target = signal * config.max_gross_exposure
    return float(
        max(
            -config.max_gross_exposure,
            min(config.max_gross_exposure, target),
        )
    )


def _apply_trade(
    quantity: float,
    average_entry: float | None,
    delta_notional: float,
    price: float,
) -> tuple[float, float | None]:
    if delta_notional == 0.0:
        return quantity, average_entry

    delta_quantity = delta_notional / price
    new_quantity = quantity + delta_quantity
    if new_quantity == 0.0:
        return 0.0, None

    if quantity == 0.0 or _direction(quantity) != _direction(new_quantity):
        return new_quantity, price

    if _direction(quantity) == _direction(delta_quantity):
        existing_notional = abs(quantity) * (average_entry if average_entry is not None else price)
        added_notional = abs(delta_quantity) * price
        entry = (existing_notional + added_notional) / (abs(quantity) + abs(delta_quantity))
        return new_quantity, entry

    return new_quantity, average_entry


def _direction(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _exposure(notional: float, equity: float) -> float:
    if equity == 0.0:
        return 0.0
    return notional / equity


def _summary(history: pd.DataFrame, config: PerpSimulatorConfig) -> dict[str, Any]:
    initial = config.starting_equity
    final = float(history["equity"].iloc[-1])
    return {
        "starting_equity": initial,
        "final_equity": final,
        "total_return": final / initial - 1.0 if initial else 0.0,
        "turnover": float(history["traded_notional"].sum()),
        "fee_drag": float(history["fees"].sum()),
        "funding_drag": float(history["funding"].sum()),
        "stop_count": int((history["exit_reason"] == "stop_loss").sum()),
        "take_profit_count": int((history["exit_reason"] == "take_profit").sum()),
    }
