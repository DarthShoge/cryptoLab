"""Single-market perpetual futures account simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from arblab.perps.venue import PerpVenueConfig


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
    venue: PerpVenueConfig | None = None
    liquidity: str = "taker"
    stop_on_liquidation: bool = False


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
    if "funding_rate" in signals.columns:
        signal_columns.append("funding_rate")
    price_columns = ["btc_close"]
    if "mark_price" in price_data.columns:
        price_columns.append("mark_price")
    aligned = price_data[price_columns].join(signals[signal_columns], how="inner")
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
        liquidation_fee = 0.0
        traded_notional = 0.0
        exit_reason = ""
        mark_price = float(row["mark_price"]) if "mark_price" in row else price
        maintenance_margin = 0.0
        initial_margin = 0.0

        if previous_price is not None and quantity != 0.0:
            realized_pnl = quantity * (price - previous_price)
            equity += realized_pnl

        current_notional = quantity * price
        mark_notional = quantity * mark_price
        if current_notional != 0.0:
            funding_rate = _funding_rate_for_row(row, config)
            if funding_rate != 0.0:
                funding = _funding_cashflow(mark_notional, funding_rate, config)
                equity -= funding

        if config.venue is not None and mark_notional != 0.0:
            maintenance_margin = config.venue.maintenance_margin(mark_notional)
            initial_margin = config.venue.initial_margin(mark_notional)
            if equity <= maintenance_margin:
                liquidation_fee = abs(mark_notional) * config.venue.liquidation_fee_rate
                equity -= liquidation_fee
                traded_notional += abs(mark_notional)
                blocked_direction = _direction(quantity)
                quantity = 0.0
                average_entry = None
                current_notional = 0.0
                mark_notional = 0.0
                exit_reason = "liquidation"

        stop_reason = _exit_reason(
            price=price,
            quantity=quantity,
            average_entry=average_entry,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
        )
        if stop_reason and not exit_reason:
            traded_notional += abs(current_notional)
            fees += abs(current_notional) * _trade_fee_rate(config)
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
        target_notional = _apply_venue_notional_cap(target_notional, equity, config)
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

        if should_trade and not (exit_reason == "liquidation" and config.stop_on_liquidation):
            delta_notional = target_notional - current_notional
            delta_notional = _apply_order_constraints(delta_notional, price, config)
            if delta_notional != 0.0:
                trade_fee = abs(delta_notional) * _trade_fee_rate(config)
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
        if config.venue is not None and quantity != 0.0:
            maintenance_margin = config.venue.maintenance_margin(quantity * mark_price)
            initial_margin = config.venue.initial_margin(quantity * mark_price)
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
                "liquidation_fee": liquidation_fee,
                "maintenance_margin": maintenance_margin,
                "initial_margin": initial_margin,
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


def _trade_fee_rate(config: PerpSimulatorConfig) -> float:
    if config.venue is not None:
        return config.venue.fees.rate(config.liquidity)
    return config.fee_rate


def _funding_rate_for_row(row: pd.Series, config: PerpSimulatorConfig) -> float:
    if "funding_rate" in row and pd.notna(row["funding_rate"]):
        return float(row["funding_rate"])
    return config.funding_rate_per_bar


def _funding_cashflow(
    position_notional: float,
    funding_rate: float,
    config: PerpSimulatorConfig,
) -> float:
    if config.venue is None:
        return abs(position_notional) * funding_rate
    return position_notional * funding_rate


def _apply_venue_notional_cap(
    target_notional: float,
    equity: float,
    config: PerpSimulatorConfig,
) -> float:
    if config.venue is None:
        return target_notional
    cap = config.venue.max_allowed_notional(equity)
    if cap is None:
        return target_notional
    return max(-cap, min(cap, target_notional))


def _apply_order_constraints(
    delta_notional: float,
    price: float,
    config: PerpSimulatorConfig,
) -> float:
    if config.venue is None:
        return delta_notional
    return config.venue.order_constraints.adjust_notional(delta_notional, price)


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
        "liquidation_count": int((history["exit_reason"] == "liquidation").sum()),
        "liquidation_fee_drag": float(history.get("liquidation_fee", pd.Series(dtype=float)).sum()),
        "stop_count": int((history["exit_reason"] == "stop_loss").sum()),
        "take_profit_count": int((history["exit_reason"] == "take_profit").sum()),
    }
