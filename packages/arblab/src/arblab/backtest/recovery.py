"""Drawdown recovery diagnostics for backtest reports."""

from __future__ import annotations

from typing import Any

import pandas as pd


def drawdown_recovery_events(
    values: pd.Series,
    min_drawdown_pct: float = 30.0,
) -> list[dict[str, Any]]:
    """Return major drawdown episodes with recovery timing.

    Recovery half-life is measured from the trough to the first bar that
    recovers half the loss from the prior peak. New-high timing requires a
    value strictly above the prior peak.
    """
    if len(values) < 2:
        return []

    clean = values.reset_index(drop=True).astype(float)
    events: list[dict[str, Any]] = []
    peak_index = 0
    peak_value = float(clean.iloc[0])
    trough_index: int | None = None
    trough_value: float | None = None

    for idx in range(1, len(clean)):
        value = float(clean.iloc[idx])
        if value > peak_value:
            if trough_index is not None and trough_value is not None:
                _append_recovery_event(
                    events,
                    clean,
                    peak_index,
                    peak_value,
                    trough_index,
                    trough_value,
                    min_drawdown_pct,
                )
            peak_index = idx
            peak_value = value
            trough_index = None
            trough_value = None
            continue

        if value < peak_value and (trough_value is None or value < trough_value):
            trough_index = idx
            trough_value = value

    if trough_index is not None and trough_value is not None:
        _append_recovery_event(
            events,
            clean,
            peak_index,
            peak_value,
            trough_index,
            trough_value,
            min_drawdown_pct,
        )

    return events


def _append_recovery_event(
    events: list[dict[str, Any]],
    values: pd.Series,
    peak_index: int,
    peak_value: float,
    trough_index: int,
    trough_value: float,
    min_drawdown_pct: float,
) -> None:
    if peak_value <= 0:
        return
    drawdown_pct = (peak_value - trough_value) / peak_value * 100.0
    if drawdown_pct < min_drawdown_pct:
        return

    half_recovery_value = trough_value + ((peak_value - trough_value) * 0.5)
    half_recovery_bars = None
    new_high_bars = None
    for idx in range(trough_index + 1, len(values)):
        value = float(values.iloc[idx])
        if half_recovery_bars is None and value >= half_recovery_value:
            half_recovery_bars = idx - trough_index
        if new_high_bars is None and value > peak_value:
            new_high_bars = idx - trough_index
        if half_recovery_bars is not None and new_high_bars is not None:
            break

    events.append(
        {
            "peak_index": peak_index,
            "trough_index": trough_index,
            "peak_value": peak_value,
            "trough_value": trough_value,
            "drawdown_pct": drawdown_pct,
            "half_recovery_bars": half_recovery_bars,
            "new_high_bars": new_high_bars,
        }
    )
