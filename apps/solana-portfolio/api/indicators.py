"""Wilder ATR and close-confirmed Supertrend; no future candle lookahead."""

import math


def supertrend(candles, period=10, multiplier=3.0):
    if (
        not 1 <= period <= 100
        or not math.isfinite(multiplier)
        or not 0.1 <= multiplier <= 20
    ):
        raise ValueError("ATR period must be 1–100; multiplier must be 0.1–20.")
    result, ranges = [], []
    atr = upper = lower = None
    direction = "bearish"
    for index, candle in enumerate(candles):
        previous_close = candles[index - 1]["close"] if index else candle["close"]
        tr = max(
            candle["high"] - candle["low"],
            abs(candle["high"] - previous_close),
            abs(candle["low"] - previous_close),
        )
        ranges.append(tr)
        if index < period - 1:
            result.append(
                {"time": candle["time"], "value": None, "direction": None, "atr": None}
            )
            continue
        atr = (
            sum(ranges[:period]) / period
            if atr is None
            else (atr * (period - 1) + tr) / period
        )
        midpoint = (candle["high"] + candle["low"]) / 2
        raw_upper, raw_lower = midpoint + multiplier * atr, midpoint - multiplier * atr
        new_upper = (
            raw_upper
            if upper is None or raw_upper < upper or previous_close > upper
            else upper
        )
        new_lower = (
            raw_lower
            if lower is None or raw_lower > lower or previous_close < lower
            else lower
        )
        if upper is not None:
            if direction == "bearish" and candle["close"] > new_upper:
                direction = "bullish"
            elif direction == "bullish" and candle["close"] < new_lower:
                direction = "bearish"
        upper, lower = new_upper, new_lower
        result.append(
            {
                "time": candle["time"],
                "value": lower if direction == "bullish" else upper,
                "direction": direction,
                "atr": atr,
            }
        )
    return result
