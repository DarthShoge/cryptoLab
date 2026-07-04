from __future__ import annotations

import pandas as pd

from arblab.perps.signal import PureSignalConfig, generate_btc_signal


def _btc_history(closes: list[float], freq: str = "1h") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq=freq, tz="UTC")
    close = pd.Series(closes, index=index, dtype=float)
    frame = pd.DataFrame(
        {
            ("BTC", "open"): close.shift(1).fillna(close.iloc[0]),
            ("BTC", "high"): close * 1.01,
            ("BTC", "low"): close * 0.99,
            ("BTC", "close"): close,
            ("BTC", "volume"): 1_000.0,
        },
        index=index,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def test_generate_btc_signal_clamps_and_applies_no_trade_zone():
    history = _btc_history([360.0 - i for i in range(260)])
    config = PureSignalConfig(
        timeframes=("1h",),
        supertrend_weight=0.0,
        ema_weight=0.0,
        rsi_weight=2.0,
        no_trade_zone=0.15,
        rsi_period=2,
        rsi_overbought=101.0,
        rsi_oversold=99.0,
    )

    signals = generate_btc_signal(history, config)

    assert signals["signal"].between(-1.0, 1.0).all()
    assert signals["signal"].iloc[-1] == 1.0

    flat_config = PureSignalConfig(
        timeframes=("1h",),
        supertrend_weight=0.0,
        ema_weight=0.0,
        rsi_weight=0.10,
        no_trade_zone=0.15,
        rsi_period=2,
        rsi_overbought=101.0,
        rsi_oversold=99.0,
    )

    flat_signals = generate_btc_signal(history, flat_config)

    assert flat_signals["signal"].iloc[-1] == 0.0


def test_generate_btc_signal_includes_explainable_components():
    history = _btc_history([100.0 + i for i in range(260)])

    signals = generate_btc_signal(
        history,
        PureSignalConfig(timeframes=("1h",), ema_fast=3, ema_slow=8, rsi_period=3),
    )

    assert {
        "btc_close",
        "signal",
        "raw_score",
        "supertrend_score",
        "ema_score",
        "rsi_score",
        "supertrend_1h",
        "ema_1h",
        "rsi_1h",
    }.issubset(signals.columns)
    assert signals.index.equals(history.index)


def test_higher_timeframe_votes_use_only_closed_candles():
    history = _btc_history([100.0] * 8)
    base_config = PureSignalConfig(
        timeframes=("4h",),
        supertrend_weight=0.0,
        ema_weight=1.0,
        rsi_weight=0.0,
        ema_fast=1,
        ema_slow=2,
        no_trade_zone=0.0,
    )

    baseline = generate_btc_signal(history, base_config)

    changed = history.copy()
    changed.loc[changed.index[-1], ("BTC", "close")] = 200.0
    changed.loc[changed.index[-1], ("BTC", "high")] = 202.0
    changed.loc[changed.index[-1], ("BTC", "low")] = 198.0

    with_in_progress_move = generate_btc_signal(changed, base_config)

    assert with_in_progress_move["ema_4h"].iloc[-1] == baseline["ema_4h"].iloc[-1]
    assert with_in_progress_move["signal"].iloc[-1] == baseline["signal"].iloc[-1]
