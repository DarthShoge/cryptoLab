from __future__ import annotations

import pandas as pd
import pytest

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


def test_default_signal_uses_calibrated_trend_and_rsi_filter_settings():
    config = PureSignalConfig()

    assert config.timeframes == ("1w", "1d", "4h")
    assert config.supertrend_params_by_timeframe == {
        "1w": (7, 4.0),
        "1d": (21, 4.0),
        "4h": (7, 4.0),
    }
    assert config.supertrend_weight == 1.0
    assert config.ema_weight == 0.0
    assert config.rsi_weight == 0.0
    assert config.rsi_filter_enabled is True
    assert config.rsi_filter_timeframe == "1d"
    assert config.rsi_filter_period == 42
    assert config.rsi_filter_oversold == 35.0
    assert config.rsi_filter_overbought == 80.0
    assert config.rsi_filter_weight == 0.5


def test_supertrend_params_can_be_calibrated_per_timeframe(monkeypatch):
    calls: list[tuple[int, float]] = []

    def fake_supertrend_direction(ohlcv, atr_period, multiplier):
        calls.append((atr_period, multiplier))
        return pd.Series(True, index=ohlcv.index)

    monkeypatch.setattr(
        "arblab.perps.signal.supertrend_direction",
        fake_supertrend_direction,
    )
    history = _btc_history([100.0 + i for i in range(120)])

    generate_btc_signal(
        history,
        PureSignalConfig(
            timeframes=("1h", "4h"),
            supertrend_params_by_timeframe={"1h": (7, 2.0), "4h": (21, 4.0)},
            supertrend_weight=1.0,
            ema_weight=0.0,
            rsi_weight=0.0,
            rsi_filter_enabled=False,
            no_trade_zone=0.0,
        ),
    )

    assert calls == [(7, 2.0), (21, 4.0)]


def test_rsi_filter_reduces_stretched_same_direction_exposure():
    rising = _btc_history([100.0 + i for i in range(80)])

    long_signals = generate_btc_signal(
        rising,
        PureSignalConfig(
            timeframes=("1h",),
            supertrend_weight=0.0,
            ema_weight=1.0,
            rsi_weight=0.0,
            ema_fast=2,
            ema_slow=5,
            rsi_filter_enabled=True,
            rsi_filter_timeframe="1h",
            rsi_filter_period=2,
            rsi_filter_oversold=1.0,
            rsi_filter_overbought=50.0,
            rsi_filter_weight=0.5,
            no_trade_zone=0.0,
        ),
    )

    assert long_signals["raw_score"].iloc[-1] == pytest.approx(1.0)
    assert long_signals["rsi_filter_signal"].iloc[-1] == -1.0
    assert long_signals["signal"].iloc[-1] == pytest.approx(0.5)

    falling = _btc_history([180.0 - i for i in range(80)])
    short_signals = generate_btc_signal(
        falling,
        PureSignalConfig(
            timeframes=("1h",),
            supertrend_weight=0.0,
            ema_weight=1.0,
            rsi_weight=0.0,
            ema_fast=2,
            ema_slow=5,
            rsi_filter_enabled=True,
            rsi_filter_timeframe="1h",
            rsi_filter_period=2,
            rsi_filter_oversold=50.0,
            rsi_filter_overbought=99.0,
            rsi_filter_weight=0.5,
            no_trade_zone=0.0,
        ),
    )

    assert short_signals["raw_score"].iloc[-1] == pytest.approx(-1.0)
    assert short_signals["rsi_filter_signal"].iloc[-1] == 1.0
    assert short_signals["signal"].iloc[-1] == pytest.approx(-0.5)


def test_trend_overlay_raises_long_target_exposure_on_confirmed_breakout(monkeypatch):
    history = _btc_history([100.0 + i for i in range(600)])

    def fake_supertrend_vote(ohlcv, config, timeframe):
        return pd.Series(1.0, index=ohlcv.index)

    monkeypatch.setattr("arblab.perps.signal._supertrend_vote", fake_supertrend_vote)

    signals = generate_btc_signal(
        history,
        PureSignalConfig(
            timeframes=("1w", "1d", "4h"),
            supertrend_weight=1.0,
            ema_weight=0.0,
            rsi_weight=0.0,
            rsi_filter_enabled=False,
            trend_overlay_enabled=True,
            trend_overlay_agreement_exposure=2.0,
            trend_overlay_breakout_exposure=3.0,
            trend_overlay_breakout_lookback=20,
            trend_overlay_ema_fast=5,
            trend_overlay_ema_slow=20,
            no_trade_zone=0.0,
        ),
    )

    assert signals["signal"].iloc[-1] == pytest.approx(1.0)
    assert bool(signals["trend_overlay_active"].iloc[-1]) is True
    assert signals["trend_overlay_reason"].iloc[-1] == "breakout"
    assert signals["target_exposure"].iloc[-1] == pytest.approx(3.0)


def test_vol_flattening_overlay_adds_micro_long_leverage(monkeypatch):
    history = _btc_history([100.0 + i for i in range(600)])

    def fake_supertrend_vote(ohlcv, config, timeframe):
        return pd.Series(1.0, index=ohlcv.index)

    def fake_vol_flattening_setup(price_data, symbol, config):
        return pd.Series(True, index=price_data.index), pd.Series(0.5, index=price_data.index)

    monkeypatch.setattr("arblab.perps.signal._supertrend_vote", fake_supertrend_vote)
    monkeypatch.setattr(
        "arblab.perps.signal._volatility_flattening_setup",
        fake_vol_flattening_setup,
    )

    signals = generate_btc_signal(
        history,
        PureSignalConfig(
            timeframes=("1w", "1d", "4h"),
            supertrend_weight=1.0,
            ema_weight=0.0,
            rsi_weight=0.0,
            rsi_filter_enabled=False,
            vol_flattening_overlay_enabled=True,
            vol_flattening_overlay_leverage=1.05,
            no_trade_zone=0.0,
        ),
    )

    assert signals["signal"].iloc[-1] == pytest.approx(1.0)
    assert bool(signals["vol_flattening_overlay_active"].iloc[-1]) is True
    assert signals["target_exposure"].iloc[-1] == pytest.approx(1.05)
    assert signals["vol_percentile"].iloc[-1] == pytest.approx(0.5)


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
