"""Pure BTC perpetual futures signal generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from arblab.strategies.sol_supertrend_short import supertrend_direction


@dataclass(frozen=True)
class PureSignalConfig:
    """Configuration for a stateless BTC exposure signal."""

    timeframes: tuple[str, ...] = ("1w", "1d", "4h")
    atr_period: int = 10
    supertrend_multiplier: float = 3.0
    supertrend_params_by_timeframe: dict[str, tuple[int, float]] = field(
        default_factory=lambda: {
            "1w": (7, 4.0),
            "1d": (21, 4.0),
            "4h": (7, 4.0),
        }
    )
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    supertrend_weight: float = 1.0
    ema_weight: float = 0.0
    rsi_weight: float = 0.0
    rsi_filter_enabled: bool = True
    rsi_filter_timeframe: str = "1d"
    rsi_filter_period: int = 42
    rsi_filter_oversold: float = 35.0
    rsi_filter_overbought: float = 80.0
    rsi_filter_weight: float = 0.5
    trend_overlay_enabled: bool = False
    trend_overlay_agreement_exposure: float = 2.0
    trend_overlay_breakout_exposure: float = 3.0
    trend_overlay_max_long_exposure: float = 5.0
    trend_overlay_breakout_timeframe: str = "1d"
    trend_overlay_breakout_lookback: int = 50
    trend_overlay_ema_fast: int = 50
    trend_overlay_ema_slow: int = 200
    vol_flattening_overlay_enabled: bool = False
    vol_flattening_overlay_leverage: float = 1.05
    vol_flattening_vol_window: int = 168
    vol_flattening_percentile_lookback: int = 2160
    vol_flattening_high_threshold: float = 0.80
    vol_flattening_drop: float = 0.25
    vol_flattening_recent_high_window: int = 336
    vol_flattening_slope_window: int = 24
    vol_target_overlay_enabled: bool = False
    vol_target_annual_vol: float = 0.50
    vol_target_window: int = 336
    vol_target_long_cap: float = 2.0
    vol_target_short_cap: float = 1.0
    vol_target_confidence_enabled: bool = False
    vol_target_strong_trend_multiplier: float = 1.20
    vol_target_mixed_trend_multiplier: float = 1.00
    vol_target_weak_trend_multiplier: float = 0.70
    vol_target_strong_trend_threshold: float = 0.99
    vol_target_mixed_trend_threshold: float = 0.33
    vol_target_bull_floor_enabled: bool = False
    vol_target_bull_floor: float = 0.50
    vol_target_bull_floor_require_4h: bool = False
    no_trade_zone: float = 0.15


def generate_btc_signal(
    price_data: pd.DataFrame,
    config: PureSignalConfig | None = None,
    symbol: str = "BTC",
) -> pd.DataFrame:
    """Return per-bar pure BTC signal components and final target exposure."""
    config = config or PureSignalConfig()
    if symbol not in price_data.columns.get_level_values(0):
        raise ValueError(f"{symbol} price data is required")

    output = pd.DataFrame(index=price_data.index)
    output["btc_close"] = price_data[(symbol, "close")].astype(float)

    supertrend_votes: list[pd.Series] = []
    ema_votes: list[pd.Series] = []
    rsi_votes: list[pd.Series] = []

    for timeframe in config.timeframes:
        ohlcv = _closed_resampled_ohlcv(price_data, symbol, timeframe)
        supertrend = _supertrend_vote(ohlcv, config, timeframe)
        ema = _ema_vote(ohlcv, config)
        rsi = _rsi_modifier(ohlcv["close"], config)

        mapped_supertrend = _map_to_base(supertrend, price_data.index)
        mapped_ema = _map_to_base(ema, price_data.index)
        mapped_rsi = _map_to_base(rsi, price_data.index)

        output[f"supertrend_{timeframe}"] = mapped_supertrend
        output[f"ema_{timeframe}"] = mapped_ema
        output[f"rsi_{timeframe}"] = mapped_rsi

        supertrend_votes.append(mapped_supertrend)
        ema_votes.append(mapped_ema)
        rsi_votes.append(mapped_rsi)

    output["supertrend_score"] = _average_votes(supertrend_votes, price_data.index)
    output["ema_score"] = _average_votes(ema_votes, price_data.index)
    output["rsi_score"] = _average_votes(rsi_votes, price_data.index)
    output["raw_score"] = (
        config.supertrend_weight * output["supertrend_score"]
        + config.ema_weight * output["ema_score"]
        + config.rsi_weight * output["rsi_score"]
    )
    output["rsi_filter_signal"] = 0.0
    output["filtered_score"] = output["raw_score"].clip(-1.0, 1.0)
    if config.rsi_filter_enabled and config.rsi_filter_weight != 0.0:
        filter_ohlcv = _closed_resampled_ohlcv(
            price_data,
            symbol,
            config.rsi_filter_timeframe,
        )
        rsi_filter = _rsi_modifier_from_params(
            filter_ohlcv["close"],
            period=config.rsi_filter_period,
            oversold=config.rsi_filter_oversold,
            overbought=config.rsi_filter_overbought,
        )
        output["rsi_filter_signal"] = _map_to_base(rsi_filter, price_data.index)
        output["filtered_score"] = _apply_rsi_filter(
            output["filtered_score"],
            output["rsi_filter_signal"],
            weight=config.rsi_filter_weight,
        )
    output["signal"] = output["filtered_score"].clip(-1.0, 1.0)
    output.loc[output["signal"].abs() < config.no_trade_zone, "signal"] = 0.0
    output["target_exposure"] = output["signal"]
    output["trend_overlay_active"] = False
    output["trend_overlay_reason"] = ""
    if config.trend_overlay_enabled:
        output = _apply_trend_overlay(output, price_data, symbol, config)
    output["vol_flattening_overlay_active"] = False
    output["vol_percentile"] = 0.0
    if config.vol_flattening_overlay_enabled:
        output = _apply_vol_flattening_overlay(output, price_data, symbol, config)
    output["vol_target_overlay_active"] = False
    output["vol_target_realized_vol"] = 0.0
    output["vol_target_multiplier"] = 1.0
    output["vol_target_confidence"] = 0.0
    output["vol_target_confidence_multiplier"] = 1.0
    output["vol_target_effective_annual_vol"] = config.vol_target_annual_vol
    output["vol_target_bull_floor_active"] = False
    if config.vol_target_overlay_enabled:
        output = _apply_vol_target_overlay(output, price_data, symbol, config)
    return output


def _average_votes(votes: list[pd.Series], index: pd.Index) -> pd.Series:
    if not votes:
        return pd.Series(0.0, index=index)
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0)


def _pandas_timeframe(timeframe: str) -> str:
    if timeframe.lower().endswith("w"):
        return timeframe[:-1] + "W"
    return timeframe


def _closed_resampled_ohlcv(
    price_data: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    base = price_data[symbol]
    rule = _pandas_timeframe(timeframe)
    resampled = pd.DataFrame(
        {
            "open": base["open"].resample(rule).first(),
            "high": base["high"].resample(rule).max(),
            "low": base["low"].resample(rule).min(),
            "close": base["close"].resample(rule).last(),
            "volume": base["volume"].resample(rule).sum(),
        }
    ).dropna()
    return resampled.shift(1).dropna()


def _map_to_base(series: pd.Series, index: pd.Index) -> pd.Series:
    mapped = series.reindex(index, method="ffill")
    return mapped.fillna(0.0).astype(float)


def _supertrend_vote(
    ohlcv: pd.DataFrame,
    config: PureSignalConfig,
    timeframe: str,
) -> pd.Series:
    atr_period, multiplier = config.supertrend_params_by_timeframe.get(
        timeframe,
        (config.atr_period, config.supertrend_multiplier),
    )
    if len(ohlcv) < max(atr_period, 2):
        return pd.Series(0.0, index=ohlcv.index)
    direction = supertrend_direction(
        ohlcv,
        atr_period=atr_period,
        multiplier=multiplier,
    )
    return direction.map({True: 1.0, False: -1.0}).astype(float)


def _ema_vote(ohlcv: pd.DataFrame, config: PureSignalConfig) -> pd.Series:
    close = ohlcv["close"].astype(float)
    fast = close.ewm(span=config.ema_fast, adjust=False, min_periods=1).mean()
    slow = close.ewm(span=config.ema_slow, adjust=False, min_periods=1).mean()
    return pd.Series(0.0, index=ohlcv.index).where(fast < slow, 1.0).where(fast >= slow, -1.0)


def _rsi_modifier(close: pd.Series, config: PureSignalConfig) -> pd.Series:
    return _rsi_modifier_from_params(
        close,
        period=config.rsi_period,
        oversold=config.rsi_oversold,
        overbought=config.rsi_overbought,
    )


def _rsi_modifier_from_params(
    close: pd.Series,
    period: int,
    oversold: float,
    overbought: float,
) -> pd.Series:
    delta = close.astype(float).diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(period, min_periods=1).mean()
    avg_loss = losses.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0.0)
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(100.0)

    modifier = pd.Series(0.0, index=close.index)
    modifier.loc[rsi <= oversold] = 1.0
    modifier.loc[rsi >= overbought] = -1.0
    return modifier


def _apply_rsi_filter(
    base: pd.Series,
    rsi_filter: pd.Series,
    weight: float,
) -> pd.Series:
    filtered = base.copy()
    long_stretched = (base > 0.0) & (rsi_filter < 0.0)
    short_stretched = (base < 0.0) & (rsi_filter > 0.0)
    filtered.loc[long_stretched] = (base.loc[long_stretched] + weight * rsi_filter.loc[long_stretched]).clip(lower=0.0)
    filtered.loc[short_stretched] = (base.loc[short_stretched] + weight * rsi_filter.loc[short_stretched]).clip(upper=0.0)
    return filtered


def _apply_trend_overlay(
    output: pd.DataFrame,
    price_data: pd.DataFrame,
    symbol: str,
    config: PureSignalConfig,
) -> pd.DataFrame:
    target = output["target_exposure"].copy()
    weekly_green = output.get("supertrend_1w", 0.0).astype(float) > 0.0
    daily_green = output.get("supertrend_1d", 0.0).astype(float) > 0.0
    four_hour_green = output.get("supertrend_4h", 0.0).astype(float) > 0.0
    long_core = output["signal"] > 0.0

    agreement = long_core & weekly_green & daily_green
    target.loc[agreement] = target.loc[agreement].clip(
        lower=config.trend_overlay_agreement_exposure
    )

    breakout = _breakout_structure(price_data, symbol, config)
    breakout_agreement = agreement & four_hour_green & breakout
    target.loc[breakout_agreement] = target.loc[breakout_agreement].clip(
        lower=config.trend_overlay_breakout_exposure
    )
    target = target.clip(
        lower=-1.0,
        upper=config.trend_overlay_max_long_exposure,
    )

    output["target_exposure"] = target
    output.loc[agreement, "trend_overlay_active"] = True
    output.loc[agreement, "trend_overlay_reason"] = "agreement"
    output.loc[breakout_agreement, "trend_overlay_reason"] = "breakout"
    return output


def _breakout_structure(
    price_data: pd.DataFrame,
    symbol: str,
    config: PureSignalConfig,
) -> pd.Series:
    ohlcv = _closed_resampled_ohlcv(
        price_data,
        symbol,
        config.trend_overlay_breakout_timeframe,
    )
    close = ohlcv["close"].astype(float)
    fast = close.ewm(
        span=config.trend_overlay_ema_fast,
        adjust=False,
        min_periods=1,
    ).mean()
    slow = close.ewm(
        span=config.trend_overlay_ema_slow,
        adjust=False,
        min_periods=1,
    ).mean()
    prior_high = (
        ohlcv["high"]
        .astype(float)
        .rolling(config.trend_overlay_breakout_lookback, min_periods=1)
        .max()
        .shift(1)
    )
    breakout = (close > prior_high) | ((close > fast) & (fast > slow))
    return _map_to_base(breakout.astype(float), price_data.index).astype(bool)


def _apply_vol_flattening_overlay(
    output: pd.DataFrame,
    price_data: pd.DataFrame,
    symbol: str,
    config: PureSignalConfig,
) -> pd.DataFrame:
    setup, vol_pct = _volatility_flattening_setup(price_data, symbol, config)
    weekly_green = output.get("supertrend_1w", 0.0).astype(float) > 0.0
    daily_green = output.get("supertrend_1d", 0.0).astype(float) > 0.0
    long_core = output["signal"] > 0.0
    active = setup & weekly_green & daily_green & long_core

    target = output["target_exposure"].copy()
    target.loc[active] = target.loc[active].clip(
        lower=config.vol_flattening_overlay_leverage
    )
    output["target_exposure"] = target.clip(-10.0, 10.0)
    output["vol_flattening_overlay_active"] = active
    output["vol_percentile"] = vol_pct.fillna(0.0)
    return output


def _apply_vol_target_overlay(
    output: pd.DataFrame,
    price_data: pd.DataFrame,
    symbol: str,
    config: PureSignalConfig,
) -> pd.DataFrame:
    close = price_data[(symbol, "close")].astype(float)
    realized_vol = _annualized_realized_vol(close, config.vol_target_window)
    confidence = _directional_trend_confidence(output)
    confidence_multiplier = _trend_confidence_multiplier(confidence, config)
    effective_target_vol = config.vol_target_annual_vol * confidence_multiplier
    multiplier = (effective_target_vol / realized_vol.where(realized_vol > 0.0)).replace(
        [float("inf"), -float("inf")],
        pd.NA,
    )
    multiplier = multiplier.reindex(output.index).ffill().fillna(1.0).astype(float)

    target = output["target_exposure"].astype(float) * multiplier
    target = target.clip(
        lower=-config.vol_target_short_cap,
        upper=config.vol_target_long_cap,
    )
    bull_floor_active = _bull_floor_active(output, config)
    if config.vol_target_bull_floor_enabled:
        floor = min(config.vol_target_bull_floor, config.vol_target_long_cap)
        target.loc[bull_floor_active] = target.loc[bull_floor_active].clip(lower=floor)
        target = target.clip(
            lower=-config.vol_target_short_cap,
            upper=config.vol_target_long_cap,
        )

    output["target_exposure"] = target
    output["vol_target_overlay_active"] = output["signal"] != 0.0
    output["vol_target_realized_vol"] = realized_vol.reindex(output.index).fillna(0.0)
    output["vol_target_multiplier"] = multiplier
    output["vol_target_confidence"] = confidence
    output["vol_target_confidence_multiplier"] = confidence_multiplier
    output["vol_target_effective_annual_vol"] = effective_target_vol
    output["vol_target_bull_floor_active"] = bull_floor_active
    return output


def _bull_floor_active(
    output: pd.DataFrame,
    config: PureSignalConfig,
) -> pd.Series:
    if not config.vol_target_bull_floor_enabled:
        return pd.Series(False, index=output.index)
    active = (
        (output["signal"] > 0.0)
        & (output.get("supertrend_1w", 0.0).astype(float) > 0.0)
        & (output.get("supertrend_1d", 0.0).astype(float) > 0.0)
    )
    if config.vol_target_bull_floor_require_4h:
        active = active & (output.get("supertrend_4h", 0.0).astype(float) > 0.0)
    return active


def _directional_trend_confidence(output: pd.DataFrame) -> pd.Series:
    signal_direction = output["signal"].astype(float).map(_direction)
    return output["supertrend_score"].astype(float) * signal_direction


def _trend_confidence_multiplier(
    confidence: pd.Series,
    config: PureSignalConfig,
) -> pd.Series:
    multiplier = pd.Series(1.0, index=confidence.index)
    if not config.vol_target_confidence_enabled:
        return multiplier

    multiplier.loc[confidence >= config.vol_target_strong_trend_threshold] = (
        config.vol_target_strong_trend_multiplier
    )
    mixed = (
        (confidence >= config.vol_target_mixed_trend_threshold)
        & (confidence < config.vol_target_strong_trend_threshold)
    )
    multiplier.loc[mixed] = config.vol_target_mixed_trend_multiplier
    multiplier.loc[confidence < config.vol_target_mixed_trend_threshold] = (
        config.vol_target_weak_trend_multiplier
    )
    return multiplier


def _direction(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _annualized_realized_vol(close: pd.Series, window: int) -> pd.Series:
    returns = close.astype(float).pct_change()
    min_periods = max(8, window // 4)
    return returns.rolling(window, min_periods=min_periods).std() * (24 * 365) ** 0.5


def _volatility_flattening_setup(
    price_data: pd.DataFrame,
    symbol: str,
    config: PureSignalConfig,
) -> tuple[pd.Series, pd.Series]:
    close = price_data[(symbol, "close")].astype(float)
    returns = close.pct_change()
    realized_vol = returns.rolling(
        config.vol_flattening_vol_window,
        min_periods=max(8, config.vol_flattening_vol_window // 4),
    ).std()
    vol_percentile = realized_vol.rolling(
        config.vol_flattening_percentile_lookback,
        min_periods=max(48, config.vol_flattening_percentile_lookback // 4),
    ).rank(pct=True)
    recent_high = vol_percentile.rolling(
        config.vol_flattening_recent_high_window,
        min_periods=1,
    ).max()
    flattening = (
        (recent_high >= config.vol_flattening_high_threshold)
        & (vol_percentile <= recent_high - config.vol_flattening_drop)
        & (vol_percentile.diff(config.vol_flattening_slope_window) <= 0.0)
    )
    return flattening.fillna(False), vol_percentile
