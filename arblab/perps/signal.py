"""Pure BTC perpetual futures signal generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from arblab.strategies.sol_supertrend_short import supertrend_direction


@dataclass(frozen=True)
class PureSignalConfig:
    """Configuration for a stateless BTC exposure signal."""

    timeframes: tuple[str, ...] = ("1h", "4h", "1d", "1w")
    atr_period: int = 10
    supertrend_multiplier: float = 3.0
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    supertrend_weight: float = 0.50
    ema_weight: float = 0.35
    rsi_weight: float = 0.15
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
        supertrend = _supertrend_vote(ohlcv, config)
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
    output["signal"] = output["raw_score"].clip(-1.0, 1.0)
    output.loc[output["signal"].abs() < config.no_trade_zone, "signal"] = 0.0
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


def _supertrend_vote(ohlcv: pd.DataFrame, config: PureSignalConfig) -> pd.Series:
    if len(ohlcv) < max(config.atr_period, 2):
        return pd.Series(0.0, index=ohlcv.index)
    direction = supertrend_direction(
        ohlcv,
        atr_period=config.atr_period,
        multiplier=config.supertrend_multiplier,
    )
    return direction.map({True: 1.0, False: -1.0}).astype(float)


def _ema_vote(ohlcv: pd.DataFrame, config: PureSignalConfig) -> pd.Series:
    close = ohlcv["close"].astype(float)
    fast = close.ewm(span=config.ema_fast, adjust=False, min_periods=1).mean()
    slow = close.ewm(span=config.ema_slow, adjust=False, min_periods=1).mean()
    return pd.Series(0.0, index=ohlcv.index).where(fast < slow, 1.0).where(fast >= slow, -1.0)


def _rsi_modifier(close: pd.Series, config: PureSignalConfig) -> pd.Series:
    delta = close.astype(float).diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(config.rsi_period, min_periods=1).mean()
    avg_loss = losses.rolling(config.rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0.0)
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(100.0)

    modifier = pd.Series(0.0, index=close.index)
    modifier.loc[rsi <= config.rsi_oversold] = 1.0
    modifier.loc[rsi >= config.rsi_overbought] = -1.0
    return modifier
