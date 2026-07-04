# BTC Pure Perp Signal Strategy Design

## Purpose

Create a BTC-only pure signal strategy that can be deployed on any venue with perpetual futures access, starting with Binance assumptions for backtesting. The strategy signal itself must not depend on account state or prior positions. It should emit a target directional exposure from `-1.0` to `+1.0` on each bar:

- `+1.0`: maximum long BTC perp exposure
- `0.0`: flat
- `-1.0`: maximum short BTC perp exposure

The backtest account is necessarily path-dependent because PnL, fees, funding, turnover, and margin evolve through time. The design must keep that account simulation separate from signal generation.

## Scope

Version 1 is BTC-only. Multi-asset portfolio weights and cross-sectional long/short ranking are future work after the single-asset signal is strong and inspectable.

## Existing Project Fit

The current project has reusable OHLCV fetching and closed-candle traffic-light helpers, but the main `BacktestEngine` models Kamino collateral, debt, health factor, and liquidation mechanics. A perp strategy should not be forced through those account abstractions. Version 1 should add a small perp-specific research path while reusing data loading and indicator conventions where practical.

## Signal Semantics

The signal is a deterministic function of market history up to the latest closed candle. It must not read current equity, current position, prior target exposure, realized PnL, drawdown, or fees.

Each bar computes component scores from four timeframes:

- Supertrend direction: bullish `+1`, bearish `-1`
- EMA trend: bullish `+1`, bearish `-1`
- RSI modifier: overbought readings reduce long bias or add short bias; oversold readings reduce short bias or add long bias

The default weighted blend is:

```text
raw_score =
  traffic_light_weight * avg(supertrend_votes)
  + ema_weight * avg(ema_votes)
  + rsi_weight * avg(rsi_modifiers)
```

Then:

```text
signal = clamp(raw_score, -1.0, 1.0)
```

Apply a no-trade zone after clamping:

```text
if abs(signal) < no_trade_zone:
    signal = 0.0
```

Initial defaults:

- Base execution timeframe: `1h`
- Signal timeframes: `1h`, `4h`, `1d`, `1w`
- Supertrend ATR period: `10`
- Supertrend multiplier: `3.0`
- EMA fast period: `50`
- EMA slow period: `200`
- RSI period: `14`
- RSI overbought threshold: `70`
- RSI oversold threshold: `30`
- Weights: Supertrend `0.50`, EMA `0.35`, RSI `0.15`
- No-trade zone: `0.15`

## Closed-Candle Rule

All higher-timeframe signals must be computed from completed candles only, then forward-filled onto the base timeframe. The strategy must not use an in-progress `4h`, `1d`, or `1w` candle while evaluating a `1h` decision bar.

## Perp Backtest Mechanics

The perp simulator starts from USD equity and tracks a single BTCUSDT perp position.

Configurable inputs:

- Starting equity, default `$10,000`
- Maximum gross exposure, default `1.0x`
- Rebalance deadband, default `0.10`
- Fee rate, default to a conservative Binance taker fee unless configured
- Funding model, initially configurable as zero funding or a simple fixed rate placeholder

Per bar:

1. Compute the closed-candle target signal `s` in `[-1, 1]`.
2. Convert signal to target notional: `target_notional = equity * max_gross_exposure * s`.
3. If the absolute target exposure change is below the rebalance deadband, keep the current position.
4. Trade the delta between current notional and target notional.
5. Apply fees on traded notional.
6. Mark the position to market using close-to-close BTC returns.
7. Apply funding when a funding model is enabled.
8. Record account state, signal components, turnover, costs, and benchmark values.

Default execution should avoid smoothing target exposure because smoothing makes execution path-dependent. It may be added later as an explicitly named execution option, not as part of the pure signal.

## Outputs

Each run should write artifacts to a new timestamped report directory and should not overwrite prior reports.

Required artifacts:

- `signals.csv`: timestamp, BTC close, final signal, component scores, and timeframe votes
- `history.csv`: equity, position notional, exposure, traded notional, fees, funding, drawdown, and BTC buy-and-hold benchmark
- `summary.json`: total return, annualized return, max drawdown, Sharpe, Sortino, turnover, fee drag, funding drag, percent time long, percent time short, percent time flat
- `report.md`: plain-English report comparing strategy performance against BTC buy-and-hold and calling out suspicious behavior

## Testing Requirements

Tests should cover:

- Signal output is always within `[-1, 1]`
- No-trade zone maps small signals to zero
- Higher-timeframe indicators use closed candles only
- Rebalance deadband suppresses tiny target changes
- Fee and PnL accounting reconcile on simple deterministic price paths
- BTC buy-and-hold benchmark uses the same start and end window as the strategy
- Generated outputs include the signal components needed to explain any bar

## Extension Points

Future work after v1:

- Real Binance funding-rate ingestion
- Signal parameter sweeps and walk-forward evaluation
- Multi-asset per-asset signals
- Cross-sectional strongest/weakest long-short portfolios
- UI integration once the artifact schema stabilizes
