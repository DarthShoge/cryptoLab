# Candidate Test Round 1

Created: 2026-06-04

## Data

Hyperliquid `1h` candles from `candleSnapshot`.

| Universe | Source folder | Markets | Notes |
|---|---|---:|---|
| Core crypto top 20 | `data/hyperliquid/core_180d` | 20 | Selected by average dollar volume from first 40 non-delisted core perps. Some fetches hit HTTP 429 and were skipped. |
| Stock perps deduped | `data/hyperliquid/stocks_180d` | 11 | Selected by average dollar volume, deduped by underlying ticker across builders. |

Fee assumption: 5 bps per unit turnover.

## Best Crypto Results

| Candidate | Total return | Annualized return | Sharpe | Sortino | Max DD | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `xsec_mom_72h_top1_reb24h` | 22.25% | 50.29% | 0.93 | 1.32 | 57.23% | 92 |
| `xsec_mom_336h_top3_reb168h` | -7.01% | -13.70% | 0.13 | 0.19 | 46.32% | 23 |
| `btc_sma_50_200` | -3.91% | -7.77% | -0.18 | -0.16 | 17.44% | 26 |

Crypto read: only the 72-hour cross-sectional winner had a positive first pass. Drawdown is too high to call it attractive yet.

## Best Stock-Perp Results

| Candidate | Total return | Annualized return | Sharpe | Sortino | Max DD | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `xsec_mom_72h_top1_reb24h` | 134.35% | 1027.08% | 3.35 | 4.36 | 21.87% | 66 |
| `xsec_mom_336h_top3_reb168h` | 50.65% | 220.76% | 2.54 | 3.10 | 19.74% | 16 |
| `xsec_mom_24h_top1_reb24h` | 40.26% | 161.75% | 1.62 | 2.00 | 25.20% | 111 |
| `xsec_mom_336h_top5_reb336h` | 31.31% | 117.01% | 2.18 | 2.66 | 17.43% | 9 |

Stock-perp read: the result is interesting enough to pursue, especially 72-hour top-1 and 336-hour top-3 momentum. Treat the annualized figures as unstable because the available HIP-3 sample is short.

## Current Ranking

1. Stock-perp cross-sectional 72-hour momentum, top 1, daily rebalance.
2. Stock-perp cross-sectional 336-hour momentum, top 3, weekly rebalance.
3. Core crypto cross-sectional 72-hour momentum, top 1, daily rebalance.
4. BTC 50/200 hourly SMA as a benchmark only.

## Next Tests

1. Add walk-forward or rolling-subperiod evaluation.
2. Add volatility targeting and max drawdown controls.
3. Add top-2/top-3 daily rebalance variants to reduce single-name concentration.
4. Fetch more stock markets with slower rate limiting and retry logic.
5. Add funding/borrow-style costs once available from the venue data.

