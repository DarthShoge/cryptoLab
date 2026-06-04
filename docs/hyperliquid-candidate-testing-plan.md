# Hyperliquid Candidate Testing Plan

Created: 2026-06-04
Branch: `research/hyperliquid-candidate-testing`

## Goal

Use Hyperliquid as the research and eventual execution venue for the candidate crypto strategies identified from the research review:

1. Cost-aware XGBoost BTC hourly.
2. Liquid-winners cross-sectional momentum.
3. Cross-sectional 3-week momentum.
4. BTC trend-following benchmark.
5. Cointegration / pairs trading.

The first milestone is not live trading. It is a clean, reproducible Hyperliquid data pipeline that can support realistic backtests across a broad asset universe.

## Venue Facts To Model

- Hyperliquid supports public market-data access through the `/info` endpoint at `https://api.hyperliquid.xyz/info`.
- The official docs list `meta`, `allMids`, `candleSnapshot`, user portfolio queries, and account-state queries under the Info API.
- `candleSnapshot` returns OHLCV-style candles and supports intervals from `1m` through `1M`, but only the most recent 5000 candles are available per request.
- Time-range responses require pagination by continuing from the last returned timestamp.
- Hyperliquid has an official Python SDK and CCXT also maintains a Hyperliquid integration.
- Cross margin is the default margin mode for perp positions.
- Portfolio margin exists, but current docs describe it as alpha/pre-alpha style infrastructure with caps, account requirements, and fallback behavior when caps are hit.
- Portfolio margin unifies spot and perps for eligible assets, but subaccounts are still treated separately.

## Portfolio Margin Implication

Portfolio margin should be treated as an execution/risk-management feature, not a research dependency for the first backtests.

Initial backtests should model:

1. Perp-only USDC-margin trading.
2. Cross-margin account-level notional and maintenance margin.
3. Funding payments.
4. Fees, spread, and slippage.
5. Strict max gross exposure and per-asset exposure caps.

Portfolio margin can be added after the basic exchange adapter is validated, especially for carry/funding or spot-perp basis strategies. For directional momentum and ML prediction strategies, a conservative cross-margin model is the right first abstraction.

## Data Pipeline Milestones

### M1: Static Market Metadata

Collect and cache:

- Perp universe from `meta`.
- Asset names, universe index, max leverage, and margin-related metadata.
- Current mids from `allMids`.
- Optional spot universe from `spotMeta` later.

Output target:

- `data/hyperliquid/metadata/perp_universe.json`
- Normalized in-memory schema for downstream strategies.

### M2: OHLCV Backfill

Collect candles using `candleSnapshot`:

- Start with intervals: `1h`, `4h`, `1d`.
- Start with assets: BTC, ETH, SOL, HYPE, plus top liquid perps by metadata/volume once available.
- Store each asset/interval as CSV or parquet.
- Respect the 5000-candle response cap by chunking requests.

Output target:

- `data/hyperliquid/ohlcv/{interval}/{coin}.csv`

Canonical columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `num_trades`
- `coin`
- `interval`
- `source`

### M3: Data Quality Checks

Validate:

- Duplicate timestamps.
- Missing bars.
- Negative or zero prices.
- Large gaps.
- Assets with insufficient history.
- Delisted or inactive assets.

Output target:

- `reports/hyperliquid/data_quality.md`

### M4: Backtest Adapter

Add a Hyperliquid data backend that can feed existing backtest workflows without hard-coding a single exchange.

Design preference:

- Keep the current `ccxt` path intact.
- Add a separate Hyperliquid loader/fetcher behind the same normalized OHLCV interface.
- Avoid live order code until research backtests are passing.

### M5: Candidate Strategy Harness

Implement in this order:

1. BTC trend-following benchmark.
2. Cross-sectional momentum on top liquid perps.
3. Cost-aware XGBoost BTC hourly.
4. Pairs trading / cointegration.

## First Implementation Slice

Build a read-only Hyperliquid client:

- `arblab/backtest/hyperliquid_data.py`

Functions:

- `fetch_perp_universe() -> list[dict]`
- `fetch_all_mids() -> dict[str, float]`
- `fetch_candles(coin, interval, start_ms, end_ms) -> pandas.DataFrame`
- `backfill_candles(coin, interval, start_ms, end_ms, out_dir) -> pandas.DataFrame`

Tests:

- Mock `/info` responses.
- Validate normalization of candle rows.
- Validate pagination boundaries.
- Validate empty or partial responses.

## Open Questions

- Do we want raw storage as CSV first, or parquet from day one?
- Should the first cross-sectional universe use all active perps or only a liquidity-screened top-N subset?
- Should the research harness model funding immediately, or after OHLCV reliability is proven?
- For live/paper trading later, do we use the official Hyperliquid Python SDK directly or CCXT for a unified exchange interface?
