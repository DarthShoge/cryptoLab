# Multi-Asset Traffic-Light Research Design

## Goal

Test whether the four-traffic-light analogy can improve drawdown by moving beyond a single SOL collateral/ETH hedge model into a curated, data-backed Kamino universe where the strategy can choose long and short legs dynamically.

## Scope

The first version is a research experiment, not a production allocator. It uses the committed curated universe only:

- Directional assets: `SOL`, `JitoSOL`, `mSOL`, `ETH`
- Stable legs: `USDC`, `USDT`

No live Kamino market discovery is included in this pass. Expansion beyond these assets requires current Kamino availability checks and a durable exchange or derived price source.

## Architecture

1. Add a reusable traffic-light signal module that can score any OHLCV asset using the same multi-timeframe Supertrend vote pattern already used by the SOL strategy.
2. Add a lightweight pair-selection research strategy that:
   - starts from USDC collateral,
   - goes long the strongest directional asset when enough green lights appear,
   - borrows the weakest directional asset as the short leg when enough red lights appear,
   - de-risks back toward USDC when signals are not strong enough,
   - keeps target debt conservative enough to evaluate drawdown before leverage maximization.
3. Add a script that runs a small parameter sweep and writes artifacts under a timestamped report directory.

## Measurement

Every candidate must report:

- final portfolio value in USD,
- final SOL equivalent,
- max drawdown,
- post-2024 drawdown,
- liquidation status and liquidation count,
- final collateral/debt exposure,
- selected long/short symbols over time.

The current SOL checkpoint remains the benchmark:

- best no-HF DD checkpoint: `496.353 SOL`, `78.991%` max DD,
- best no-HF Pareto checkpoint: `506.680 SOL`, `79.349%` max DD.

## First Hypothesis

If sharp SOL path dependence is the failure mode, then a cross-asset selector should improve drawdown by shifting collateral toward stronger assets and shorting weaker assets during adverse regimes. The first experiment should prioritize honest diagnostics over parameter breadth.
