# Hedge-Failure Circuit Breaker Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Does pausing re-risking, and optionally selling a small SOL slice, help when SOL materially underperforms ETH while a defensive overlay is active?

## Ranking

| scenario | sell | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | events | active% | maxUSDC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_breaker |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 | 0.000 | 0.000 |
| hedge_failure_pause_only | 0.000 | 423.812 | 52955.345 | 81.318 | 64.827 | 2.289 | 1.601 | 0 | 21.157 | 0.000 |
| hedge_failure_sell5pct | 0.050 | 412.754 | 51573.650 | 81.162 | 64.722 | 2.284 | 1.635 | 1 | 21.157 | 201.010 |

## Readout

This is the first candidate with a constructive trade-off, though it is not radical enough by itself.

The pause-only circuit breaker is almost neutral: final SOL-equivalent slips from `424.902 SOL` to `423.812 SOL`, while max drawdown improves slightly from `81.344%` to `81.318%` and 2024+ peak-to-trough improves from `64.839%` to `64.827%`. The `5%` sell variant is more useful defensively: max drawdown improves to `81.162%`, 2024+ peak-to-trough improves to `64.722%`, and min health factor improves to `1.635`. The cost is real but not catastrophic: final SOL-equivalent falls to `412.754 SOL`.

The circuit breaker is active about `21.157%` of the backtest, but the sell variant only executes one small SOL sale, with max protected USDC of about `$201`. This suggests the relative SOL/ETH failure signal is directionally right but too timid in current sizing. It improves the book without the severe opportunity-cost damage seen in reserve-heavy and CPPI variants.

Conclusion: keep this candidate as a useful overlay, but it needs stronger sizing or pairing with a smarter re-entry rule before it can radically improve drawdown.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/hedge_failure_circuit_breaker_sweep_20260602_115413/comparison.csv`
- Ranked CSV: `reports/sol_supertrend_3y/hedge_failure_circuit_breaker_sweep_20260602_115413/ranked.csv`
- Summary JSON: `reports/sol_supertrend_3y/hedge_failure_circuit_breaker_sweep_20260602_115413/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/hedge_failure_circuit_breaker_sweep_20260602_115413`
