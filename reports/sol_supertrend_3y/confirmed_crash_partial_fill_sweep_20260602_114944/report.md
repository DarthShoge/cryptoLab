# Confirmed-Crash Partial Fill Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Does requiring a confirmed crash state (`crisis=true` and `green <= 1`) rescue fast-break partial fills by preventing early-cycle over-hedging?

## Ranking

| scenario | crisis | max_green | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | events | maxPF$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 | 0.000 |
| crisis_gated_partial_fill | True |  | 233.351 | 29157.245 | 82.727 | 70.004 | 2.160 | 1.376 | 56 | 7990.306 |
| confirmed_crash_partial_fill | True | 1 | 239.423 | 29915.912 | 81.754 | 70.403 | 2.157 | 1.377 | 35 | 13867.129 |

## Readout

Confirmed-crash gating improves the older crisis-gated partial-fill variant, but not enough to justify the mechanism.

Adding `fast_break_partial_fill_max_green = 1` reduces partial-fill events from `56` to `35` and improves max drawdown relative to the old crisis-gated partial fill (`82.727%` to `81.754%`). It still loses to the v2 no-partial-fill control on every important portfolio metric: final SOL-equivalent falls from `424.902 SOL` to `239.423 SOL`, final USD falls from `$53,091.484` to `$29,915.912`, Sortino drops from `2.290` to `2.157`, and 2024+ peak-to-trough worsens from `64.839%` to `70.403%`.

Conclusion: the strategy still should not add ETH debt as a partial-fill workaround. Even when fills are delayed until confirmed crash states, they arrive when the book is already stressed and they impair recovery.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/confirmed_crash_partial_fill_sweep_20260602_114944/comparison.csv`
- Ranked CSV: `reports/sol_supertrend_3y/confirmed_crash_partial_fill_sweep_20260602_114944/ranked.csv`
- Summary JSON: `reports/sol_supertrend_3y/confirmed_crash_partial_fill_sweep_20260602_114944/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/confirmed_crash_partial_fill_sweep_20260602_114944`
