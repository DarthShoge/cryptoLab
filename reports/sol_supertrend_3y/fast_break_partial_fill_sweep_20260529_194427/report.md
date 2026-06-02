# Fast-Break Partial-Fill Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can fast-break partial hedge filling fix the v2 failure mode where the signal activates but the requested ETH hedge jump is too large to execute atomically?

## Scenario Set

The sweep keeps the current stateful-profit-lock incumbent and the best v2 fast-break no-partial-fill candidate as controls.

- Base trigger: `-8%` 24h SOL return with `2.50x` realized-vol expansion.
- Requested hedge floor: `1.00`.
- Add HF for normal fast-break adds: `2.50`.
- Staged decay: `1.00 -> 0.75 -> 0.35`.
- Partial-fill min HF grid: `1.50`, `1.75`, `2.00`, `2.25`, `2.50`.
- Partial-fill episode budget grid: `10%`, `20%`, `35%`, `50%`, `75%` of SOL collateral value.
- Cross-checks: stricter `-10% / 2.50x` and wider `-8% / 2.00x` trigger families.

## Ranking By Sortino

| scenario | pfHF | budget | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | ETH debt | active% | pf events | pf max $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 28530.476 | 7.262 | 0 | 0.000 |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 27013.323 | 0.000 | 0 | 0.000 |
| pf_ret8_vol2.50_hf1.50_budget0.75 | 1.500 | 0.750 | 237.886 | 29723.820 | 82.369 | 68.122 | 2.176 | 1.383 | 23842.465 | 7.262 | 16 | 34169.929 |
| pf_ret8_vol2.50_hf1.50_budget0.10 | 1.500 | 0.100 | 240.177 | 30010.111 | 82.960 | 67.405 | 2.174 | 1.374 | 26522.734 | 7.262 | 114 | 9030.401 |
| pf_ret8_vol2.50_hf1.75_budget0.10 | 1.750 | 0.100 | 239.951 | 29981.859 | 82.962 | 67.439 | 2.173 | 1.374 | 26439.055 | 7.262 | 102 | 9030.401 |
| pf_ret8_vol2.50_hf2.50_budget0.35 | 2.500 | 0.350 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.50 | 2.500 | 0.500 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.75 | 2.500 | 0.750 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.20 | 2.500 | 0.200 | 238.081 | 29748.238 | 82.703 | 70.023 | 2.164 | 1.376 | 22661.591 | 7.262 | 70 | 8153.282 |
| pf_ret8_vol2.50_hf2.00_budget0.10 | 2.000 | 0.100 | 222.987 | 27862.257 | 82.950 | 70.064 | 2.150 | 1.347 | 20871.691 | 7.262 | 76 | 9030.401 |
| pf_ret8_vol2.50_hf2.25_budget0.10 | 2.250 | 0.100 | 220.196 | 27513.512 | 82.878 | 70.737 | 2.143 | 1.343 | 21495.546 | 7.262 | 71 | 9081.710 |
| pf_ret8%_vol2.00_hf2.00_budget0.35 | 2.000 | 0.350 | 196.277 | 24524.870 | 84.545 | 73.566 | 2.131 | 1.317 | 21221.260 | 10.869 | 75 | 13774.189 |

## Ranking By Max Drawdown

| scenario | pfHF | budget | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | ETH debt | active% | pf events | pf max $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 28530.476 | 7.262 | 0 | 0.000 |
| pf_ret8_vol2.50_hf1.50_budget0.50 | 1.500 | 0.500 | 144.909 | 18106.431 | 81.607 | 80.464 | 2.016 | 1.147 | 26492.157 | 7.262 | 50 | 48094.731 |
| pf_ret8_vol2.50_hf2.50_budget0.35 | 2.500 | 0.350 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.50 | 2.500 | 0.500 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.75 | 2.500 | 0.750 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf1.50_budget0.75 | 1.500 | 0.750 | 237.886 | 29723.820 | 82.369 | 68.122 | 2.176 | 1.383 | 23842.465 | 7.262 | 16 | 34169.929 |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 27013.323 | 0.000 | 0 | 0.000 |
| pf_ret8_vol2.50_hf2.50_budget0.10 | 2.500 | 0.100 | 209.882 | 26224.795 | 82.673 | 71.974 | 2.128 | 1.313 | 21564.858 | 7.262 | 74 | 7141.094 |
| pf_ret8_vol2.50_hf2.50_budget0.20 | 2.500 | 0.200 | 238.081 | 29748.238 | 82.703 | 70.023 | 2.164 | 1.376 | 22661.591 | 7.262 | 70 | 8153.282 |
| pf_ret8_vol2.50_hf2.00_budget0.35 | 2.000 | 0.350 | 181.894 | 22727.661 | 82.835 | 74.914 | 2.090 | 1.288 | 20200.835 | 7.262 | 70 | 12567.259 |
| pf_ret8_vol2.50_hf2.25_budget0.10 | 2.250 | 0.100 | 220.196 | 27513.512 | 82.878 | 70.737 | 2.143 | 1.343 | 21495.546 | 7.262 | 71 | 9081.710 |
| pf_ret8_vol2.50_hf2.00_budget0.10 | 2.000 | 0.100 | 222.987 | 27862.257 | 82.950 | 70.064 | 2.150 | 1.347 | 20871.691 | 7.262 | 76 | 9030.401 |

## Ranking By 2024+ Peak-To-Trough

| scenario | pfHF | budget | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | ETH debt | active% | pf events | pf max $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pf_ret10%_vol2.50_hf1.75_budget0.50 | 1.750 | 0.500 | 175.637 | 21945.885 | 87.244 | 64.805 | 2.068 | 1.366 | 19457.845 | 5.088 | 88 | 18297.992 |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 28530.476 | 7.262 | 0 | 0.000 |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 27013.323 | 0.000 | 0 | 0.000 |
| pf_ret10%_vol2.50_hf1.75_budget0.35 | 1.750 | 0.350 | 174.101 | 21753.888 | 87.195 | 65.042 | 2.069 | 1.368 | 19335.080 | 5.088 | 79 | 12757.787 |
| pf_ret10%_vol2.50_hf1.75_budget0.20 | 1.750 | 0.200 | 170.381 | 21289.156 | 86.919 | 67.257 | 2.058 | 1.360 | 19127.686 | 5.088 | 63 | 12543.622 |
| pf_ret8_vol2.50_hf1.50_budget0.10 | 1.500 | 0.100 | 240.177 | 30010.111 | 82.960 | 67.405 | 2.174 | 1.374 | 26522.734 | 7.262 | 114 | 9030.401 |
| pf_ret8_vol2.50_hf1.75_budget0.10 | 1.750 | 0.100 | 239.951 | 29981.859 | 82.962 | 67.439 | 2.173 | 1.374 | 26439.055 | 7.262 | 102 | 9030.401 |
| pf_ret8_vol2.50_hf1.50_budget0.75 | 1.500 | 0.750 | 237.886 | 29723.820 | 82.369 | 68.122 | 2.176 | 1.383 | 23842.465 | 7.262 | 16 | 34169.929 |
| pf_ret8%_vol2.00_hf1.75_budget0.20 | 1.750 | 0.200 | 136.032 | 16997.225 | 89.925 | 69.287 | 2.030 | 1.333 | 16240.507 | 10.869 | 133 | 10285.372 |
| pf_ret8_vol2.50_hf2.50_budget0.35 | 2.500 | 0.350 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.50 | 2.500 | 0.500 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.75 | 2.500 | 0.750 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |

## Ranking By Final SOL-Equivalent

| scenario | pfHF | budget | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | ETH debt | active% | pf events | pf max $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 28530.476 | 7.262 | 0 | 0.000 |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 27013.323 | 0.000 | 0 | 0.000 |
| pf_ret8_vol2.50_hf1.50_budget0.10 | 1.500 | 0.100 | 240.177 | 30010.111 | 82.960 | 67.405 | 2.174 | 1.374 | 26522.734 | 7.262 | 114 | 9030.401 |
| pf_ret8_vol2.50_hf1.75_budget0.10 | 1.750 | 0.100 | 239.951 | 29981.859 | 82.962 | 67.439 | 2.173 | 1.374 | 26439.055 | 7.262 | 102 | 9030.401 |
| pf_ret8_vol2.50_hf2.50_budget0.20 | 2.500 | 0.200 | 238.081 | 29748.238 | 82.703 | 70.023 | 2.164 | 1.376 | 22661.591 | 7.262 | 70 | 8153.282 |
| pf_ret8_vol2.50_hf1.50_budget0.75 | 1.500 | 0.750 | 237.886 | 29723.820 | 82.369 | 68.122 | 2.176 | 1.383 | 23842.465 | 7.262 | 16 | 34169.929 |
| pf_ret8_vol2.50_hf2.50_budget0.35 | 2.500 | 0.350 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.50 | 2.500 | 0.500 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.75 | 2.500 | 0.750 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 71 | 8137.138 |
| pf_ret8_vol2.50_hf2.00_budget0.10 | 2.000 | 0.100 | 222.987 | 27862.257 | 82.950 | 70.064 | 2.150 | 1.347 | 20871.691 | 7.262 | 76 | 9030.401 |
| pf_ret8_vol2.50_hf2.25_budget0.10 | 2.250 | 0.100 | 220.196 | 27513.512 | 82.878 | 70.737 | 2.143 | 1.343 | 21495.546 | 7.262 | 71 | 9081.710 |
| pf_ret8_vol2.50_hf2.50_budget0.10 | 2.500 | 0.100 | 209.882 | 26224.795 | 82.673 | 71.974 | 2.128 | 1.313 | 21564.858 | 7.262 | 74 | 7141.094 |

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | Partial-fill events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent | incumbent_stateful_best | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0 |
| V2 no partial fill | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |
| Best Sortino | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |
| Best Max DD | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |
| Best 2024 DD | pf_ret10%_vol2.50_hf1.75_budget0.50 | 175.637 | 21945.885 | 87.244 | 64.805 | 2.068 | 1.366 | 88 |
| Best SOL-eq | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |

## Forensic Notes

Decision: no promotion.

Fast-break partial fill, as implemented in this sweep, is a clear negative result. The best no-partial-fill v2 control remains the top row by Sortino, max drawdown, and final SOL-equivalent. It ends at `424.90 SOL`, `81.34%` max drawdown, `2.290` Sortino, and `1.598` min HF. Every partial-fill row gives up a large amount of final SOL-equivalent and Sortino.

The headline failure is not subtle. The best partial-fill row by Sortino, `pf_ret8_vol2.50_hf1.50_budget0.75`, finishes at only `237.89 SOL` versus `424.90 SOL` for v2 no partial fill. The partial-fill group average is only `189.51 SOL`. This is not an acceptable trade-off for a drawdown system whose purpose is to protect long-term SOL compounding.

Partial fill also fails the original red-mark objective. The v2 no-partial-fill control has a 2024+ peak-to-trough drawdown of `64.84%`. The best partial-fill row by 2024+ peak-to-trough, `pf_ret10%_vol2.50_hf1.75_budget0.50`, improves that number by only `0.03` percentage points to `64.81%`, while final SOL-equivalent collapses to `175.64 SOL` and max drawdown worsens to `87.24%`. That is not a real solution.

The mechanism did fire, but it fired in the wrong places. For example, `pf_ret8_vol2.50_hf2.50_budget0.35` records `71` fast-break partial-fill events. Several occur in April and May 2021 while the vote still has `3` green timeframes. Those additions increase ETH short exposure during a still-constructive regime, which then hurts the portfolio during the later SOL rally. The report confirms our fear: a fast signal plus automatic partial execution can become an over-trading machine.

The 2025 diagnostics are more nuanced. In `pf_ret8_vol2.50_hf2.50_budget0.35`, partial fill does add roughly `$4.1k` during the January/February 2025 fast-break episode, and the 2025 trough value is higher in absolute dollars than v2 no partial fill. But the peak is also much lower because prior partial fills had already damaged the compounding path. The measured 2024+ peak-to-trough gets worse: `69.99%` versus `64.84%`.

Looser partial-fill HF settings are unsafe. The `hf1.50` and `hf1.75` rows sometimes have superficially competitive max drawdown, but their min HF drops into the `1.1x` to `1.3x` area and final SOL-equivalent suffers badly. Those rows violate the spirit of the health-factor guardrail.

The higher-HF rows are safer but still poor. `hf2.50` limits the maximum partial-fill amount to roughly `$8k` in the main grid, but it still creates dozens of partial-fill events and ends around `238 SOL`, far below both controls. That means the issue is not only too much size; it is also too little selectivity.

The lesson is sharp: partial fill should not be attached directly to the generic fast-break trigger. The trigger is good enough to raise a target floor, but not good enough to authorize incremental borrowing on its own. We need a second gate for execution.

Next hypothesis: keep v2 no-partial-fill as the current best and test a guarded partial-fill variant only when fast-break overlaps with a stronger damage state. Candidate gates:

1. fast-break active and crisis mode active,
2. fast-break active and `under_hedged_crisis = true`,
3. fast-break active and green votes are at most `1`,
4. fast-break active and 3d or 1w trend is bearish.

The simplest next test is probably `fast_break_partial_fill_requires_crisis = true`, with no fills while the vote still has 2-3 green timeframes. That directly targets the 2025 execution bottleneck without letting the mechanism short every early bull-market shakeout.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/fast_break_partial_fill_sweep_20260529_194427/comparison.csv`
- Top-by-drawdown CSV: `reports/sol_supertrend_3y/fast_break_partial_fill_sweep_20260529_194427/top_by_drawdown.csv`
- Top-by-2024-drawdown CSV: `reports/sol_supertrend_3y/fast_break_partial_fill_sweep_20260529_194427/top_by_2024_drawdown.csv`
- Top-by-SOL-equivalent CSV: `reports/sol_supertrend_3y/fast_break_partial_fill_sweep_20260529_194427/top_by_sol_equiv.csv`
- Summary JSON: `reports/sol_supertrend_3y/fast_break_partial_fill_sweep_20260529_194427/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/fast_break_partial_fill_sweep_20260529_194427`
