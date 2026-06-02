# TIPP Profit-Lock Reserve Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can a stateful, high-watermark-gated SOL reserve reduce late-cycle drawdown without the reserve churn that damaged the earlier profit-lock reserve experiment?

## Scenario Set

The sweep compares the current v2 fast-break no-reserve control against episode-bounded profit-lock reserve variants.

- Entry: profit lock active, portfolio up at least `100%` from initial, and within `10%` of the trailing high.
- Episode rule: one initial slice and at most one escalation slice per completed high-watermark episode.
- Re-entry rule: after a full reserve rebuy, a new episode requires portfolio value above the completed episode high by the reset gap.
- Rebuy rule: 1w/1d/3d recovery plus a cooldown from the last reserve sale.
- Minimum SOL collateral: `100 SOL`.
- First sell fraction grid: `5%`, `10%`.
- Escalation sell fraction grid: `0%`, `5%`, `10%`.
- Max reserve fraction grid: `15%`, `25%`.
- Rebuy fraction grid: `25%`, `50%`.
- Cooldown: `168` bars.
- Reset gap: `2%`.

## Ranking By Sortino

| scenario | sell | esc | max | rebuy | cool | reset | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | sell/esc/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0.000 | 0/0/0 |
| incumbent_stateful_best |  |  |  |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0.000 | 0/0/0 |
| tipp_sell0.05_esc0.00_max0.15_rebuy0.50_cool168_reset0.02 | 0.050 | 0.000 | 0.150 | 0.500 | 168 | 0.020 | 305.424 | 38162.703 | 84.155 | 64.053 | 2.206 | 1.702 | 12.817 | 7/0/284 |
| tipp_sell0.05_esc0.00_max0.25_rebuy0.50_cool168_reset0.02 | 0.050 | 0.000 | 0.250 | 0.500 | 168 | 0.020 | 305.424 | 38162.703 | 84.155 | 64.053 | 2.206 | 1.702 | 12.817 | 7/0/284 |
| tipp_sell0.05_esc0.00_max0.15_rebuy0.25_cool168_reset0.02 | 0.050 | 0.000 | 0.150 | 0.250 | 168 | 0.020 | 287.911 | 35974.496 | 84.449 | 63.858 | 2.192 | 1.702 | 35.712 | 7/0/649 |
| tipp_sell0.05_esc0.00_max0.25_rebuy0.25_cool168_reset0.02 | 0.050 | 0.000 | 0.250 | 0.250 | 168 | 0.020 | 287.911 | 35974.496 | 84.449 | 63.858 | 2.192 | 1.702 | 35.712 | 7/0/649 |
| tipp_sell0.10_esc0.00_max0.15_rebuy0.50_cool168_reset0.02 | 0.100 | 0.000 | 0.150 | 0.500 | 168 | 0.020 | 249.049 | 31118.634 | 83.952 | 60.721 | 2.169 | 1.702 | 12.723 | 7/0/291 |
| tipp_sell0.10_esc0.00_max0.25_rebuy0.50_cool168_reset0.02 | 0.100 | 0.000 | 0.250 | 0.500 | 168 | 0.020 | 249.049 | 31118.634 | 83.952 | 60.721 | 2.169 | 1.702 | 12.723 | 7/0/291 |
| tipp_sell0.10_esc0.00_max0.15_rebuy0.25_cool168_reset0.02 | 0.100 | 0.000 | 0.150 | 0.250 | 168 | 0.020 | 236.642 | 29568.459 | 84.153 | 61.310 | 2.154 | 1.702 | 16.815 | 6/0/596 |
| tipp_sell0.10_esc0.00_max0.25_rebuy0.25_cool168_reset0.02 | 0.100 | 0.000 | 0.250 | 0.250 | 168 | 0.020 | 236.642 | 29568.459 | 84.153 | 61.310 | 2.154 | 1.702 | 16.815 | 6/0/596 |
| tipp_sell0.10_esc0.05_max0.15_rebuy0.25_cool168_reset0.02 | 0.100 | 0.050 | 0.150 | 0.250 | 168 | 0.020 | 235.576 | 29435.181 | 83.982 | 61.269 | 2.154 | 1.702 | 21.532 | 7/6/843 |
| tipp_sell0.10_esc0.05_max0.25_rebuy0.25_cool168_reset0.02 | 0.100 | 0.050 | 0.250 | 0.250 | 168 | 0.020 | 235.576 | 29435.181 | 83.982 | 61.269 | 2.154 | 1.702 | 21.532 | 7/6/843 |

## Ranking By Max Drawdown

| scenario | sell | esc | max | rebuy | cool | reset | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | sell/esc/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0.000 | 0/0/0 |
| incumbent_stateful_best |  |  |  |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0.000 | 0/0/0 |
| tipp_sell0.10_esc0.10_max0.25_rebuy0.50_cool168_reset0.02 | 0.100 | 0.100 | 0.250 | 0.500 | 168 | 0.020 | 178.810 | 22342.363 | 83.737 | 68.469 | 2.093 | 1.379 | 12.467 | 6/4/286 |
| tipp_sell0.10_esc0.10_max0.25_rebuy0.25_cool168_reset0.02 | 0.100 | 0.100 | 0.250 | 0.250 | 168 | 0.020 | 125.754 | 15712.917 | 83.831 | 75.698 | 1.978 | 1.252 | 17.350 | 6/5/751 |
| tipp_sell0.10_esc0.10_max0.15_rebuy0.50_cool168_reset0.02 | 0.100 | 0.100 | 0.150 | 0.500 | 168 | 0.020 | 181.459 | 22673.259 | 83.845 | 67.852 | 2.086 | 1.374 | 13.527 | 7/5/329 |
| tipp_sell0.05_esc0.10_max0.15_rebuy0.50_cool168_reset0.02 | 0.050 | 0.100 | 0.150 | 0.500 | 168 | 0.020 | 222.613 | 27815.436 | 83.858 | 60.199 | 2.153 | 1.702 | 13.346 | 6/6/294 |
| tipp_sell0.05_esc0.10_max0.25_rebuy0.50_cool168_reset0.02 | 0.050 | 0.100 | 0.250 | 0.500 | 168 | 0.020 | 222.613 | 27815.436 | 83.858 | 60.199 | 2.153 | 1.702 | 13.346 | 6/6/294 |
| tipp_sell0.10_esc0.05_max0.15_rebuy0.50_cool168_reset0.02 | 0.100 | 0.050 | 0.150 | 0.500 | 168 | 0.020 | 177.524 | 22181.683 | 83.858 | 68.715 | 2.079 | 1.364 | 13.518 | 7/5/328 |
| tipp_sell0.10_esc0.05_max0.25_rebuy0.50_cool168_reset0.02 | 0.100 | 0.050 | 0.250 | 0.500 | 168 | 0.020 | 177.524 | 22181.683 | 83.858 | 68.715 | 2.079 | 1.364 | 13.518 | 7/5/328 |
| tipp_sell0.10_esc0.00_max0.15_rebuy0.50_cool168_reset0.02 | 0.100 | 0.000 | 0.150 | 0.500 | 168 | 0.020 | 249.049 | 31118.634 | 83.952 | 60.721 | 2.169 | 1.702 | 12.723 | 7/0/291 |
| tipp_sell0.10_esc0.00_max0.25_rebuy0.50_cool168_reset0.02 | 0.100 | 0.000 | 0.250 | 0.500 | 168 | 0.020 | 249.049 | 31118.634 | 83.952 | 60.721 | 2.169 | 1.702 | 12.723 | 7/0/291 |
| tipp_sell0.05_esc0.05_max0.15_rebuy0.50_cool168_reset0.02 | 0.050 | 0.050 | 0.150 | 0.500 | 168 | 0.020 | 137.569 | 17189.247 | 83.964 | 77.326 | 1.990 | 1.262 | 14.438 | 7/6/355 |

## Ranking By 2024+ Peak-To-Trough

| scenario | sell | esc | max | rebuy | cool | reset | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | sell/esc/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tipp_sell0.05_esc0.10_max0.15_rebuy0.50_cool168_reset0.02 | 0.050 | 0.100 | 0.150 | 0.500 | 168 | 0.020 | 222.613 | 27815.436 | 83.858 | 60.199 | 2.153 | 1.702 | 13.346 | 6/6/294 |
| tipp_sell0.05_esc0.10_max0.25_rebuy0.50_cool168_reset0.02 | 0.050 | 0.100 | 0.250 | 0.500 | 168 | 0.020 | 222.613 | 27815.436 | 83.858 | 60.199 | 2.153 | 1.702 | 13.346 | 6/6/294 |
| tipp_sell0.05_esc0.10_max0.15_rebuy0.25_cool168_reset0.02 | 0.050 | 0.100 | 0.150 | 0.250 | 168 | 0.020 | 218.115 | 27253.469 | 83.982 | 60.378 | 2.145 | 1.702 | 16.160 | 5/5/540 |
| tipp_sell0.05_esc0.10_max0.25_rebuy0.25_cool168_reset0.02 | 0.050 | 0.100 | 0.250 | 0.250 | 168 | 0.020 | 218.115 | 27253.469 | 83.982 | 60.378 | 2.145 | 1.702 | 16.160 | 5/5/540 |
| tipp_sell0.10_esc0.00_max0.15_rebuy0.50_cool168_reset0.02 | 0.100 | 0.000 | 0.150 | 0.500 | 168 | 0.020 | 249.049 | 31118.634 | 83.952 | 60.721 | 2.169 | 1.702 | 12.723 | 7/0/291 |
| tipp_sell0.10_esc0.00_max0.25_rebuy0.50_cool168_reset0.02 | 0.100 | 0.000 | 0.250 | 0.500 | 168 | 0.020 | 249.049 | 31118.634 | 83.952 | 60.721 | 2.169 | 1.702 | 12.723 | 7/0/291 |
| tipp_sell0.10_esc0.05_max0.15_rebuy0.25_cool168_reset0.02 | 0.100 | 0.050 | 0.150 | 0.250 | 168 | 0.020 | 235.576 | 29435.181 | 83.982 | 61.269 | 2.154 | 1.702 | 21.532 | 7/6/843 |
| tipp_sell0.10_esc0.05_max0.25_rebuy0.25_cool168_reset0.02 | 0.100 | 0.050 | 0.250 | 0.250 | 168 | 0.020 | 235.576 | 29435.181 | 83.982 | 61.269 | 2.154 | 1.702 | 21.532 | 7/6/843 |
| tipp_sell0.10_esc0.00_max0.15_rebuy0.25_cool168_reset0.02 | 0.100 | 0.000 | 0.150 | 0.250 | 168 | 0.020 | 236.642 | 29568.459 | 84.153 | 61.310 | 2.154 | 1.702 | 16.815 | 6/0/596 |
| tipp_sell0.10_esc0.00_max0.25_rebuy0.25_cool168_reset0.02 | 0.100 | 0.000 | 0.250 | 0.250 | 168 | 0.020 | 236.642 | 29568.459 | 84.153 | 61.310 | 2.154 | 1.702 | 16.815 | 6/0/596 |
| tipp_sell0.05_esc0.00_max0.15_rebuy0.25_cool168_reset0.02 | 0.050 | 0.000 | 0.150 | 0.250 | 168 | 0.020 | 287.911 | 35974.496 | 84.449 | 63.858 | 2.192 | 1.702 | 35.712 | 7/0/649 |
| tipp_sell0.05_esc0.00_max0.25_rebuy0.25_cool168_reset0.02 | 0.050 | 0.000 | 0.250 | 0.250 | 168 | 0.020 | 287.911 | 35974.496 | 84.449 | 63.858 | 2.192 | 1.702 | 35.712 | 7/0/649 |

## Ranking By Final SOL-Equivalent

| scenario | sell | esc | max | rebuy | cool | reset | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | sell/esc/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0.000 | 0/0/0 |
| incumbent_stateful_best |  |  |  |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0.000 | 0/0/0 |
| tipp_sell0.05_esc0.00_max0.15_rebuy0.50_cool168_reset0.02 | 0.050 | 0.000 | 0.150 | 0.500 | 168 | 0.020 | 305.424 | 38162.703 | 84.155 | 64.053 | 2.206 | 1.702 | 12.817 | 7/0/284 |
| tipp_sell0.05_esc0.00_max0.25_rebuy0.50_cool168_reset0.02 | 0.050 | 0.000 | 0.250 | 0.500 | 168 | 0.020 | 305.424 | 38162.703 | 84.155 | 64.053 | 2.206 | 1.702 | 12.817 | 7/0/284 |
| tipp_sell0.05_esc0.00_max0.15_rebuy0.25_cool168_reset0.02 | 0.050 | 0.000 | 0.150 | 0.250 | 168 | 0.020 | 287.911 | 35974.496 | 84.449 | 63.858 | 2.192 | 1.702 | 35.712 | 7/0/649 |
| tipp_sell0.05_esc0.00_max0.25_rebuy0.25_cool168_reset0.02 | 0.050 | 0.000 | 0.250 | 0.250 | 168 | 0.020 | 287.911 | 35974.496 | 84.449 | 63.858 | 2.192 | 1.702 | 35.712 | 7/0/649 |
| tipp_sell0.10_esc0.00_max0.15_rebuy0.50_cool168_reset0.02 | 0.100 | 0.000 | 0.150 | 0.500 | 168 | 0.020 | 249.049 | 31118.634 | 83.952 | 60.721 | 2.169 | 1.702 | 12.723 | 7/0/291 |
| tipp_sell0.10_esc0.00_max0.25_rebuy0.50_cool168_reset0.02 | 0.100 | 0.000 | 0.250 | 0.500 | 168 | 0.020 | 249.049 | 31118.634 | 83.952 | 60.721 | 2.169 | 1.702 | 12.723 | 7/0/291 |
| tipp_sell0.10_esc0.00_max0.15_rebuy0.25_cool168_reset0.02 | 0.100 | 0.000 | 0.150 | 0.250 | 168 | 0.020 | 236.642 | 29568.459 | 84.153 | 61.310 | 2.154 | 1.702 | 16.815 | 6/0/596 |
| tipp_sell0.10_esc0.00_max0.25_rebuy0.25_cool168_reset0.02 | 0.100 | 0.000 | 0.250 | 0.250 | 168 | 0.020 | 236.642 | 29568.459 | 84.153 | 61.310 | 2.154 | 1.702 | 16.815 | 6/0/596 |
| tipp_sell0.10_esc0.05_max0.15_rebuy0.25_cool168_reset0.02 | 0.100 | 0.050 | 0.150 | 0.250 | 168 | 0.020 | 235.576 | 29435.181 | 83.982 | 61.269 | 2.154 | 1.702 | 21.532 | 7/6/843 |
| tipp_sell0.10_esc0.05_max0.25_rebuy0.25_cool168_reset0.02 | 0.100 | 0.050 | 0.250 | 0.250 | 168 | 0.020 | 235.576 | 29435.181 | 83.982 | 61.269 | 2.154 | 1.702 | 21.532 | 7/6/843 |

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | Reserve sell/esc/rebuy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 no reserve | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0/0 |
| Best Sortino | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0/0 |
| Best Max DD | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0/0 |
| Best 2024 DD | tipp_sell0.05_esc0.10_max0.15_rebuy0.50_cool168_reset0.02 | 222.613 | 27815.436 | 83.858 | 60.199 | 2.153 | 1.702 | 6/6/294 |
| Best SOL-eq | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0/0 |

## Readout

This candidate is not acceptable as the radical drawdown fix.

The episode gate fixed the worst mechanical problem from the earlier profit-lock reserve experiment: the strategy now sells only a handful of episode slices instead of repeatedly firing an initial sell. However, rebuy churn remains high because the reserve is still dripped back in many small recovered-trend bars. The best 2024+ peak-to-trough result, `tipp_sell0.05_esc0.10_max0.15_rebuy0.50_cool168_reset0.02`, improves 2024+ drawdown from `64.839%` to `60.199%`, but it cuts final SOL-equivalent from `424.902 SOL` to `222.613 SOL` and worsens full-window max drawdown from `81.344%` to `83.858%`.

The best capital-retention TIPP variants are the no-escalation `5%` sell versions, but those only improve 2024+ drawdown by less than one point while still dropping final SOL-equivalent to about `305.424 SOL`. Larger reserves protect the local late-cycle trough better, but the lost SOL exposure and repeated rebuy fees dominate.

Conclusion: keep the tested episode machinery as useful infrastructure, but do not promote this candidate. A SOL reserve can modestly reduce a local drawdown, yet the opportunity cost is far too high for the portfolio objective.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/tipp_profit_lock_reserve_sweep_20260602_111944/comparison.csv`
- Top-by-drawdown CSV: `reports/sol_supertrend_3y/tipp_profit_lock_reserve_sweep_20260602_111944/top_by_drawdown.csv`
- Top-by-2024-drawdown CSV: `reports/sol_supertrend_3y/tipp_profit_lock_reserve_sweep_20260602_111944/top_by_2024_drawdown.csv`
- Top-by-SOL-equivalent CSV: `reports/sol_supertrend_3y/tipp_profit_lock_reserve_sweep_20260602_111944/top_by_sol_equiv.csv`
- Summary JSON: `reports/sol_supertrend_3y/tipp_profit_lock_reserve_sweep_20260602_111944/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/tipp_profit_lock_reserve_sweep_20260602_111944`
