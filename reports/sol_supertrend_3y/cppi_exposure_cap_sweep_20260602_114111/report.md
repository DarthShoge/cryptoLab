# CPPI Exposure Cap Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can a high-watermark CPPI/TIPP exposure cap reduce drawdown by continuously limiting SOL collateral value to a multiple of the cushion above a protected floor?

## Scenario Set

- Activation gain: `1.5x` initial portfolio value.
- Protected floor grid: `55%`, `65%`, `75%` of portfolio high-watermark.
- Cushion multiplier grid: `1.5`, `2.0`.
- Rebuy confirmation grid: `3` or `4` green votes with no bearish 1d/3d/1w votes.
- Core SOL minimum: `100 SOL`.
- Per-bar sell step cap: `10%` of current SOL collateral.
- Rebuy fraction: `50%` of permitted exposure gap.

## Ranking By Sortino

| scenario | protect | mult | green | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | sell/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0.000 | 0/0 |
| incumbent_stateful_best |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0.000 | 0/0 |
| cppi_protect0.55_mult1.5_green3 | 0.550 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult1.5_green4 | 0.550 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult2.0_green3 | 0.550 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult2.0_green4 | 0.550 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult1.5_green3 | 0.650 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult1.5_green4 | 0.650 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult2.0_green3 | 0.650 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult2.0_green4 | 0.650 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult1.5_green3 | 0.750 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult1.5_green4 | 0.750 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |

## Ranking By Max Drawdown

| scenario | protect | mult | green | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | sell/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cppi_protect0.55_mult1.5_green3 | 0.550 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult1.5_green4 | 0.550 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult2.0_green3 | 0.550 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult2.0_green4 | 0.550 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult1.5_green3 | 0.650 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult1.5_green4 | 0.650 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult2.0_green3 | 0.650 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult2.0_green4 | 0.650 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult1.5_green3 | 0.750 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult1.5_green4 | 0.750 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult2.0_green3 | 0.750 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult2.0_green4 | 0.750 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |

## Ranking By 2024+ Peak-To-Trough

| scenario | protect | mult | green | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | sell/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0.000 | 0/0 |
| incumbent_stateful_best |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0.000 | 0/0 |
| cppi_protect0.55_mult1.5_green3 | 0.550 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult1.5_green4 | 0.550 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult2.0_green3 | 0.550 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult2.0_green4 | 0.550 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult1.5_green3 | 0.650 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult1.5_green4 | 0.650 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult2.0_green3 | 0.650 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult2.0_green4 | 0.650 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult1.5_green3 | 0.750 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult1.5_green4 | 0.750 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |

## Ranking By Final SOL-Equivalent

| scenario | protect | mult | green | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | sell/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0.000 | 0/0 |
| incumbent_stateful_best |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0.000 | 0/0 |
| cppi_protect0.55_mult1.5_green3 | 0.550 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult1.5_green4 | 0.550 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult2.0_green3 | 0.550 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.55_mult2.0_green4 | 0.550 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult1.5_green3 | 0.650 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult1.5_green4 | 0.650 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult2.0_green3 | 0.650 | 2.000 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.65_mult2.0_green4 | 0.650 | 2.000 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult1.5_green3 | 0.750 | 1.500 | 3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |
| cppi_protect0.75_mult1.5_green4 | 0.750 | 1.500 | 4 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 68.783 | 39/0 |

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | CPPI sell/rebuy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 no reserve | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |
| Best Sortino | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |
| Best Max DD | cppi_protect0.55_mult1.5_green3 | 66.280 | 8281.626 | 74.869 | 70.167 | 1.795 | 1.350 | 39/0 |
| Best 2024 DD | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |
| Best SOL-eq | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |

## Readout

This candidate proves the structural point, but fails the strategy objective.

The CPPI cap is the only tested mechanism so far that materially reduces full-window max drawdown: every CPPI variant lowers max drawdown from `81.344%` to `74.869%`. The cost is severe. Final SOL-equivalent collapses from `424.902 SOL` to `66.280 SOL`, final USD falls from `$53,091.484` to `$8,281.626`, Sortino drops to `1.795`, and 2024+ peak-to-trough worsens from `64.839%` to `70.167%`.

All tested variants converged to the same result because the core `100 SOL` minimum became the binding constraint. Once the cap forced SOL down to the core floor, the cushion never expanded enough to permit CPPI rebuys under the rule set. That is the classic portfolio-insurance trap: it protects an old drawdown window by permanently de-risking the asset that later drives recovery.

Conclusion: CPPI/TIPP exposure caps can radically lower headline max drawdown, but this implementation is too blunt for a SOL-accumulation objective. Do not promote it. If revisited, it needs a higher core floor, delayed activation, or a rebuy rule based on SOL-equivalent recovery rather than only cushion expansion.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/cppi_exposure_cap_sweep_20260602_114111/comparison.csv`
- Top-by-drawdown CSV: `reports/sol_supertrend_3y/cppi_exposure_cap_sweep_20260602_114111/top_by_drawdown.csv`
- Top-by-2024-drawdown CSV: `reports/sol_supertrend_3y/cppi_exposure_cap_sweep_20260602_114111/top_by_2024_drawdown.csv`
- Top-by-SOL-equivalent CSV: `reports/sol_supertrend_3y/cppi_exposure_cap_sweep_20260602_114111/top_by_sol_equiv.csv`
- Summary JSON: `reports/sol_supertrend_3y/cppi_exposure_cap_sweep_20260602_114111/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/cppi_exposure_cap_sweep_20260602_114111`
