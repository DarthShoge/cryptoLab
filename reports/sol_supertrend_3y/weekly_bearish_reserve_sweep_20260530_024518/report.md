# Weekly Bearish Reserve Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can selling SOL collateral into USDC during weekly-bearish SOL regimes reduce the 2022 and 2025 drawdowns without destroying the long-SOL accumulation goal?

## Scenario Set

The sweep compares the current v2 fast-break no-reserve control against weekly-bearish reserve variants.

- Weekly bearish trigger: 1w SOL Supertrend bearish.
- Action: sell a capped slice of SOL collateral into USDC collateral.
- Minimum SOL collateral: `100 SOL`.
- Sell fraction grid: `5%`, `10%`, `15%`, `20%`, `25%`.
- Max reserve fraction grid: `20%`, `30%`, `40%`, `50%` of episode SOL.
- Rebuy fraction grid: `25%`, `50%`, `100%` when 1w recovers and 1d/3d are not both bearish.

## Ranking By Sortino

| scenario | sell | max | rebuy | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | SOL | USDC | ETH debt | active% | sell events | rebuy events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 375.669 | 34682.087 | 28530.476 | 0.000 | 0 | 0 |
| incumbent_stateful_best |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 355.692 | 32807.438 | 27013.323 | 0.000 | 0 | 0 |
| wbr_sell0.15_max0.30_rebuy0.50 | 0.150 | 0.300 | 0.500 | 157.550 | 19685.881 | 82.955 | 74.815 | 2.017 | 1.561 | 169.097 | 13648.380 | 15091.124 | 23.665 | 7 | 82 |
| wbr_sell0.20_max0.30_rebuy0.50 | 0.200 | 0.300 | 0.500 | 157.467 | 19675.561 | 82.955 | 74.828 | 2.016 | 1.561 | 169.140 | 13638.798 | 15097.252 | 23.665 | 5 | 82 |
| wbr_sell0.10_max0.30_rebuy0.50 | 0.100 | 0.300 | 0.500 | 157.412 | 19668.590 | 82.955 | 74.837 | 2.016 | 1.559 | 168.900 | 13627.714 | 15063.125 | 23.665 | 9 | 82 |
| wbr_sell0.25_max0.30_rebuy0.50 | 0.250 | 0.300 | 0.500 | 157.392 | 19666.154 | 82.955 | 74.841 | 2.016 | 1.561 | 169.173 | 13629.964 | 15102.001 | 23.665 | 5 | 82 |
| wbr_sell0.15_max0.30_rebuy0.25 | 0.150 | 0.300 | 0.250 | 157.579 | 19689.521 | 83.274 | 74.496 | 2.015 | 1.563 | 167.140 | 14023.094 | 15217.668 | 24.334 | 7 | 195 |
| wbr_sell0.20_max0.30_rebuy0.25 | 0.200 | 0.300 | 0.250 | 157.500 | 19679.578 | 83.274 | 74.509 | 2.015 | 1.563 | 167.183 | 14014.191 | 15224.099 | 24.334 | 5 | 195 |
| wbr_sell0.25_max0.30_rebuy0.25 | 0.250 | 0.300 | 0.250 | 157.427 | 19670.480 | 83.274 | 74.522 | 2.015 | 1.564 | 167.216 | 14005.898 | 15229.082 | 24.334 | 5 | 195 |
| wbr_sell0.10_max0.30_rebuy0.25 | 0.100 | 0.300 | 0.250 | 157.309 | 19655.761 | 83.274 | 74.544 | 2.015 | 1.561 | 166.942 | 13862.236 | 15065.938 | 24.334 | 10 | 195 |
| wbr_sell0.05_max0.30_rebuy0.50 | 0.050 | 0.300 | 0.500 | 155.878 | 19476.994 | 82.955 | 75.091 | 2.014 | 1.551 | 168.129 | 13264.133 | 14794.860 | 23.665 | 15 | 82 |
| wbr_sell0.05_max0.30_rebuy0.25 | 0.050 | 0.300 | 0.250 | 155.905 | 19480.361 | 83.274 | 74.778 | 2.012 | 1.554 | 166.172 | 13576.403 | 14859.190 | 24.334 | 15 | 195 |

## Ranking By Max Drawdown

| scenario | sell | max | rebuy | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | SOL | USDC | ETH debt | active% | sell events | rebuy events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 375.669 | 34682.087 | 28530.476 | 0.000 | 0 | 0 |
| incumbent_stateful_best |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 355.692 | 32807.438 | 27013.323 | 0.000 | 0 | 0 |
| wbr_sell0.15_max0.30_rebuy0.50 | 0.150 | 0.300 | 0.500 | 157.550 | 19685.881 | 82.955 | 74.815 | 2.017 | 1.561 | 169.097 | 13648.380 | 15091.124 | 23.665 | 7 | 82 |
| wbr_sell0.20_max0.30_rebuy0.50 | 0.200 | 0.300 | 0.500 | 157.467 | 19675.561 | 82.955 | 74.828 | 2.016 | 1.561 | 169.140 | 13638.798 | 15097.252 | 23.665 | 5 | 82 |
| wbr_sell0.10_max0.30_rebuy0.50 | 0.100 | 0.300 | 0.500 | 157.412 | 19668.590 | 82.955 | 74.837 | 2.016 | 1.559 | 168.900 | 13627.714 | 15063.125 | 23.665 | 9 | 82 |
| wbr_sell0.25_max0.30_rebuy0.50 | 0.250 | 0.300 | 0.500 | 157.392 | 19666.154 | 82.955 | 74.841 | 2.016 | 1.561 | 169.173 | 13629.964 | 15102.001 | 23.665 | 5 | 82 |
| wbr_sell0.05_max0.30_rebuy0.50 | 0.050 | 0.300 | 0.500 | 155.878 | 19476.994 | 82.955 | 75.091 | 2.014 | 1.551 | 168.129 | 13264.133 | 14794.860 | 23.665 | 15 | 82 |
| wbr_sell0.10_max0.20_rebuy0.50 | 0.100 | 0.200 | 0.500 | 152.168 | 19013.388 | 82.955 | 75.881 | 2.009 | 1.460 | 202.101 | 9907.895 | 16146.997 | 23.658 | 7 | 82 |
| wbr_sell0.15_max0.20_rebuy0.50 | 0.150 | 0.200 | 0.500 | 152.084 | 19002.901 | 82.955 | 75.894 | 2.009 | 1.460 | 202.144 | 9897.790 | 16152.743 | 23.658 | 5 | 82 |
| wbr_sell0.20_max0.20_rebuy0.50 | 0.200 | 0.200 | 0.500 | 152.004 | 18992.862 | 82.955 | 75.908 | 2.009 | 1.460 | 202.182 | 9888.089 | 16157.832 | 23.658 | 4 | 82 |
| wbr_sell0.05_max0.20_rebuy0.50 | 0.050 | 0.200 | 0.500 | 151.098 | 18879.704 | 82.955 | 76.055 | 2.007 | 1.457 | 202.153 | 9809.333 | 16188.679 | 23.655 | 11 | 81 |
| wbr_sell0.25_max0.50_rebuy0.50 | 0.250 | 0.500 | 0.500 | 118.440 | 14799.018 | 82.955 | 81.264 | 1.925 | 1.595 | 109.097 | 14583.360 | 13415.989 | 23.662 | 7 | 83 |

## Ranking By 2024+ Peak-To-Trough

| scenario | sell | max | rebuy | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | SOL | USDC | ETH debt | active% | sell events | rebuy events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 375.669 | 34682.087 | 28530.476 | 0.000 | 0 | 0 |
| incumbent_stateful_best |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 355.692 | 32807.438 | 27013.323 | 0.000 | 0 | 0 |
| wbr_sell0.15_max0.30_rebuy0.25 | 0.150 | 0.300 | 0.250 | 157.579 | 19689.521 | 83.274 | 74.496 | 2.015 | 1.563 | 167.140 | 14023.094 | 15217.668 | 24.334 | 7 | 195 |
| wbr_sell0.20_max0.30_rebuy0.25 | 0.200 | 0.300 | 0.250 | 157.500 | 19679.578 | 83.274 | 74.509 | 2.015 | 1.563 | 167.183 | 14014.191 | 15224.099 | 24.334 | 5 | 195 |
| wbr_sell0.25_max0.30_rebuy0.25 | 0.250 | 0.300 | 0.250 | 157.427 | 19670.480 | 83.274 | 74.522 | 2.015 | 1.564 | 167.216 | 14005.898 | 15229.082 | 24.334 | 5 | 195 |
| wbr_sell0.10_max0.30_rebuy0.25 | 0.100 | 0.300 | 0.250 | 157.309 | 19655.761 | 83.274 | 74.544 | 2.015 | 1.561 | 166.942 | 13862.236 | 15065.938 | 24.334 | 10 | 195 |
| wbr_sell0.05_max0.30_rebuy0.25 | 0.050 | 0.300 | 0.250 | 155.905 | 19480.361 | 83.274 | 74.778 | 2.012 | 1.554 | 166.172 | 13576.403 | 14859.190 | 24.334 | 15 | 195 |
| wbr_sell0.15_max0.30_rebuy0.50 | 0.150 | 0.300 | 0.500 | 157.550 | 19685.881 | 82.955 | 74.815 | 2.017 | 1.561 | 169.097 | 13648.380 | 15091.124 | 23.665 | 7 | 82 |
| wbr_sell0.20_max0.30_rebuy0.50 | 0.200 | 0.300 | 0.500 | 157.467 | 19675.561 | 82.955 | 74.828 | 2.016 | 1.561 | 169.140 | 13638.798 | 15097.252 | 23.665 | 5 | 82 |
| wbr_sell0.10_max0.30_rebuy0.50 | 0.100 | 0.300 | 0.500 | 157.412 | 19668.590 | 82.955 | 74.837 | 2.016 | 1.559 | 168.900 | 13627.714 | 15063.125 | 23.665 | 9 | 82 |
| wbr_sell0.25_max0.30_rebuy0.50 | 0.250 | 0.300 | 0.500 | 157.392 | 19666.154 | 82.955 | 74.841 | 2.016 | 1.561 | 169.173 | 13629.964 | 15102.001 | 23.665 | 5 | 82 |
| wbr_sell0.05_max0.30_rebuy0.50 | 0.050 | 0.300 | 0.500 | 155.878 | 19476.994 | 82.955 | 75.091 | 2.014 | 1.551 | 168.129 | 13264.133 | 14794.860 | 23.665 | 15 | 82 |

## Ranking By Final SOL-Equivalent

| scenario | sell | max | rebuy | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | SOL | USDC | ETH debt | active% | sell events | rebuy events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 375.669 | 34682.087 | 28530.476 | 0.000 | 0 | 0 |
| incumbent_stateful_best |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 355.692 | 32807.438 | 27013.323 | 0.000 | 0 | 0 |
| wbr_sell0.15_max0.30_rebuy0.25 | 0.150 | 0.300 | 0.250 | 157.579 | 19689.521 | 83.274 | 74.496 | 2.015 | 1.563 | 167.140 | 14023.094 | 15217.668 | 24.334 | 7 | 195 |
| wbr_sell0.15_max0.30_rebuy0.50 | 0.150 | 0.300 | 0.500 | 157.550 | 19685.881 | 82.955 | 74.815 | 2.017 | 1.561 | 169.097 | 13648.380 | 15091.124 | 23.665 | 7 | 82 |
| wbr_sell0.20_max0.30_rebuy0.25 | 0.200 | 0.300 | 0.250 | 157.500 | 19679.578 | 83.274 | 74.509 | 2.015 | 1.563 | 167.183 | 14014.191 | 15224.099 | 24.334 | 5 | 195 |
| wbr_sell0.20_max0.30_rebuy0.50 | 0.200 | 0.300 | 0.500 | 157.467 | 19675.561 | 82.955 | 74.828 | 2.016 | 1.561 | 169.140 | 13638.798 | 15097.252 | 23.665 | 5 | 82 |
| wbr_sell0.25_max0.30_rebuy0.25 | 0.250 | 0.300 | 0.250 | 157.427 | 19670.480 | 83.274 | 74.522 | 2.015 | 1.564 | 167.216 | 14005.898 | 15229.082 | 24.334 | 5 | 195 |
| wbr_sell0.10_max0.30_rebuy0.50 | 0.100 | 0.300 | 0.500 | 157.412 | 19668.590 | 82.955 | 74.837 | 2.016 | 1.559 | 168.900 | 13627.714 | 15063.125 | 23.665 | 9 | 82 |
| wbr_sell0.25_max0.30_rebuy0.50 | 0.250 | 0.300 | 0.500 | 157.392 | 19666.154 | 82.955 | 74.841 | 2.016 | 1.561 | 169.173 | 13629.964 | 15102.001 | 23.665 | 5 | 82 |
| wbr_sell0.10_max0.30_rebuy0.25 | 0.100 | 0.300 | 0.250 | 157.309 | 19655.761 | 83.274 | 74.544 | 2.015 | 1.561 | 166.942 | 13862.236 | 15065.938 | 24.334 | 10 | 195 |
| wbr_sell0.05_max0.30_rebuy0.25 | 0.050 | 0.300 | 0.250 | 155.905 | 19480.361 | 83.274 | 74.778 | 2.012 | 1.554 | 166.172 | 13576.403 | 14859.190 | 24.334 | 15 | 195 |
| wbr_sell0.05_max0.30_rebuy0.50 | 0.050 | 0.300 | 0.500 | 155.878 | 19476.994 | 82.955 | 75.091 | 2.014 | 1.551 | 168.129 | 13264.133 | 14794.860 | 23.665 | 15 | 82 |

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | Reserve sell/rebuy events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 no reserve | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |
| Best Sortino | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |
| Best Max DD | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |
| Best 2024 DD | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |
| Best SOL-eq | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0 |

## Forensic Notes

Decision: no promotion. The current v2 fast-break no-reserve control remains the best result across every promotion axis: final SOL-equivalent, final USD value, Sortino, max drawdown, and 2024+ peak-to-trough drawdown.

This sweep is a useful negative result. Weekly-bearish reserve does reduce raw SOL exposure, but it does it too late and then keeps too much capital out of SOL during the next compounding phase. The best reserve Sortino candidate, `wbr_sell0.15_max0.30_rebuy0.50`, finishes at only `157.55` SOL-equivalent versus `424.90` for no-reserve. It also worsens max drawdown from `81.34%` to `82.96%` and worsens the 2024+ peak-to-trough drawdown from `64.84%` to `74.81%`.

The event log explains the failure. The first reserve sell in the best reserve row occurs on `2022-07-25`, after the main 2022 crash damage has already happened. At that point SOL has already collapsed, so the rule is effectively selling weakness rather than preserving frothy gains. The strategy then spends much of early 2023 slowly rebuying, which delays SOL re-accumulation into the next major uptrend.

The 2025 window shows the trade-off clearly. On `2025-01-19`, the no-reserve control reaches about `$103.0k` with `375.67 SOL` collateral. The reserve row reaches only about `$76.1k` with `274.88 SOL` collateral because it had missed earlier compounding. By `2025-04-07`, the reserve row has less SOL beta and a lower dollar trough, but the lower starting peak and weaker participation make its measured 2024+ peak-to-trough drawdown worse, not better.

This means the principle is still right but the trigger is wrong. Selling SOL only after the weekly Supertrend turns bearish is too delayed for this objective. The next SOL de-risk experiment should sell earlier, while the portfolio is still near highs, and then use weekly bearishness as a confirmation to hold the reserve rather than as the first sell trigger.

Recommended next hypothesis:

- enter a SOL reserve from profit-lock or near-high conditions, not from 1w bearish alone;
- require the portfolio to be up meaningfully from initial value;
- sell a smaller first slice before weekly bearish confirmation;
- if 1w turns bearish, stop rebuying and possibly add one additional capped reserve slice;
- rebuy only after stronger trend recovery, but not in tiny hourly trickles.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/weekly_bearish_reserve_sweep_20260530_024518/comparison.csv`
- Top-by-drawdown CSV: `reports/sol_supertrend_3y/weekly_bearish_reserve_sweep_20260530_024518/top_by_drawdown.csv`
- Top-by-2024-drawdown CSV: `reports/sol_supertrend_3y/weekly_bearish_reserve_sweep_20260530_024518/top_by_2024_drawdown.csv`
- Top-by-SOL-equivalent CSV: `reports/sol_supertrend_3y/weekly_bearish_reserve_sweep_20260530_024518/top_by_sol_equiv.csv`
- Summary JSON: `reports/sol_supertrend_3y/weekly_bearish_reserve_sweep_20260530_024518/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/weekly_bearish_reserve_sweep_20260530_024518`
