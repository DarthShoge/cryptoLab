# Froth Reserve Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can small tiered SOL-to-USDC rotations in frothy/profit-lock regimes reduce drawdown without destroying long-term SOL accumulation?

## Ranking

| scenario                | froth_reserve_tiers    | froth_reserve_rebuy_drawdown_threshold | froth_reserve_rebuy_fraction | final_sol_equiv | final_portfolio_value_usd | max_drawdown_pct | sortino_ratio | min_health_factor | giveback_from_2024_peak_pct | froth_rotate_events | froth_rebuy_events |
| ----------------------- | ---------------------- | -------------------------------------- | ---------------------------- | --------------- | ------------------------- | ---------------- | ------------- | ----------------- | --------------------------- | ------------------- | ------------------ |
| current_stateful_best   | {1.0: 0.05, 3.0: 0.05} | 0.350                                  | 0.250                        | 402.064         | 50237.887                 | 82.386           | 2.274         | 1.591             | -48.486                     | 0                   | 0                  |
| tier_2x5_4x5_rebuy50_50 | {1.0: 0.05, 3.0: 0.05} | 0.500                                  | 0.500                        | 379.921         | 47471.190                 | 83.227           | 2.259         | 1.598             | -48.469                     | 2                   | 501                |
| tier_2x5_rebuy50_50     | {1.0: 0.05}            | 0.500                                  | 0.500                        | 374.407         | 46782.186                 | 83.710           | 2.254         | 1.585             | -48.501                     | 1                   | 503                |
| tier_2x10_rebuy50_50    | {1.0: 0.1}             | 0.500                                  | 0.500                        | 374.407         | 46782.186                 | 83.710           | 2.254         | 1.585             | -48.501                     | 1                   | 503                |
| tier_2x5_rebuy35_25     | {1.0: 0.05}            | 0.350                                  | 0.250                        | 372.790         | 46580.139                 | 83.708           | 2.249         | 1.645             | -45.866                     | 1                   | 1608               |
| tier_2x10_rebuy35_25    | {1.0: 0.1}             | 0.350                                  | 0.250                        | 372.790         | 46580.139                 | 83.708           | 2.249         | 1.645             | -45.866                     | 1                   | 1608               |
| tier_2x5_4x5_rebuy35_25 | {1.0: 0.05, 3.0: 0.05} | 0.350                                  | 0.250                        | 371.771         | 46452.729                 | 83.902           | 2.248         | 1.635             | -45.947                     | 2                   | 1631               |

## Readout

Best by final SOL-equivalent: `current_stateful_best` at 402.06 SOL-equivalent, Sortino 2.274, max drawdown 82.39%.
Control stateful incumbent: 402.06 SOL-equivalent, Sortino 2.274, max drawdown 82.39%.

## Forensic Notes

Decision: no promotion.

This first sweep keeps rotations small and enforces the 100 SOL core. The idea worked only in the narrow sense that some variants reduced the late-cycle giveback from about -48.5% to about -45.9%. But every froth-reserve challenger materially damaged final SOL-equivalent and worsened max drawdown.

The failure mode is the rebuy policy. Once a drawdown threshold is crossed, the reserve repeatedly rebuys while the market is still falling. That turns dry powder back into SOL too early, increasing SOL exposure before the drawdown has finished. The `rebuy35_25` variants show this most clearly: over 1,600 rebuy events, better late giveback, but final SOL-equivalent drops from 402.06 to about 372 SOL and max drawdown worsens.

The conclusion is not that SOL-to-USDC rotation is bad. The conclusion is that the reserve needs a stricter redeployment rule. It should probably rebuy on recovery confirmation or tiered deeper drawdowns with one-shot buckets, not continuously whenever SOL is below a threshold.

Next candidate: keep the rotation side, but make rebuy stateful and bucketed:

- rotate once per froth tier,
- split reserve into fixed rebuy buckets,
- allow each bucket to deploy only once,
- require either a deeper drawdown tier such as 35%, 50%, 65%, or a recovery signal,
- stop reserve rebuys while crisis mode is still worsening.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/froth_reserve_sweep_20260529_132927/comparison.csv`
- Summary JSON: `reports/sol_supertrend_3y/froth_reserve_sweep_20260529_132927/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/froth_reserve_sweep_20260529_132927`
