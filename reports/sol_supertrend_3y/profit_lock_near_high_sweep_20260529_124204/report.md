# Profit Lock Near-High Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can profit lock reduce the 2021/2025 giveback by hedging while the portfolio is still near a trailing high, instead of waiting for the existing drawdown trigger?

## Scenarios

All runs keep the current best configuration unless listed in the comparison table. Near-high `0.00` means disabled.

## Ranking

| scenario             | profit_lock_near_high_threshold | profit_lock_drawdown_threshold | profit_lock_hedge_floor | final_sol_equiv | final_portfolio_value_usd | max_drawdown_pct | sortino_ratio | min_health_factor | giveback_from_2021_peak_pct | giveback_from_2024_peak_pct | profit_lock_events |
| -------------------- | ------------------------------- | ------------------------------ | ----------------------- | --------------- | ------------------------- | ---------------- | ------------- | ----------------- | --------------------------- | --------------------------- | ------------------ |
| near_0.03_floor_0.25 | 0.030                           | 0.050                          | 0.250                   | 395.534         | 49421.977                 | 83.204           | 2.245         | 1.557             | -49.135                     | -49.135                     | 65                 |
| near_0.02_dd_0.08    | 0.020                           | 0.080                          | 0.350                   | 392.469         | 49038.987                 | 82.907           | 2.247         | 1.560             | -49.776                     | -49.776                     | 89                 |
| near_0.01_dd_0.05    | 0.010                           | 0.050                          | 0.350                   | 392.140         | 48997.833                 | 83.056           | 2.247         | 1.564             | -48.581                     | -48.581                     | 83                 |
| near_0.05_dd_0.05    | 0.050                           | 0.050                          | 0.350                   | 392.011         | 48981.801                 | 83.002           | 2.249         | 1.572             | -48.531                     | -48.531                     | 76                 |
| near_0.08_dd_0.05    | 0.080                           | 0.050                          | 0.350                   | 392.011         | 48981.801                 | 83.002           | 2.249         | 1.572             | -48.531                     | -48.531                     | 76                 |
| near_0.08_dd_0.08    | 0.080                           | 0.080                          | 0.350                   | 392.011         | 48981.801                 | 83.002           | 2.249         | 1.572             | -48.531                     | -48.531                     | 76                 |
| current_best         | 0.000                           | 0.050                          | 0.350                   | 391.925         | 48971.024                 | 82.900           | 2.249         | 1.573             | -48.562                     | -48.562                     | 80                 |
| near_0.03_dd_0.08    | 0.030                           | 0.080                          | 0.350                   | 391.020         | 48858.002                 | 82.968           | 2.245         | 1.559             | -49.778                     | -49.778                     | 89                 |
| near_0.05_dd_0.08    | 0.050                           | 0.080                          | 0.350                   | 388.533         | 48547.136                 | 83.092           | 2.244         | 1.559             | -49.747                     | -49.747                     | 96                 |
| near_0.01_dd_0.08    | 0.010                           | 0.080                          | 0.350                   | 386.777         | 48327.740                 | 83.007           | 2.244         | 1.566             | -49.762                     | -49.762                     | 89                 |
| near_0.03_dd_0.05    | 0.030                           | 0.050                          | 0.350                   | 384.320         | 48020.762                 | 83.021           | 2.242         | 1.583             | -48.537                     | -48.537                     | 83                 |
| near_0.02_dd_0.05    | 0.020                           | 0.050                          | 0.350                   | 382.776         | 47827.843                 | 83.065           | 2.241         | 1.586             | -48.530                     | -48.530                     | 83                 |
| near_0.03_floor_0.50 | 0.030                           | 0.050                          | 0.500                   | 380.328         | 47521.937                 | 83.594           | 2.230         | 1.643             | -44.589                     | -44.589                     | 73                 |

## Readout

Best by final SOL-equivalent: `near_0.03_floor_0.25` at 395.53 SOL-equivalent, Sortino 2.245, max drawdown 83.20%.
Current best control: 391.92 SOL-equivalent, Sortino 2.249, max drawdown 82.90%.

## Forensic Notes

Near-high produced one narrow final SOL-equivalent improvement, but it did not beat the current control on the full promotion rule. The best SOL-equivalent variant used a lower `0.25` profit-lock floor and ended at 395.53 SOL-equivalent, but Sortino slipped from 2.249 to 2.245, max drawdown worsened from 82.90% to 83.20%, and min HF fell from 1.573 to 1.557. The 2024/2025 giveback also became slightly worse.

The most important finding is mixed but not decisive: early profit-lock hedging is easy to over-apply. It can increase final SOL-equivalent in a narrow setting, but the extra timing sensitivity does not convert into a cleaner risk-adjusted result here.

## Next Hypothesis

Test a stateful peak-protection regime that activates after a large gain, then remains active until either SOL-equivalent makes a new high or higher-timeframe structure confirms recovery. That is different from this near-high trigger because it should persist through the mid-drawdown zone instead of firing only very near the peak or after the normal drawdown threshold.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/profit_lock_near_high_sweep_20260529_124204/comparison.csv`
- Summary JSON: `reports/sol_supertrend_3y/profit_lock_near_high_sweep_20260529_124204/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/profit_lock_near_high_sweep_20260529_124204`
