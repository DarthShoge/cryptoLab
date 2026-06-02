# Stateful Profit Lock Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can a stateful profit-lock regime reduce late-cycle giveback by keeping the hedge floor active after the first profit-lock trigger, instead of dropping back to the normal ladder in the middle drawdown zone?

## Ranking

| scenario                      | profit_lock_near_high_threshold | profit_lock_stateful_exit_gap | profit_lock_hedge_floor | final_sol_equiv | final_portfolio_value_usd | max_drawdown_pct | sortino_ratio | min_health_factor | giveback_from_2024_peak_pct | profit_lock_events |
| ----------------------------- | ------------------------------- | ----------------------------- | ----------------------- | --------------- | ------------------------- | ---------------- | ------------- | ----------------- | --------------------------- | ------------------ |
| stateful_near_0.01_exit_0.05  | 0.010                           | 0.050                         | 0.350                   | 402.064         | 50237.887                 | 82.386           | 2.274         | 1.591             | -48.486                     | 48                 |
| stateful_near_0.03_exit_0.05  | 0.030                           | 0.050                         | 0.350                   | 402.064         | 50237.887                 | 82.386           | 2.274         | 1.591             | -48.486                     | 48                 |
| stateful_near_0.05_exit_0.05  | 0.050                           | 0.050                         | 0.350                   | 402.064         | 50237.887                 | 82.386           | 2.274         | 1.591             | -48.486                     | 48                 |
| current_best                  | 0.000                           | 0.020                         | 0.350                   | 391.925         | 48971.024                 | 82.900           | 2.249         | 1.573             | -48.562                     | 80                 |
| stateful_near_0.01_exit_0.00  | 0.010                           | 0.000                         | 0.350                   | 374.995         | 46855.683                 | 83.425           | 2.254         | 1.586             | -48.526                     | 47                 |
| stateful_near_0.03_exit_0.00  | 0.030                           | 0.000                         | 0.350                   | 374.995         | 46855.683                 | 83.425           | 2.254         | 1.586             | -48.526                     | 47                 |
| stateful_near_0.05_exit_0.00  | 0.050                           | 0.000                         | 0.350                   | 374.995         | 46855.683                 | 83.425           | 2.254         | 1.586             | -48.526                     | 47                 |
| stateful_near_0.03_floor_0.50 | 0.030                           | 0.020                         | 0.500                   | 371.213         | 46383.026                 | 83.469           | 2.234         | 1.565             | -44.246                     | 40                 |
| stateful_near_0.01_exit_0.02  | 0.010                           | 0.020                         | 0.350                   | 365.710         | 45695.443                 | 84.185           | 2.242         | 1.562             | -48.583                     | 51                 |
| stateful_near_0.03_exit_0.02  | 0.030                           | 0.020                         | 0.350                   | 365.710         | 45695.443                 | 84.185           | 2.242         | 1.562             | -48.583                     | 51                 |
| stateful_near_0.05_exit_0.02  | 0.050                           | 0.020                         | 0.350                   | 365.710         | 45695.443                 | 84.185           | 2.242         | 1.562             | -48.583                     | 51                 |
| stateful_near_0.03_floor_0.25 | 0.030                           | 0.020                         | 0.250                   | 112.848         | 14100.312                 | 96.104           | 1.900         | 1.690             | -48.816                     | 26                 |

## Readout

Best by final SOL-equivalent: `stateful_near_0.01_exit_0.05` at 402.06 SOL-equivalent, Sortino 2.274, max drawdown 82.39%.
Current best control: 391.92 SOL-equivalent, Sortino 2.249, max drawdown 82.90%.

## Forensic Notes

Stateful profit lock changes the holding behavior, not only the initial trigger. This sweep clears the promotion rule: the best stateful run improves final SOL-equivalent from 391.92 to 402.06, Sortino from 2.249 to 2.274, max drawdown from 82.90% to 82.39%, and min HF from 1.573 to 1.591.

The red-mark giveback improvement is real but small: -48.56% to -48.49%. So this is a better incumbent, but it still does not solve the peak-giveback problem outright.

The winning near-high thresholds from 1% to 5% produced identical outcomes in this sweep, which implies the stateful holding/exiting rule mattered more than the exact early trigger width. We promote the conservative `0.01` near-high threshold to avoid broadening the trigger unnecessarily.

The exit gap mattered a lot. `0.05` was strongly better than `0.02` or `0.00`, suggesting that forcing the system to wait for a perfect new high before relaxing the hedge is too rigid, while a 5% recovery band gets the strategy out of protection without clinging to stale shorts.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/stateful_profit_lock_sweep_20260529_125323/comparison.csv`
- Summary JSON: `reports/sol_supertrend_3y/stateful_profit_lock_sweep_20260529_125323/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/stateful_profit_lock_sweep_20260529_125323`
