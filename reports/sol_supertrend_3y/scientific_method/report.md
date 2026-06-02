# SOL Supertrend Scientific Method Report

Run date: 2026-05-28

Dataset: Binance SOL/USDT and ETH/USDT, 1h candles, 2023-05-28 00:00:00+00:00 through 2026-05-28 00:00:00+00:00 (26305 bars).

## Question

Can the SOL Supertrend Kamino strategy improve SOL-relative USD compounding by isolating core ETH hedging, full-short mode, and USDC releverage as separate modules?

## Hypotheses

1. Core ETH hedging can improve drawdown and possibly beat SOL buy-and-hold in USD.

2. Full-short mode may help during deep SOL downtrends, but can leave residual ETH debt if exit timing is late.

3. USDC releverage is likely to increase tail risk and should only survive if it clears the tiered SOL benchmark gate.

4. ETH borrow carry should reduce but not necessarily erase hedge-only performance.

## Benchmark

Initial account: 100.00 SOL = $2,063.00. Buy-and-hold final: $8,271.00, return 300.92%, max drawdown 73.62%.

## Experiment Design

Tested 30 combinations across three signal regimes, three hedge ladders, explicit full-short variants, and USDC releverage variants. USDC releverage is opt-in. Full-short is isolated with `enable_full_short_mode`.

## Results

Top by benchmark tier then Sortino:

| run_id  | experiment_group | profile                        | signal    | benchmark_tier | final_portfolio_value_usd | total_return_pct | tracking_gap_vs_sol_pct | final_sol_equiv | max_drawdown_pct | sortino_ratio | liquidations | bars_below_hf_1 |
| ------- | ---------------- | ------------------------------ | --------- | -------------- | ------------------------- | ---------------- | ----------------------- | --------------- | ---------------- | ------------- | ------------ | --------------- |
| run_015 | core_full_short  | soft_hedge_capped_full_short   | base_10_3 | pass           | 17,272.21                 | 737.238          | 436.317                 | 208.829         | 43.154           | 1.823         | 0            | 0               |
| run_014 | core_full_short  | soft_hedge_standard_full_short | base_10_3 | pass           | 17,097.47                 | 728.767          | 427.846                 | 206.716         | 43.154           | 1.817         | 0            | 0               |
| run_016 | core_full_short  | base_hedge_standard_full_short | base_10_3 | pass           | 14,668.93                 | 611.049          | 310.128                 | 177.354         | 49.345           | 1.748         | 0            | 0               |
| run_017 | core_full_short  | base_hedge_capped_full_short   | base_10_3 | pass           | 14,481.08                 | 601.943          | 301.022                 | 175.083         | 50.450           | 1.739         | 0            | 0               |
| run_012 | core_hedge       | base_hedge                     | base_10_3 | pass           | 12,801.51                 | 520.529          | 219.608                 | 154.776         | 55.516           | 1.661         | 0            | 0               |
| run_011 | core_hedge       | soft_hedge                     | base_10_3 | pass           | 12,704.10                 | 515.807          | 214.886                 | 153.598         | 57.096           | 1.636         | 0            | 0               |
| run_022 | core_hedge       | base_hedge                     | slow_14_4 | pass           | 12,070.94                 | 485.116          | 184.195                 | 145.943         | 59.821           | 1.633         | 0            | 0               |
| run_013 | core_hedge       | aggressive_hedge               | base_10_3 | pass           | 12,126.96                 | 487.831          | 186.910                 | 146.620         | 56.839           | 1.630         | 0            | 0               |

Top by final USD:

| run_id  | experiment_group | profile                        | signal    | benchmark_tier | final_portfolio_value_usd | total_return_pct | tracking_gap_vs_sol_pct | final_sol_equiv | max_drawdown_pct | sortino_ratio | liquidations | bars_below_hf_1 |
| ------- | ---------------- | ------------------------------ | --------- | -------------- | ------------------------- | ---------------- | ----------------------- | --------------- | ---------------- | ------------- | ------------ | --------------- |
| run_015 | core_full_short  | soft_hedge_capped_full_short   | base_10_3 | pass           | 17,272.21                 | 737.238          | 436.317                 | 208.829         | 43.154           | 1.823         | 0            | 0               |
| run_014 | core_full_short  | soft_hedge_standard_full_short | base_10_3 | pass           | 17,097.47                 | 728.767          | 427.846                 | 206.716         | 43.154           | 1.817         | 0            | 0               |
| run_016 | core_full_short  | base_hedge_standard_full_short | base_10_3 | pass           | 14,668.93                 | 611.049          | 310.128                 | 177.354         | 49.345           | 1.748         | 0            | 0               |
| run_017 | core_full_short  | base_hedge_capped_full_short   | base_10_3 | pass           | 14,481.08                 | 601.943          | 301.022                 | 175.083         | 50.450           | 1.739         | 0            | 0               |
| run_012 | core_hedge       | base_hedge                     | base_10_3 | pass           | 12,801.51                 | 520.529          | 219.608                 | 154.776         | 55.516           | 1.661         | 0            | 0               |
| run_011 | core_hedge       | soft_hedge                     | base_10_3 | pass           | 12,704.10                 | 515.807          | 214.886                 | 153.598         | 57.096           | 1.636         | 0            | 0               |
| run_013 | core_hedge       | aggressive_hedge               | base_10_3 | pass           | 12,126.96                 | 487.831          | 186.910                 | 146.620         | 56.839           | 1.630         | 0            | 0               |
| run_022 | core_hedge       | base_hedge                     | slow_14_4 | pass           | 12,070.94                 | 485.116          | 184.195                 | 145.943         | 59.821           | 1.633         | 0            | 0               |

Group medians:

| experiment_group | final_portfolio_value_usd | total_return_pct | final_sol_equiv | tracking_gap_vs_sol_pct | max_drawdown_pct | sortino_ratio | liquidations | bars_below_hf_1 |
| ---------------- | ------------------------- | ---------------- | --------------- | ----------------------- | ---------------- | ------------- | ------------ | --------------- |
| core_full_short  | 9,144.85                  | 343.279          | 110.565         | 42.358                  | 69.885           | 1.440         | 0.000        | 0.000           |
| core_hedge       | 11,236.63                 | 444.674          | 135.856         | 143.753                 | 62.052           | 1.556         | 0.000        | 0.000           |
| usdc_releverage  | 6,789.54                  | 229.110          | 82.089          | -71.811                 | 95.605           | 1.408         | 5.000        | 1,274.00        |

Tier counts:

| experiment_group | benchmark_tier | count |
| ---------------- | -------------- | ----- |
| core_full_short  | acceptable     | 1     |
| core_full_short  | pass           | 7     |
| core_full_short  | reject         | 4     |
| core_hedge       | pass           | 6     |
| core_hedge       | reject         | 3     |
| usdc_releverage  | reject         | 9     |

## Ending Position Of Best Tiered Candidate

| run_id  | experiment_group | profile                      | signal    | final_portfolio_value_usd | final_sol_equiv | final_collateral_SOL | final_collateral_SOL_value_usd | final_collateral_USDC | final_collateral_USDC_value_usd | final_debt_ETH | final_debt_ETH_value_usd | final_debt_USDC | final_debt_USDC_value_usd |
| ------- | ---------------- | ---------------------------- | --------- | ------------------------- | --------------- | -------------------- | ------------------------------ | --------------------- | ------------------------------- | -------------- | ------------------------ | --------------- | ------------------------- |
| run_015 | core_full_short  | soft_hedge_capped_full_short | base_10_3 | 17,272.21                 | 208.829         | 100.000              | 8,271.00                       | 19,339.96             | 19,339.96                       | 5.097          | 10,338.75                | 0.000           | 0.000                     |

## ETH Carry Sensitivity

Applied to best non-USDC run `run_015` (soft_hedge_capped_full_short, base_10_3).

| eth_borrow_apy | final_portfolio_value_usd | total_return_pct | final_sol_equiv | max_drawdown_pct | sortino_ratio | interest_paid | liquidations | bars_below_hf_1 |
| -------------- | ------------------------- | ---------------- | --------------- | ---------------- | ------------- | ------------- | ------------ | --------------- |
| 0.000          | 17,272.21                 | 737.238          | 208.829         | 43.154           | 1.823         | 0.000         | 0            | 0               |
| 0.050          | 16,550.41                 | 702.250          | 200.102         | 44.424           | 1.794         | 704.951       | 0            | 0               |
| 0.100          | 15,694.55                 | 660.764          | 189.754         | 46.754           | 1.759         | 1,410.63      | 0            | 0               |
| 0.200          | 14,144.57                 | 585.631          | 171.014         | 51.074           | 1.692         | 2,827.62      | 0            | 0               |

## Interpretation

Best tiered candidate: `run_015` with core_full_short / soft_hedge_capped_full_short / base_10_3. It ended at $17,272.21, 208.83 SOL-equivalent, drawdown 43.15%, Sortino 1.823, and tier `pass`.

Best raw USD candidate: `run_015` with $17,272.21; it is also tier `pass` with drawdown 43.15%.

The scientific update is: full-short mode, once properly isolated, slightly improved the prior best hedge-only result in this sample. USDC releverage still looks like the dangerous module: it can produce positive USD returns, but the group-level medians show weaker SOL-relative outcomes and worse health-factor stress than core ETH hedge/full-short modules.

The best candidate ends with 100 SOL collateral, a large USDC collateral balance from ETH short proceeds, and open ETH debt. That is economically coherent, but it means the final mark is still exposed to ETH rally risk after the backtest window. The next experiment should focus on exit discipline and residual ETH debt limits.

## Next Hypotheses

1. Optimize the soft/base hedge ladder around the best `base_10_3` signal before reintroducing USDC debt.

2. Add ETH borrow-rate scenarios to every grid run, not only sensitivity checks.

3. Test full-short exits separately: the cover ladder may need earlier de-risking or a residual short cap.

4. Add walk-forward splits so the best 3-year profile is not just fit to this one market path.

## Files Generated

- `reports/sol_supertrend_3y/scientific_method/best_non_usdc_eth_carry_sensitivity.csv`
- `reports/sol_supertrend_3y/scientific_method/run_014_events.csv`
- `reports/sol_supertrend_3y/scientific_method/run_014_history.csv`
- `reports/sol_supertrend_3y/scientific_method/run_015_events.csv`
- `reports/sol_supertrend_3y/scientific_method/run_015_history.csv`
- `reports/sol_supertrend_3y/scientific_method/run_016_events.csv`
- `reports/sol_supertrend_3y/scientific_method/run_016_history.csv`
- `reports/sol_supertrend_3y/scientific_method/run_025_events.csv`
- `reports/sol_supertrend_3y/scientific_method/run_025_history.csv`
- `reports/sol_supertrend_3y/scientific_method/scientific_summary.json`
- `reports/sol_supertrend_3y/scientific_method/scientific_sweep.csv`