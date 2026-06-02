# Quarterly Drawdown Review

## Scope

Run reviewed: promoted stateful profit lock `stateful_near_0.01_exit_0.05` from `reports/sol_supertrend_3y/stateful_profit_lock_sweep_20260529_125323/`.

Comparison run: previous incumbent `current_best` from the same sweep folder.

Objective: identify which quarters contribute to the still-unacceptable max drawdown and what programmatic rules might cut that drawdown without destroying SOL accumulation.

## Headline

The promoted run improves the old incumbent, but the drawdown profile is still dominated by violent SOL beta. The worst drawdown trough is `2023-06-10` at -82.39%. The preceding portfolio peak was `2021-11-06` at $22,927.54.

Late-cycle giveback remains visible too: from the overall portfolio high on `2025-01-19` ($97,521.87) to the subsequent trough on `2025-04-07` ($34,270.34), the strategy gave back -64.86%.

## Quarter Table

| quarter | portfolio_end_usd | portfolio_return_pct | quarter_max_drawdown_pct | buy_hold_return_pct | tracking_gap_pct | sol_equiv_end | sol_collateral_end | eth_debt_value_end | effective_hedge_target_avg | profit_lock_pct_bars | crisis_pct_bars |
| ------- | ----------------- | -------------------- | ------------------------ | ------------------- | ---------------- | ------------- | ------------------ | ------------------ | -------------------------- | -------------------- | --------------- |
| 2021Q1  | $1,874.00         | 1113.57%             | -30.93%                  | 1155.41%            | -41.84%          | 96.67 SOL     | 100.00             | $700.96            | 0.23                       | 65.57%               | 0.00%           |
| 2021Q2  | $3,267.13         | 74.25%               | -59.39%                  | 82.63%              | -8.38%           | 92.01 SOL     | 100.00             | $1,735.51          | 0.45                       | 79.03%               | 42.91%          |
| 2021Q3  | $13,300.28        | 307.81%              | -41.91%                  | 301.36%             | 6.45%            | 94.08 SOL     | 100.00             | $2,288.37          | 0.81                       | 82.74%               | 100.00%         |
| 2021Q4  | $16,218.84        | 20.44%               | -38.05%                  | 18.72%              | 1.71%            | 95.41 SOL     | 100.00             | $9,595.04          | 0.60                       | 93.25%               | 66.85%          |
| 2022Q1  | $12,330.04        | -24.52%              | -41.32%                  | -28.78%             | 4.26%            | 100.43 SOL    | 100.00             | $9,486.47          | 1.07                       | 100.00%              | 99.95%          |
| 2022Q2  | $9,302.86         | -25.19%              | -43.51%                  | -72.86%             | 47.67%           | 275.56 SOL    | 100.00             | $4,278.28          | 1.09                       | 100.00%              | 99.95%          |
| 2022Q3  | $8,004.25         | -13.72%              | -26.29%                  | -3.60%              | -10.12%          | 240.73 SOL    | 113.36             | $4,778.69          | 1.25                       | 100.00%              | 100.00%         |
| 2022Q4  | $4,831.22         | -39.32%              | -42.93%                  | -69.63%             | 30.31%           | 484.58 SOL    | 137.77             | $1,707.86          | 1.21                       | 100.00%              | 95.29%          |
| 2023Q1  | $6,387.31         | 32.05%               | -35.43%                  | 111.81%             | -79.76%          | 301.86 SOL    | 355.69             | $4,322.44          | 0.67                       | 82.68%               | 64.43%          |
| 2023Q2  | $5,302.80         | -16.94%              | -45.34%                  | -10.87%             | -6.07%           | 281.17 SOL    | 355.69             | $4,588.85          | 0.80                       | 100.00%              | 99.95%          |
| 2023Q3  | $6,819.45         | 29.84%               | -36.03%                  | 14.16%              | 15.68%           | 319.11 SOL    | 355.69             | $3,964.99          | 0.85                       | 97.83%               | 100.00%         |
| 2023Q4  | $33,949.49        | 399.68%              | -24.26%                  | 377.11%             | 22.57%           | 333.75 SOL    | 355.69             | $5,414.84          | 0.80                       | 67.80%               | 100.00%         |
| 2024Q1  | $66,543.02        | 95.70%               | -32.41%                  | 98.56%              | -2.86%           | 328.69 SOL    | 355.69             | $8,650.21          | 0.75                       | 83.65%               | 100.00%         |
| 2024Q2  | $47,172.67        | -28.99%              | -41.27%                  | -27.42%             | -1.57%           | 321.76 SOL    | 355.69             | $8,158.69          | 0.75                       | 99.73%               | 100.00%         |
| 2024Q3  | $51,247.79        | 7.67%                | -41.24%                  | 3.15%               | 4.52%            | 336.07 SOL    | 355.69             | $6,175.05          | 0.75                       | 96.47%               | 100.00%         |
| 2024Q4  | $62,598.93        | 20.83%               | -34.03%                  | 22.79%              | -1.96%           | 330.67 SOL    | 355.69             | $7,920.49          | 0.75                       | 81.25%               | 100.00%         |
| 2025Q1  | $43,156.63        | -32.24%              | -58.96%                  | -35.32%             | 3.08%            | 346.53 SOL    | 355.69             | $4,324.59          | 0.89                       | 98.56%               | 100.00%         |
| 2025Q2  | $52,350.06        | 21.40%               | -30.86%                  | 24.34%              | -2.94%           | 338.16 SOL    | 355.69             | $5,897.98          | 0.96                       | 96.61%               | 100.00%         |
| 2025Q3  | $69,253.41        | 32.38%               | -24.60%                  | 34.84%              | -2.46%           | 331.86 SOL    | 355.69             | $36,625.05         | 0.72                       | 79.94%               | 89.13%          |
| 2025Q4  | $50,237.89        | -27.00%              | -35.40%                  | -39.79%             | 12.80%           | 402.06 SOL    | 355.69             | $27,013.32         | 0.97                       | 100.00%              | 89.11%          |

## Worst Quarters By Intra-Quarter Drawdown

| quarter | portfolio_return_pct | quarter_max_drawdown_pct | buy_hold_return_pct | tracking_gap_pct | effective_hedge_target_avg | profit_lock_pct_bars | crisis_pct_bars | hf_min |
| ------- | -------------------- | ------------------------ | ------------------- | ---------------- | -------------------------- | -------------------- | --------------- | ------ |
| 2021Q2  | 74.25%               | -59.39%                  | 82.63%              | -8.38%           | 0.45                       | 79.03%               | 42.91%          | 1.72   |
| 2025Q1  | -32.24%              | -58.96%                  | -35.32%             | 3.08%            | 0.89                       | 98.56%               | 100.00%         | 6.71   |
| 2023Q2  | -16.94%              | -45.34%                  | -10.87%             | -6.07%           | 0.80                       | 100.00%              | 99.95%          | 1.59   |
| 2022Q2  | -25.19%              | -43.51%                  | -72.86%             | 47.67%           | 1.09                       | 100.00%              | 99.95%          | 1.80   |
| 2022Q4  | -39.32%              | -42.93%                  | -69.63%             | 30.31%           | 1.21                       | 100.00%              | 95.29%          | 1.81   |
| 2021Q3  | 307.81%              | -41.91%                  | 301.36%             | 6.45%            | 0.81                       | 82.74%               | 100.00%         | 1.74   |
| 2022Q1  | -24.52%              | -41.32%                  | -28.78%             | 4.26%            | 1.07                       | 100.00%              | 99.95%          | 1.70   |
| 2024Q2  | -28.99%              | -41.27%                  | -27.42%             | -1.57%           | 0.75                       | 99.73%               | 100.00%         | 4.49   |

## Promoted Versus Previous Incumbent

| quarter | return_delta_pct | qdd_delta_pct | sol_equiv_delta | promoted_profit_lock_pct_bars | incumbent_profit_lock_pct_bars |
| ------- | ---------------- | ------------- | --------------- | ----------------------------- | ------------------------------ |
| 2021Q1  | 71.76%           | 5.26%         | 5.72 SOL        | 65.57%                        | 47.91%                         |
| 2021Q2  | -3.45%           | 1.33%         | 3.54 SOL        | 79.03%                        | 74.30%                         |
| 2021Q3  | -8.58%           | -2.80%        | 1.73 SOL        | 82.74%                        | 68.39%                         |
| 2021Q4  | 0.21%            | 0.55%         | 2.01 SOL        | 93.25%                        | 82.07%                         |
| 2022Q1  | -0.01%           | 0.67%         | 2.09 SOL        | 100.00%                       | 94.21%                         |
| 2022Q2  | -0.12%           | 0.71%         | 5.18 SOL        | 100.00%                       | 97.89%                         |
| 2022Q3  | 0.47%            | 0.50%         | 5.82 SOL        | 100.00%                       | 81.25%                         |
| 2022Q4  | -0.01%           | -0.09%        | 11.63 SOL       | 100.00%                       | 94.61%                         |
| 2023Q1  | 1.66%            | 0.98%         | 10.95 SOL       | 82.68%                        | 66.23%                         |
| 2023Q2  | 0.24%            | 0.54%         | 10.97 SOL       | 100.00%                       | 92.31%                         |
| 2023Q3  | -0.50%           | 0.36%         | 11.30 SOL       | 97.83%                        | 81.61%                         |
| 2023Q4  | -5.21%           | 0.10%         | 8.48 SOL        | 67.80%                        | 47.83%                         |
| 2024Q1  | -0.25%           | 0.07%         | 7.95 SOL        | 83.65%                        | 72.16%                         |
| 2024Q2  | 0.04%            | 0.05%         | 7.95 SOL        | 99.73%                        | 94.46%                         |
| 2024Q3  | -0.01%           | 0.06%         | 8.25 SOL        | 96.47%                        | 81.66%                         |
| 2024Q4  | -0.04%           | 0.03%         | 8.02 SOL        | 81.25%                        | 68.48%                         |
| 2025Q1  | 0.04%            | 0.05%         | 8.62 SOL        | 98.56%                        | 94.26%                         |
| 2025Q2  | -0.04%           | 0.04%         | 8.29 SOL        | 96.61%                        | 83.88%                         |
| 2025Q3  | 0.01%            | 0.03%         | 8.16 SOL        | 79.94%                        | 67.44%                         |
| 2025Q4  | 0.05%            | 0.06%         | 10.14 SOL       | 100.00%                       | 100.00%                        |

## Quarter Commentary Flags

| quarter | commentary                                                                                                                                                      | event_summary                                                 |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 2021Q1  | large intra-quarter drawdown; materially lagged buy-and-hold SOL; profit lock active often; ended strong quarter still carrying hedge                           | cooldown_skip:96, profit_lock_hedge_up:17, hedge_down:14      |
| 2021Q2  | severe intra-quarter drawdown; profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                    | cooldown_skip:75, profit_lock_hedge_up:12, hedge_down:11      |
| 2021Q3  | large intra-quarter drawdown; profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                     |                                                               |
| 2021Q4  | large intra-quarter drawdown; profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                     | cooldown_skip:33, profit_lock_hedge_up:5, hedge_down:5        |
| 2022Q1  | large intra-quarter drawdown; profit lock active often; crisis mode active often                                                                                | full_short_cover:1, crisis_enter:1                            |
| 2022Q2  | large intra-quarter drawdown; profit lock active often; crisis mode active often                                                                                | cooldown_skip:25, full_short_up:6, crisis_hedge_up:3          |
| 2022Q3  | large intra-quarter drawdown; profit lock active often; crisis mode active often                                                                                | cooldown_skip:45, crisis_hedge_down:7, crisis_hedge_up:5      |
| 2022Q4  | large intra-quarter drawdown; profit lock active often; crisis mode active often                                                                                | cooldown_skip:96, full_short_up:18, crisis_hedge_down:4       |
| 2023Q1  | large intra-quarter drawdown; materially lagged buy-and-hold SOL; profit lock active often; crisis mode active often; ended strong quarter still carrying hedge | cooldown_skip:122, surplus_reinvestment:20, crisis_hedge_up:7 |
| 2023Q2  | severe intra-quarter drawdown; profit lock active often; crisis mode active often                                                                               | crisis_exit:1, crisis_enter:1                                 |
| 2023Q3  | large intra-quarter drawdown; profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                     |                                                               |
| 2023Q4  | profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                                                   |                                                               |
| 2024Q1  | large intra-quarter drawdown; profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                     |                                                               |
| 2024Q2  | large intra-quarter drawdown; profit lock active often; crisis mode active often                                                                                |                                                               |
| 2024Q3  | large intra-quarter drawdown; profit lock active often; crisis mode active often                                                                                |                                                               |
| 2024Q4  | large intra-quarter drawdown; profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                     |                                                               |
| 2025Q1  | severe intra-quarter drawdown; profit lock active often; crisis mode active often                                                                               |                                                               |
| 2025Q2  | large intra-quarter drawdown; profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                     |                                                               |
| 2025Q3  | profit lock active often; crisis mode active often; ended strong quarter still carrying hedge                                                                   | cooldown_skip:24, profit_lock_hedge_up:4, hedge_up:3          |
| 2025Q4  | large intra-quarter drawdown; profit lock active often; crisis mode active often                                                                                | cooldown_skip:12, profit_lock_hedge_up:2, hedge_up:2          |

## Diagnosis

1. The biggest max-drawdown problem is not simply that the hedge never exists. In multiple bad quarters the hedge/profit-lock/crisis machinery is active, but the account still carries large net SOL beta and drawdown stays huge.

2. Stateful profit lock helps because it prevents some premature relaxation, but its floor is still only 35%. In a true distribution/collapse quarter, a 35% ETH hedge cannot cut portfolio drawdown enough if SOL is the dominant collateral and SOL falls harder than ETH.

3. Crisis mode enters after damage is observable. It is useful for not getting liquidated, but it is not a fast enough peak-protection rule by itself.

4. There is a trade-off between drawdown control and SOL accumulation. The 0.50 floor stateful variant cut the late giveback more, but it lost too much final SOL-equivalent in the sweep. This suggests we need a conditional floor escalation, not a permanently higher floor.

## Programmatic Rule Candidates

1. **Drawdown-speed escalation:** if portfolio value drops more than X% from a 7d or 14d high while profit lock is already active, raise the hedge floor from 0.35 to 0.50 or 0.75 for a fixed cooldown window.

2. **SOL underperformance trigger:** if SOL drawdown from a recent high is worse than ETH drawdown by more than X%, treat ETH as an insufficient hedge and move some USDC collateral into debt repayment or a higher hedge floor.

3. **Peak-to-trough breaker:** after a new portfolio high and prior gain threshold, if running drawdown crosses 10-15%, freeze SOL buying/reinvestment and prioritize hedge/debt cleanup until recovery.

4. **Conditional high floor:** use the 0.50 floor only when stateful profit lock is active and either the 1d plus 3d trend are bearish or the quarter-to-date drawdown exceeds a threshold. This targets the red marks without making every pause expensive.

## Artifacts

- Quarterly metrics CSV: `reports/sol_supertrend_3y/quarterly_drawdown_review_20260529/quarterly_metrics.csv`
- Promoted-vs-incumbent CSV: `reports/sol_supertrend_3y/quarterly_drawdown_review_20260529/quarterly_promoted_vs_incumbent.csv`
- Commentary flags CSV: `reports/sol_supertrend_3y/quarterly_drawdown_review_20260529/quarterly_commentary_flags.csv`
