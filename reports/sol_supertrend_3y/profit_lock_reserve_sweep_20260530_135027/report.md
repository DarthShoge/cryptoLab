# Profit-Lock Reserve Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can SOL reserve selling from profit-lock / near-high conditions reduce the 2022 and 2025 drawdowns without waiting until the weekly trend has already turned bearish?

## Scenario Set

The sweep compares the current v2 fast-break no-reserve control against profit-lock reserve variants.

- Entry: profit lock active, portfolio up at least `100%` from initial, and within `10%` of the trailing high.
- First trigger: 1d bearish, green votes at or below 2, or fast-break active.
- Escalation trigger: 3d bearish or portfolio drawdown from high at least `15%`.
- Weekly bearish rule: blocks reserve rebuy, but does not initiate the first sale.
- Minimum SOL collateral: `100 SOL`.
- First sell fraction grid: `5%`, `10%`, `15%`.
- Escalation sell fraction grid: `5%`, `10%`.
- Max reserve fraction grid: `20%`, `30%`, `40%`.
- Rebuy fraction grid: `50%`, `100%` after 1w/1d/3d recovery.

## Ranking By Sortino

| scenario | sell | esc | max | rebuy | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | SOL | USDC | ETH debt | active% | sell/esc/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 375.669 | 34682.087 | 28530.476 | 0.000 | 0/0/0 |
| incumbent_stateful_best |  |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 355.692 | 32807.438 | 27013.323 | 0.000 | 0/0/0 |
| plr_sell0.10_esc0.05_max0.30_rebuy0.50 | 0.100 | 0.050 | 0.300 | 0.500 | 198.377 | 24787.245 | 83.854 | 59.860 | 2.117 | 1.576 | 160.214 | 21210.413 | 16441.846 | 29.385 | 13/149/661 |
| plr_sell0.05_esc0.10_max0.40_rebuy1.00 | 0.050 | 0.100 | 0.400 | 1.000 | 220.243 | 27519.374 | 83.708 | 63.979 | 2.107 | 1.691 | 133.492 | 31996.795 | 21157.192 | 18.654 | 63/19/63 |
| plr_sell0.10_esc0.10_max0.30_rebuy1.00 | 0.100 | 0.100 | 0.300 | 1.000 | 211.114 | 26378.693 | 83.520 | 63.582 | 2.103 | 1.698 | 162.252 | 22881.398 | 16776.115 | 18.670 | 63/12/64 |
| plr_sell0.05_esc0.10_max0.30_rebuy1.00 | 0.050 | 0.100 | 0.300 | 1.000 | 210.798 | 26339.258 | 83.690 | 63.659 | 2.102 | 1.689 | 163.921 | 22805.994 | 16948.653 | 18.652 | 62/12/63 |
| plr_sell0.10_esc0.05_max0.40_rebuy1.00 | 0.100 | 0.050 | 0.400 | 1.000 | 214.947 | 26857.662 | 83.464 | 63.912 | 2.102 | 1.702 | 132.462 | 31300.534 | 20993.936 | 18.672 | 64/30/64 |
| plr_sell0.10_esc0.10_max0.40_rebuy1.00 | 0.100 | 0.100 | 0.400 | 1.000 | 214.794 | 26838.461 | 83.494 | 63.840 | 2.101 | 1.701 | 129.799 | 31192.036 | 20571.981 | 18.677 | 64/15/64 |
| plr_sell0.10_esc0.05_max0.30_rebuy1.00 | 0.100 | 0.050 | 0.300 | 1.000 | 208.887 | 26100.468 | 83.470 | 63.585 | 2.100 | 1.660 | 162.902 | 22352.071 | 16606.193 | 18.670 | 63/21/64 |
| plr_sell0.05_esc0.05_max0.30_rebuy1.00 | 0.050 | 0.050 | 0.300 | 1.000 | 208.665 | 26072.723 | 83.667 | 63.976 | 2.099 | 1.671 | 164.468 | 22045.571 | 16523.161 | 18.663 | 62/25/63 |
| plr_sell0.05_esc0.05_max0.40_rebuy1.00 | 0.050 | 0.050 | 0.400 | 1.000 | 212.620 | 26566.814 | 83.686 | 64.206 | 2.098 | 1.692 | 134.025 | 30753.901 | 20933.457 | 18.654 | 63/34/63 |
| plr_sell0.05_esc0.10_max0.40_rebuy0.50 | 0.050 | 0.100 | 0.400 | 0.500 | 185.533 | 23182.303 | 83.810 | 60.374 | 2.092 | 1.515 | 127.968 | 23580.302 | 16387.618 | 33.770 | 12/236/673 |

## Ranking By Max Drawdown

| scenario | sell | esc | max | rebuy | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | SOL | USDC | ETH debt | active% | sell/esc/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 375.669 | 34682.087 | 28530.476 | 0.000 | 0/0/0 |
| incumbent_stateful_best |  |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 355.692 | 32807.438 | 27013.323 | 0.000 | 0/0/0 |
| plr_sell0.10_esc0.05_max0.40_rebuy1.00 | 0.100 | 0.050 | 0.400 | 1.000 | 214.947 | 26857.662 | 83.464 | 63.912 | 2.102 | 1.702 | 132.462 | 31300.534 | 20993.936 | 18.672 | 64/30/64 |
| plr_sell0.10_esc0.05_max0.30_rebuy1.00 | 0.100 | 0.050 | 0.300 | 1.000 | 208.887 | 26100.468 | 83.470 | 63.585 | 2.100 | 1.660 | 162.902 | 22352.071 | 16606.193 | 18.670 | 63/21/64 |
| plr_sell0.10_esc0.10_max0.40_rebuy1.00 | 0.100 | 0.100 | 0.400 | 1.000 | 214.794 | 26838.461 | 83.494 | 63.840 | 2.101 | 1.701 | 129.799 | 31192.036 | 20571.981 | 18.677 | 64/15/64 |
| plr_sell0.10_esc0.10_max0.30_rebuy1.00 | 0.100 | 0.100 | 0.300 | 1.000 | 211.114 | 26378.693 | 83.520 | 63.582 | 2.103 | 1.698 | 162.252 | 22881.398 | 16776.115 | 18.670 | 63/12/64 |
| plr_sell0.15_esc0.05_max0.40_rebuy1.00 | 0.150 | 0.050 | 0.400 | 1.000 | 133.637 | 16697.901 | 83.538 | 73.802 | 1.998 | 1.261 | 244.971 | 846.750 | 14758.003 | 9.195 | 38/18/38 |
| plr_sell0.15_esc0.10_max0.40_rebuy1.00 | 0.150 | 0.100 | 0.400 | 1.000 | 133.784 | 16716.284 | 83.550 | 73.799 | 1.999 | 1.261 | 245.206 | 851.068 | 14773.254 | 9.195 | 38/10/38 |
| plr_sell0.15_esc0.05_max0.30_rebuy1.00 | 0.150 | 0.050 | 0.300 | 1.000 | 138.347 | 17286.518 | 83.561 | 73.536 | 2.009 | 1.264 | 250.612 | 1165.711 | 15193.186 | 9.195 | 38/12/38 |
| plr_sell0.15_esc0.10_max0.30_rebuy1.00 | 0.150 | 0.100 | 0.300 | 1.000 | 138.801 | 17343.205 | 83.575 | 73.530 | 2.010 | 1.264 | 251.372 | 1175.482 | 15241.259 | 9.195 | 38/6/38 |
| plr_sell0.05_esc0.05_max0.40_rebuy0.50 | 0.050 | 0.050 | 0.400 | 0.500 | 172.336 | 21533.445 | 83.636 | 63.047 | 2.064 | 1.658 | 131.992 | 23400.586 | 18359.572 | 33.572 | 11/375/566 |
| plr_sell0.10_esc0.05_max0.40_rebuy0.50 | 0.100 | 0.050 | 0.400 | 0.500 | 174.681 | 21826.350 | 83.647 | 62.338 | 2.073 | 1.702 | 134.851 | 18756.612 | 13779.864 | 29.009 | 12/267/548 |

## Ranking By 2024+ Peak-To-Trough

| scenario | sell | esc | max | rebuy | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | SOL | USDC | ETH debt | active% | sell/esc/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plr_sell0.10_esc0.05_max0.30_rebuy0.50 | 0.100 | 0.050 | 0.300 | 0.500 | 198.377 | 24787.245 | 83.854 | 59.860 | 2.117 | 1.576 | 160.214 | 21210.413 | 16441.846 | 29.385 | 13/149/661 |
| plr_sell0.05_esc0.10_max0.40_rebuy0.50 | 0.050 | 0.100 | 0.400 | 0.500 | 185.533 | 23182.303 | 83.810 | 60.374 | 2.092 | 1.515 | 127.968 | 23580.302 | 16387.618 | 33.770 | 12/236/673 |
| plr_sell0.15_esc0.05_max0.40_rebuy0.50 | 0.150 | 0.050 | 0.400 | 0.500 | 177.321 | 22156.229 | 83.718 | 61.698 | 2.084 | 1.476 | 130.648 | 24091.987 | 18260.231 | 28.627 | 12/181/572 |
| plr_sell0.10_esc0.10_max0.40_rebuy0.50 | 0.100 | 0.100 | 0.400 | 0.500 | 174.150 | 21760.043 | 83.653 | 61.867 | 2.075 | 1.413 | 135.896 | 18666.500 | 13886.720 | 31.930 | 12/166/610 |
| plr_sell0.10_esc0.05_max0.40_rebuy0.50 | 0.100 | 0.050 | 0.400 | 0.500 | 174.681 | 21826.350 | 83.647 | 62.338 | 2.073 | 1.702 | 134.851 | 18756.612 | 13779.864 | 29.009 | 12/267/548 |
| plr_sell0.05_esc0.10_max0.30_rebuy0.50 | 0.050 | 0.100 | 0.300 | 0.500 | 181.060 | 22623.420 | 83.648 | 62.560 | 2.088 | 1.586 | 158.036 | 14878.958 | 12002.196 | 30.934 | 12/160/652 |
| plr_sell0.05_esc0.05_max0.40_rebuy0.50 | 0.050 | 0.050 | 0.400 | 0.500 | 172.336 | 21533.445 | 83.636 | 63.047 | 2.064 | 1.658 | 131.992 | 23400.586 | 18359.572 | 33.572 | 11/375/566 |
| plr_sell0.15_esc0.05_max0.30_rebuy0.50 | 0.150 | 0.050 | 0.300 | 0.500 | 172.777 | 21588.434 | 83.736 | 63.171 | 2.079 | 1.459 | 159.341 | 13780.045 | 12101.268 | 26.001 | 13/99/629 |
| plr_sell0.05_esc0.05_max0.30_rebuy0.50 | 0.050 | 0.050 | 0.300 | 0.500 | 172.752 | 21585.411 | 83.826 | 63.330 | 2.077 | 1.476 | 158.138 | 13835.976 | 12009.903 | 28.027 | 13/99/666 |
| plr_sell0.15_esc0.10_max0.40_rebuy0.50 | 0.150 | 0.100 | 0.400 | 0.500 | 177.607 | 22191.991 | 83.735 | 63.384 | 2.087 | 1.449 | 137.809 | 24105.887 | 19133.075 | 27.442 | 14/76/616 |
| plr_sell0.15_esc0.10_max0.30_rebuy0.50 | 0.150 | 0.100 | 0.300 | 0.500 | 176.268 | 22024.707 | 83.752 | 63.580 | 2.085 | 1.443 | 165.085 | 13934.835 | 12537.504 | 24.921 | 13/34/572 |
| plr_sell0.10_esc0.10_max0.30_rebuy1.00 | 0.100 | 0.100 | 0.300 | 1.000 | 211.114 | 26378.693 | 83.520 | 63.582 | 2.103 | 1.698 | 162.252 | 22881.398 | 16776.115 | 18.670 | 63/12/64 |

## Ranking By Final SOL-Equivalent

| scenario | sell | esc | max | rebuy | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | SOL | USDC | ETH debt | active% | sell/esc/rebuy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_reserve |  |  |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 375.669 | 34682.087 | 28530.476 | 0.000 | 0/0/0 |
| incumbent_stateful_best |  |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 355.692 | 32807.438 | 27013.323 | 0.000 | 0/0/0 |
| plr_sell0.05_esc0.10_max0.40_rebuy1.00 | 0.050 | 0.100 | 0.400 | 1.000 | 220.243 | 27519.374 | 83.708 | 63.979 | 2.107 | 1.691 | 133.492 | 31996.795 | 21157.192 | 18.654 | 63/19/63 |
| plr_sell0.10_esc0.05_max0.40_rebuy1.00 | 0.100 | 0.050 | 0.400 | 1.000 | 214.947 | 26857.662 | 83.464 | 63.912 | 2.102 | 1.702 | 132.462 | 31300.534 | 20993.936 | 18.672 | 64/30/64 |
| plr_sell0.10_esc0.10_max0.40_rebuy1.00 | 0.100 | 0.100 | 0.400 | 1.000 | 214.794 | 26838.461 | 83.494 | 63.840 | 2.101 | 1.701 | 129.799 | 31192.036 | 20571.981 | 18.677 | 64/15/64 |
| plr_sell0.05_esc0.05_max0.40_rebuy1.00 | 0.050 | 0.050 | 0.400 | 1.000 | 212.620 | 26566.814 | 83.686 | 64.206 | 2.098 | 1.692 | 134.025 | 30753.901 | 20933.457 | 18.654 | 63/34/63 |
| plr_sell0.10_esc0.10_max0.30_rebuy1.00 | 0.100 | 0.100 | 0.300 | 1.000 | 211.114 | 26378.693 | 83.520 | 63.582 | 2.103 | 1.698 | 162.252 | 22881.398 | 16776.115 | 18.670 | 63/12/64 |
| plr_sell0.05_esc0.10_max0.30_rebuy1.00 | 0.050 | 0.100 | 0.300 | 1.000 | 210.798 | 26339.258 | 83.690 | 63.659 | 2.102 | 1.689 | 163.921 | 22805.994 | 16948.653 | 18.652 | 62/12/63 |
| plr_sell0.10_esc0.05_max0.30_rebuy1.00 | 0.100 | 0.050 | 0.300 | 1.000 | 208.887 | 26100.468 | 83.470 | 63.585 | 2.100 | 1.660 | 162.902 | 22352.071 | 16606.193 | 18.670 | 63/21/64 |
| plr_sell0.05_esc0.05_max0.30_rebuy1.00 | 0.050 | 0.050 | 0.300 | 1.000 | 208.665 | 26072.723 | 83.667 | 63.976 | 2.099 | 1.671 | 164.468 | 22045.571 | 16523.161 | 18.663 | 62/25/63 |
| plr_sell0.10_esc0.05_max0.30_rebuy0.50 | 0.100 | 0.050 | 0.300 | 0.500 | 198.377 | 24787.245 | 83.854 | 59.860 | 2.117 | 1.576 | 160.214 | 21210.413 | 16441.846 | 29.385 | 13/149/661 |
| plr_sell0.05_esc0.10_max0.40_rebuy0.50 | 0.050 | 0.100 | 0.400 | 0.500 | 185.533 | 23182.303 | 83.810 | 60.374 | 2.092 | 1.515 | 127.968 | 23580.302 | 16387.618 | 33.770 | 12/236/673 |

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | Reserve sell/esc/rebuy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 no reserve | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0/0 |
| Best Sortino | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0/0 |
| Best Max DD | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0/0 |
| Best 2024 DD | plr_sell0.10_esc0.05_max0.30_rebuy0.50 | 198.377 | 24787.245 | 83.854 | 59.860 | 2.117 | 1.576 | 13/149/661 |
| Best SOL-eq | v2_best_no_reserve | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0/0/0 |

## Forensic Notes

Decision: no promotion. The current v2 no-reserve control remains best by final SOL-equivalent, final USD value, Sortino, and full-period max drawdown.

This is still an important directional result. Unlike the weekly-bearish reserve sweep, profit-lock reserve does improve the late-cycle 2024+ drawdown. The best 2024+ drawdown candidate, `plr_sell0.10_esc0.05_max0.30_rebuy0.50`, cuts the 2024+ peak-to-trough from `64.84%` to `59.86%`. That is the first tested SOL-reserve variant that actually attacks the red-mark problem in the intended window.

The cost is too high. The same candidate finishes at only `198.38` SOL-equivalent versus `424.90` for the no-reserve control, lowers Sortino from `2.290` to `2.117`, and worsens full-period max drawdown from `81.34%` to `83.85%`. That is not acceptable for the stated objective of maximizing long-term SOL while controlling drawdown.

The event log shows the main failure mode: churn. The best 2024+ drawdown row fires `13` initial reserve sells, `149` escalations, and `661` rebuys. The rule is behaving more like a high-frequency flip-flop than a stateful reserve regime. In many places it sells and rebuys repeatedly during the same broad trend zone. That burns compounding and leaves the strategy underexposed during recoveries.

The best final SOL-equivalent reserve row, `plr_sell0.05_esc0.10_max0.40_rebuy1.00`, preserves more upside than the best drawdown row but still ends at only `220.24` SOL-equivalent and does not materially beat the control on drawdown. It is also not promotable.

Conclusion: the entry direction is better than weekly-bearish reserve, but the state machine is too loose. Profit-lock reserve should not be allowed to repeatedly enter and exit on every local weakening/recovery cycle.

Next hypothesis:

- make profit-lock reserve stateful by high-watermark episode;
- allow only one initial reserve sale per episode;
- allow at most one escalation per episode;
- require a multi-day cooldown before rebuy;
- require stronger recovery before rebuy, such as 1w green plus 3 or 4 green base votes;
- optionally only reset the episode after a fresh portfolio high.

In plain terms: keep the idea, but turn it from a trading rule into a slow reserve policy.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/profit_lock_reserve_sweep_20260530_135027/comparison.csv`
- Top-by-drawdown CSV: `reports/sol_supertrend_3y/profit_lock_reserve_sweep_20260530_135027/top_by_drawdown.csv`
- Top-by-2024-drawdown CSV: `reports/sol_supertrend_3y/profit_lock_reserve_sweep_20260530_135027/top_by_2024_drawdown.csv`
- Top-by-SOL-equivalent CSV: `reports/sol_supertrend_3y/profit_lock_reserve_sweep_20260530_135027/top_by_sol_equiv.csv`
- Summary JSON: `reports/sol_supertrend_3y/profit_lock_reserve_sweep_20260530_135027/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/profit_lock_reserve_sweep_20260530_135027`
