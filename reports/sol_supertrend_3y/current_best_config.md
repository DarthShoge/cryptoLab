# Current Best Configuration

Last updated: 2026-05-29

## Incumbent Best

The current best known configuration is `stateful_near_0.01_exit_0.05` from:

`reports/sol_supertrend_3y/stateful_profit_lock_sweep_20260529_125323/`

It includes stateful profit lock and keeps the partial crisis fill budget at `0.00`.

The previous best was `floor_0.35_dd_0.05` from:

`reports/sol_supertrend_3y/profit_lock_sweep_20260529_121217/`

The earlier crisis-mode best was `min_hf_2.00` from:

`reports/sol_supertrend_3y/crisis_mode_sweep_20260529_103000/`

The stateful profit-lock challenger improved final SOL-equivalent, final USD value, Sortino, max drawdown, and min HF. It only slightly improved the red-mark giveback, so it is a better incumbent rather than a complete solution.

Latest fast-break v2 challenger:

`reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349/report.md`

Decision: provisionally best fast-break configuration, but not yet a complete answer to the red-mark drawdown problem.

The balanced v2 winner is `v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50`. It improves final SOL-equivalent, Sortino, max drawdown, and min HF versus the incumbent, but the 2025 peak-to-trough drawdown barely changes. Keep the finding active, and test fast-break partial hedge filling next.

## Metrics

| Metric | Value |
|---|---:|
| Final portfolio value | $50,237.89 |
| Final SOL-equivalent | 402.06 SOL |
| Total return | 32,433.28% |
| Max drawdown | 82.39% |
| Sortino | 2.274 |
| Min HF | 1.591 |
| Final LTV | 35.02% |
| Final SOL collateral | 355.6925 SOL |
| Final ETH debt | $27,029.15 |

## Key Parameters

| Parameter | Value |
|---|---:|
| `supertrend_atr_period` | 10 |
| `supertrend_multiplier` | 3.0 |
| `min_rebalance_hf` | 2.00 |
| `rebalance_threshold` | 0.10 |
| `rebalance_cooldown_bars` | 4 |
| `enable_crisis_mode` | true |
| `crisis_sol_drawdown_threshold` | 0.25 |
| `crisis_portfolio_drawdown_threshold` | 0.20 |
| `crisis_hedge_floor_base` | 0.75 |
| `crisis_hedge_floor_3d` | 1.00 |
| `crisis_hedge_floor_3d_1w` | 1.25 |
| `crisis_exit_sol_equiv_recovery_gap` | 0.10 |
| `enable_profit_lock` | true |
| `profit_lock_metric` | portfolio |
| `profit_lock_min_gain_pct` | 0.25 |
| `profit_lock_drawdown_threshold` | 0.05 |
| `profit_lock_near_high_threshold` | 0.01 |
| `profit_lock_stateful` | true |
| `profit_lock_stateful_exit_gap` | 0.05 |
| `profit_lock_hedge_floor` | 0.35 |
| `profit_lock_max_green` | 3 |
| `enable_fast_break_overlay` | true |
| `fast_break_return_lookback_bars` | 24 |
| `fast_break_return_threshold` | -0.08 |
| `fast_break_vol_lookback_bars` | 24 |
| `fast_break_vol_median_bars` | 720 |
| `fast_break_vol_multiplier` | 2.5 |
| `fast_break_hedge_floor` | 1.00 |
| `fast_break_hold_bars` | 72 |
| `fast_break_add_min_hf` | 2.50 |
| `fast_break_decay_enabled` | true |
| `fast_break_decay_floors` | [0.75, 0.35] |
| `enable_full_short_mode` | true |
| `full_short_lower_bound` | 0.75 |
| `full_short_upper_bound` | 1.25 |
| `enable_surplus_usdc_reinvestment` | true |
| `realized_hedge_profit_gate_pct` | 0.10 |
| `max_surplus_reinvestment_pct_of_sol_collateral` | 0.05 |
| `surplus_reinvestment_min_hf` | 2.00 |
| `crisis_partial_fill_budget_pct` | 0.00 |

## Active Challenger

Latest promoted challenger:

`reports/sol_supertrend_3y/stateful_profit_lock_sweep_20260529_125323/report.md`

Decision: promote.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024/2025 Giveback |
|---|---:|---:|---:|---:|---:|
| previous incumbent / `current_best` | 391.92 SOL | 2.249 | 82.90% | 1.573 | -48.56% |
| promoted / `stateful_near_0.01_exit_0.05` | 402.06 SOL | 2.274 | 82.39% | 1.591 | -48.49% |

Latest tested challenger:

`reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349/report.md`

Decision: provisional fast-break promotion, but keep red-mark follow-up open.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024+ peak-to-trough |
|---|---:|---:|---:|---:|---:|
| incumbent / `incumbent_stateful_best` | 402.06 SOL | 2.274 | 82.39% | 1.591 | 64.86% |
| v2 fast-break / `v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50` | 424.90 SOL | 2.290 | 81.34% | 1.598 | 64.84% |

This challenger is numerically better on the broad promotion metrics, but it does not materially reduce the 2025 red-mark drawdown. The forensic finding is that fast-break can become active while the requested hedge add is too large to execute as one atomic rebalance. Next test: fast-break partial fills bounded by health factor and existing cooldown.

Latest tested challenger:

`reports/sol_supertrend_3y/profit_lock_reserve_sweep_20260530_135027/report.md`

Decision: no promotion.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024+ peak-to-trough |
|---|---:|---:|---:|---:|---:|
| v2 no reserve / `v2_best_no_reserve` | 424.90 SOL | 2.290 | 81.34% | 1.598 | 64.84% |
| best 2024 DD / `plr_sell0.10_esc0.05_max0.30_rebuy0.50` | 198.38 SOL | 2.117 | 83.85% | 1.576 | 59.86% |
| best reserve SOL-eq / `plr_sell0.05_esc0.10_max0.40_rebuy1.00` | 220.24 SOL | 2.107 | 83.71% | 1.691 | 63.98% |

Profit-lock reserve is a better trigger family than weekly-bearish reserve because it can reduce the 2024+ red-mark drawdown. However, the first implementation churns too much: the best drawdown row produced 13 initial sells, 149 escalations, and 661 rebuys. Keep disabled. The next version should be a stateful high-watermark reserve with one initial sale, one escalation, and slower recovery/rebuy rules.

Latest tested challenger:

`reports/sol_supertrend_3y/weekly_bearish_reserve_sweep_20260530_024518/report.md`

Decision: no promotion.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024+ peak-to-trough |
|---|---:|---:|---:|---:|---:|
| v2 no reserve / `v2_best_no_reserve` | 424.90 SOL | 2.290 | 81.34% | 1.598 | 64.84% |
| best reserve Sortino / `wbr_sell0.15_max0.30_rebuy0.50` | 157.55 SOL | 2.017 | 82.96% | 1.561 | 74.81% |
| best reserve 2024 DD / `wbr_sell0.15_max0.30_rebuy0.25` | 157.58 SOL | 2.015 | 83.27% | 1.563 | 74.50% |

Weekly-bearish SOL reserve is implemented behind a disabled toggle, but this first sweep was a clear negative result. Selling SOL only after the 1w Supertrend turns bearish is too late; it sells after crash damage is visible and keeps too little SOL exposure into the following recovery. The next SOL de-risk test should enter reserve from profit-lock or near-high conditions, then use weekly bearishness to hold or extend the reserve.

Latest tested challenger:

`reports/sol_supertrend_3y/fast_break_crisis_gated_partial_fill_sweep_20260530_004941/report.md`

Decision: no promotion.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024+ peak-to-trough |
|---|---:|---:|---:|---:|---:|
| v2 no partial fill / `v2_best_no_partial_fill` | 424.90 SOL | 2.290 | 81.34% | 1.598 | 64.84% |
| best crisis-gated partial-fill Sortino / `pf_ret10%_vol2.50_hf1.75_budget0.20` | 244.07 SOL | 2.178 | 82.86% | 1.357 | 67.37% |
| best crisis-gated 2024 DD / `pf_ret10%_vol2.50_hf1.75_budget0.35` | 172.40 SOL | 2.083 | 87.41% | 1.371 | 64.85% |

Crisis-gating partial fill still damaged compounding and did not improve the 2025 red-mark drawdown enough to justify the added ETH debt. Crisis mode is too broad as an execution permission. The next partial-fill test should require crisis overlap plus much weaker trend confirmation, such as green votes at or below 1 or 3d/1w bearish confirmation.

Latest tested challenger:

`reports/sol_supertrend_3y/fast_break_partial_fill_sweep_20260529_194427/report.md`

Decision: no promotion.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024+ peak-to-trough |
|---|---:|---:|---:|---:|---:|
| v2 no partial fill / `v2_best_no_partial_fill` | 424.90 SOL | 2.290 | 81.34% | 1.598 | 64.84% |
| best partial-fill Sortino / `pf_ret8_vol2.50_hf1.50_budget0.75` | 237.89 SOL | 2.176 | 82.37% | 1.383 | 68.12% |
| best partial-fill 2024 DD / `pf_ret10%_vol2.50_hf1.75_budget0.50` | 175.64 SOL | 2.068 | 87.24% | 1.366 | 64.81% |

Generic fast-break partial fill damaged compounding because it fired during still-constructive shakeouts. Keep disabled. The next variant should require crisis/under-hedged crisis or much weaker trend confirmation before executing partial fills.

Previous tested challenger:

`reports/sol_supertrend_3y/froth_reserve_sweep_20260529_132927/report.md`

Decision: no promotion.

| Scenario | Final SOL-equiv | Sortino | Max DD | 2024/2025 Giveback |
|---|---:|---:|---:|---:|
| incumbent / `current_stateful_best` | 402.06 SOL | 2.274 | 82.39% | -48.49% |
| best froth challenger / `tier_2x5_4x5_rebuy50_50` | 379.92 SOL | 2.259 | 83.23% | -48.47% |
| best giveback challenger / `tier_2x5_rebuy35_25` | 372.79 SOL | 2.249 | 83.71% | -45.87% |

The froth-reserve concept is implemented behind a disabled toggle, but this first rebuy policy redeployed too often during ongoing drawdowns. Keep it disabled until rebuy buckets/recovery confirmation are tested.

Previous tested challenger:

`reports/sol_supertrend_3y/drawdown_containment_isolated_sweep_20260529_140056/report.md`

Decision: no promotion yet; safety-constrained follow-up required.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024/2025 Giveback |
|---|---:|---:|---:|---:|---:|
| incumbent / `incumbent_stateful_best` | 402.06 SOL | 2.274 | 82.39% | 1.591 | -48.49% |
| raw challenger / `floor_only_dd0.20_floor1.25` | 434.46 SOL | 2.330 | 80.72% | 1.407 | -48.41% |
| safer challenger / `floor_only_dd0.25_floor0.50` | 373.28 SOL | 2.260 | 83.29% | 1.559 | -48.46% |

The isolated sweep showed that drawdown containment was not inherently broken. Floor-only containment can improve max drawdown, final SOL-equivalent, and Sortino. The blocker policy was the problem. However, the raw winner violates the current min-HF promotion guardrail, so it should not replace the incumbent until a safety-constrained floor-only sweep clears min HF >= 1.50.

Previous tested challenger:

`reports/sol_supertrend_3y/drawdown_containment_sweep_20260529_134448/report.md`

Decision: no promotion.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024/2025 Giveback |
|---|---:|---:|---:|---:|---:|
| incumbent / `incumbent_stateful_best` | 402.06 SOL | 2.274 | 82.39% | 1.591 | -48.49% |
| best drawdown challenger / `contain_dd0.20_floor1.00_exit0.10` | 301.93 SOL | 2.232 | 83.75% | 1.444 | -49.13% |
| best giveback challenger / `contain_dd0.15_floor0.75_exit0.10` | 303.16 SOL | 2.203 | 84.06% | 1.480 | -44.17% |

The simple drawdown containment state is implemented behind a disabled toggle, but the first sweep showed that reacting after portfolio drawdown is visible is too late and too sticky. It worsened max drawdown and materially reduced final SOL-equivalent value.

Previous tested challenger:

`reports/sol_supertrend_3y/profit_lock_near_high_sweep_20260529_124204/report.md`

Decision: no promotion.

| Scenario | Final SOL-equiv | Sortino | Max DD | Min HF | 2024/2025 Giveback |
|---|---:|---:|---:|---:|---:|
| incumbent / `current_best` | 391.92 SOL | 2.249 | 82.90% | 1.573 | -48.56% |
| challenger / `near_0.03_floor_0.25` | 395.53 SOL | 2.245 | 83.20% | 1.557 | -49.14% |

The challenger improves final SOL-equivalent, but it does not clear the promotion rule because Sortino, max drawdown, min HF, and the red-mark giveback all move the wrong way.

Previous challenger family:

- `partial_fill_min_hf = 2.50`
- `crisis_partial_fill_budget_pct` sweep: `0.00`, `0.05`, `0.10`, `0.15`, `0.25`

Promotion rule: do not replace the incumbent unless a challenger improves final SOL-equivalent and Sortino while keeping min HF above 1.50 and avoiding a materially worse max drawdown.

## Previous Challenger Result

Sweep report:

`reports/sol_supertrend_3y/partial_crisis_budget_sweep_20260529_115758/report.md`

Decision: no promotion.

| Scenario | Final SOL-equiv | Sortino | Max DD | 2025 Giveback |
|---|---:|---:|---:|---:|
| incumbent / `budget_0.00` | 381.55 SOL | 2.227 | 83.61% | -64.98% |
| closest challenger / `budget_0.05` | 345.55 SOL | 2.215 | 84.19% | -62.34% |

The 5% budget improves the 2025 giveback slightly, but it does not justify the loss in SOL-equivalent value or the weaker risk metrics.
