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
