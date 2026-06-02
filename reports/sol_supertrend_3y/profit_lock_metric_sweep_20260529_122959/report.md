# Profit-Lock Metric Sweep

Run date: 2026-05-29
Window: 2021-01-01 to 2025-12-31, 43,787 hourly bars
Initial account: 100 SOL collateral
Fixed settings: current best config, `profit_lock_hedge_floor = 0.35`, `profit_lock_min_gain_pct = 0.25`, `profit_lock_max_green = 3`, `crisis_partial_fill_budget_pct = 0.00`
Sweep axis: `profit_lock_metric = portfolio, sol_equivalent, both` and drawdown triggers `0.03, 0.05, 0.08`

## Executive Summary

No promotion.

The current best configuration remains `portfolio` metric with a 5% profit-lock drawdown trigger. None of the metric variants improved the current best on final SOL-equivalent, Sortino, and max drawdown.

| Scenario | Metric | Trigger | Final SOL-equiv | Max DD | Sortino | 2025 Giveback |
|---|---|---:|---:|---:|---:|---:|
| `current_best` | portfolio | 5% | 391.92 SOL | 82.90% | 2.249 | -64.92% |
| `portfolio_dd_0.03` | portfolio | 3% | 369.11 SOL | 83.56% | 2.235 | -64.84% |
| `portfolio_dd_0.08` | portfolio | 8% | 385.85 SOL | 83.05% | 2.244 | -64.95% |
| `sol_equivalent_dd_0.03` | SOL-equiv | 3% | 348.41 SOL | 84.33% | 2.205 | -64.82% |
| `sol_equivalent_dd_0.05` | SOL-equiv | 5% | 344.27 SOL | 84.33% | 2.202 | -64.82% |
| `sol_equivalent_dd_0.08` | SOL-equiv | 8% | 382.04 SOL | 83.32% | 2.228 | -64.93% |
| `both_dd_0.03` | both | 3% | 348.41 SOL | 84.33% | 2.205 | -64.82% |
| `both_dd_0.05` | both | 5% | 344.27 SOL | 84.33% | 2.202 | -64.82% |
| `both_dd_0.08` | both | 8% | 381.59 SOL | 83.58% | 2.227 | -64.97% |

The faster 3% triggers slightly improve 2025 giveback, but they cost too much final SOL-equivalent. The SOL-equivalent and both-metric triggers underperform the portfolio trigger.

## Findings

1. **Portfolio remains the best trigger metric.**
   The current best uses portfolio value and still wins this sweep.

2. **SOL-equivalent triggers were too sparse or too late.**
   They produced only 8-15 profit-lock events versus 80 events for the current best, and they did not improve the red-mark problem enough to matter.

3. **Faster drawdown triggers are not enough.**
   The 3% trigger modestly improved 2025 giveback, but final SOL-equivalent fell sharply.

4. **The red marks are still not solved.**
   All 2025 givebacks remain around -64.8% to -65.0%. This mechanism improves the whole path more than it protects the exact peak.

## Promotion Decision

No promotion.

Keep the current best:

- `profit_lock_metric = portfolio`
- `profit_lock_drawdown_threshold = 0.05`
- `profit_lock_hedge_floor = 0.35`

## Interpretation

Changing the profit-lock metric does not solve the main issue. The strategy still reaches the 2025 peak with only about a 7.3% hedge ratio, so the profit-lock logic is not meaningfully active at the actual marked top.

This suggests the next experiment should not be another trailing drawdown threshold. We likely need a rule that acts while the account is still near the high, not after it has already drawn down.

## Next Hypothesis

Test a **near-high weakening trigger**:

- portfolio is within a configured distance of its trailing high, for example within 3-5%,
- portfolio has gained at least 25% from initial value,
- short-term trend weakens, for example green votes <= 3 or 1d bearish,
- impose a modest hedge floor.

That differs from the current profit-lock rule because it does not require a drawdown from the high first. It would attempt to hedge earlier around distribution zones.

## Artifacts

Output directory:

`reports/sol_supertrend_3y/profit_lock_metric_sweep_20260529_122959/`

Files:

- `profit_lock_metric_sweep_results.csv`
- `summary.json`
- per-scenario `history.csv`
- per-scenario `strategy_events.csv`
- `report.md`
