# Partial Crisis Budget Sweep

Run date: 2026-05-29
Window: 2021-01-01 to 2025-12-31, 43,787 hourly bars
Initial account: 100 SOL collateral
Fixed settings: current UI defaults, `partial_fill_min_hf = 2.50`
Sweep axis: `crisis_partial_fill_budget_pct = 0.00, 0.05, 0.10, 0.15, 0.25`

## Executive Summary

The incumbent still wins.

The best configuration remains the old `min_hf_2.00` / no-partial-fill behavior, represented here by `budget_0.00`. Smaller partial-fill budgets improve the 2025 red-mark drawdown a little, but every nonzero budget gives up final SOL-equivalent value and Sortino versus the incumbent.

| Scenario | Budget | Final USD | Final SOL-equiv | Max DD | Sortino | Min HF | 2025 Giveback |
|---|---:|---:|---:|---:|---:|---:|---:|
| `budget_0.00` | 0% | $47,674 | 381.55 SOL | 83.61% | 2.227 | 1.553 | -64.98% |
| `budget_0.05` | 5% | $43,177 | 345.55 SOL | 84.19% | 2.215 | 1.557 | -62.34% |
| `budget_0.10` | 10% | $30,427 | 243.51 SOL | 88.05% | 2.125 | 1.578 | -60.10% |
| `budget_0.15` | 15% | $35,397 | 283.29 SOL | 85.18% | 2.188 | 1.558 | -56.90% |
| `budget_0.25` | 25% | $20,728 | 165.89 SOL | 85.47% | 2.042 | 1.281 | -51.27% |

The 5% budget is the closest challenger. It improves the 2025 giveback from -64.98% to -62.34%, but costs about 36 SOL-equivalent and slightly worsens max drawdown and Sortino. That is not a good trade yet.

## Current Best Note

The incumbent best configuration has been recorded separately:

`reports/sol_supertrend_3y/current_best_config.md`

Do not promote any partial-fill configuration from this sweep. The current best remains `budget_0.00` / `min_hf_2.00`.

## What The Sweep Says

1. **Partial fill has a cost even at 5%.**
   A small budget triggered 186 partial-fill events and reduced final SOL-equivalent from 381.55 to 345.55 SOL.

2. **Bigger budgets improve 2025 giveback but damage the objective.**
   The 25% budget produced the best 2025 giveback, but final SOL-equivalent collapsed to 165.89 SOL.

3. **The relationship is not monotonic.**
   The 10% budget was worse than 15% on final value and drawdown. That means path dependency is strong; we should not assume simple linear tuning.

4. **This does not solve the 2021 top.**
   2021 giveback stays roughly in the -55% to -57% region across the sweep. Partial crisis fill is late-stage repair, not early peak protection.

5. **Crisis persistence remains the parent problem.**
   Crisis mode is active around 84-86% of bars across these runs. Any repeated crisis-only action is risky while the regime itself is this sticky.

## Promotion Decision

No promotion.

The promotion rule was:

- improve final SOL-equivalent,
- improve or preserve Sortino,
- keep min HF above 1.50,
- avoid materially worse max drawdown.

No nonzero budget passes. `budget_0.05` is closest but fails final SOL-equivalent, Sortino, and max drawdown.

## Interpretation

The partial-fill mechanism is useful diagnostically: it proved that under-hedged no-action states were real. But as a production rule, it is not yet good enough. Even small amounts of repeated partial filling erode SOL-equivalent compounding because the parent crisis state lasts too long.

This points away from more partial-fill tuning and toward a different problem: earlier profit protection before the waterfall. The red marks are not only about inability to hedge once crisis is active. They are about waiting too long to treat a large rally as something worth protecting.

## Next Hypothesis

Pause partial-fill promotion. Keep the code configurable, but leave the best configuration at `crisis_partial_fill_budget_pct = 0.00`.

The next experiment should be a simple profit-lock or rollover mode:

- trigger after portfolio/SOL-equivalent makes a large trailing gain,
- require short-term trend deterioration,
- impose a small hedge floor before crisis mode,
- avoid repeated debt additions by using a fixed floor rather than a repeated partial-fill budget.

That directly targets the 2021 and 2025 red marks. Partial fill only addresses the later “why did we not act once already under-hedged?” symptom.

## Artifacts

Output directory:

`reports/sol_supertrend_3y/partial_crisis_budget_sweep_20260529_115758/`

Files:

- `budget_sweep_results.csv`
- `summary.json`
- per-scenario `history.csv`
- per-scenario `strategy_events.csv`
- `report.md`
