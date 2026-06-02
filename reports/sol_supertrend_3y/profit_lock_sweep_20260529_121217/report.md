# Profit-Lock Hedge Floor Sweep

Run date: 2026-05-29
Window: 2021-01-01 to 2025-12-31, 43,787 hourly bars
Initial account: 100 SOL collateral
Fixed settings: current UI defaults, `crisis_partial_fill_budget_pct = 0.00`, `profit_lock_metric = portfolio`, `profit_lock_min_gain_pct = 0.25`, `profit_lock_max_green = 3`
Sweep axis: `profit_lock_hedge_floor = 0.25, 0.35, 0.50` and `profit_lock_drawdown_threshold = 0.05, 0.10, 0.15`

## Executive Summary

Profit lock produced a new best configuration, but it did not solve the red-mark giveback.

The best run was `floor_0.35_dd_0.05`:

| Metric | Incumbent | `floor_0.35_dd_0.05` |
|---|---:|---:|
| Final portfolio value | $47,674 | $48,971 |
| Final SOL-equivalent | 381.55 SOL | 391.92 SOL |
| Max drawdown | 83.61% | 82.90% |
| Sortino | 2.227 | 2.249 |
| Min HF | 1.553 | 1.573 |
| Final LTV | 35.53% | 35.02% |
| Final SOL collateral | 345.89 SOL | 347.55 SOL |
| 2025 giveback | -64.98% | -64.92% |

This passes the current promotion rule: better final SOL-equivalent, better Sortino, better max drawdown, and min HF above 1.50.

But the red-mark problem remains. The 2025 giveback barely changed. Profit lock seems to improve overall path quality and compounding, not materially protect the marked tops yet.

## Result Table

| Scenario | Floor | Drawdown Trigger | Final SOL-equiv | Max DD | Sortino | 2021 Giveback | 2025 Giveback |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | off | off | 381.55 SOL | 83.61% | 2.227 | -56.43% | -64.98% |
| `floor_0.25_dd_0.05` | 0.25 | 5% | 378.60 SOL | 83.54% | 2.238 | -55.19% | -64.87% |
| `floor_0.25_dd_0.10` | 0.25 | 10% | 368.13 SOL | 83.88% | 2.225 | -56.02% | -64.94% |
| `floor_0.25_dd_0.15` | 0.25 | 15% | 382.09 SOL | 83.63% | 2.231 | -56.32% | -64.98% |
| `floor_0.35_dd_0.05` | 0.35 | 5% | 391.92 SOL | 82.90% | 2.249 | -55.73% | -64.92% |
| `floor_0.35_dd_0.10` | 0.35 | 10% | 365.62 SOL | 84.16% | 2.226 | -56.98% | -65.00% |
| `floor_0.35_dd_0.15` | 0.35 | 15% | 369.39 SOL | 84.70% | 2.222 | -57.40% | -65.12% |
| `floor_0.50_dd_0.05` | 0.50 | 5% | 361.06 SOL | 84.49% | 2.221 | -57.59% | -65.45% |
| `floor_0.50_dd_0.10` | 0.50 | 10% | 346.70 SOL | 84.33% | 2.203 | -57.28% | -65.07% |
| `floor_0.50_dd_0.15` | 0.50 | 15% | 294.70 SOL | 87.00% | 2.165 | -59.57% | -65.18% |

## Interpretation

The best profit-lock setting is surprisingly modest:

- floor: 35% ETH hedge against SOL collateral value,
- drawdown trigger: 5% from trailing portfolio high,
- prior gain requirement: 25%,
- trend weakening: green votes <= 3 or 1d bearish.

This helped the account compound slightly better. It likely caught some smaller rollovers and avoided a few bad unhedged paths. However, it did not meaningfully increase hedge exposure at the major 2025 peak: peak hedge ratio remained about 7.3%, the same as the incumbent.

So this is a better candidate configuration, but not a complete fix for the marked givebacks.

## Promotion Decision

Promote `floor_0.35_dd_0.05` as the new current best candidate.

Promotion rationale:

- higher final SOL-equivalent,
- higher final USD value,
- higher Sortino,
- lower max drawdown,
- slightly higher min HF,
- no liquidations.

Important caveat:

This should be promoted as an incremental improvement, not as the final answer to peak giveback. The red marks remain unsolved.

## Next Hypotheses

1. Test profit lock using a more sensitive trigger metric.
   The portfolio trigger improved overall path quality but did not catch the 2025 peak. Try `profit_lock_metric = sol_equivalent` and `both`.

2. Add a trailing-high proximity rule.
   Instead of waiting for drawdown from peak, activate when the account is near a large trailing high and short-term Supertrend weakens. This may catch tops earlier.

3. Test a short-term drawdown-speed trigger.
   The red marks may require “fast damage” detection over 1-7 days rather than slow trailing drawdown logic.

4. Keep partial crisis fill off for now.
   Its budget sweeps did not beat the incumbent. Profit lock should be tested independently before recombining with partial fill.

## Artifacts

Output directory:

`reports/sol_supertrend_3y/profit_lock_sweep_20260529_121217/`

Files:

- `profit_lock_sweep_results.csv`
- `summary.json`
- per-scenario `history.csv`
- per-scenario `strategy_events.csv`
- `report.md`
