# Partial Crisis Fill Budget 25% Report

Run date: 2026-05-29
Window: 2021-01-01 to 2025-12-31, 43,787 hourly bars
Initial account: 100 SOL collateral
Configuration: UI best-in-class defaults plus `partial_fill_min_hf = 2.50` and `crisis_partial_fill_budget_pct = 0.25`.

## Executive Summary

The per-crisis budget helped, but not enough to beat the prior no-partial-fill baseline.

Compared with the HF-only partial-fill run, the 25% per-crisis budget improved final value, final SOL-equivalent, max drawdown, Sortino, min HF, and final LTV. That means the diagnosis was right: repeated unconstrained partial filling was acting like a debt ratchet.

But the old `min_hf_2.00` run is still better overall. The budgeted partial-fill rule improves the 2025 red-mark drawdown, but the strategy pays for it through lower long-run compounding and a still-large final ETH debt burden.

| Metric | Prior `min_hf_2.00` | HF 2.50 only | HF 2.50 + 25% budget |
|---|---:|---:|---:|
| Final portfolio value | $47,674 | $14,016 | $20,728 |
| Final SOL-equivalent | 381.55 SOL | 112.18 SOL | 165.89 SOL |
| Max drawdown | 83.61% | 87.98% | 85.47% |
| Sortino | 2.227 | 1.932 | 2.042 |
| Min HF | 1.553 | 1.182 | 1.281 |
| Final LTV | 35.53% | 55.31% | 48.16% |
| Final SOL collateral | 345.89 SOL | 251.01 SOL | 320.02 SOL |
| Final ETH debt | $26,268 | $17,347 | $19,258 |

This is progress relative to the failed partial-fill variants, but it is not yet a better strategy.

## Position Reconciliation

Final mark-to-market portfolio:

| Item | Amount | Value |
|---|---:|---:|
| SOL collateral | 320.0182 SOL | $39,986.28 |
| USDC collateral | 0.00 USDC | $0.00 |
| ETH debt | 6.4920 ETH | -$19,257.88 |
| USDC debt | 0.00 USDC | $0.00 |
| Net portfolio value |  | $20,728.40 |
| SOL-equivalent |  | 165.89 SOL |

The accounting adds up:

`$39,986.28 - $19,257.88 = $20,728.40`

The strategy still beats 100 SOL buy-and-hold in final USD value, but the SOL-equivalent result is weak versus the prior no-partial-fill baseline.

## Red-Mark Windows

| Window | Prior `min_hf_2.00` | HF 2.50 only | HF 2.50 + 25% budget |
|---|---:|---:|---:|
| 2021 top giveback | -56.43% | -54.90% | -55.05% |
| 2025 top giveback | -64.98% | -51.52% | -51.27% |

The 25% budget preserves the 2025 improvement but does almost nothing for the 2021 top. That makes sense: the 2021 problem is earlier detection; the 2025 problem was under-hedged crisis execution.

### 2021

- Peak: 2021-11-06, $23,870.04
- Trough: 2022-02-27, $10,730.51
- Giveback: -55.05%
- Hedge at peak: 7.6% of SOL collateral value
- Hedge near trough: 108.5%
- HF near trough: 1.769

The strategy still entered the fall with very little hedge. Partial fills arrived after the move had already started.

### 2025

- Peak: 2025-01-19, $81,175.39
- Trough: 2025-04-07, $39,553.15
- Giveback: -51.27%
- Hedge at peak: 37.2% of SOL collateral value
- Hedge near trough: 54.5%
- HF near trough: 2.727

This is the best evidence for keeping the mechanism in some form. The budgeted rule materially improved the 2025 red-mark drawdown without pushing HF as low as the uncapped version.

## What Improved

1. The budget reduced the debt-ratchet damage.
   Final value improved from $14,016 in the HF-only partial-fill run to $20,728.

2. Final SOL collateral recovered.
   The HF-only run ended with 251.01 SOL. The budgeted run ended with 320.02 SOL.

3. Final LTV improved.
   HF-only ended at 55.31% LTV. The budgeted run ended at 48.16%.

4. 2025 drawdown protection held.
   The budgeted run still cut the 2025 red-mark giveback from -64.98% to -51.27%.

## What Still Fails

1. The old baseline is still better.
   The prior `min_hf_2.00` run ended at $47,674 and 381.55 SOL-equivalent. The budgeted run ended at $20,728 and 165.89 SOL-equivalent.

2. Crisis mode is still too persistent.
   Crisis mode was active for about 84.1% of bars. A partial-fill rule remains dangerous when the parent state is active most of the time.

3. The 2021 top remains unsolved.
   Partial fill cannot protect a top that the system has not identified yet.

4. The minimum HF is still too low.
   Min HF was 1.281, and there were 1,217 bars below HF 1.5. This is better than the failed variants, but still not where the strategy should live.

## Interpretation

The per-crisis budget is a better architecture than HF-only partial fill, but the default 25% budget is probably still too permissive for a state that lasts this long.

There are now two distinct workstreams:

1. **Crisis execution:** keep partial fill, but sweep smaller budgets such as 5%, 10%, and 15%.
2. **Peak protection:** add an earlier profit-lock/rollover mode, because crisis repair is inherently late for 2021-style tops.

The next scientific step is not another single default. It should be a budget sweep.

## Next Experiment

Run a focused sweep:

| Parameter | Values |
|---|---|
| `crisis_partial_fill_budget_pct` | 0.00, 0.05, 0.10, 0.15, 0.25 |
| `partial_fill_min_hf` | 2.50 |
| other settings | current UI defaults |

Success criteria:

- final SOL-equivalent closer to or above the prior 381.55 SOL,
- 2025 giveback meaningfully better than -64.98%,
- min HF above 1.50,
- final LTV below 45%,
- no liquidation events.

## Artifacts

Output directory:

`reports/sol_supertrend_3y/partial_crisis_budget_025_20260529_115101/`

Files:

- `history.csv`
- `strategy_events.csv`
- `summary.json`
- `report.md`
