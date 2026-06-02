# Partial Crisis Fill HF 2.50 Report

Run date: 2026-05-29
Window: 2021-01-01 to 2025-12-31, 43,787 hourly bars
Initial account: 100 SOL collateral
Configuration: UI best-in-class defaults plus `partial_fill_min_hf = 2.50`.

## Executive Summary

The stricter partial-fill HF floor helped the safety profile versus uncapped partial fill, but it did not rescue the idea.

Compared with the uncapped partial-fill run, `partial_fill_min_hf = 2.50` reduced partial-fill events from 335 to 180, improved minimum HF from 1.064 to 1.182, cut final ETH debt from $27,561 to $17,347, and lowered final LTV from 65.56% to 55.31%.

But compared with the prior `min_hf_2.00` baseline, the result is still materially worse:

| Metric | Prior `min_hf_2.00` | Partial fill uncapped | Partial fill HF 2.50 |
|---|---:|---:|---:|
| Final portfolio value | $47,674 | $14,476 | $14,016 |
| Final SOL-equivalent | 381.55 SOL | 115.85 SOL | 112.18 SOL |
| Total return | 30,773.05% | 9,274.21% | 8,976.72% |
| Max drawdown | 83.61% | 85.69% | 87.98% |
| Sortino | 2.227 | 1.948 | 1.932 |
| Min HF | 1.553 | 1.064 | 1.182 |
| Final LTV | 35.53% | 65.56% | 55.31% |
| Partial-fill events | 0 | 335 | 180 |

The key lesson: a higher partial-fill HF floor reduces the debt ratchet, but the mechanism still over-hedges into a persistent crisis regime. The problem is not only sizing; it is repeated permission to add hedge while crisis remains active for most of the test.

## Position Reconciliation

Final mark-to-market portfolio:

| Item | Amount | Value |
|---|---:|---:|
| SOL collateral | 251.0059 SOL | $31,363.19 |
| USDC collateral | 0.00 USDC | $0.00 |
| ETH debt | 5.8478 ETH | -$17,346.91 |
| USDC debt | 0.00 USDC | $0.00 |
| Net portfolio value |  | $14,016.28 |
| SOL-equivalent |  | 112.18 SOL |

The accounting adds up:

`$31,363.19 - $17,346.91 = $14,016.28`

This still beats the 100 SOL buy-and-hold final USD value of $12,495, but only modestly, and with a much larger final debt load.

## Red-Mark Windows

| Window | Prior `min_hf_2.00` | Partial fill uncapped | Partial fill HF 2.50 |
|---|---:|---:|---:|
| 2021 top giveback | -56.43% | -47.35% | -54.90% |
| 2025 top giveback | -64.98% | -52.31% | -51.52% |

The stricter HF floor retained most of the 2025 improvement but gave back much of the 2021 improvement.

### 2021

The 2021 peak-to-trough path improved only slightly versus the prior baseline:

- Prior giveback: -56.43%
- HF 2.50 giveback: -54.90%
- Hedge at 2021 peak: 7.4% of SOL collateral value
- Hedge near 2022 trough: 108.5%
- HF near trough: 1.775

This confirms that partial fill does not solve the top-detection problem. By the time crisis filling becomes active, a large part of the damage is already done.

### 2025

The 2025 drawdown window improved:

- Prior giveback: -64.98%
- HF 2.50 giveback: -51.52%
- Hedge at 2025 peak: 36.9% of SOL collateral value
- Hedge near trough: 61.4%
- HF near trough: 2.500

This is the strongest argument for the rule. It forced protection into the 2025 drawdown while respecting the stricter HF floor.

The problem is the cost: the strategy entered 2025 with a lower peak value than the prior run, meaning earlier partial fills had already dragged performance down before the red mark.

## Diagnosis

The strategy now has three separate problems:

1. **Under-hedged crisis no-action was real.**
   Partial fill fixed that mechanical bug.

2. **Uncapped partial fill was too aggressive.**
   HF 2.50 reduced this, but not enough to beat the prior baseline.

3. **Crisis mode is too persistent for any repeated-fill mechanic.**
   Crisis mode was active for about 85.4% of bars. Any rule that can repeatedly add ETH debt during an 85% active regime becomes a structural short, not a crisis repair.

The current result says we should not simply tune `partial_fill_min_hf` upward and hope. It is treating a state problem as a sizing problem.

## Next Hypothesis

Keep the partial-fill mechanism, but add a per-crisis budget.

The budget should be stateful and reset only when crisis exits. A simple first version:

- track `crisis_partial_fill_added_usd`,
- allow at most `crisis_partial_fill_budget_pct` of current SOL collateral value per crisis episode,
- keep `partial_fill_min_hf = 2.50`,
- leave normal hedge changes untouched.

This is cleaner than only raising HF because it says: crisis repair may add protection, but it cannot keep borrowing indefinitely just because crisis mode remains active.

Suggested first budget sweep:

| Budget | Purpose |
|---|---|
| 0.10 | Very conservative repair |
| 0.25 | Moderate repair |
| 0.50 | Aggressive repair |

Success criteria:

- retain most of the 2025 giveback improvement,
- avoid final SOL-equivalent collapse,
- keep min HF above 1.50,
- reduce partial-fill events from hundreds to a small number per crisis episode.

## Artifacts

Output directory:

`reports/sol_supertrend_3y/partial_crisis_fill_hf250_20260529_114347/`

Files:

- `history.csv`
- `strategy_events.csv`
- `summary.json`
- `report.md`
