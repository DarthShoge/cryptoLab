# Partial Crisis Hedge Fill Report

Run date: 2026-05-29
Window: 2021-01-01 to 2025-12-31, 43,787 hourly bars
Initial account: 100 SOL collateral
Configuration: UI best-in-class defaults, including `min_rebalance_hf = 2.00`, after adding partial crisis hedge fills.

## Executive Summary

Partial crisis hedge fill fixed the immediate no-action problem, but the first full rerun is not a keeper.

The new logic did what it was supposed to do locally: when crisis mode wanted more hedge than could be safely reached in one rebalance, the strategy took the maximum safe ETH short instead of idling. That reduced the red-mark peak-to-trough giveback windows:

| Window | Prior `min_hf_2.00` | Partial-fill rerun |
|---|---:|---:|
| 2021 top giveback | -56.43% | -47.35% |
| 2025 top giveback | -64.98% | -52.31% |

But the global result worsened sharply:

| Metric | Prior `min_hf_2.00` | Partial-fill rerun |
|---|---:|---:|
| Final portfolio value | $47,674 | $14,476 |
| Final SOL-equivalent | 381.55 SOL | 115.85 SOL |
| Total return | 30,773.05% | 9,274.21% |
| Max drawdown | 83.61% | 85.69% |
| Sortino | 2.227 | 1.948 |
| Min HF | 1.553 | 1.064 |
| Final LTV | 35.53% | 65.56% |
| Liquidations | 0 | 0 |

The conclusion is uncomfortable but useful: partial filling is directionally right as a mechanic, but uncapped repeated partial fills can turn crisis mode into a debt ratchet. The strategy protected peaks better, then carried too much ETH debt and gave back long-term SOL-equivalent performance.

## What Changed

Before this rerun, under-hedged crisis mode could get stuck:

- target hedge was above current hedge,
- health rules prevented reaching the full target in one step,
- there was no USDC debt to clean up,
- so the strategy emitted no action.

The new behavior is:

- keep USDC debt cleanup first,
- if cleanup is not available, add the maximum safe ETH short,
- record the event as `under_hedged_crisis_partial_fill`.

Focused and full verification before the report:

`tests/test_sol_supertrend_short_strategy.py tests/test_backtest_app_helpers.py`: 48 passed.
Full suite: 244 passed, 1 skipped.

## Position Reconciliation

Final mark-to-market portfolio:

| Item | Amount | Value |
|---|---:|---:|
| SOL collateral | 336.4288 SOL | $42,036.78 |
| USDC collateral | 0.00 USDC | $0.00 |
| ETH debt | 9.2911 ETH | -$27,561.13 |
| USDC debt | 0.00 USDC | $0.00 |
| Net portfolio value |  | $14,475.65 |
| SOL-equivalent |  | 115.85 SOL |

The accounting adds up:

`$42,036.78 - $27,561.13 = $14,475.65`

This is above 100 SOL buy-and-hold in USD terms at the final price, but only barely in SOL-equivalent terms. Buy-and-hold ended at $12,495; this rerun ended at $14,476 with a much larger risk footprint.

## Peak Giveback Forensics

### 2021 Top

Prior `min_hf_2.00`:

- Peak: 2021-11-06, $22,896.74
- Trough: 2022-02-24, $9,975.40
- Giveback: -56.43%
- Hedge at peak: 11.4% of SOL collateral value

Partial-fill rerun:

- Peak: 2021-11-06, $24,520.93
- Trough: 2022-02-27, $12,910.35
- Giveback: -47.35%
- Hedge at peak: 5.1% of SOL collateral value
- Hedge near trough: 109.2% of SOL collateral value

The improvement came from later crisis hedging becoming more complete. It did not solve the pre-top timing problem: the strategy was still not meaningfully hedged at the top.

### 2025 Top

Prior `min_hf_2.00`:

- Peak: 2025-01-19, $94,659.55
- Trough: 2025-04-07, $33,152.05
- Giveback: -64.98%
- Hedge at peak: 7.3% of SOL collateral value
- No events from 2024-10-01 to 2025-05-01

Partial-fill rerun:

- Peak: 2025-01-19, $79,621.71
- Trough: 2025-02-25, $37,969.72
- Giveback: -52.31%
- Hedge at peak: 17.3% of SOL collateral value
- Hedge near trough: 78.4% of SOL collateral value
- 66 partial-fill events from 2024-10-01 to 2025-05-01

This confirms the previous diagnosis: the old strategy wanted protection but did not act. The new strategy acted, and the local drawdown improved. The problem is that it kept pushing the account toward the health boundary.

## Suspicious Behavior

1. Partial fills fired too often.
   The rerun produced 335 `under_hedged_crisis_partial_fill` events. That is not a one-time repair; it is a recurring leverage loop.

2. Health factor was pushed too low.
   The minimum HF fell from 1.553 to 1.064, with 5,378 bars below HF 1.5. No liquidations occurred, but the account spent too much time near the danger zone.

3. Final debt burden is too high.
   The final position held 336.43 SOL but also 9.29 ETH debt worth $27,561. The net result was only 115.85 SOL-equivalent.

4. Crisis mode remains too persistent.
   Crisis mode was active for about 85.1% of bars. Partial fills make that persistence more dangerous because every under-hedged crisis can add debt.

5. The maximum drawdown moved back to the 2021-to-2023 path.
   The deepest drawdown was from $24,520.93 on 2021-11-06 to $3,508.73 on 2023-06-10.

## Interpretation

The mechanic is useful, but it needs a budget.

The old behavior was too passive: if the target hedge was unreachable, the strategy often did nothing. The new behavior is too eager: it keeps taking every safe inch of borrowing capacity whenever crisis mode remains active and under-hedged.

That creates a bad interaction with persistent crisis state. A state that lasts for years should not be allowed to repeatedly max-fill the hedge. Otherwise the strategy gradually turns a protective overlay into a large residual ETH short.

## Next Hypotheses

1. Keep partial fill, but cap it by health.
   Use a stricter partial-fill HF floor than the general rebalance floor, for example `partial_fill_min_hf = 2.25` or `2.50`. This would preserve the no-idle repair without letting HF grind toward 1.0.

2. Add a per-crisis hedge budget.
   Allow partial fills only up to a maximum added hedge per crisis episode, such as 25% or 50% of SOL collateral value. Reset the budget only after crisis exit.

3. Add a partial-fill throttle.
   Use the existing rebalance cooldown, but also require a larger gap before repeated partial fills, for example only fill if target-current exceeds 20% or 25%.

4. Separate peak protection from crisis repair.
   The 2021 top still needs earlier protection. Partial fill fixes the inability to act after crisis has started; it does not identify tops sooner.

## Artifacts

Output directory:

`reports/sol_supertrend_3y/partial_crisis_fill_20260529_113801/`

Files:

- `history.csv`
- `strategy_events.csv`
- `summary.json`
- `report.md`
