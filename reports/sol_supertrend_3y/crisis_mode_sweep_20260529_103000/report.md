# SOL Supertrend Crisis Mode Sweep

Run date: 2026-05-29
Window: 2021-01-01 to 2025-12-31, 43,787 hourly bars
Initial account: 100 SOL collateral
Scenario count: 25

## Executive Summary

Crisis mode is directionally useful, but not solved.

The old post-reinvestment baseline ended with high headline return but suffered a 95.20% max drawdown and a catastrophic 2022Q4 trough. The new crisis-mode default improved max drawdown to 86.59%, lifted final value from $33,397 to $38,483, and improved final SOL-equivalent from 267.29 SOL to 307.99 SOL. That is a real improvement.

The best tested profile was `min_hf_2.00`, which raised the minimum rebalance health factor from 1.75 to 2.00. It ended at $47,674, 381.55 SOL-equivalent, Sortino 2.227, and max drawdown 83.61%.

Bad news: even the best run still has an 83.61% drawdown. Crisis mode also spends about 86% of bars active in the leading variants and re-enters near the end of 2025, leaving a large open ETH short. So the current rule protects the 2022 waterfall better, but it behaves more like a long-lived risk regime than a crisp crisis overlay.

## What Changed

Compared with the prior post-reinvestment report, this run includes the new drawdown-driven crisis mode:

- stateful crisis entry/exit,
- crisis hedge floors,
- target override via `max(normal_target, crisis_floor)`,
- under-hedged crisis cleanup,
- crisis observability in events/history,
- crisis on/off grid-search comparison.

Focused verification before the report:

`tests/test_sol_supertrend_short_strategy.py tests/test_backtest_app_helpers.py`: 47 passed.

## Benchmark

| Item | Value |
|---|---:|
| SOL initial price | $1.5442 |
| SOL final price | $124.95 |
| 100 SOL buy-and-hold final value | $12,495 |
| SOL buy-and-hold return | 7,991.57% |
| SOL buy-and-hold max drawdown | 96.80% |

Every leading strategy candidate beat SOL buy-and-hold in USD and SOL-equivalent terms, but that alone is not enough. The drawdown is still severe.

## Key Results

| Scenario | Return | Max DD | Sortino | Min HF | Final USD | Final SOL-equiv | Actual SOL | ETH debt USD | 2022Q4 DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_crisis_off` | 21,527.55% | 95.20% | 2.106 | 1.366 | $33,397 | 267.29 | 254.01 | $23,648 | -81.61% |
| `current_crisis_default` | 24,821.08% | 86.59% | 2.199 | 1.297 | $38,483 | 307.99 | 365.67 | $27,771 | -52.78% |
| `min_hf_2.00` | 30,773.05% | 83.61% | 2.227 | 1.553 | $47,674 | 381.55 | 345.89 | $26,268 | -43.12% |
| `heavy_floors` | 25,487.46% | 85.24% | 2.190 | 1.436 | $39,512 | 316.22 | 284.00 | $21,569 | -43.27% |
| `combo_s0.15_p0.10_heavy_floors_g0.05` | 26,096.33% | 84.49% | 2.197 | 1.389 | $40,452 | 323.75 | 289.23 | $22,653 | -43.07% |
| `exit_gap_0.05` | 12,247.84% | 84.81% | 2.013 | 1.111 | $19,068 | 152.60 | 395.79 | $30,387 | -52.68% |

## Interpretation

### 1. Crisis mode fixed the most obvious 2022 failure

The baseline crisis-off run finished 2022Q4 with only $1,426.89 of portfolio value and an intra-quarter drawdown of -81.61%. The current crisis default finished 2022Q4 with $5,004.70 and a -52.78% intra-quarter drawdown.

That is the strongest evidence in favor of the crisis rule. It held more protection through the 2022 waterfall instead of rapidly covering ETH shorts during false rallies.

### 2. The best result came from a higher minimum rebalance HF

`min_hf_2.00` was best by Sortino and max drawdown. Raising the minimum rebalance HF produced:

- better final USD value,
- better SOL-equivalent value,
- lower max drawdown,
- healthier final LTV,
- much better 2022Q4 drawdown.

This is counterintuitive but important. The higher HF setting likely forced the strategy into cleaner, more defensible hedge sizing and debt cleanup behavior. It also produced 43 under-hedged crisis events, which means the rule was frequently constrained by safety logic.

This is not necessarily bad. It may mean the health constraint is acting as a useful governor.

### 3. Lighter crisis floors are bad

`lighter_floors` ended with worse drawdown and much lower return. The heavy-floor variants were consistently more compelling than the light-floor variants.

For this market window, the crisis hedge floor needs to be meaningful. A timid crisis floor does not solve the waterfall problem.

### 4. Tight exit reduces drawdown but can damage the objective

`exit_gap_0.05` had one of the better drawdowns, but final SOL-equivalent collapsed to 152.60 SOL despite holding 395.79 actual SOL collateral. It ended with a large ETH debt burden and high current LTV.

This is a warning: optimizing the exit recovery gap in isolation can make the account look safer on drawdown while producing a worse SOL-relative compounding result.

### 5. Crisis mode is too persistent

Current crisis default:

- crisis active for about 86.0% of bars,
- final state still in crisis,
- last crisis re-entry occurred in October 2025,
- final ETH debt remains large.

That is suspicious. Crisis mode was intended as a protective regime, not the default operating state. The strategy is now much better at surviving drawdowns, but the rule may be too easy to enter or too hard to exit.

### 6. The max drawdown moved, but did not disappear

Baseline max drawdown occurred around 2022-12-29. Crisis default and `min_hf_2.00` improved the 2022 damage, but the deepest drawdown moved into 2023-06.

That suggests the next problem is not only crash protection. It is post-crisis opportunity cost: the account remains defensively hedged while SOL recovery is still early and fragile.

## Forensic Notes

### 2022 Quarter Ends

Baseline crisis-off:

| Quarter | Portfolio | SOL collateral | USDC collateral | ETH debt | HF | Crisis |
|---|---:|---:|---:|---:|---:|---|
| 2022Q1 | $11,691.76 | 100.00 | $0 | $585.24 | 15.73 | no |
| 2022Q2 | $6,937.29 | 183.04 | $3,710.63 | $2,952.88 | 2.70 | no |
| 2022Q3 | $5,913.97 | 183.04 | $0 | $172.22 | 26.50 | no |
| 2022Q4 | $1,426.89 | 183.04 | $1,270.96 | $1,669.01 | 1.51 | no |

Current crisis default:

| Quarter | Portfolio | SOL collateral | USDC collateral | ETH debt | HF | Crisis |
|---|---:|---:|---:|---:|---:|---|
| 2022Q1 | $15,626.92 | 100.00 | $16,906.20 | $13,556.28 | 1.80 | yes |
| 2022Q2 | $12,188.40 | 173.34 | $13,548.77 | $7,212.19 | 2.30 | yes |
| 2022Q3 | $9,724.39 | 188.77 | $11,528.53 | $8,080.84 | 1.87 | yes |
| 2022Q4 | $5,004.70 | 218.50 | $5,534.94 | $2,708.66 | 2.44 | yes |

`min_hf_2.00`:

| Quarter | Portfolio | SOL collateral | USDC collateral | ETH debt | HF | Crisis |
|---|---:|---:|---:|---:|---:|---|
| 2022Q1 | $12,059.47 | 100.00 | $8,842.48 | $9,060.01 | 1.89 | yes |
| 2022Q2 | $9,100.75 | 100.00 | $10,003.03 | $4,278.28 | 2.70 | yes |
| 2022Q3 | $7,758.26 | 115.55 | $8,862.63 | $4,946.56 | 2.20 | yes |
| 2022Q4 | $4,671.20 | 133.75 | $4,995.78 | $1,658.06 | 3.31 | yes |

The crisis default preserved more USD value and accumulated SOL through the crash. `min_hf_2.00` accumulated less SOL during 2022 than the current crisis default, but ended the whole window with better net value and SOL-equivalent.

## Suspicious Behavior

1. Crisis mode is active too often.
   The leading variants are in crisis for roughly 86% of the test window. That is not a clean overlay.

2. The strategy ends with large open ETH debt.
   The best run still ends with $26,268 of ETH debt. Current crisis default ends with $27,771 of ETH debt.

3. Final actual SOL and final SOL-equivalent diverge.
   `exit_gap_0.05` ends with 395.79 actual SOL but only 152.60 SOL-equivalent because the open short burden is large. Actual SOL alone is not enough.

4. Under-hedged crisis events are meaningful.
   `min_hf_2.00` has 43 under-hedged crisis events. This may be acceptable, but it means crisis performance is health-cap constrained.

5. Max drawdown is still unacceptable for a strategy that is meant to avoid sharp drawdowns.
   83.61% is better than 95.20%, but still brutal.

## Best Current Candidate

For now, the best candidate from this sweep is:

`min_hf_2.00`

Key config deltas:

- `min_rebalance_hf = 2.00`
- crisis defaults otherwise unchanged

Why:

- best Sortino,
- best max drawdown,
- best final USD value,
- best final SOL-equivalent value,
- no liquidations,
- healthier final LTV than current crisis default.

Why not blindly adopt it:

- it has many under-hedged crisis events,
- it stays in crisis for most of the test,
- it ends with large ETH debt,
- it still has an 83.61% drawdown.

## Next Hypotheses

1. Add a crisis persistence control.
   Crisis mode needs either stricter re-entry rules, a refractory period after exit, or an additional higher-regime condition so it does not become the default state.

2. Test `min_rebalance_hf` as a first-class search axis.
   The most important result in this sweep was that `2.00` beat the default. Test 1.85, 2.00, 2.15, 2.30 with the same crisis defaults and heavy floors.

3. Add a stale-crisis diagnostic before adding more rules.
   Track crisis duration, number of crisis episodes, average hedge ratio during crisis, and final crisis state. This will tell us whether we are reducing drawdown by simply staying short forever.

4. Investigate post-crisis accumulation separately.
   The crisis rule protects 2022 better, but the 2023 recovery still shows opportunity cost. This connects to the deferred recovery accumulation idea, but it should not be mixed into the crisis rule yet.

5. Re-run quarterly forensic commentary for `min_hf_2.00`.
   The original memo was for the post-reinvestment baseline. The new best candidate deserves the same quarter-by-quarter review, especially across 2022Q2 through 2023Q4.

## Artifacts

Main output directory:

`reports/sol_supertrend_3y/crisis_mode_sweep_20260529_103000/`

Key files:

- `scenario_results.csv`
- `top_by_sortino.csv`
- `top_by_drawdown.csv`
- `summary.json`
- `baseline_crisis_off/history.csv`
- `current_crisis_default/history.csv`
- `min_hf_2.00/history.csv`
- `combo_s0.15_p0.10_heavy_floors_g0.05/history.csv`

Note: `reports/sol_supertrend_3y/crisis_mode_sweep_20260529_101728/` is an incomplete earlier attempt and should not be used for conclusions.
