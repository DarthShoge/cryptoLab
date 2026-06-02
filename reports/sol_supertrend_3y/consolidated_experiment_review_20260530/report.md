# Consolidated SOL Supertrend Experiment Review

Date: `2026-05-30`

Window covered by the major sweeps: `2021-01-01` to `2025-12-31`, 1h Binance SOL/ETH data.

## Why This Consolidation Exists

We have reached the point where the report corpus is itself a research asset. The failed experiments are not noise: they describe which protective mechanisms actually help, which ones merely look attractive in one metric, and which ones damage the central objective of growing SOL-equivalent wealth.

The immediate storage issue is real. The largest sweep folders contain hundreds of megabytes to more than a gigabyte each, mostly from per-scenario `history.csv` and `strategy_events.csv` files. This review consolidates the learnings first. Cleanup should happen as a separate, explicit step after we decide which raw traces still need to be inspectable.

## Current Baseline

The current best-in-class configuration remains the fast-break v2 no-partial-fill candidate:

`reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349/`

Representative scenario:

`v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50`

| Metric | Result |
|---|---:|
| Final SOL-equivalent | 424.90 SOL |
| Final USD value | $53,091.48 |
| Sortino | 2.290 |
| Max drawdown | 81.34% |
| 2024+ peak-to-trough | 64.84% |
| Min HF | 1.598 |

This is the strongest broad result so far, but it has not solved the specific red-mark problem. It improves full-period max drawdown mostly by helping the earlier crash window, while the late-cycle giveback remains almost unchanged.

## What We Learned

### 1. The ETH Hedge Helps, But It Is Not Enough

The core ETH hedge is useful and should stay, but it cannot fully neutralize SOL beta in severe SOL-specific declines. When SOL is the objective asset and ETH is only a correlated hedge, the hedge will sometimes under-protect precisely when SOL underperforms.

Implication: future drawdown controls cannot rely only on "borrow more ETH" as the universal answer. At times, we must either reduce SOL collateral exposure, increase USDC reserve, or detect when the ETH hedge is not tracking the SOL downside.

### 2. Fast-Break V2 Is The Best Current Overlay

The v2 fast-break trigger is useful because it reacts to sharp return plus volatility expansion before slower Supertrend regimes fully confirm. The better rows activate for a small fraction of bars, which keeps the overlay emergency-like rather than turning it into a permanent second strategy.

The important detail is that the best row uses a stricter add-health-factor cap. That avoided the large residual ETH debt problem from earlier fast-break variants.

Implication: keep fast-break v2 as the baseline. Future tests should modify execution around it, not scrap the signal.

### 3. Partial Hedge Filling Failed In Its Generic Form

Generic fast-break partial fill was supposed to fix the all-or-nothing execution problem. Instead, it fired during still-constructive shakeouts and borrowed too often. The result was lower compounding, lower final SOL-equivalent, worse Sortino, and only trivial improvement in the target drawdown window.

The crisis-gated partial-fill variant also failed. Crisis mode was too broad as a permission to add debt.

Implication: partial fill should not be attached directly to a fast-break trigger. It needs a second, stricter damage gate before it can borrow.

### 4. Weekly-Bearish SOL Reserve Is Too Late

Selling SOL only after the weekly Supertrend turns bearish is not a drawdown solution. In the 2022 path, the first sell came after much of the crash damage had already occurred. The strategy then held too little SOL into the recovery, which destroyed the long-term SOL-equivalent objective.

Implication: weekly bearishness is better as a "do not rebuy yet" or "hold reserve" confirmation, not as the first trigger to sell SOL.

### 5. Profit-Lock Reserve Is The Right Family, But The First State Machine Churned

Profit-lock reserve was the first SOL reserve family that materially reduced the late-cycle 2024+ peak-to-trough drawdown. The best drawdown row cut that number from 64.84% to 59.86%.

But the cost was unacceptable. It finished with only 198.38 SOL-equivalent versus 424.90 for the v2 control. The event counts explain why: 13 initial sells, 149 escalations, and 661 rebuys. That is not a reserve policy; it is a churn machine.

Implication: the next reserve experiment should be stateful by high-watermark episode. One initial sale, at most one escalation, slow rebuy, reset only after a fresh high or strong recovery.

### 6. Froth Reserve Was Directionally Sensible But Too Eager To Rebuy

The froth reserve experiment supported the idea that SOL exposure should be reduced during extreme profit conditions. But the tested rebuy policy redeployed too often during ongoing drawdowns, so it did not preserve enough capital when the market continued to fall.

Implication: reserve entry and reserve exit are separate problems. Most failed reserve tests were less about the initial sell and more about premature redeployment.

### 7. Drawdown Containment Reacted Too Late When Bundled

The bundled drawdown-containment sweep failed. Reacting after portfolio drawdown is already visible is too late, especially when the policy also blocks or distorts other mechanisms.

The isolated floor-only drawdown containment result was more interesting: it improved final SOL-equivalent, Sortino, and max drawdown, but violated the min-HF guardrail.

Implication: floor-only containment may deserve a safety-constrained retest, but the blocker-style containment state should not be promoted.

### 8. The Main Failure Mode Has Split Into Two Problems

The 2021-2023 crash and the 2024-2025 red-mark giveback are no longer the same problem.

Fast-break v2 helps the earlier crash window enough to improve full-period max drawdown. It does not materially improve the late-cycle giveback. Reserve-style rules can improve the late-cycle giveback, but the first versions paid too much in lost compounding.

Implication: reports should keep segment metrics. A single max-drawdown number is too blunt now.

## Experiment Family Verdicts

| Family | Verdict | Keep Learning |
|---|---|---|
| Fast-break v2 | Current best | Sharp drop plus volatility expansion is useful; keep as baseline. |
| Stateful profit lock | Former incumbent | Solid lineage; still useful as the base state model. |
| Profit-lock reserve | Failed but important | Early SOL reserve can reduce 2024+ drawdown; current rebuy loop churns. |
| Weekly-bearish reserve | Failed | Weekly bearish trigger is too delayed for first sell. |
| Generic partial fill | Failed | Execution without strong damage confirmation over-trades. |
| Crisis-gated partial fill | Failed | Crisis mode is too broad as an execution permission. |
| Froth reserve | Failed but concept alive | Needs much slower, stronger rebuy confirmation. |
| Drawdown containment bundled | Failed | Late and sticky state worsened the path. |
| Drawdown containment isolated | Promising but unsafe | Floor-only can help, but current winner violates HF guardrail. |
| Partial crisis budget | Failed | Small giveback improvements were not worth SOL-equivalent loss. |

## Current Best Configuration Notes

Keep the current UI/default strategy anchored to the v2 fast-break no-partial-fill candidate until a challenger beats it on the full promotion rule:

- improves final SOL-equivalent;
- improves or preserves Sortino;
- does not materially worsen full-period max drawdown;
- keeps min HF above the safety guardrail;
- materially improves the 2024+ red-mark giveback.

The last condition is the open problem. We should not promote a configuration that only improves final SOL while leaving the red-mark drawdown untouched if the experiment was explicitly meant to solve that drawdown.

## Next Research Queue

1. Stateful high-watermark profit-lock reserve.

   Enter reserve only after a major profit state and early deterioration. Allow one initial sale and at most one escalation per episode. Rebuy only after strong recovery and a multi-day cooldown. Reset the episode only after a fresh high or clear recovery.

2. Reserve rebuy hysteresis.

   Test rebuy as a slow policy, not an hourly reaction. Candidate gates: 1w green, 3d green, 3 or 4 base votes green, portfolio within a small distance of a fresh high, and existing cooldown.

3. SOL-underperformance damage gate.

   Detect when SOL is falling harder than ETH or when the ETH hedge is not protecting the portfolio. This directly addresses the "ETH is close but not identical" problem.

4. Safety-constrained floor-only containment.

   Retest the isolated floor-only containment branch with min-HF constraints high enough to clear the promotion guardrail.

5. Segment-first reporting.

   Every future report should show full-period metrics, 2021-2023 crash metrics, 2024+ peak-to-trough metrics, final native balances, and event counts for reserve/hedge mechanisms.

## Cleanup Plan

Cleanup was performed on `2026-05-31`. Raw trace CSVs matching `history*.csv` and `strategy_events*.csv` were removed, and empty scenario directories left behind by that deletion were pruned.

The safe retention policy is now:

1. Keep every `report.md`, `summary.json`, `comparison.csv`, and top-ranked CSV.
2. Keep `current_best_config.md`, this consolidated review, and the strategy documentation.
3. Regenerate raw traces only for a specific scenario if we need to inspect bar-level behavior again.

The pre-cleanup inventory confirmed that most large folders were almost entirely raw traces. For example, `weekly_bearish_reserve_sweep_20260530_024518` was about `1.0GB`, of which about `1.0GB` was `history.csv` plus `strategy_events.csv`; its compact report artifacts were only about `254KB`. `fast_break_v2_sweep_20260529_163349` showed the same pattern: about `985MB` total, with about `985MB` in raw traces and only about `316KB` in compact report artifacts.

The post-cleanup inventory records the remaining artifact sizes and trace counts.

## Artifacts

- Manifest: `reports/sol_supertrend_3y/consolidated_experiment_review_20260530/artifact_manifest.csv`
- Pre-cleanup inventory: `reports/sol_supertrend_3y/consolidated_experiment_review_20260530/pre_cleanup_inventory.csv`
- Post-cleanup inventory: `reports/sol_supertrend_3y/consolidated_experiment_review_20260530/post_cleanup_inventory.csv`
- Current best notes: `reports/sol_supertrend_3y/current_best_config.md`
- Strategy documentation: `docs/sol-supertrend-strategy-current.md`
