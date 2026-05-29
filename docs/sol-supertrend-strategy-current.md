# SOL Supertrend Kamino Strategy

Last updated: 2026-05-29

This document describes the current SOL Supertrend strategy as implemented in the backtester after the crisis-mode, partial-fill, surplus-reinvestment, profit-lock, froth-reserve, and drawdown-containment experiments.

The strategy is a Kamino-style lending account strategy. It starts from SOL collateral, may borrow ETH to create a short ETH hedge, sells borrowed ETH into USDC collateral, and may use realized hedge profits to buy more SOL. The goal is not simply to maximize USD value; the strategy is judged by USD value, SOL-equivalent value, drawdown, health factor, and final open exposures.

## Current Best Configuration

The current best known configuration is recorded in:

`reports/sol_supertrend_3y/current_best_config.md`

Current promoted candidate:

- `enable_profit_lock = true`
- `profit_lock_metric = portfolio`
- `profit_lock_min_gain_pct = 0.25`
- `profit_lock_drawdown_threshold = 0.05`
- `profit_lock_hedge_floor = 0.35`
- `profit_lock_max_green = 3`
- `enable_crisis_mode = true`
- `min_rebalance_hf = 2.00`
- `crisis_partial_fill_budget_pct = 0.00`

Important: partial crisis fill is implemented and configurable, but the latest promoted configuration keeps its budget at zero because nonzero budgets improved some local drawdowns while hurting SOL-equivalent compounding.

Drawdown containment and froth reserve are also implemented behind disabled toggles. Their first sweeps did not improve the promoted incumbent.

## Portfolio Model

The simulated account has:

- SOL collateral,
- USDC collateral,
- ETH debt,
- USDC debt.

An ETH hedge is modeled as a real Kamino-style short:

1. borrow ETH,
2. sell ETH into USDC,
3. deposit the USDC as collateral.

ETH hedge size is measured as:

`ETH debt value / SOL collateral value`

So a 35% hedge means the ETH debt notional is 35% of the current SOL collateral mark-to-market value.

## Signal Model

The strategy computes SOL Supertrend votes over four timeframes:

- 1h,
- 4h,
- 8h,
- 1d.

The number of green votes controls the normal hedge target:

| Green votes | Normal ETH hedge target |
|---:|---:|
| 4 | 0% |
| 3 | 10% |
| 2 | 25% |
| 1 | 50% |
| 0 | 50% |

The strategy also computes higher-timeframe bearish flags:

- 1d bearish,
- 3d bearish,
- 1w bearish.

Those higher-timeframe flags are used by crisis mode, full-short mode, and profit-lock confirmation.

## Target Resolution

Each bar resolves a single effective ETH hedge target.

The target is built in layers:

1. normal Supertrend hedge ladder,
2. profit-lock hedge floor,
3. optional drawdown-containment hedge floor,
4. crisis hedge floor,
5. full-short mode where applicable.

The effective target generally becomes the maximum protective target from the active layers. The strategy then compares that target with the current ETH short ratio and rebalances if the gap exceeds the rebalance threshold and cooldown permits.

## Normal Hedge Mode

Normal hedge mode is the base strategy.

It uses the four-timeframe SOL Supertrend vote to set the target ETH hedge. More red votes increase the ETH hedge. More green votes reduce it.

This is always present unless overridden by stronger states. It is the default behavior when profit lock, crisis mode, and full-short mode are inactive.

## Profit-Lock Mode

Profit lock is an early protective overlay. It tries to protect large gains before crisis mode would normally activate.

Current promoted settings:

- trigger metric: portfolio value,
- required gain from initial value: 25%,
- trailing drawdown from recent high: 5%,
- near-high trigger: 1%,
- stateful mode: enabled,
- stateful exit gap: 5% from the rolling high with strong trend,
- trend weakening: green votes at or below 3, or 1d bearish,
- hedge floor: 35%.

When active, profit lock raises the target ETH hedge to at least the configured profit-lock floor. Stateful mode keeps that floor active after the first trigger until the selected metric recovers near its rolling high and trend strength returns.

The optional near-high trigger allows profit lock to activate while the selected metric is still within a configured distance of its rolling high. Near-high alone was not strong enough to promote, but stateful near-high profit lock improved the incumbent in the first sweep.

What it improved:

- final USD value,
- final SOL-equivalent value,
- Sortino,
- max drawdown.

What it did not solve:

- the large 2025 red-mark giveback barely improved in the first sweep.

So profit lock is currently a better default, but not the final answer to peak protection.

## Drawdown Containment

Drawdown containment is an experimental defensive state, disabled in the promoted configuration.

It can enter when portfolio value falls a configured percentage below its rolling high. While active it can:

- raise the ETH hedge target to a configured floor,
- block froth-reserve SOL rebuys,
- block surplus USDC reinvestment,
- block bullish USDC releverage.

The initial tested policy exits only after the portfolio recovers near its rolling high and the vote is strongly green.

First sweep result:

`reports/sol_supertrend_3y/drawdown_containment_sweep_20260529_134448/report.md`

Decision: no promotion. The bundled rule made max drawdown worse and reduced final SOL-equivalent value. Follow-up debugging showed the damaging part was not the hedge floor itself; it was blocking surplus reinvestment while containment was active.

Isolated follow-up:

`reports/sol_supertrend_3y/drawdown_containment_isolated_sweep_20260529_140056/report.md`

The isolated floor-only version produced a raw challenger that improved max drawdown, final SOL-equivalent, and Sortino, but its minimum health factor fell below the current promotion guardrail. This keeps drawdown containment in the experimental bucket until a safety-constrained floor-only sweep is run.

## Full-Short Mode

Full-short mode is a separate bearish experiment.

It can enter when:

- full-short mode is enabled,
- all four standard Supertrend votes are red,
- 3d SOL Supertrend is bearish.

While full-short mode is active:

- if all four votes are green, full-short mode exits and target hedge goes to 0%,
- if all four votes are red, target hedge uses the configured full-short bounds,
- if 1w is bearish, the strategy uses the upper full-short bound,
- otherwise it uses the lower full-short bound.

Current promoted bounds:

- lower bound: 75%,
- upper bound: 125%.

Full-short mode does not mean shorting SOL. The strategy never intentionally shorts SOL. It increases ETH debt as the hedge/short instrument.

## Crisis Mode

Crisis mode is a stateful protective regime. It is designed to enter after meaningful damage appears, before or alongside full-short confirmation.

It can enter from either:

1. SOL price drawdown:
   - SOL drawdown from rolling high is at least 25%,
   - 1d SOL Supertrend is bearish,
   - and either 3d is bearish or portfolio drawdown is at least 20%.

2. SOL-equivalent drawdown:
   - portfolio SOL-equivalent value is down at least 25% from its trailing high,
   - and either 3d or 1w SOL Supertrend is bearish.

While active, crisis mode imposes hedge floors:

| Condition | Crisis hedge floor |
|---|---:|
| Base crisis | 75% |
| 3d bearish | 100% |
| 3d and 1w bearish | 125% |

Crisis mode persists until an exit trigger fires:

- 1w SOL Supertrend turns green, or
- SOL-equivalent value recovers close enough to its trailing high.

The current recovery gap is 10%.

Crisis mode does not force an immediate full cover when it exits. It removes the crisis floor, and normal hedge targets plus cooldown govern the unwind.

## Under-Hedged Crisis

Under-hedged crisis means crisis mode wants a larger ETH hedge than the account can safely add under health-factor constraints.

The strategy marks this state explicitly so reports can distinguish:

- a bad signal,
- a capacity-limited hedge,
- or a rule that chose not to act.

Current behavior:

1. Try USDC debt cleanup first.
2. If partial crisis fill budget is available, add a capped ETH short using `partial_fill_min_hf`.
3. If no safe action is available, record under-hedged crisis and wait.

Partial-fill settings:

- `partial_fill_min_hf = 2.50`,
- `crisis_partial_fill_budget_pct` is configurable.

Current promoted configuration sets:

`crisis_partial_fill_budget_pct = 0.00`

Reason: sweeps showed nonzero partial-fill budgets improved the 2025 drawdown window but degraded final SOL-equivalent performance and overall ranking.

## USDC Debt

The strategy can model USDC debt, but current promoted settings disable USDC releverage:

`enable_usdc_releverage = false`

Conceptually:

- USDC debt may be carried during bearish weekly regimes,
- green weekly regimes prioritize USDC debt repayment,
- crisis mode blocks new USDC borrowing to buy SOL.

The current best runs are mostly about ETH hedge mechanics and surplus reinvestment, not active USDC debt expansion.

## Surplus Reinvestment

Surplus reinvestment buys more SOL with eligible realized hedge profits.

It is separate from USDC releverage. It uses existing USDC collateral/profit, not newly borrowed USDC.

Eligibility:

- ETH short exposure must first be reduced to the current target,
- realized hedge PnL must be positive after prior losses,
- spendable hedge profit must exceed a gate.

Current promoted settings:

- realized hedge profit gate: 10% of SOL collateral value,
- buy with 25% of eligible surplus at 3 green votes,
- buy with 50% of eligible surplus at 4 green votes,
- cap each reinvestment at 5% of SOL collateral value,
- projected health factor must remain above 2.00.

This is how the strategy attempts to turn defensive hedge gains into more SOL collateral.

## Rebalance Ordering

The strategy does not perform every possible action on every bar. The rough priority order is:

1. compute normal hedge target,
2. apply profit-lock floor,
3. apply optional drawdown-containment floor,
4. apply crisis floor,
5. respect cooldown unless health is below the minimum rebalance HF,
6. increase/decrease ETH hedge toward target,
7. handle defensive USDC debt repayment when very bearish,
8. clean up USDC debt in green weekly regimes,
9. bullish USDC releverage if enabled and not blocked by containment,
10. froth-reserve rebuy/rotation if enabled and not blocked by containment,
11. surplus USDC reinvestment if eligible and not blocked by containment.

This ordering matters. For example, the strategy should reduce excess ETH short before using USDC to buy more SOL during recovery.

## Observability

The strategy writes key state into history and event logs.

Important fields include:

- `in_crisis_mode`,
- `crisis_hedge_floor`,
- `under_hedged_crisis`,
- `crisis_partial_fill_added_usd`,
- `in_profit_lock_mode`,
- `profit_lock_hedge_floor`,
- `in_drawdown_containment`,
- `drawdown_containment_hedge_floor`,
- `froth_reserve_usdc`,
- `effective_hedge_target`,
- current ETH short ratio,
- health factor,
- realized hedge PnL fields.

These are needed because headline returns alone are misleading. A run can look good in USD but end with too much ETH debt, too little SOL-equivalent value, or a fragile health factor.

## Known Open Problems

1. **Peak giveback remains too high.**
   Profit lock improved the overall best configuration, but it barely improved the 2025 red-mark giveback.

2. **Crisis mode is sticky.**
   Prior reports showed crisis mode active for most bars. This makes any repeated crisis action dangerous.

3. **Partial crisis fill is not promoted.**
   It fixed a real no-action issue, but nonzero budgets hurt SOL-equivalent compounding in sweeps.

4. **The 2021 top remains mostly unsolved.**
   This likely needs earlier peak/rally protection, not more crisis repair.

5. **The current strategy is rule-heavy.**
   Further changes should be tested one mechanism at a time, with current best configuration preserved as the incumbent.

## Current Research Direction

The next promising direction is not more partial crisis fill. It is better peak protection:

- profit-lock metric variants: portfolio vs SOL-equivalent vs both,
- further peak-protection beyond stateful profit lock,
- drawdown-speed circuit breakers,
- rules that hedge earlier without becoming permanently short.

Any challenger should be compared against the current best configuration in `current_best_config.md`, not against an outdated baseline.
