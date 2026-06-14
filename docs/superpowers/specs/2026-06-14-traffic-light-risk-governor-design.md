# Traffic-Light Risk Governor Design

Date: `2026-06-14`

Branch: `research/radical-drawdown-candidates`

## Objective

Build and test radical SOL Supertrend / Kamino strategy variants that use the four Supertrend traffic lights as the central portfolio governor, not just as a hedge input.

The mandate is investor-grade risk control without abandoning the compounding objective:

- Final SOL-equivalent must be greater than `420 SOL`.
- Max drawdown should target `<= 50%`.
- Recovery speed after major drawdowns must improve versus the current benchmark.
- Minimum health factor must remain visible, with `>= 1.50` as the default hard safety gate.

The current best constrained candidate, `hedge_failure_thr0.10_sell0.01`, finishes at `420.951 SOL` but still has `81.157%` max drawdown. It is useful evidence, not a solution.

## Problem Statement

The current model can compound through severe losses and still finish above `420 SOL`, but the path is not investable. A capital allocator will not tolerate `80%+` drawdowns even if the final number is acceptable.

The current architecture is path-specific:

- It can add risk too late in deteriorating regimes.
- It can remain exposed through sharp SOL-specific drawdowns.
- ETH short exposure does not reliably hedge SOL collateral beta.
- Defensive reserves and profit locks reduce drawdown only by starving compounding.
- Existing overlays fight each other because they are bolted on after the base hedge ladder.

The traffic-light idea should become a single governing state machine. Each light state should answer:

- How much SOL beta is allowed?
- How much ETH short exposure is required or allowed?
- Is new USDC leverage allowed?
- Is surplus reinvestment allowed?
- Should realized hedge profits compound into SOL, sit in USDC, repay debt, or buy recovery buckets?
- How does the strategy re-enter after damage?

## Current Benchmark

Use two benchmark rows in every report:

| Benchmark | Final SOL-eq | Max DD | 2024+ DD | Sortino | Min HF |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2_best_no_overlays` | `424.902` | `81.344%` | `64.839%` | `2.290` | `1.598` |
| `hedge_failure_thr0.10_sell0.01` | `420.951` | `81.157%` | `64.781%` | `2.288` | `1.616` |

The new work must beat the drawdown path meaningfully while preserving the `>420 SOL` ending requirement.

## Traffic-Light Model

The four lights are the existing four Supertrend timeframes:

- `1h`
- `4h`
- `8h`
- `1d`

The governing state is the number of green votes, with optional confirmation from volatility, 3d/1w trend, drawdown, and SOL/ETH relative behavior.

| Green votes | Name | Portfolio posture | Default permission model |
| ---: | --- | --- | --- |
| 4 | Green | Compound | Allow SOL accumulation, allow bullish USDC releverage if safety gates pass, unwind excess hedge. |
| 3 | Lime | Participate | Allow surplus SOL reinvestment, modest ETH hedge, no emergency protection unless volatility or drawdown confirms. |
| 2 | Yellow | Defend | No new USDC leverage, keep or raise hedge floor, route some hedge profits to USDC/debt cleanup. |
| 1 | Orange | Preserve | Block new SOL buys except recovery buckets, raise hedge floor, freeze bullish releverage, protect realized gains. |
| 0 | Red | Survival/recovery | No new leverage, maximum allowed defensive behavior, use protected capital only through strict recovery or deep-value buckets. |

This does not mean every red state sells SOL. The core research question is whether the light state can choose between selling SOL, increasing ETH hedge, pausing reinvestment, repaying debt, or saving USDC for recovery.

## Experiment Family A: Traffic-Light Exposure Governor

### Thesis

Most drawdown damage comes from being too exposed when the light stack deteriorates and too slow to re-risk when the lights recover. A first radical model should replace loosely connected overlays with one exposure policy.

### Mechanics

Each light state controls:

- ETH hedge floor.
- USDC releverage permission.
- surplus SOL reinvestment permission.
- crisis accumulation permission.
- maximum incremental debt add per rebalance.

Initial grid:

| Green votes | Hedge floor candidates | Reinvestment | USDC releverage |
| ---: | --- | --- | --- |
| 4 | `0.00`, `0.10` | allowed | allowed if HF target passes |
| 3 | `0.10`, `0.25`, `0.35` | allowed | disabled or reduced |
| 2 | `0.35`, `0.50`, `0.75` | reduced or USDC-routed | disabled |
| 1 | `0.75`, `1.00`, `1.25` | blocked except recovery buckets | disabled |
| 0 | `1.00`, `1.25`, `1.50` | blocked except recovery buckets | disabled |

Health-factor constraints still cap executable hedge size. If a target cannot be safely reached, the run must record under-hedged state explicitly.

### Expected Failure Mode

If hedge floors are too high, recovery will lag because the ETH short remains painful during rebounds. This family needs explicit staged cover rules.

## Experiment Family B: Traffic-Light Recovery Engine

### Thesis

Reducing drawdown is not enough. The strategy must recover quickly while preserving final SOL-equivalent value. The previous reserve experiments failed because they sold or withheld capital without a disciplined path back into SOL.

### Mechanics

The recovery engine controls when protected USDC, reserve USDC, or hedge profit can buy SOL.

Recovery buy triggers should be bucketed, not continuous:

| Trigger type | Candidate values |
| --- | --- |
| Light recovery | `3 green`, `4 green`, `3 green + 3d green`, `4 green + 1w not bearish` |
| Volatility cooldown | `24h vol < 1.25x median`, `72h vol < 1.25x median` |
| Drawdown bucket | buy at `35%`, `45%`, `55%` portfolio drawdown only if lights improve |
| Recovery bucket size | `25%`, `50%`, `100%` of protected/recovery USDC |
| Re-entry cooldown | `72h`, `168h`, `336h` |

Recovery speed metrics:

- bars to recover `50%` of each major drawdown;
- bars to recover prior SOL-equivalent high;
- bars to recover prior USD high;
- final SOL-equivalent after recovery episodes.

### Expected Failure Mode

Strict recovery confirmation may protect the account but miss the V-bottom. Loose confirmation may recreate the existing path-specific drawdown.

## Experiment Family C: Traffic-Light Insurance Book

### Thesis

The strategy currently mixes compounding and protection in one balance sheet. The insurance book separates future realized gains from existing SOL collateral so the model can protect capital without repeatedly selling core SOL.

### Mechanics

When the model is in green/lime states, realized hedge profits can still compound into SOL. As lights deteriorate, a percentage of realized hedge profits and/or surplus USDC is diverted to protected USDC.

Initial grid:

| Light state | Protected routing candidates |
| --- | --- |
| Green | `0%` |
| Lime | `0%`, `10%`, `25%` |
| Yellow | `25%`, `50%` |
| Orange | `50%`, `75%`, `100%` |
| Red | `100%` |

Protected USDC can be used only for:

- health-factor defense;
- debt cleanup;
- recovery bucket buys governed by Family B;
- optional red-state deep-value SOL buys if the recovery engine allows it.

This differs from earlier protected-book tests because allocation is state-dependent and paired with a recovery engine.

### Expected Failure Mode

If routing begins too early, it will starve compounding and miss the `>420 SOL` gate. This family should be tested with small routing percentages first.

## Experiment Family D: Traffic-Light SOL/ETH Hedge-Failure Override

### Thesis

ETH is not a reliable hedge when SOL-specific downside dominates. A red/orange traffic-light state should diagnose whether the ETH hedge is actually helping.

### Mechanics

Trigger hedge-failure override only when:

- green votes are `<= 2`, and
- SOL underperforms ETH by a threshold over `24h`, `72h`, or `7d`, or
- SOL/ETH breaks a trailing low, and
- portfolio drawdown or drawdown speed confirms damage.

Actions by severity:

| Severity | Action |
| --- | --- |
| Stage 1 | pause surplus SOL buys, route hedge profit to USDC/debt cleanup |
| Stage 2 | raise ETH hedge only if projected HF passes |
| Stage 3 | sell a small SOL slice into protected USDC |
| Stage 4 | stop new risk until light recovery |

Candidate sale slices should be tiny at first: `0.5%`, `1%`, `2.5%`, `5%`, with cumulative caps of `3%`, `5%`, `8%`.

### Expected Failure Mode

SOL can underperform ETH immediately before a SOL-led rebound. This override needs the recovery engine; otherwise it will preserve USD at the expense of the `>420 SOL` gate.

## Experiment Family E: Regime-Specific Objective Function

### Thesis

One scalar ranking metric hides failure. Investor-grade evaluation needs to rank only valid candidates and explain why invalid candidates failed.

### Validity Gates

A candidate is valid only if:

- final SOL-equivalent `> 420`;
- min health factor `>= 1.50`;
- no liquidation event;
- final open debt and collateral reconcile to final portfolio value.

### Ranking Among Valid Candidates

Rank by:

1. max drawdown, lower is better;
2. 2024+ peak-to-trough drawdown, lower is better;
3. major drawdown recovery half-life, lower is better;
4. final SOL-equivalent, higher is better;
5. Sortino, higher is better;
6. lower final ETH debt and lower final LTV as tie-breakers.

### Research Frontier Reporting

Also report near-valid candidates separately:

- `400-420 SOL` final SOL-equivalent;
- min HF `1.35-1.50`;
- max drawdown `<=50%` but misses SOL gate.

Near-valid rows are diagnostic only. They cannot be promoted.

## Metrics To Add

Every radical traffic-light report should include:

- final USD value;
- final SOL-equivalent value;
- max drawdown;
- 2024+ peak-to-trough drawdown;
- Sortino;
- min health factor;
- final LTV;
- final SOL collateral;
- final USDC collateral;
- final ETH debt value;
- final USDC debt value;
- liquidation count;
- percent of bars in each light state;
- percent of bars in each overlay/governor state;
- count of hedge-up, hedge-down, reinvestment, debt-cleanup, protected-routing, recovery-buy, and SOL-sale events;
- recovery half-life for every drawdown greater than `30%`;
- time to new high after every drawdown greater than `30%`;
- SOL-equivalent recovery half-life;
- drawdown decomposition by 2021-2023 crash and 2024+ red-mark period.

## First Implementation Slice

The first implementation should not attempt every family at once. It should build the measurement and state plumbing required to test them cleanly.

Recommended first slice:

1. Add traffic-light state observability to history/events if not already complete.
2. Add recovery metrics to reporting.
3. Implement Family A as a configurable traffic-light exposure governor.
4. Run a constrained sweep over Family A with existing recovery behavior.
5. Implement Family B recovery buckets only after Family A identifies whether the traffic-light exposure idea can create valid `>420 SOL` candidates.

## First Sweep Proposal

Start with a small, interpretable sweep:

| Axis | Values |
| --- | --- |
| Yellow floor | `0.35`, `0.50`, `0.75` |
| Orange floor | `0.75`, `1.00`, `1.25` |
| Red floor | `1.00`, `1.25`, `1.50` |
| 3-green reinvestment | `allowed`, `reduced` |
| 2-green reinvestment | `allowed`, `blocked`, `USDC-routed` |
| 1/0-green reinvestment | `blocked` |
| USDC releverage below 4 green | `disabled` |
| Projected min HF for new hedge adds | `1.50`, `1.75`, `2.00` |
| Cover rule | `normal`, `staged on 3 green`, `aggressive on 4 green` |

This sweep should be intentionally smaller than the previous brute-force runs. The goal is to understand whether traffic-light governance changes the frontier, not to hide a fragile winner in a giant grid.

## Reporting Output

Create a new timestamped report directory under:

`reports/sol_supertrend_3y/traffic_light_governor_<timestamp>/`

Required artifacts:

- `report.md`
- `comparison.csv`
- `valid_configs.csv`
- `near_valid_configs.csv`
- `all_by_drawdown.csv`
- `all_by_sol.csv`
- `family_best.csv`
- `summary.json`
- per-scenario `history.csv`
- per-scenario `strategy_events.csv`

## Promotion Criteria

A candidate can be promoted only if it satisfies all:

- final SOL-equivalent `> 420`;
- min health factor `>= 1.50`;
- max drawdown improves by at least `10` percentage points versus `v2_best_no_overlays`;
- 2024+ drawdown improves by at least `5` percentage points versus `v2_best_no_overlays`;
- recovery half-life improves for at least one major drawdown window;
- no evidence of unreconciled final exposure or accounting distortion.

The aspirational institutional target remains:

- max drawdown `<= 50%`;
- final SOL-equivalent `> 420`.

If no candidate reaches `<=50%` drawdown while preserving `>420 SOL`, the report must say so plainly and identify the best achievable frontier.

## Non-Goals

- Do not add an HMM or ML regime model in this slice.
- Do not change the data source.
- Do not optimize for USD-only returns.
- Do not promote near-valid candidates.
- Do not hide health-factor deterioration behind better drawdown.
- Do not short SOL unless the user explicitly approves expanding the strategy universe beyond current constraints.

## Open Questions

1. Whether SOL-native hedging is allowed in a later phase if ETH-short governance cannot reach the target.
2. Whether a `>420 SOL` candidate with `55%-60%` max drawdown is acceptable as an interim promotion if the `50%` target is not reachable.
3. Whether investor reporting should use full-period drawdown, post-2024 drawdown, or both as hard gates.

## Expected Research Outcome

The most likely useful result is a clearer frontier:

- If traffic-light governance produces `>420 SOL` and materially lower drawdown, continue refining recovery and insurance book mechanics.
- If it preserves `>420 SOL` but cannot get below roughly `70%` drawdown, the strategy needs SOL-native protection or a different hedge asset.
- If it gets below `50%` drawdown only by falling below `420 SOL`, then the compounding objective and investor drawdown target are incompatible under the current mechanics.
