# Quarterly Position Forensic Memo

Source timeline: `temp/2026-05-28T21-54_export.csv`
Event log: `reports/sol_supertrend_3y/post_reinvestment_report/strategy_events.csv`
Run window: 2021-01-01 to 2025-12-31

## Executive Diagnosis

The strategy eventually succeeds at the long-term objective: it ends with 254.01 actual SOL and 267.29 SOL-equivalent net value. But the path exposes a major defect in drawdown control. The worst damage is not that the strategy failed to hedge; it is that the hedge/full-short system was too reactive, too signal-fragile during the decline, and too willing to carry high ETH short exposure after the account had already been deeply impaired.

The deepest wound is 2022Q4. The account entered the quarter with 177.9 SOL-equivalent value and ended at 143.1 SOL-equivalent. Intra-quarter drawdown was -81.6%, min HF reached 1.366, and max drawdown for the whole run occurred on 2022-12-29. By then the account had already accumulated more SOL, but it had not protected the newly enlarged SOL stack well enough during the final leg down.

The likely rule-design lesson: Supertrend vote alone is too slow for crash control. We need a programmatic override that responds to drawdown velocity, SOL-relative impairment, and higher-regime deterioration before the normal full-short entry has fully confirmed.

## Quarter-End Position Table

| Quarter | Net USD | Q Return | SOL Eq | Actual SOL | USDC | ETH Debt | ETH/SOL | HF | Min HF | Intra-Q DD | Key Events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2021Q1 | $1,801 | 1,066.5% | 92.9 | 100.0 | $31 | $168 | 0.09x | 8.81 | 2.14 | -34.6% | hedge_up:24, hedge_down:3 |
| 2021Q2 | $3,035 | 68.0% | 85.5 | 100.0 | $0 | $516 | 0.15x | 5.16 | 1.92 | -64.1% | hedge_up:35 |
| 2021Q3 | $12,912 | 328.6% | 91.3 | 100.0 | $213 | $1,438 | 0.10x | 7.51 | 1.65 | -40.5% | hedge_up:22, full_short_up:1 |
| 2021Q4 | $14,426 | 10.3% | 84.9 | 100.0 | $6,157 | $8,730 | 0.51x | 2.10 | 1.94 | -43.5% | hedge_up:34, hedge_down:3 |
| 2022Q1 | $11,692 | -19.7% | 95.2 | 100.0 | $0 | $585 | 0.05x | 15.73 | 1.61 | -43.2% | full_short_up:12, full_short_cover:40 |
| 2022Q2 | $6,937 | -41.5% | 205.5 | 183.0 | $3,711 | $2,953 | 0.48x | 2.70 | 1.61 | -53.8% | full_short_up:25, full_short_cover:22, hedge_up:18, surplus_reinvestment:13 |
| 2022Q3 | $5,914 | -15.5% | 177.9 | 183.0 | $0 | $172 | 0.03x | 26.50 | 1.62 | -35.9% | hedge_up:40, full_short_up:12, full_short_cover:16 |
| 2022Q4 | $1,427 | -75.6% | 143.1 | 183.0 | $1,271 | $1,669 | 0.91x | 1.51 | 1.37 | -81.6% | full_short_cover:24, hedge_up:13, full_short_up:2 |
| 2023Q1 | $3,103 | 116.5% | 146.6 | 183.0 | $0 | $770 | 0.20x | 3.77 | 1.49 | -38.6% | hedge_up:24 |
| 2023Q2 | $2,455 | -20.9% | 130.2 | 183.0 | $167 | $1,164 | 0.34x | 2.35 | 1.70 | -54.2% | hedge_up:27, full_short_up:7 |
| 2023Q3 | $2,982 | 22.7% | 139.5 | 183.0 | $343 | $1,273 | 0.33x | 2.55 | 1.63 | -43.6% | hedge_up:20, full_short_up:3 |
| 2023Q4 | $17,133 | 476.9% | 168.4 | 183.0 | $397 | $1,883 | 0.10x | 7.61 | 1.99 | -25.2% | hedge_up:20 |
| 2024Q1 | $33,121 | 92.9% | 163.6 | 183.0 | $0 | $3,936 | 0.11x | 7.06 | 2.13 | -32.5% | hedge_up:40 |
| 2024Q2 | $23,596 | -28.6% | 160.9 | 183.0 | $0 | $3,240 | 0.12x | 6.21 | 1.99 | -41.6% | hedge_up:46 |
| 2024Q3 | $28,118 | 18.1% | 184.4 | 183.0 | $205 | $0 | 0.00x | inf | 2.16 | -30.7% | hedge_up:34 |
| 2024Q4 | $34,064 | 19.9% | 179.9 | 183.0 | $16,738 | $17,326 | 0.50x | 2.37 | 2.24 | -31.2% | hedge_up:41 |
| 2025Q1 | $32,231 | -6.6% | 258.8 | 183.0 | $38,292 | $28,857 | 1.27x | 1.79 | 1.72 | -42.5% | hedge_up:23, full_short_up:6 |
| 2025Q2 | $40,671 | 26.8% | 262.7 | 254.0 | $5,114 | $3,766 | 0.10x | 9.05 | 1.75 | -25.8% | surplus_reinvestment:7, full_short_up:5 |
| 2025Q3 | $48,431 | 19.2% | 232.1 | 254.0 | $21,579 | $26,155 | 0.49x | 2.26 | 2.17 | -21.3% | hedge_up:40 |
| 2025Q4 | $33,397 | -30.6% | 267.3 | 254.0 | $25,306 | $23,648 | 0.75x | 1.97 | 1.64 | -38.7% | full_short_up:24, full_short_cover:37 |

## Quarter Commentary

### 2021Q1: Early Bull, Hedge Drag Appears Immediately

The account rode the early SOL rally from a tiny starting value to $1,801, but SOL-equivalent value ended at 92.9, below the starting 100 SOL baseline. That means the hedge system was already creating opportunity-cost drag during a bullish environment. The position ended with only a small ETH debt value, so the issue was not catastrophic leverage; it was signal noise creating hedge actions in an uptrend.

What went well: health factor stayed strong and there was no dangerous debt state.

What did not go well: the strategy underperformed SOL in SOL terms despite a huge USD gain. This is the first warning that USD return alone is not enough.

Possible rule: suppress small hedge changes in strong bull regimes unless SOL-relative drawdown or volatility confirms danger.

### 2021Q2: More Hedge Drag During a Rising Market

The account ended at $3,035, but SOL-equivalent dropped further to 85.5. The quarter had 35 hedge-up events and no full-short events. This means the strategy repeatedly added ETH hedge exposure during a period where the larger trend was still favorable enough that the account should probably have prioritized SOL participation.

What went well: ending HF remained healthy at 5.16.

What did not go well: intra-quarter drawdown was -64.1%, and the strategy still had not created any SOL accumulation mechanism.

Possible rule: make hedge ladder less sensitive when higher-regime trend is green and price remains above a longer moving peak threshold.

### 2021Q3: Strong USD Gains, Still Poor SOL Capture

The account reached $12,912, but still ended at only 91.3 SOL-equivalent. Actual SOL was unchanged at 100. The strategy was participating in the bull market mostly as a marked-to-market SOL holder, but repeated hedge actions still prevented it from keeping pace with pure SOL.

What went well: the system survived volatility and did not over-short the rally.

What did not go well: the strategy still failed the philosophical hurdle. A strategy that cannot keep its SOL-equivalent base near 100 during a bull phase is already spending too much insurance premium.

Possible rule: add a bull-market participation guardrail: if 1w is green and portfolio SOL-equivalent is below starting SOL, limit hedge ratio unless drawdown velocity breaches a hard threshold.

### 2021Q4: The Peak and the First Real Warning

The account hit its pre-crash high near the November 2021 SOL peak, then ended the quarter at $14,426 and only 84.9 SOL-equivalent. The hedge did eventually build to 0.51x ETH/SOL by quarter end, with $6,157 USDC collateral and $8,730 ETH debt. But the important timing problem remains: after the 2021 SOL high on 2021-11-06, the first hedge-up after the ATH did not occur until 2021-11-12, already -10.5% from the high. Full-short did not arrive until 2022-01-08, about -45.8% from the high.

What went well: the account had started creating defensive USDC by the end of the quarter.

What did not go well: the strategy let SOL-equivalent fall below 85 while still treating the environment as mostly hedge, not crisis. The normal full-short gate was too slow.

Possible rule: introduce a peak-drawdown override. If SOL falls more than 20-25% from a rolling 60-day high while 1d and 3d trend are bearish, escalate to a minimum 75-100% hedge even before all four standard timeframes are bearish.

### 2022Q1: Full-Short Arrives, But Too Late and Too Choppy

The account lost 19.7% in USD during the quarter, but SOL-equivalent recovered to 95.2. That is better than the prior quarter and suggests full-short behavior was finally adding defensive value. However, the quarter had 12 full-short-up events and 40 full-short-cover events. That is a sign of churn: the strategy entered and exited full-short repeatedly rather than staying in a durable crisis posture.

What went well: ending ETH debt was small, HF was excellent, and SOL-equivalent improved.

What did not go well: the system covered too often during an unfolding macro downtrend. The short-cover ladder likely responded too eagerly to short-term green votes.

Possible rule: once a crisis override is active, require a higher bar to exit than to trim. For example, do not fully exit crisis hedge until 1w turns green or portfolio recovers above a SOL-equivalent trailing floor.

### 2022Q2: Best Defensive Quarter, But Also the Setup for Later Risk

This was the most constructive quarter. The account fell in USD to $6,937, but SOL-equivalent jumped to 205.5 and actual SOL increased from 100 to 183.0. Surplus reinvestment fired 13 times from 2022-06-25 to 2022-06-27, converting realized hedge profit into cheap SOL.

What went well: this is the strategy's philosophical win. Hedge profit became SOL accumulation after a severe repricing.

What did not go well: the account still suffered -53.8% intra-quarter drawdown, and it ended with a 0.48x ETH/SOL hedge ratio. Buying SOL was correct, but the newly larger SOL stack needed an explicit post-reinvestment protection rule.

Possible rule: after surplus reinvestment increases actual SOL by more than a threshold, impose a temporary protective hedge floor for the enlarged SOL stack until higher-regime recovery is confirmed.

### 2022Q3: Defensive Calm, But Not Enough Recovery Confirmation

The account lost another 15.5% and ended at 177.9 SOL-equivalent. ETH debt was nearly gone by quarter end, and HF was very high. That looks safe, but it may also indicate the strategy reduced hedge exposure while the broader bear market had not truly ended.

What went well: no liquidation pressure, and actual SOL remained at 183.

What did not go well: the strategy let the crisis posture fade while the weekly regime was still vulnerable. The account had a larger SOL stack after Q2 and therefore more downside sensitivity.

Possible rule: distinguish "health is safe" from "market regime is safe." High HF should not be a reason to lower hedge exposure if rolling SOL drawdown remains severe and 1w/3d are not repaired.

### 2022Q4: The Core Failure

This is the quarter to design around. Net value fell -75.6%, SOL-equivalent fell to 143.1, and intra-quarter drawdown was -81.6%. The max drawdown for the whole run occurred on 2022-12-29, when portfolio value fell to $1,117 from a prior peak of $23,266, a -95.2% drawdown.

The ending account had 183 SOL, $1,271 USDC, and $1,669 ETH debt. ETH/SOL was 0.91x, so the hedge was large by quarter end. The problem is that the hedge did not prevent the damage soon enough, and the strategy still had 24 full-short-cover events versus only 2 full-short-up events. That suggests the system was trimming/covering too much during a continuing waterfall.

What went well: the account avoided liquidation, barely. Min HF was 1.366.

What did not go well: the account gave back a large part of the Q2 SOL-equivalent gain. The full-short cover logic was too permissive in a regime where preserving the newly accumulated SOL should have been the priority.

Possible rule: add a drawdown-lock. If portfolio value is down more than 50% from a 180-day high, or SOL is down more than 60% from a rolling high, lock the account into a minimum hedge ratio until either 1w turns green or portfolio SOL-equivalent recovers above a configured threshold.

### 2023Q1-Q3: Recovery Churn Without SOL Accumulation

The account recovered from the Q4 low but remained volatile. SOL-equivalent moved from 146.6 to 130.2 to 139.5. Actual SOL stayed fixed at 183. The strategy survived, but did not compound. It repeatedly hedged and occasionally entered full-short, yet it did not create additional SOL accumulation.

What went well: survival after the worst drawdown.

What did not go well: no accumulation despite lower prices and repeated volatility. The surplus reinvestment gate required realized hedge profit, so no moderate accumulation path existed for non-hedge-funded buying.

Possible rule: add a non-debt accumulation module for excess USDC collateral when 1w is green or recovering, separate from realized hedge profit surplus.

### 2023Q4-2024Q2: Bull Recovery, But Still No Added SOL

The account recovered strongly in USD, reaching $33,121 in 2024Q1. But actual SOL stayed at 183, and SOL-equivalent was still only around 160-168 through much of the recovery. The strategy participated, but not as aggressively as the objective would suggest.

What went well: drawdowns improved versus 2022, and HF stayed strong.

What did not go well: the strategy missed a major SOL accumulation window. It had no rule saying, "the bear market survived, the account has stabilized, now rebuild SOL exposure gradually."

Possible rule: when weekly trend turns green after a crisis and ETH short ratio is below target, allocate a fixed percentage of excess USDC collateral to SOL over multiple rebalance windows.

### 2024Q3-Q4: Hedge Rebuilds Into Strength

By 2024Q3 ETH debt was zero, then by 2024Q4 the account carried $17,326 ETH debt against $16,738 USDC collateral. The ending ETH/SOL ratio was 0.50x. This is not necessarily wrong, but it shows the strategy rebuilt hedge exposure into a still-profitable account.

What went well: the hedge created dry powder and the account stayed healthy.

What did not go well: no SOL was bought. The strategy was building defensive structure, but not converting any of the strength into additional SOL.

Possible rule: require hedge profit harvesting to have an accumulation plan when higher-regime damage is not present. Otherwise USDC accumulates without increasing SOL.

### 2025Q1-Q4: Late Accumulation and Persistent Open Short Risk

The strategy finally bought another 70.97 SOL in 2025Q2, ending with 254 SOL. This is good. But the account also carried large ETH debt in Q1, Q3, and Q4, with ETH/SOL reaching 1.27x in Q1 and ending at 0.75x in Q4.

What went well: final SOL stack improved materially.

What did not go well: final net value depends heavily on an open ETH short. This is acceptable only if the report clearly separates realized gains from open short exposure.

Possible rule: add an open-short sensitivity report and a rule that trims terminal/open short risk when SOL-equivalent objective is already met.

## Failure Patterns

### 1. Full-Short Entry Is Too Late

The strategy waits for severe confirmation before full-short escalation. In 2021-2022 this meant full-short arrived after SOL had already fallen roughly 45.8% from its high. That is too late for sharp drawdown avoidance.

Rule candidate: earlier crisis entry when rolling drawdown and higher-timeframe deterioration agree, even if the standard four-timeframe vote has not fully collapsed.

### 2. Full-Short Exit Is Too Eager During Crisis

2022Q4 had 24 full-short-cover events and only 2 full-short-up events. During a waterfall, short-term green votes should not cause aggressive covering unless higher-regime recovery is real.

Rule candidate: crisis-mode hysteresis. Entry can be fast, exit must be slow. Require 1w green, or 3d green plus portfolio drawdown recovery, before full cover is allowed.

### 3. Surplus Reinvestment Needs Post-Buy Protection

Q2 2022 buying was good. The failure was not buying SOL; it was failing to protect the larger SOL stack in Q4. After actual SOL increased from 100 to 183, the account's downside sensitivity increased.

Rule candidate: after surplus reinvestment adds more than 20% to actual SOL, activate a temporary hedge floor until the account is no longer in a crisis regime.

### 4. Health Factor Alone Is Not a Regime Signal

Q3 2022 looked safe by health factor, but the market regime was not safe. The account reduced hedge exposure while broader crypto risk remained elevated.

Rule candidate: separate account safety from market safety. HF controls liquidation risk; drawdown velocity and weekly regime control hedge persistence.

### 5. SOL-Equivalent Drawdown Needs Its Own Guardrail

The strategy's philosophical objective is SOL-relative USD compounding. Therefore SOL-equivalent impairment should be a first-class signal. Falling from 205.5 SOL-equivalent in 2022Q2 to 143.1 in 2022Q4 should have triggered a defensive override.

Rule candidate: if SOL-equivalent falls more than 25-30% from a trailing high, prevent hedge reduction and pause new SOL accumulation until stabilized.

## Candidate Programmatic Rules

### Rule A: Crisis Drawdown Override

Trigger if:

- SOL is down at least 25% from a rolling 60-day high, and
- 1d Supertrend is red, and
- either 3d Supertrend is red or the portfolio is down 20% from a rolling 30-day high.

Action:

- Raise minimum ETH hedge ratio to 75%.
- If 3d is also red, raise minimum ETH hedge ratio to 100%.
- Cap by health factor as usual.

Why this helps: it would have started protection before the full 45.8% drawdown from the 2021 high.

### Rule B: Crisis Exit Hysteresis

Trigger if crisis mode is already active.

Action:

- Do not allow full hedge removal from short-term green votes alone.
- Permit partial cover, but preserve at least 50-75% hedge until 1w turns green or SOL recovers above a rolling drawdown threshold.

Why this helps: it addresses 2022Q4's excessive full-short-cover behavior.

### Rule C: Post-Reinvestment Protection

Trigger if surplus reinvestment has increased actual SOL by at least 20% within the last 30 days.

Action:

- Enforce a temporary hedge floor while 3d or 1w remains bearish.
- Do not allow the enlarged SOL stack to sit effectively unprotected during a still-bearish regime.

Why this helps: Q2 2022 accumulation was correct, but Q4 2022 exposed the larger SOL stack to another drawdown.

### Rule D: SOL-Equivalent Damage Guard

Trigger if portfolio SOL-equivalent falls more than 25% from its trailing 90-day high.

Action:

- Pause surplus reinvestment.
- Prevent hedge-down actions unless health factor requires it.
- Escalate to crisis mode if higher-timeframe signals are also bearish.

Why this helps: it aligns risk control with the actual objective, not only USD PnL.

### Rule E: Recovery Accumulation Plan

Trigger if:

- 1w turns green after crisis mode,
- ETH short ratio is at or below the normal target,
- health factor is above the safety threshold,
- USDC collateral remains after required hedge unwind capital.

Action:

- Convert a small percentage of excess USDC collateral to SOL per rebalance or per day.

Why this helps: it targets the 2023-2024 missed accumulation window without using USDC debt.

## Recommended Next Experiment

Start with Rule A and Rule B together. They directly target the 2022 drawdown:

1. Add a crisis mode state.
2. Enter crisis mode from drawdown velocity plus 1d/3d deterioration.
3. Enforce a minimum hedge floor while crisis mode is active.
4. Exit crisis mode only with higher-regime confirmation, not short-term green votes.

Then rerun the same quarterly forensic memo. Success should not be judged only by final USD value. The target improvement is:

- reduce 2022Q4 intra-quarter drawdown,
- keep SOL-equivalent above the 2022Q2 post-reinvestment level by more than the current result did,
- avoid materially reducing final actual SOL,
- avoid liquidation or excessive HF stress.

This is the right next scientific question: can we keep the SOL accumulation win from Q2 2022 while preventing the Q4 2022 giveback?

## Deferred Idea: Recovery Accumulation

Keep the recovery accumulation idea, but do not mix it into the first crisis-mode implementation. It targets a different failure: the strategy missed the 2023-2024 SOL accumulation window after surviving the 2022 crash.

Return to this after the crisis-mode report. The hypothesis to test later is: once crisis mode has exited, ETH short exposure is at or below target, health factor is safely above threshold, and excess USDC collateral remains, the strategy may convert a small amount of USDC into SOL over repeated rebalance windows without borrowing new USDC.
