# Research Synthesis: Fast Downside Positioning Overlay

Date: 2026-05-29

## Executive Summary

The literature supports a simple conclusion for the SOL Supertrend / Kamino strategy: downside protection should be triggered by a combination of trend deterioration and volatility expansion, then exited with explicit rebound discipline. A trailing portfolio drawdown alone is too late. A permanent higher hedge floor is too expensive. A broad "block all reinvestment" state fights the strategy's SOL-accumulation engine.

The current incumbent, `stateful_near_0.01_exit_0.05`, is a better default than the prior profit-lock setup, but it still leaves unacceptable drawdown: final value `$50,237.89`, final SOL-equivalent `402.06 SOL`, max drawdown `82.39%`, Sortino `2.274`, min HF `1.591`, and final ETH debt `$27,029.15`. The 2025 peak-to-trough giveback remains severe. The best clue from the existing sweeps is that floor-only drawdown containment can help when it does not block reinvestment, but the raw winner got there with min HF `1.407`, below the current promotion guardrail.

The next rules should therefore be short-lived and conditional:

- Enter before portfolio damage is large, using fast SOL break signals.
- Escalate the ETH hedge floor only when realized volatility confirms the break.
- Preserve surplus reinvestment unless a separate rule proves that a narrow pause is beneficial.
- Exit quickly or stage down when rebound risk appears, because momentum crash literature warns that high-volatility panic states can reverse violently.
- Report every candidate by final USD value, final SOL-equivalent, max drawdown, Sortino, min HF, final ETH debt, and the 2024/2025 giveback.

The first experiments should not attempt a full statistical regime model. The right first pass is a four-state proxy: trend up/down crossed with volatility low/high. That is auditable, aligns with the Hidden Markov literature's state idea, and fits the existing Supertrend signal stack.

## Paper-By-Paper Notes

### Moskowitz, Ooi, Pedersen: Time Series Momentum

Local file: `time-series-momentum-moskowitz-ooi-pedersen.pdf`

Source: [SSRN abstract](https://ssrn.com/abstract=2089463)

- The paper documents return persistence over 1- to 12-month horizons across futures markets.
- It also finds partial reversal over longer horizons, which argues against permanently defensive states after every break.
- Time-series momentum performs best in extreme markets, supporting the idea that trend-following protection is useful during large market moves.
- For this codebase, the useful translation is not "short SOL"; the strategy constraint forbids that. The translation is "increase ETH hedge floor or slow SOL acquisition when SOL's own trend breaks."
- The existing four-timeframe Supertrend ladder is already a trend-following structure. The gap is speed and conditional escalation, not lack of trend signal.
- Since the study uses own-asset return history, SOL-specific fast-break triggers are more defensible than generic portfolio drawdown triggers.
- Practical rule implication: add a fast own-price downside trigger such as short-lookback return/channel break plus at least one weakening Supertrend vote.

### Barroso, Santa-Clara: Momentum Has Its Moments

Local file: `momentum-has-its-moments-barroso-santa-clara.pdf`

Source: [PDF copy](https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/2d050545-c67a-4955-bac7-cba45a3375ec.pdf)

- The paper's central finding is that momentum crash risk is time-varying and predictable.
- It scales momentum exposure using realized volatility, targeting a steadier risk level.
- The risk-managed version materially reduces left-tail outcomes and improves Sharpe in the paper's tests.
- For this project, the analogous mechanism is not to scale a pure momentum portfolio. It is to scale defensive intensity by SOL realized volatility or SOL/ETH relative volatility.
- A volatility-only trigger is likely too blunt because crypto volatility rises in both selloffs and rallies.
- The stronger implementation is a conjunctive rule: negative return or trend break plus volatility expansion.
- Practical rule implication: when SOL realized volatility exceeds its trailing median or percentile and SOL short-term return is negative, raise the ETH hedge floor temporarily.

### Daniel, Moskowitz: Momentum Crashes

Local file: `momentum-crashes-daniel-moskowitz.pdf`

Source: [SSRN abstract](https://ssrn.com/abstract=2371227)

- Momentum strategies have negative skew and occasional severe crash strings.
- The paper identifies "panic" states after market declines and during high volatility, often contemporaneous with rebounds.
- This matters because the strategy's defensive hedge can be right during the initial downside break but wrong if it stays max hedged into a sharp rebound.
- The paper supports dynamic exposure based on expected mean and variance rather than a static defensive posture.
- For this codebase, that argues for explicit hedge de-escalation after the panic state changes, not a sticky full-short-like state.
- It also argues against adding a crash overlay with only a slow near-high recovery exit.
- Practical rule implication: any fast-break hedge escalation should have a staged cover rule, such as cover one tier after a 2-3 bar recovery with falling realized volatility, and remove the overlay when 4h/8h trend recovers.

### Moreira, Muir: Volatility Managed Portfolios

Local file: `volatility-managed-portfolios-moreira-muir-nber.pdf`

Source: [NBER page](https://www.nber.org/papers/w22208)

- The paper finds that reducing risk when volatility is high can improve factor Sharpe ratios because higher volatility is not fully offset by higher expected returns.
- It is directly relevant to the health-factor problem: high volatility is exactly when the same notional hedge can consume more liquidation capacity.
- The paper supports volatility-managed sizing, not only volatility-managed entry.
- For this strategy, that suggests scaling target hedge floors or add-size budgets based on realized volatility and min-HF capacity.
- The translation should be asymmetric: high volatility plus falling SOL increases defensive urgency, but high volatility plus rebounding SOL should trigger de-escalation.
- This is a cleaner research direction than simply lowering `min_rebalance_hf`, because it explicitly reports whether extra protection came from unsafe leverage.
- Practical rule implication: cap incremental ETH hedge adds by both realized-volatility state and projected health factor, then rank results with min HF visible rather than hidden.

### Harvey et al.: An Investor's Guide To Crypto

Local file: `investors-guide-to-crypto-ssrn-id4124576.pdf`

Source: [SSRN abstract](https://ssrn.com/abstract=4124576)

- The paper emphasizes that crypto returns are highly volatile, but volatility targeting and trend-following have performed well.
- It also notes that crypto correlations with traditional risk assets can rise in left-tail environments.
- For this project, the relevant point is that passive crypto beta can dominate unless active rules intentionally control risk.
- Because SOL is the accumulation asset, the overlay must reduce drawdown without abandoning long-term SOL compounding.
- Intraday data is useful in crypto because the sample history is short; this supports using the repo's 1h bars for fast-break diagnostics.
- The paper's active-versus-buy-and-hold framing matches the user's success metric: USD return alone is not enough if SOL-equivalent accumulation fails.
- Practical rule implication: use hourly fast-break signals, but evaluate them against both USD and SOL-equivalent outcomes.

### Bui, Nguyen: AdaptiveTrend Crypto

Local file: `adaptive-trend-crypto-arxiv-2602-11708.pdf`

Source: [arXiv abstract](https://arxiv.org/abs/2602.11708)

- The paper proposes a crypto trend-following framework using 6-hour intervals, volatility-regime-aware trailing stops, rolling Sharpe selection, and asymmetric long-short allocation.
- Its main implementation-relevant idea is not the full framework; it is the combination of trend signal plus volatility-calibrated stop/position sizing.
- The 6-hour cadence is relevant because this strategy already uses 1h, 4h, 8h, and 1d Supertrend votes. A 4h/8h break overlay is a natural fit.
- The asymmetric long-short allocation is consistent with this project's constraint: the strategy is structurally long SOL and uses ETH only as a hedge.
- It reinforces that crypto trend rules need regime-conditional evaluation across bull, bear, and sideways periods.
- Practical rule implication: test a 4h/8h fast-break overlay before adding a separate model. For example, if 4h turns red, 8h slope weakens, and realized volatility spikes, impose a temporary 75%-100% hedge floor.

### Koki, Leonardos, Piliouras: Bayesian HMM Crypto Predictability

Local file: `bayesian-hmm-crypto-predictability.pdf`

Source: [arXiv abstract](https://arxiv.org/abs/2011.03741)

- The paper finds that multi-state hidden Markov models can distinguish crypto regimes with different return and risk characteristics.
- The four-state result is useful conceptually: bull, bear, calm, and higher-risk states map well to trend and volatility dimensions.
- A full HMM is too complex for the next experiment because it adds calibration risk, fitting choices, and explainability overhead.
- The useful first approximation is a deterministic state proxy: trend up/down crossed with volatility low/high.
- That proxy can be computed from existing history fields and audited in event logs.
- Practical rule implication: create a four-state overlay using existing Supertrend votes plus realized volatility percentile. Let only "trend down + volatility high" escalate the hedge; let "trend up + volatility falling" exit.

## Candidate Strategy Mechanisms

### 1. Fast-Break Volatility Overlay

- Trigger: SOL short-lookback break plus volatility expansion. Start with `24h return <= -8%` or `close below 7d Donchian low`, and `24h realized volatility >= 1.5x` its 30d median. Require at least one trend weakening signal: green votes `<= 3`, 1d bearish, or 4h/8h Supertrend flip.
- Action: raise ETH hedge floor to `0.75` for a short window. If 1d is bearish or 3d is bearish, allow `1.00`. Do not block surplus reinvestment in the first version.
- Exit: staged exit after either 48-72 hours, volatility falls below `1.1x` median, or 4h/8h recover green. Step floor down from `1.00` to `0.75` to `0.35`, rather than full cover.
- Expected trade-off: should catch waterfalls earlier than crisis mode, but may over-hedge sharp washouts. The explicit time decay is there to control rebound risk.

### 2. Volatility-Scaled Hedge Add Budget

- Trigger: normal rebalance wants more hedge, profit lock is active, or fast-break overlay is active.
- Action: scale incremental ETH hedge add size by realized volatility and projected health factor. Example: allow larger hedge adds in high-volatility downside states only if projected HF remains above a research floor such as `1.35`, while still reporting all min-HF outcomes.
- Exit: normal target resolution, but with no special sticky state. This is a sizing rule, not an entry rule.
- Expected trade-off: may reproduce the drawdown benefit of floor-only containment without letting min HF collapse silently. It may also reduce final SOL-equivalent if it keeps the strategy from using profitable hedge capacity during real crashes.

### 3. Four-State Trend/Volatility Regime Proxy

- Trigger: classify each bar into `trend_up_vol_low`, `trend_up_vol_high`, `trend_down_vol_low`, or `trend_down_vol_high`. Trend can be green votes `>= 3` versus `<= 2`, with 1d bearish as an override. Volatility can be 24h or 72h realized volatility above its 30d percentile threshold.
- Action: only `trend_down_vol_high` imposes a new defensive floor. `trend_down_vol_low` keeps normal Supertrend/crisis behavior. `trend_up_vol_high` forbids new defensive escalation and allows staged cover.
- Exit: state transition out of `trend_down_vol_high`, plus one-bar or multi-bar confirmation to reduce churn.
- Expected trade-off: more auditable than a full HMM, but threshold choices can be fragile. It should be tested across multiple volatility lookbacks before promotion.

### 4. Fast Drawdown-Speed Escalation While Profit Lock Is Active

- Trigger: profit lock is active and portfolio drops `7%-10%` from a 7d or 14d high within `24h-72h`, especially if SOL is also below a short moving average or channel low.
- Action: raise profit-lock floor from `0.35` to `0.50` or `0.75` for a fixed window.
- Exit: time-based decay or 4h/8h recovery, not recovery to old portfolio high.
- Expected trade-off: targeted at frothy peak giveback. It could still be too late if the drawdown-speed trigger waits for portfolio damage instead of SOL break confirmation.

### 5. Bucketed Froth Reserve Redeployment

- Trigger: existing froth reserve rotation condition after large run-up or profit-lock state.
- Action: keep small SOL-to-USDC reserve rotations, but redeploy only in one-shot buckets.
- Exit/rebuy condition: either deeper drawdown buckets such as `35%`, `50%`, `65%` from SOL high, or recovery confirmation such as 4h/8h trend reclaim plus volatility falling. No continuous rebuy loop.
- Expected trade-off: may reduce late-cycle giveback without repeating the prior failure mode. It is less directly tied to "fast break" than the hedge overlays, so it should come after the first fast-break tests.

### 6. SOL/ETH Relative Breakdown Guard

- Trigger: SOL 24h/72h return underperforms ETH by a threshold while SOL trend is weakening. This detects periods when ETH is a poor hedge because SOL is falling harder than ETH.
- Action: increase ETH hedge target only if projected HF is acceptable; otherwise prioritize USDC debt cleanup and avoid new SOL buying for a narrow window.
- Exit: SOL/ETH relative return stabilizes or SOL trend recovers.
- Expected trade-off: can prevent false confidence in the ETH hedge, but may be noisy because SOL/ETH relative moves can mean SOL-specific upside as well as downside.

## Recommended First 3 Experiments

### 1. Fast-Break Volatility Overlay

Priority: highest.

Why: it most directly answers the research question. It acts before crisis mode by combining an own-price break with volatility expansion. It is simple, auditable, and literature-backed by time-series momentum and volatility-managed exposure.

Suggested sweep:

| Axis | Values |
|---|---|
| SOL break trigger | `24h return <= -6%, -8%, -10%`; `close < 7d Donchian low` |
| Vol confirmation | `24h realized vol >= 1.25x, 1.5x, 2.0x 30d median` |
| Floor | `0.50, 0.75, 1.00` |
| Hold / decay | `48h, 72h, until 4h+8h recovery` |
| HF research floor | report all, but flag `<1.50` and `<1.35` |

### 2. Safety-Constrained Floor-Only Containment

Priority: high, because existing evidence is already strong.

Why: the isolated sweep found a raw challenger with `434.46 SOL`, max drawdown `80.72%`, Sortino `2.330`, but min HF `1.407`. This should be rerun with explicit safety constraints before inventing more complex states.

Suggested sweep:

| Axis | Values |
|---|---|
| Trigger | `15%, 20%, 25%` portfolio drawdown |
| Floor | `1.00, 1.05, 1.10, 1.15, 1.20, 1.25` |
| Projected min HF for new hedge adds | `1.35, 1.50, 1.75` |
| Reinvestment blocking | disabled |

### 3. Four-State Trend/Volatility Proxy

Priority: medium-high.

Why: it takes the regime-detection idea seriously without introducing a full HMM. It can also unify the fast-break overlay, profit-lock escalation, and rebound exit logic under a small state table.

Suggested state table:

| State | Trigger | Action |
|---|---|---|
| Trend up / vol low | green votes `>= 3`, vol below threshold | normal strategy |
| Trend up / vol high | green votes `>= 3`, vol high | no new escalation; allow staged cover |
| Trend down / vol low | green votes `<= 2` or 1d bearish, vol low | normal Supertrend/crisis only |
| Trend down / vol high | trend down and vol high | fast-break hedge floor |

## Anti-Patterns / Things Not To Try Next

- Do not test another plain profit-lock drawdown threshold first. The metric and near-high sweeps already showed that faster trailing triggers do not solve the red-mark giveback.
- Do not block surplus reinvestment broadly during drawdown containment. Existing evidence shows this destroyed SOL accumulation and made the deepest trough worse.
- Do not continuously rebuy froth reserve merely because a drawdown threshold is crossed. The prior froth-reserve sweep produced hundreds to thousands of rebuy events during ongoing drawdowns.
- Do not add a full HMM as the next step. The deterministic four-state proxy should be tested first so the backtest remains explainable.
- Do not promote a rule that improves final USD but loses badly versus SOL buy-and-hold in SOL-equivalent terms.
- Do not hide health-factor degradation as a research success. It is acceptable to explore lower min-HF zones, but min HF and final debt must be headline metrics.
- Do not use a sticky high hedge floor with only a near-high recovery exit. Momentum crash research warns that defensive trend positions can be punished during panic rebounds.
- Do not treat ETH hedge size as equivalent to SOL de-risking. If SOL falls harder than ETH, the hedge can underperform exactly when needed.

## Evaluation Checklist For Each Candidate

Every experiment should report:

- Final portfolio value in USD.
- Final SOL-equivalent value.
- Max drawdown.
- Sortino.
- Min health factor.
- Final LTV.
- Final SOL collateral.
- Final ETH debt value.
- 2021-2023 global max-drawdown trough date and value.
- 2024/2025 peak-to-trough giveback.
- Percent of bars in the new overlay state.
- Count of hedge-up, hedge-down, skipped, and under-hedged events.

## Bottom Line

The best next direction is not "more profit lock." It is a fast-break overlay that only escalates when SOL trend deterioration and volatility expansion agree, then exits or decays quickly enough to avoid staying over-hedged into rebounds. The first implementation should be intentionally boring: a few threshold rules, explicit event logs, no full HMM, no reinvestment blocker, and hard reporting of the health-factor cost.
