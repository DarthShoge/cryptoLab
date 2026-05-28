# CryptoLab Context

CryptoLab models Kamino lending positions and strategy backtests for crypto assets. The language below keeps strategy ideas distinct from generic portfolio allocation.

## Language

**Kamino position strategy**:
A strategy whose state is a Kamino-style lending account with collateral, debt, health factor, liquidation risk, and optional cash reserve.
_Avoid_: Portfolio allocation strategy when collateral and debt mechanics matter.

**Long asset**:
The asset the strategy primarily wants positive exposure to by default.
_Avoid_: Risk asset, base asset

**Hedge asset**:
The asset borrowed and sold to reduce or offset risk when the long asset receives a negative signal. In this strategy ETH is the hedge asset.
_Avoid_: Signal-only hedge asset

**Short exposure**:
Inverse exposure created by borrowing an asset and selling it for USD value while retaining the obligation to repay that asset later.
_Avoid_: Hedge when the borrowed-and-sold mechanics are important.

**Cash reserve**:
USD-denominated value held after selling borrowed or withdrawn assets.
_Avoid_: Cash position when it could mean an on-chain stablecoin collateral deposit.

**Hedge proceeds**:
USDC value received by selling borrowed hedge asset. In this strategy hedge proceeds are deposited as USDC collateral inside the Kamino position.
_Avoid_: Off-account cash reserve

**Short proceeds**:
USDC value received by selling any borrowed short asset. Short proceeds are posted as USDC collateral inside the Kamino position.
_Avoid_: Off-account short cash

**Defensive mode**:
A bearish SOL Supertrend vote state, starting at one bullish and three bearish SOL Supertrend states, where hedge proceeds may be used to repay USDC debt after the target ETH hedge is established.
_Avoid_: Mild hedge state

**Full short mode**:
A core bearish strategy mode where ETH short exposure is no longer only a hedge against SOL collateral and may express net short crypto direction. SOL is never shorted. The harness keeps full short mode explicitly switchable so hedge-only and hedge-plus-full-short experiments can be compared.
_Avoid_: Hedge mode

**SOL-relative USD compounding objective**:
The strategy's ultimate goal of growing USD account value while clearing the opportunity-cost hurdle of simply holding SOL. USD growth alone is not success if SOL buy-and-hold massively outperforms over the same window. SOL may be bought, held, or used as collateral, but not shorted.
_Avoid_: USD-only return objective; SOL-only accumulation objective

**Higher-regime confirmation**:
A slower SOL trend confirmation required before hedge mode can escalate into full short mode. The initial confirmations are 3d and 1w SOL Supertrend states, and they gate entry and scale-up rather than exit.
_Avoid_: Extra timeframe when it specifically gates full short mode.

**Full short scale-up**:
The staged increase from the lower full-short exposure bound to the maximum full-short exposure as higher-regime confirmations strengthen.
_Avoid_: Instant max short

**Full short exposure bounds**:
The ETH short targets used in full short mode. Initial bounds are 100% of SOL collateral value with 3d bearish confirmation and 150% with both 3d and 1w bearish confirmations.
_Avoid_: Hedge ladder targets

**Short-cover ladder**:
The bullish-vote exit ladder used while already in full short mode. Increasing bullish votes reduce ETH short exposure, and a four-bullish-vote state cuts the ETH short aggressively to zero.
_Avoid_: Hedge ladder

**Hedge ratio**:
The target ETH short notional expressed as a fraction of the current SOL collateral value.
_Avoid_: Allocation weight

**Signal-discipline hedge unwind**:
Reducing ETH short exposure to the hedge ladder target when the SOL Supertrend vote improves, regardless of whether the ETH short is profitable.
_Avoid_: Profit-driven hedge hold

**Supertrend vote**:
The count of bullish and bearish Supertrend states across the strategy's four configured SOL timeframes.
_Avoid_: Signal when referring specifically to the cross-timeframe count.

**Hedge ladder**:
The mapping from Supertrend vote state to target hedge ratio. The initial four-timeframe ladder is 0%, 25%, 50%, 75%, and 75% for increasingly bearish SOL vote states.
_Avoid_: Risk model

**Closed-candle signal**:
A signal computed only from candles whose interval has fully completed before the decision bar. Higher-timeframe Supertrend states are forward-filled from the latest completed candle.
_Avoid_: In-progress candle signal

**Minimum rebalance health factor**:
The lowest acceptable Kamino health factor after a hedge rebalance. If the desired hedge would breach this value, the executable hedge is reduced.
_Avoid_: Stop loss

**Emergency de-risk**:
A safety-driven rebalance used when health factor falls below the minimum rebalance health factor. It stops new debt, reduces excess ETH debt, repays USDC debt toward defensive targets, sells SOL to repay USDC debt if needed, and lets survival override signal targets.
_Avoid_: Normal rebalance

**Bullish rebalance**:
A rebalance triggered by an improving SOL Supertrend vote. The priority is to reduce ETH short exposure first, then increase SOL collateral using available USDC value and, when permitted, additional USDC debt.
_Avoid_: Unwind when additional SOL buying is included.

**Surplus USDC reinvestment**:
Using existing USDC collateral or realized hedge proceeds to buy and deposit more SOL without borrowing new USDC. It is distinct from USDC releverage because it converts existing account value rather than increasing debt.
_Avoid_: Releverage; borrowing to buy SOL

**Bullish reinvestment priority**:
The bullish recovery ordering where ETH short exposure is reduced to the current hedge target first, a USDC safety buffer is preserved second, and only remaining surplus USDC is used to buy and deposit SOL.
_Avoid_: Buying SOL before hedge unwind

**Hedge unwind capital**:
USDC value reserved or spent to buy back ETH debt until ETH short exposure reaches the current hedge target.
_Avoid_: Reinvestable cash; surplus

**Realized hedge profit surplus**:
Positive cumulative hedge realized PnL remaining after ETH short exposure has actually been reduced to the current hedge target and the USDC safety buffer has been preserved. It is the only hedge-derived USDC eligible for surplus reinvestment, and prior realized hedge losses must be recovered before the surplus can be positive.
_Avoid_: Paper hedge profit; raw USDC collateral

**Hedge realized PnL ledger**:
The accounting record that tracks ETH short proceeds when exposure is opened and ETH cover cost when exposure is reduced. Realized hedge profit surplus is derived from cumulative realized PnL in this ledger rather than inferred from raw USDC balances.
_Avoid_: Balance-derived hedge profit

**Spendable hedge profit pool**:
The portion of positive cumulative hedge realized PnL that has not already been consumed by surplus reinvestment. Buying SOL through surplus reinvestment reduces this pool, while lifetime realized PnL remains available for reporting.
_Avoid_: Reusing the same hedge profit across multiple buys

**Hedge profit spend policy**:
The configurable choice of whether USDC debt cleanup consumes the spendable hedge profit pool. The conservative default treats both surplus reinvestment and hedge-profit-funded debt cleanup as spending from the same pool.
_Avoid_: Implicitly double-counting hedge profit

**Aggregate ETH short basis**:
The averaged entry basis used by the hedge realized PnL ledger for the open ETH short. It tracks total borrowed ETH amount and average sale price rather than FIFO lots.
_Avoid_: Tax-lot accounting

**Realized hedge profit gate**:
A configurable threshold requiring realized hedge profit surplus to exceed a chosen fraction of SOL collateral value before surplus USDC reinvestment can buy more SOL.
_Avoid_: Reinvesting every positive hedge PnL

**Surplus reinvestment ladder**:
The bullish-vote sizing rule for converting eligible realized hedge profit surplus into additional SOL collateral. The initial ladder reinvests 25% of eligible surplus at three bullish votes and 50% at four bullish votes after the realized hedge profit gate is met.
_Avoid_: All-in reinvestment; raw surplus spend

**Surplus reinvestment cap**:
The maximum amount of eligible realized hedge profit surplus that can be converted into SOL collateral in a single rebalance, expressed as a fraction of SOL collateral value. The initial cap is 5% of SOL collateral value per rebalance.
_Avoid_: Spending the whole eligible surplus at once

**Surplus reinvestment cadence**:
Surplus USDC reinvestment uses the existing rebalance cooldown rather than a separate cooldown clock.
_Avoid_: Independent reinvestment cooldown

**Bullish re-lever**:
The part of a bullish rebalance that borrows USDC, buys SOL, and deposits that SOL as collateral.
_Avoid_: Buy SOL when the purchase is funded by new debt.

**Strong bullish vote**:
A four-timeframe SOL Supertrend vote where all configured timeframes are bullish.
_Avoid_: Mostly bullish

**Target bullish health factor**:
The Kamino health factor the strategy aims for after adding USDC debt to buy SOL during a bullish re-lever. The initial value is 1.35.
_Avoid_: Minimum health factor

**Dynamic USDC debt budget**:
A growth-linked cap on USDC debt, expressed as a multiple of account equity rather than as a fixed dollar ceiling. The initial cap is 1.0x equity.
_Avoid_: Fixed debt ceiling

**USDC releverage module**:
The optional high-risk module that borrows USDC to buy and deposit more SOL during strong bullish regimes. It is separate from the core ETH hedge harness and should be tested as an explicitly labeled aggressive growth experiment.
_Avoid_: Default strategy behavior; hedge behavior; surplus USDC reinvestment

**USDC debt repayment threshold**:
A high-health-factor threshold above which the strategy may trim USDC debt. USDC debt is growth leverage and is otherwise repaid only when required for safety.
_Avoid_: Hedge unwind threshold

**Defensive USDC debt target**:
The configurable target USDC debt level in defensive mode, expressed as a fraction of the dynamic USDC debt budget. Initial defaults are 50% at one bullish vote and 0% at zero bullish votes.
_Avoid_: Fixed repayment amount; initial optimization parameter

**Defensive carried USDC debt**:
USDC debt that may remain open while the 1w SOL Supertrend is bearish, provided the account is protected by an aggressive ETH hedge or full-short posture.
_Avoid_: Growth debt; unhedged dollar debt

**Aggressive hedge floor**:
The minimum ETH short exposure required when defensive carried USDC debt exists in a bearish 1w SOL regime. The initial floor is 100% of SOL collateral value, capped by health factor.
_Avoid_: Mild hedge while carrying USDC debt

**Under-hedged defensive debt**:
Defensive carried USDC debt that cannot be protected by the aggressive hedge floor within health factor limits. New USDC debt stops immediately, and existing USDC debt is reduced until the remaining debt can be defended by the maximum safe ETH hedge.
_Avoid_: Carrying oversized USDC debt without hedge capacity

**Green-regime USDC debt repayment**:
The rule that a bullish 1w SOL Supertrend makes USDC debt repayment a priority before surplus reinvestment or new USDC releverage.
_Avoid_: Waiting for all short-term votes to be green before repaying USDC debt

**USDC-first debt cleanup**:
Routine USDC debt repayment preserves the USDC safety buffer, then uses remaining USDC collateral or hedge-derived USDC before selling SOL. SOL is sold to repay USDC debt only when required for safety.
_Avoid_: Selling SOL for normal green-regime repayment

**USDC safety buffer**:
USDC collateral value preserved before debt cleanup or surplus reinvestment so projected health factor remains above the configured safety threshold. The initial surplus reinvestment threshold is a projected health factor of 2.0.
_Avoid_: Fixed cash buffer

**Close-price execution**:
Backtest execution that fills swaps at the decision bar close price with a configurable fee or slippage haircut.
_Avoid_: Intrabar execution

**Optimization objective**:
The metric used to rank hyperparameter runs in the harness. For this strategy, optimization is two-layered: first apply a SOL benchmark gate, then rank surviving candidates by USD risk-adjusted growth such as Sortino or CAGR-to-drawdown.
_Avoid_: Best result without naming the metric; USD Sortino without a SOL benchmark gate

**SOL benchmark gate**:
A pass/fail hurdle requiring a strategy candidate to beat, or remain within an explicitly accepted tracking gap of, unlevered SOL buy-and-hold over the same backtest window before USD risk-adjusted returns can rank it highly.
_Avoid_: Benchmark as an informational chart only

**Tiered SOL benchmark gate**:
The SOL benchmark gate expressed as candidate tiers instead of a single cutoff. A pass beats SOL buy-and-hold in USD while reducing drawdown. An acceptable candidate may trail SOL by a limited configured gap if it materially improves drawdown or liquidation risk. A reject trails SOL beyond the configured gap regardless of USD Sortino.
_Avoid_: All-or-nothing benchmark comparison

**Capital preservation tier**:
A separate candidate tier for strategies that trail SOL beyond the normal acceptable gap but avoid catastrophic loss or materially reduce drawdown. These runs may be useful defensively, but they are not labeled as successful SOL-relative USD compounding.
_Avoid_: Treating defensive underperformance as a normal pass

**Rebalance cooldown**:
The minimum number of decision bars the strategy waits after a rebalance before trading again, except for safety-driven defensive action.
_Avoid_: Signal delay

**Initial position**:
The Kamino position at the start of a backtest before any valid four-timeframe signal exists. The initial position is unlevered SOL collateral only.
_Avoid_: Starting allocation

**Primary benchmark**:
The baseline used to judge whether the strategy adds value. The primary benchmark is unlevered SOL buy-and-hold with the same starting SOL amount.
_Avoid_: Benchmark without naming the baseline.

**Strategy event log**:
A separate record of hedge, unwind, re-lever, safety-cap, and cooldown decisions with pre- and post-trade exposure and health factor context.
_Avoid_: History when referring to decision-level explanations.

**Strategy price universe**:
The market data required by the strategy. The initial universe is aligned SOL and ETH OHLCV, with USDC treated as fixed at one USD.
_Avoid_: Asset universe when referring specifically to required price inputs.

**Strategy operation**:
A domain-level rebalance action such as opening an ETH short into USDC collateral, repaying ETH short from USDC collateral, or borrowing USDC to buy SOL.
_Avoid_: Raw action when the intent spans multiple account changes.

## Example Dialogue

Dev: "Is this a portfolio allocation strategy?"

Domain expert: "No. It is a Kamino position strategy. SOL is the long asset, ETH is the hedge asset, and short exposure means borrowing ETH and selling it into USDC collateral."

Dev: "What controls hedge size?"

Domain expert: "The SOL Supertrend vote controls the hedge ratio, which targets ETH short notional as a fraction of SOL collateral value."

Dev: "Can ETH hedge proceeds repay USDC debt?"

Domain expert: "Yes, but only in defensive mode. Early hedge proceeds stay as USDC collateral; deeper bearish states may use them to reduce bullish USDC debt."

Dev: "Where do proceeds from full short positions go?"

Domain expert: "Short proceeds are posted as USDC collateral inside the Kamino position."

Dev: "Can full short mode short SOL?"

Domain expert: "No. The objective is to maximize SOL, so SOL is never shorted."

Dev: "When can hedge mode escalate into full short mode?"

Domain expert: "Only when all standard SOL Supertrend timeframes are bearish and a higher-regime confirmation is also bearish."

Dev: "How do the 3d and 1w confirmations affect full short size?"

Domain expert: "Use full short scale-up: 3d bearish unlocks the lower full-short bound, and 1w bearish allows the maximum full-short exposure."

Dev: "How large can full short mode get?"

Domain expert: "Initial full short exposure bounds are 100% and 150% of SOL collateral value, still capped by account health."

Dev: "How does full short mode close?"

Domain expert: "Use the short-cover ladder. In full short mode, bullish standard votes work in the other direction and reduce ETH short exposure; four bullish votes cut the short aggressively."

Dev: "Can 3d or 1w bearish signals prevent covering a full short?"

Domain expert: "No. Higher-regime confirmations gate entry and scale-up, not exit. The short-cover ladder can reduce exposure immediately."

Dev: "Can a profitable ETH short stay open after SOL signals improve?"

Domain expert: "No. Signal discipline controls hedge sizing; hedge PnL is reported but does not override the target hedge ratio."

Dev: "How much ETH should be shorted?"

Domain expert: "Use the hedge ladder: fully bullish SOL has no ETH short, and increasingly bearish four-timeframe votes target 25%, 50%, 75%, then 75% of SOL collateral value."

Dev: "Can the daily Supertrend use today's incomplete candle?"

Domain expert: "No. Signals are closed-candle signals; a 1h decision bar can only see the latest completed 4h, 8h, or 1d candle."

Dev: "What wins if the hedge ladder wants more ETH short than the account can safely carry?"

Domain expert: "The minimum rebalance health factor wins. The strategy scales down the executable hedge rather than breaching the safety cap."

Dev: "What wins if the account becomes unsafe?"

Domain expert: "Emergency de-risk wins. Survival overrides the signal target."

Dev: "What happens when SOL turns bullish again?"

Domain expert: "Run a bullish rebalance: unwind the ETH short as the first priority, then try to buy additional SOL with available USDC value."

Dev: "Can hedge profit buy SOL before the ETH debt is actually reduced?"

Domain expert: "No. Hedge-derived USDC becomes realized hedge profit surplus only after ETH short exposure has been reduced to the current hedge target and the safety buffer is preserved."

Dev: "How is hedge profit measured?"

Domain expert: "Use the hedge realized PnL ledger. Track ETH short proceeds when opening exposure and cover cost when reducing it, rather than inferring profit from raw USDC balances."

Dev: "Do realized hedge losses affect future reinvestment?"

Domain expert: "Yes. The ledger uses cumulative realized PnL, so prior realized hedge losses must be recovered before realized hedge profit surplus can become positive."

Dev: "Does buying SOL consume hedge profit surplus?"

Domain expert: "Yes. Surplus reinvestment spends from the spendable hedge profit pool, while lifetime realized PnL remains available for reporting."

Dev: "Does USDC debt cleanup consume hedge profit surplus?"

Domain expert: "That is controlled by the hedge profit spend policy. The conservative setting consumes the spendable hedge profit pool when realized hedge profit is used for either SOL buying or USDC debt cleanup."

Dev: "Does the hedge ledger need FIFO lots?"

Domain expert: "No. Use aggregate ETH short basis: total borrowed ETH amount and average sale price are enough for strategy backtesting."

Dev: "How much realized hedge profit surplus can be reinvested?"

Domain expert: "Only after the realized hedge profit gate is met. The initial surplus reinvestment ladder buys SOL with 25% of eligible surplus at three bullish votes and 50% at four bullish votes."

Dev: "Can a large hedge win be reinvested all at once?"

Domain expert: "No. The surplus reinvestment cap limits each rebalance to an initial maximum of 5% of SOL collateral value."

Dev: "Does surplus reinvestment have a separate cooldown?"

Domain expert: "No. It uses the existing rebalance cooldown."

Dev: "Can bullish rebalancing add new debt?"

Domain expert: "Yes. After reducing ETH short exposure, the strategy may borrow USDC to buy and deposit additional SOL."

Dev: "When can the strategy add new USDC debt?"

Domain expert: "Only on a strong bullish vote. Mixed bullish states may maintain existing USDC debt but do not add more."

Dev: "How much USDC should the strategy borrow when bullish?"

Domain expert: "Borrow enough USDC to buy SOL until the account approaches the target bullish health factor, while respecting the minimum rebalance health factor."

Dev: "Should USDC debt use a fixed dollar ceiling?"

Domain expert: "No. Use a dynamic USDC debt budget so the allowable debt can grow with account equity. Repay USDC debt only at a very healthy level or when required for safety."

Dev: "How much USDC debt should hedge proceeds repay in defensive mode?"

Domain expert: "Repay toward configurable defensive USDC debt targets, not an all-or-nothing amount."

Dev: "Can USDC debt stay open while the weekly SOL regime is red?"

Domain expert: "Yes, but treat it as defensive carried USDC debt. It must be paired with aggressive ETH hedge protection or full-short posture."

Dev: "How much ETH hedge protection is required for defensive carried USDC debt?"

Domain expert: "Use the aggressive hedge floor: initially at least 100% of SOL collateral value in ETH short exposure, capped by health factor."

Dev: "What if the account cannot safely reach the aggressive hedge floor?"

Domain expert: "Then the position has under-hedged defensive debt. Stop adding USDC debt and reduce existing USDC debt until the remaining debt can be defended by the maximum safe ETH hedge."

Dev: "When does USDC debt repayment become a priority?"

Domain expert: "When the 1w SOL Supertrend turns green. Green-regime USDC debt repayment happens before surplus reinvestment or new USDC releverage."

Dev: "Can routine green-regime USDC debt repayment sell SOL?"

Domain expert: "No. Use USDC-first debt cleanup; preserve the USDC safety buffer, then use remaining USDC, and sell SOL only when required for safety."

Dev: "How is the USDC safety buffer sized?"

Domain expert: "Use projected health factor rather than fixed dollars. Initially, preserve enough USDC collateral so cleanup or reinvestment leaves projected health factor at or above 2.0."

Dev: "How are swaps priced in the first harness?"

Domain expert: "Use close-price execution with a configurable fee or slippage haircut."

Dev: "How should the harness choose the best hyperparameters?"

Domain expert: "Rank runs by Sortino ratio, while still showing liquidation, health factor, drawdown, and return metrics."

Dev: "Can the strategy rebalance every hour?"

Domain expert: "No. Use a rebalance cooldown to reduce churn, while allowing immediate safety-driven defensive action."

Dev: "Does the backtest start already hedged or levered?"

Domain expert: "No. The initial position is unlevered SOL collateral only; the first valid four-timeframe signal builds any hedge or leverage."

Dev: "What should the strategy be compared against?"

Domain expert: "Use unlevered SOL buy-and-hold as the primary benchmark."

Dev: "How do we explain why one hyperparameter run won?"

Domain expert: "Use the strategy event log to inspect votes, target hedge ratios, executed actions, exposure changes, and health factor before and after rebalances."

Dev: "Which prices does the strategy need?"

Domain expert: "Use aligned SOL and ETH OHLCV. SOL drives Supertrend votes, ETH prices size and value the short, and USDC is fixed at one USD."

Dev: "Should strategy decisions be described as raw deposits, borrows, and repays?"

Domain expert: "No. Use strategy operations for multi-step decisions, then account for their effects on collateral, debt, and health factor."
