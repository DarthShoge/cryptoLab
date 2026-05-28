# Require Actual Hedge Unwind Before Reinvestment

Hedge-derived USDC will not be treated as reinvestable profit until ETH short exposure has actually been reduced to the current hedge target and the safety buffer is preserved. We chose this stricter rule over reserving USDC on paper because it keeps bullish recovery from carrying stale ETH debt while also buying more SOL, making the account's risk state easier to reason about.

## Consequences

The strategy may reinvest later than a more aggressive model, especially during fast V-shaped recoveries. In exchange, reported reinvestment is tied to realized hedge unwinds rather than a raw USDC balance that may still be needed to cover ETH debt.
