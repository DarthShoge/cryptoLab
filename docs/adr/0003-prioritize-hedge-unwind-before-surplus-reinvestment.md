# Prioritize Hedge Unwind Before Surplus Reinvestment

During bullish recovery, the SOL Supertrend strategy will first reduce ETH short exposure toward the current hedge ladder target, then preserve a USDC safety buffer, and only then reinvest surplus USDC into SOL collateral. We chose this ordering because buying more SOL while excess ETH debt remains can leave the account simultaneously long SOL and over-short ETH, which obscures whether bullish recovery is reducing risk or adding a new directional bet.

## Consequences

Surplus USDC reinvestment should be gated by bullish confirmation and available safety buffer, not by raw USDC balance alone. The strategy may miss some early upside while it unwinds hedge risk, but the resulting position is easier to reason about and less likely to carry stale full-short exposure into a bull regime.
