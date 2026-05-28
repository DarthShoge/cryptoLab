# Require Aggressive Hedge Floor For Defensive USDC Debt

When USDC debt is carried in a bearish 1w SOL regime, ETH short exposure must target an aggressive hedge floor, initially 100% of SOL collateral value and capped by health factor. This keeps defensive carried USDC debt from becoming unhedged growth leverage during a higher-regime downtrend.

## Consequences

If the account cannot safely support the aggressive hedge floor, the strategy should stop adding USDC debt and reduce risk. This can make bearish regimes more conservative, but it preserves the distinction between protected defensive debt and bullish USDC releverage.
