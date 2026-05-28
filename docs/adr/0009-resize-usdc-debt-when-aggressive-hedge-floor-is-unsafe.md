# Resize USDC Debt When Aggressive Hedge Floor Is Unsafe

If defensive carried USDC debt exists in a bearish 1w SOL regime but the account cannot safely reach the aggressive hedge floor, the strategy will stop adding USDC debt and reduce existing USDC debt until the remaining debt can be defended by the maximum safe ETH hedge. We chose debt resizing over forcing the hedge because health factor limits must remain the hard safety boundary.

## Consequences

The strategy may repay USDC debt in bearish conditions when hedge capacity is insufficient. This sacrifices some optionality, but avoids a state where the account carries both dollar debt and inadequate hedge protection.
