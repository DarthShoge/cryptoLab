# Cap Surplus Reinvestment Per Rebalance

Surplus USDC reinvestment will be capped per rebalance, initially at 5% of SOL collateral value, even after the realized hedge profit gate and surplus reinvestment ladder allow buying SOL. We chose this over unrestricted reinvestment because a large hedge win can occur near volatile regime transitions, and spending it all on the first bullish recovery signal would turn a defensive payoff into concentrated timing risk.

## Consequences

The strategy will accumulate SOL more gradually after profitable hedges. This may trail fast recoveries, but it makes surplus reinvestment less dependent on a single recovery candle or cooldown window.
