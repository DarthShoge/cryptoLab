# Use Realized Hedge Profit Gate For Surplus Reinvestment

Surplus USDC reinvestment will require realized hedge profit surplus to exceed an initial 10% of SOL collateral value before buying more SOL. When the gate is met, the initial surplus reinvestment ladder reinvests 25% of eligible surplus at three bullish Supertrend votes and 50% at four bullish votes, favoring staged compounding over spending all defensive gains on the first recovery signal.

## Consequences

Small or noisy hedge wins stay as USDC collateral instead of immediately increasing SOL exposure. The threshold and ladder should be treated as optimization parameters, but the default posture is to make the hedge prove it has paid for a meaningful SOL accumulation opportunity.
