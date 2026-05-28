# Use Realized PnL Ledger For Hedge Profit

Hedge profit will be measured with a realized PnL ledger that records ETH short proceeds when exposure is opened and ETH cover cost when exposure is reduced. We chose this over deriving profit from current balances because USDC collateral can also represent safety buffer, debt repayment capacity, reinvested proceeds, fees, or unrelated cash flows.

## Consequences

The backtest must maintain explicit accounting state for hedge lots or aggregate realized PnL. This adds implementation complexity, but it gives surplus reinvestment a defensible source of truth.
