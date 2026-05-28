# Use USDC-First Debt Cleanup

Routine green-regime USDC debt repayment will use USDC collateral or hedge-derived USDC before selling SOL. We chose this because the strategy's objective is SOL-relative USD compounding, so selling SOL during recovery should be reserved for safety rather than normal debt cleanup.

## Consequences

USDC debt may remain open longer if available USDC collateral is insufficient. This preserves SOL exposure during recovery, while emergency de-risking can still sell SOL when survival requires it.
