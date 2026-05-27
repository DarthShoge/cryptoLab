# Model ETH Hedge as a Kamino Short

The SOL Supertrend strategy will model its ETH hedge as true short exposure inside the Kamino account: borrow ETH, sell it for USDC, and deposit the USDC as collateral. We chose this over signal-only hedging, SOL-to-ETH rotation, or off-account cash because it preserves the lending account's health factor, collateral, debt, and liquidation mechanics in the backtest harness.

## Considered Options

- Signal-only ETH hedge: simpler, but it would not model the economic exposure the strategy intends to carry.
- Rotate SOL exposure into ETH collateral: useful for relative-strength allocation, but it is not a short hedge.
- Borrow ETH and keep proceeds as off-account cash: easier to implement with the current engine, but it would understate Kamino collateral and health-factor effects.
- Borrow ETH, sell to USDC, and deposit USDC collateral: more accounting work, but it best matches the intended Kamino position.

## Consequences

The strategy needs domain-level operations for opening and closing ETH short exposure, swap fee/slippage assumptions, ETH and SOL price data, and risk caps that prevent target hedges from breaching the minimum rebalance health factor.
