# Make Hedge Profit Spend Policy Configurable

Whether USDC debt cleanup consumes the spendable hedge profit pool will be configurable. The conservative setting treats both surplus reinvestment and hedge-profit-funded USDC debt cleanup as spending from the same pool, while the alternative lets debt cleanup occur without reducing later SOL reinvestment capacity.

## Consequences

This keeps the backtest honest about the double-counting risk while allowing experiments with a more aggressive interpretation of hedge profits. Reports should label the selected hedge profit spend policy so results remain comparable.
