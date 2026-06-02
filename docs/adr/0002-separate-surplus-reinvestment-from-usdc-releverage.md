# Separate Surplus Reinvestment from USDC Releverage

The SOL Supertrend strategy will treat buying SOL with existing USDC collateral as **surplus USDC reinvestment**, separate from **USDC releverage**, which borrows new USDC to buy SOL. We chose this split because both actions increase SOL collateral, but they have different risk profiles: reinvestment converts existing account value, while releverage increases debt, health-factor pressure, and liquidation risk.

## Consequences

The harness should expose and optimize these modules separately. A strategy can accumulate more SOL without enabling USDC releverage, and USDC releverage remains an explicitly labeled high-risk experiment rather than the default path for bullish SOL buying.
