# Curated Kamino Research Universe

Date: 2026-06-16

This universe is the starting point for multi-asset traffic-light research. It is intentionally narrower than the live Kamino market list: every asset must already have modeled Kamino risk parameters in the backtester and a durable price source suitable for repeatable historical tests.

## V1 Universe

| Symbol | Directional research | Price source | Role |
| --- | --- | --- | --- |
| SOL | Yes | `SOL/USDT` exchange OHLCV | Primary benchmark, collateral, borrow leg |
| JitoSOL | Yes | `JITOSOL/USDT` exchange OHLCV | SOL liquid staking token collateral/borrow leg |
| mSOL | Yes | `MSOL/USDT` exchange OHLCV | SOL liquid staking token collateral/borrow leg |
| ETH | Yes | `ETH/USDT` exchange OHLCV | Major exchange asset and existing hedge leg |
| USDC | No | USD peg | Quote, reserve, collateral, debt, and cash leg |
| USDT | No | USD peg | Stable collateral/debt capacity |

## Inclusion Rules

1. The asset must be configured in `MarketParams.kamino_defaults()`.
2. Directional long/short candidates need both collateral and borrow support in the model.
3. Non-stable assets need a major-exchange OHLCV symbol or a documented derived-price method.
4. Stablecoins can be included as funding, reserve, and debt legs, but not as directional alpha candidates.
5. Live Kamino availability must be re-verified before expanding beyond this repo-supported V1 set.

## Research Implication

This gives us a clean first portfolio search surface without pretending we have solved live market discovery. The next step is to build a portfolio strategy that assigns traffic-light scores per directional asset, then chooses long collateral and short borrow legs from this curated list while preserving the current SOL checkpoint as the baseline.
