# Kamino Liquidation Risk Simulator - Status

## Completed
The simulator is fully working end-to-end against live Kamino on-chain data.

### What was done
1. **Program ID discovered**: `KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD` (Kamino Lending mainnet)
2. **IDL obtained**: Downloaded from `Kamino-Finance/klend-sdk` repo (`src/idl/klend.json`) → saved as `kamino_idl.json`
3. **Code updated** (`arblab/kamino_onchain.py`):
   - Fixed anchorpy API: `Idl.from_json()` takes a string, `AccountsCoder.decode()` takes only bytes
   - Account names are PascalCase (`Obligation`, `Reserve`) in Anchor IDL
   - anchorpy decodes fields to snake_case attributes (not dicts)
   - Amounts use `Sf` (Scale Factor) 128-bit fixed-point format (scale = 2^60)
   - Obligation deposits/borrows are fixed-size arrays; empty slots filtered by zero amount
   - Reserve config uses `loan_to_value_pct` and `liquidation_threshold_pct` (u8 percentages)
   - Price is `market_price_sf` in reserve liquidity (also Sf format)
   - Added well-known Solana token mint → symbol mapping (SOL, USDC, USDT, PENGU, USDG, etc.)
4. **CLI defaults**: program-id now defaults to mainnet KLend address


### User's wallet & obligations
- Wallet: `F8ir9FxMgi17DpnLDbkM6mxy5GmS1o8ynmtP73HuHzQL`
- Obligation 1: `J3cQ7pkaR7xLXCPEV1xgyGFvryhMXJU4fy8ZSCHxaZSU`
  - Collateral: USDG + SOL (~$5,123)
  - Debt: USDC (~$3,669)
  - Health factor: ~1.07
- Obligation 2: `3HUVJerBFwycMVkrSSdBKs2z5LYzKkmxHBvayga5nq1j`
  - Collateral: PENGU (~$247)
  - Debt: USDC (~$95)
  - Health factor: ~1.30

## Usage

```bash
# Run against a specific obligation (program-id defaults to mainnet KLend):
python kamino_sim.py \
  --obligation J3cQ7pkaR7xLXCPEV1xgyGFvryhMXJU4fy8ZSCHxaZSU \
  --idl kamino_idl.json

# Or with JSON input:
python kamino_sim.py --input positions.json
```

## Possible future improvements
- Resolve all token symbols dynamically via on-chain token metadata (Metaplex)
- Add `--wallet` flag to auto-discover obligation accounts for a wallet
- Monte Carlo price simulation for liquidation probability estimation
- Support for elevation groups and e-mode LTV overrides

## Files
- `arblab/kamino_onchain.py` - On-chain data loading via Solana RPC + anchorpy
- `arblab/kamino_risk.py` - Core risk models and liquidation calculations
- `kamino_sim.py` - CLI entry point
- `kamino_idl.json` - Kamino Lending Anchor IDL (from klend-sdk)
- `requirements.txt` - Python dependencies
