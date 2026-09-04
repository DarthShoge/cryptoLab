# Kamino Liquidation Risk Simulator - Status

> **Archived historical handoff (2026-02-07).** This document records a point-in-time development status. Its balances, health factors, addresses, and live-operation claims are stale and must not be used operationally. See the repository [README](../../README.md) for canonical setup, commands, and current project structure.

## Completed
The simulator is fully working end-to-end against live Kamino on-chain data.

### What was done
1. **Program ID discovered**: `KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD` (Kamino Lending mainnet)
2. **IDL obtained**: Downloaded from `Kamino-Finance/klend-sdk` repo (`src/idl/klend.json`) → saved as `data/fixtures/kamino_idl.json`
3. **Code updated** (`packages/arblab/src/arblab/kamino_onchain.py`):
   - Fixed anchorpy API: `Idl.from_json()` takes a string, `AccountsCoder.decode()` takes only bytes
   - Account names are PascalCase (`Obligation`, `Reserve`) in Anchor IDL
   - anchorpy decodes fields to snake_case attributes (not dicts)
   - Amounts use `Sf` (Scale Factor) 128-bit fixed-point format (scale = 2^60)
   - Obligation deposits/borrows are fixed-size arrays; empty slots filtered by zero amount
   - Reserve config uses `loan_to_value_pct` and `liquidation_threshold_pct` (u8 percentages)
   - Price is `market_price_sf` in reserve liquidity (also Sf format)
   - Added well-known Solana token mint → symbol mapping (SOL, USDC, USDT, PENGU, USDG, etc.)
4. **CLI defaults**: program-id now defaults to mainnet KLend address


### Point-in-time live-account details

The wallet, obligation addresses, balances, and health factors originally recorded here were intentionally removed because they are not durable project documentation. The prior values remain available in Git history.

## Usage

```bash
# Run against a specific obligation (program-id defaults to mainnet KLend):
uv run python -m kamino_simulator.cli \
  --obligation <obligation-address> \
  --idl data/fixtures/kamino_idl.json

# No arguments runs the bundled offline sample:
uv run python -m kamino_simulator.cli

# Or select that input explicitly:
uv run python -m kamino_simulator.cli --input data/fixtures/kamino_sample.json
```

## Possible future improvements
- Resolve all token symbols dynamically via on-chain token metadata (Metaplex)
- Add `--wallet` flag to auto-discover obligation accounts for a wallet
- Monte Carlo price simulation for liquidation probability estimation
- Support for elevation groups and e-mode LTV overrides

## Files
- `packages/arblab/src/arblab/kamino_onchain.py` - On-chain data loading via Solana RPC + anchorpy
- `packages/arblab/src/arblab/kamino_risk.py` - Core risk models and liquidation calculations
- `apps/kamino-simulator/src/kamino_simulator/cli.py` - CLI entry point
- `data/fixtures/kamino_idl.json` - Kamino Lending Anchor IDL (from klend-sdk)
- `pyproject.toml` and `uv.lock` - Python workspace metadata and locked dependencies
