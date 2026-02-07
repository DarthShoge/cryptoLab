# Continuation Prompt for Next Agent

You are continuing work on the Kamino liquidation risk simulator in `/workspace/cryptoLab`.
The current code can load on-chain obligations via Solana RPC + Anchor IDL, but it still
needs real program/IDL discovery and a clean end-to-end run against a specific obligation.

## Current status
- CLI supports JSON input or on-chain loading:
  - `kamino_sim.py --input <file>`
  - `kamino_sim.py --obligation <addr> --program-id <id> --idl <idl.json> --rpc-url <rpc>`
- On-chain loader is in `arblab/kamino_onchain.py` and uses `anchorpy` to decode accounts.
- Dependencies added: `anchorpy`, `solana` (see `requirements.txt`).

## Goal
Run the simulator against the user's obligation account:
`F8ir9FxMgi17DpnLDbkM6mxy5GmS1o8ynmtP73HuHzQL`
and produce liquidation metrics using real on-chain data.

## What you must do next
1) **Find the correct Kamino lending program ID**:
   - Use Solana RPC `getAccountInfo` on the obligation address.
   - The account `owner` field is the program ID.

2) **Obtain the Anchor IDL JSON for that program**:
   - Try `anchor idl fetch <program_id> -o kamino_idl.json`.
   - If that fails, look in Kamino’s GitHub repos for an `idl/` or `target/idl/` JSON.
   - If the IDL is not published on-chain, you may need to locate a public repo or
     reach out for the IDL from Kamino’s sources.

3) **Run the CLI**:
   ```bash
   python kamino_sim.py \
     --obligation F8ir9FxMgi17DpnLDbkM6mxy5GmS1o8ynmtP73HuHzQL \
     --program-id <PROGRAM_ID> \
     --idl kamino_idl.json \
     --rpc-url https://api.mainnet-beta.solana.com
   ```
   Capture output and confirm collateral/debt entries look reasonable.

4) **If decoding fails**:
   - Inspect actual account layout and adjust field names in `kamino_onchain.py`.
   - Key expected fields: obligation `deposits`, `borrows`, reserve `config`, `liquidity`.

## Notes / constraints from previous environment
- GitHub and some RPC endpoints returned HTTP 403 here; the next environment should
  have better internet access.
- Kamino’s token mint (KMNO) is NOT the program ID. You must query the obligation
  account owner to find the program ID.

## Files to review
- `arblab/kamino_onchain.py`
- `arblab/kamino_risk.py`
- `kamino_sim.py`
- `requirements.txt`
