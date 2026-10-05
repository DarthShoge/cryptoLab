"""Recognize deployed swap instructions, never infer them from program presence.

Primary sources checked 2026-09-30:
https://github.com/orca-so/whirlpools/blob/main/programs/whirlpool/src/lib.rs
https://github.com/orca-so/typescript-sdk/blob/main/src/public/utils/constants.ts
https://github.com/orca-so/typescript-sdk/blob/main/src/public/utils/web3/instructions/pool-instructions.ts
https://github.com/solana-labs/solana-program-library/blob/master/token-swap/js/src/index.ts
https://github.com/raydium-io/raydium-amm/blob/master/program/src/instruction.rs
https://github.com/raydium-io/raydium-cp-swap/blob/master/programs/cp-swap/src/lib.rs
https://github.com/raydium-io/raydium-clmm/blob/master/programs/amm/src/lib.rs
https://github.com/jup-ag/jupiter-cpi/blob/main/idl.json
https://t.me/s/jup_dev?before=157 (official V2 instruction announcement)
Anchor prefixes are SHA256(global:<snake_case_instruction>)[:8]. Minimum
payload sizes include required fixed fields and optional/vector length tags.
"""

import hashlib

import base58

JUPITER_PROGRAM = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"
WHIRLPOOL_PROGRAM = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
ORCA_TOKEN_SWAP_PROGRAM = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"
RAYDIUM_AMM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
RAYDIUM_CP_PROGRAM = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C"
RAYDIUM_CL_PROGRAM = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"


def _anchor_specs(specs):
    return {
        hashlib.sha256(f"global:{name}".encode()).digest()[:8]: (name, length)
        for name, length in specs.items()
    }


ANCHOR_PROGRAMS = {
    WHIRLPOOL_PROGRAM: (
        "Orca",
        _anchor_specs(
            {
                "swap": 42,
                "swap_v2": 43,
                "two_hop_swap": 59,
                "two_hop_swap_v2": 60,
            }
        ),
    ),
    RAYDIUM_CP_PROGRAM: (
        "Raydium",
        _anchor_specs({"swap_base_input": 24, "swap_base_output": 24}),
    ),
    RAYDIUM_CL_PROGRAM: (
        "Raydium",
        _anchor_specs({"swap": 41, "swap_v2": 41, "swap_router_base_in": 24}),
    ),
    JUPITER_PROGRAM: (
        "Jupiter",
        _anchor_specs(
            {
                "route": 31,
                "shared_accounts_route": 32,
                "exact_out_route": 23,
                "shared_accounts_exact_out_route": 32,
                "route_with_token_ledger": 23,
                "shared_accounts_route_with_token_ledger": 24,
                "route_v2": 34,
                "exact_out_route_v2": 34,
                "shared_accounts_route_v2": 35,
                "shared_accounts_exact_out_route_v2": 35,
            }
        ),
    ),
}
LEGACY_PROGRAMS = {
    ORCA_TOKEN_SWAP_PROGRAM: ("Orca", {1: "swap"}),
    RAYDIUM_AMM_PROGRAM: (
        "Raydium",
        {
            9: "swap_base_in",
            11: "swap_base_out",
            16: "swap_base_in_v2",
            17: "swap_base_out_v2",
        },
    ),
}
EVENT_PREFIX = bytes.fromhex("e445a52e51cb9a1d")


def swap_evidence(instructions):
    """Return named swap calls and whether a composite DEX action is ambiguous.

    Other instructions of a direct DEX (liquidity, rewards, position operations,
    unknown versions) invalidate aggregate execution quantities. Anchor event CPI
    instructions carry no asset transfer and are ignored. Jupiter setup/event
    calls are not execution evidence; a named route is required for attribution.
    """
    swaps, composite = [], False
    for instruction in instructions:
        program = instruction.get("programId")
        if program not in ANCHOR_PROGRAMS and program not in LEGACY_PROGRAMS:
            continue
        try:
            encoded = instruction.get("data")
            data = base58.b58decode(encoded) if isinstance(encoded, str) else b""
        except (ValueError, TypeError):
            data = b""
        if program in ANCHOR_PROGRAMS:
            protocol, specs = ANCHOR_PROGRAMS[program]
            if len(data) >= 16 and data[:8] == EVENT_PREFIX:
                continue
            spec = specs.get(data[:8])
            name = spec[0] if spec and len(data) >= spec[1] else None
        else:
            protocol, specs = LEGACY_PROGRAMS[program]
            name = specs.get(data[0]) if len(data) >= 17 else None
        if name:
            swaps.append(
                {"program": program, "protocol": protocol, "instruction": name}
            )
        elif program != JUPITER_PROGRAM:
            composite = True
    return swaps, composite
