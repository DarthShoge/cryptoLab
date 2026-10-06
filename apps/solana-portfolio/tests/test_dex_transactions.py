import hashlib

import base58
import pytest

from test_transactions import tx
from api.transactions import parse_transaction

WHIRLPOOL = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
ORCA_LEGACY = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"
RAYDIUM_AMM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
RAYDIUM_CP = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C"
RAYDIUM_CL = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"
SOL = "So11111111111111111111111111111111111111112"


def instruction(program, name, length):
    prefix = (
        bytes([name])
        if isinstance(name, int)
        else hashlib.sha256(f"global:{name}".encode()).digest()[:8]
    )
    return {
        "programId": program,
        "data": base58.b58encode(prefix + bytes(length - len(prefix))).decode(),
    }


@pytest.mark.parametrize(
    "program,name,length,protocol",
    [
        (WHIRLPOOL, "swap", 42, "Orca"),
        (WHIRLPOOL, "swap_v2", 43, "Orca"),
        (WHIRLPOOL, "two_hop_swap", 59, "Orca"),
        (WHIRLPOOL, "two_hop_swap_v2", 60, "Orca"),
        (ORCA_LEGACY, 1, 17, "Orca"),
        (RAYDIUM_AMM, 9, 17, "Raydium"),
        (RAYDIUM_AMM, 11, 17, "Raydium"),
        (RAYDIUM_AMM, 16, 17, "Raydium"),
        (RAYDIUM_AMM, 17, 17, "Raydium"),
        (RAYDIUM_CP, "swap_base_input", 24, "Raydium"),
        (RAYDIUM_CP, "swap_base_output", 24, "Raydium"),
        (RAYDIUM_CL, "swap", 41, "Raydium"),
        (RAYDIUM_CL, "swap_v2", 41, "Raydium"),
        (RAYDIUM_CL, "swap_router_base_in", 24, "Raydium"),
    ],
)
def test_direct_swap_instructions_classify_actual_balance_deltas(
    program, name, length, protocol
):
    data = tx(program)
    data["transaction"]["message"]["instructions"] = [
        instruction(program, name, length)
    ]
    record = parse_transaction(data, "wallet", "signature")
    assert record["type"] == "buy"
    assert record["protocol"] == protocol
    assert record["amount"] == 1
    assert record["price"] == 100
    assert record["quote"] == "USDC"
    assert len(record["executionLegs"]) == 2
    assert {leg["side"] for leg in record["executionLegs"]} == {"buy", "sell"}
    assert {leg["signedAmount"] for leg in record["executionLegs"]} == {1, -100}


@pytest.mark.parametrize(
    "program,name,length",
    [
        (WHIRLPOOL, "increase_liquidity", 40),
        (WHIRLPOOL, "decrease_liquidity_v2", 41),
        (RAYDIUM_AMM, 3, 25),
        (RAYDIUM_AMM, 4, 9),
        (ORCA_LEGACY, 2, 25),
        (RAYDIUM_CP, "deposit", 32),
        (RAYDIUM_CL, "increase_liquidity", 40),
    ],
)
def test_liquidity_instruction_is_not_a_swap_despite_two_opposite_deltas(
    program, name, length
):
    data = tx(program)
    data["transaction"]["message"]["instructions"] = [
        instruction(program, name, length)
    ]
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"


def test_composite_swap_and_liquidity_action_remains_unclassified():
    data = tx(WHIRLPOOL)
    data["transaction"]["message"]["instructions"] = [
        instruction(WHIRLPOOL, "swap", 42),
        instruction(WHIRLPOOL, "increase_liquidity", 40),
    ]
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"


def test_jupiter_route_through_orca_retains_aggregator_attribution():
    data = tx()
    data["meta"]["innerInstructions"] = [
        {"index": 0, "instructions": [instruction(WHIRLPOOL, "swap", 42)]}
    ]
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "buy"
    assert record["protocol"] == "Jupiter"
    assert record["venues"] == ["Orca"]


def test_program_touch_and_truncated_swap_payload_do_not_prove_execution():
    data = tx(WHIRLPOOL)
    data["transaction"]["message"]["instructions"] = [{"programId": WHIRLPOOL}]
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"
    data["transaction"]["message"]["instructions"] = [instruction(WHIRLPOOL, "swap", 8)]
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"
    data = tx()
    data["transaction"]["message"]["instructions"][0].pop("data")
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"


def test_token_sol_execution_is_priced_in_sol_and_preserves_both_legs():
    data = tx(ORCA_LEGACY)
    data["transaction"]["message"]["instructions"] = [instruction(ORCA_LEGACY, 1, 17)]
    for state in ("preTokenBalances", "postTokenBalances"):
        data["meta"][state][1]["mint"] = "unknown-token-mint"
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "sell"
    assert record["mint"] == "unknown-token-mint"
    assert record["amount"] == 100
    assert record["quote"] == "SOL"
    assert record["price"] == 0.01
    assert record["value"] == 1
    assert (
        next(leg for leg in record["executionLegs"] if leg["mint"] == SOL)["side"]
        == "buy"
    )


def test_unstable_token_pair_has_two_actual_legs_without_an_invented_quote():
    data = tx(WHIRLPOOL)
    data["transaction"]["message"]["instructions"] = [
        instruction(WHIRLPOOL, "swap", 42)
    ]
    for state in ("preTokenBalances", "postTokenBalances"):
        data["meta"][state][0]["mint"] = "token-a"
        data["meta"][state][1]["mint"] = "token-b"
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "swap"
    assert record["price"] is None
    assert record["quote"] is None
    assert {leg["mint"] for leg in record["executionLegs"]} == {"token-a", "token-b"}


def test_direct_orca_swap_separates_explicit_external_sol_outflow():
    data = tx(WHIRLPOOL)
    data["transaction"]["message"]["instructions"] = [
        instruction(WHIRLPOOL, "swap", 42),
        {
            "program": "system",
            "parsed": {
                "type": "transfer",
                "info": {
                    "source": "wallet",
                    "destination": "external-recipient",
                    "lamports": 3428,
                },
            },
        },
    ]
    data["meta"]["postBalances"][0] -= 3428
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "buy"
    assert record["amount"] == 1
    assert record["price"] == 100
    assert record["externalSolOutflowSol"] == 0.000003428
    assert record["feeSol"] == 0.000005


def test_direct_native_input_to_unowned_wrapping_account_is_not_guessed_as_a_tip():
    data = tx(WHIRLPOOL)
    data["transaction"]["message"]["instructions"] = [
        instruction(WHIRLPOOL, "swap", 42),
        {
            "program": "system",
            "parsed": {
                "type": "transfer",
                "info": {
                    "source": "wallet",
                    "destination": "other-wrapped",
                    "lamports": 3428,
                },
            },
        },
        {
            "program": "spl-token",
            "parsed": {
                "type": "initializeAccount",
                "info": {"owner": "program-authority", "account": "other-wrapped"},
            },
        },
    ]
    data["meta"]["postBalances"][0] -= 3428
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"


def test_jupiter_external_transfer_guard_still_rejects_composite_outflows():
    data = tx()
    data["transaction"]["message"]["instructions"].append(
        {
            "program": "system",
            "parsed": {
                "type": "transfer",
                "info": {
                    "source": "wallet",
                    "destination": "external-recipient",
                    "lamports": 3428,
                },
            },
        }
    )
    data["meta"]["postBalances"][0] -= 3428
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"


def test_token_stable_swap_is_recovered_when_third_sol_delta_is_exact_external_outflow():
    data = tx(WHIRLPOOL)
    for state in ("preTokenBalances", "postTokenBalances"):
        data["meta"][state][0]["mint"] = "traded-token"
    data["transaction"]["message"]["instructions"] = [
        instruction(WHIRLPOOL, "swap", 42),
        {
            "program": "system",
            "parsed": {
                "type": "transfer",
                "info": {
                    "source": "wallet",
                    "destination": "external-recipient",
                    "lamports": 3428,
                },
            },
        },
    ]
    data["meta"]["postBalances"][0] -= 3428
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "buy"
    assert record["mint"] == "traded-token"
    assert record["amount"] == 1
    assert record["price"] == 100
    assert "SOL" not in record["deltas"]
    assert len(record["executionLegs"]) == 2


def test_genuine_third_token_delta_stays_unclassified():
    data = tx(WHIRLPOOL)
    data["transaction"]["message"]["instructions"] = [
        instruction(WHIRLPOOL, "swap", 42)
    ]
    data["meta"]["postTokenBalances"].append(
        {
            "owner": "wallet",
            "mint": "third-token",
            "uiTokenAmount": {"amount": "1000000", "decimals": 6},
        }
    )
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "unclassified"
    assert len(record["deltas"]) == 3
