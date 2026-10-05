import sys
import hashlib
from pathlib import Path

import base58

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.transactions import parse_transaction


def tx(program="JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4", failed=False):
    return {
        "blockTime": 100,
        "transaction": {
            "message": {
                "accountKeys": [{"pubkey": "wallet"}],
                "instructions": [
                    {
                        "programId": program,
                        "data": base58.b58encode(
                            hashlib.sha256(b"global:route").digest()[:8] + bytes(24)
                        ).decode(),
                    }
                ],
            }
        },
        "meta": {
            "err": "failed" if failed else None,
            "fee": 5000,
            "preBalances": [1000000000],
            "postBalances": [999995000],
            "preTokenBalances": [
                {
                    "owner": "wallet",
                    "mint": "So11111111111111111111111111111111111111112",
                    "uiTokenAmount": {"amount": "1000000000", "decimals": 9},
                },
                {
                    "owner": "wallet",
                    "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "uiTokenAmount": {"amount": "1000000000", "decimals": 6},
                },
            ],
            "postTokenBalances": [
                {
                    "owner": "wallet",
                    "mint": "So11111111111111111111111111111111111111112",
                    "uiTokenAmount": {"amount": "2000000000", "decimals": 9},
                },
                {
                    "owner": "wallet",
                    "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "uiTokenAmount": {"amount": "900000000", "decimals": 6},
                },
            ],
        },
    }


def test_swap_has_actual_amount_and_execution_price():
    result = parse_transaction(tx(), "wallet", "signature")
    assert result["type"] == "buy"
    assert result["asset"] == "SOL"
    assert result["amount"] == 1
    assert result["price"] == 100
    assert result["funding"] == "unknown"


def test_transfers_and_borrows_are_not_guessed_as_trades():
    assert (
        parse_transaction(tx(program="other"), "wallet", "sig")["type"]
        == "unclassified"
    )
    assert (
        parse_transaction(
            tx(program="KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD"), "wallet", "sig"
        )["type"]
        == "kamino"
    )


def test_failed_transactions_do_not_create_execution_markers():
    assert parse_transaction(tx(failed=True), "wallet", "sig") is None


def test_native_sol_swap_removes_network_fee_from_execution_quantity():
    data = tx()
    data["meta"]["preTokenBalances"] = [data["meta"]["preTokenBalances"][1]]
    data["meta"]["postTokenBalances"] = [data["meta"]["postTokenBalances"][1]]
    data["meta"]["postBalances"] = [1_999_995_000]
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "buy"
    assert record["asset"] == "SOL"
    assert record["amount"] == 1
    assert record["price"] == 100


def test_created_token_account_rent_is_not_counted_as_sol_sold():
    data = tx()
    data["transaction"]["message"]["accountKeys"].append({"pubkey": "ata"})
    data["meta"]["preTokenBalances"] = []
    data["meta"]["postTokenBalances"] = [
        {
            "accountIndex": 1,
            "owner": "wallet",
            "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "uiTokenAmount": {"amount": "100000000", "decimals": 6},
        }
    ]
    data["meta"]["preBalances"] = [2_000_000_000, 0]
    data["meta"]["postBalances"] = [997_955_720, 2_039_280]
    data["transaction"]["message"]["instructions"].append(
        {
            "program": "system",
            "parsed": {
                "type": "createAccount",
                "info": {
                    "source": "wallet",
                    "newAccount": "ata",
                    "lamports": 2_039_280,
                },
            },
        }
    )
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "sell"
    assert record["amount"] == 1
    assert record["price"] == 100


def test_unwrapping_sol_is_not_an_execution():
    data = tx()
    data["transaction"]["message"]["accountKeys"].append({"pubkey": "wrapped-account"})
    data["meta"]["preTokenBalances"] = [
        {
            "accountIndex": 1,
            "owner": "wallet",
            "mint": "So11111111111111111111111111111111111111112",
            "uiTokenAmount": {"amount": "1000000000", "decimals": 9},
        }
    ]
    data["meta"]["postTokenBalances"] = []
    data["meta"]["preBalances"] = [1_000_000_000, 1_002_039_280]
    data["meta"]["postBalances"] = [2_002_034_280, 0]
    data["transaction"]["message"]["instructions"].append(
        {
            "program": "spl-token",
            "parsed": {
                "type": "closeAccount",
                "info": {"account": "wrapped-account", "destination": "wallet"},
            },
        }
    )
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"


def test_sponsored_token_account_rent_is_not_added_to_wallet_sol_delta():
    data = tx()
    data["transaction"]["message"]["accountKeys"] = [
        {"pubkey": key} for key in ("wallet", "wrapped", "stable-ata", "sponsor")
    ]
    data["meta"]["preTokenBalances"] = [
        {
            "accountIndex": 1,
            "owner": "wallet",
            "mint": "So11111111111111111111111111111111111111112",
            "uiTokenAmount": {"amount": "1000000000", "decimals": 9},
        }
    ]
    data["meta"]["postTokenBalances"] = [
        {
            "accountIndex": 1,
            "owner": "wallet",
            "mint": "So11111111111111111111111111111111111111112",
            "uiTokenAmount": {"amount": "0", "decimals": 9},
        },
        {
            "accountIndex": 2,
            "owner": "wallet",
            "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "uiTokenAmount": {"amount": "100000000", "decimals": 6},
        },
    ]
    data["meta"]["preBalances"] = [1_000_000_000, 1_002_039_280, 0, 1_000_000_000]
    data["meta"]["postBalances"] = [999_995_000, 2_039_280, 2_039_280, 997_960_720]
    data["transaction"]["message"]["instructions"].append(
        {
            "program": "system",
            "parsed": {
                "type": "createAccount",
                "info": {
                    "source": "sponsor",
                    "newAccount": "stable-ata",
                    "lamports": 2_039_280,
                },
            },
        }
    )
    record = parse_transaction(data, "wallet", "sig")
    assert record["type"] == "sell"
    assert record["amount"] == 1
    assert record["price"] == 100
    data["transaction"]["message"]["instructions"].pop()
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"


def test_incoming_unrelated_sol_transfer_does_not_distort_execution_price():
    data = tx()
    data["meta"]["preTokenBalances"] = [data["meta"]["preTokenBalances"][1]]
    data["meta"]["postTokenBalances"] = [data["meta"]["postTokenBalances"][1]]
    data["meta"]["postBalances"] = [3_999_995_000]
    data["transaction"]["message"]["instructions"].append(
        {
            "program": "system",
            "parsed": {
                "type": "transfer",
                "info": {
                    "source": "someone",
                    "destination": "wallet",
                    "lamports": 2_000_000_000,
                },
            },
        }
    )
    assert parse_transaction(data, "wallet", "sig")["type"] == "unclassified"
