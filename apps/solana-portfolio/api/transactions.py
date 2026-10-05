"""Conservative DEX swap decoding with owned deltas and SOL fee/rent normalization."""

from collections import defaultdict
from decimal import Decimal

from arblab.kamino_onchain import KNOWN_MINTS
from .dex_instructions import JUPITER_PROGRAM, swap_evidence

PARSER_VERSION = 2

KAMINO_PROGRAM = "KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD"
SOL_MINT = "So11111111111111111111111111111111111111112"
STABLE_MINTS = {
    mint for mint, symbol in KNOWN_MINTS.items() if symbol in ("USDC", "USDT")
}


def owned_balances(entries, wallet):
    result = defaultdict(Decimal)
    for entry in entries:
        if entry.get("owner") == wallet:
            token = entry["uiTokenAmount"]
            result[entry["mint"]] += (
                Decimal(token["amount"]) / Decimal(10) ** token["decimals"]
            )
    return result


def native_sol_change(meta, message, wallet, instructions):
    keys = [
        entry["pubkey"] if isinstance(entry, dict) else entry
        for entry in message.get("accountKeys", [])
    ]
    pre, post = meta.get("preBalances", []), meta.get("postBalances", [])
    if wallet not in keys or len(pre) != len(keys) or len(post) != len(keys):
        return None
    index = keys.index(wallet)
    change = Decimal(
        post[index] - pre[index] + (meta.get("fee", 0) if index == 0 else 0)
    )
    # Neutralize rent moving between wallet and its token accounts. Wrapped SOL
    # principal is accounted for by the owned SPL amount delta instead of rent.
    rent_changes = defaultdict(int)
    for sign, entries, balances in (
        (-1, meta.get("preTokenBalances", []), pre),
        (1, meta.get("postTokenBalances", []), post),
    ):
        for entry in entries:
            account_index = entry.get("accountIndex")
            if (
                entry.get("owner") != wallet
                or account_index is None
                or account_index >= len(balances)
            ):
                continue
            wrapped = (
                int(entry["uiTokenAmount"]["amount"])
                if entry["mint"] == SOL_MINT
                else 0
            )
            rent_changes[keys[account_index]] += sign * (
                balances[account_index] - wrapped
            )
    funders, refunds = {}, {}
    for instruction in instructions:
        parsed = instruction.get("parsed", {})
        if not isinstance(parsed, dict):
            continue
        info = parsed.get("info", {})
        if (
            parsed.get("type") in ("createAccount", "createAccountWithSeed")
            and instruction.get("program") == "system"
        ):
            funders[info.get("newAccount")] = info.get("source")
        elif (
            parsed.get("type") in ("create", "createIdempotent")
            and instruction.get("program") == "spl-associated-token-account"
        ):
            funders[info.get("account")] = info.get("source")
        elif parsed.get("type") == "closeAccount" and instruction.get("program") in (
            "spl-token",
            "spl-token-2022",
        ):
            refunds[info.get("account")] = info.get("destination")
    for account, rent_change in rent_changes.items():
        if not rent_change:
            continue
        counterparty = funders.get(account) if rent_change > 0 else refunds.get(account)
        if not counterparty:
            return None
        if counterparty == wallet:
            change += rent_change
    return change / Decimal(10**9)


def parse_transaction(tx, wallet, signature):
    if (
        not tx
        or not tx.get("meta")
        or tx["meta"].get("err") is not None
        or not tx.get("blockTime")
    ):
        return None
    meta, message = tx["meta"], tx["transaction"]["message"]
    instructions = list(message.get("instructions", []))
    for inner in meta.get("innerInstructions", []):
        instructions.extend(inner.get("instructions", []))
    programs = {instruction.get("programId") for instruction in instructions}
    before = owned_balances(meta.get("preTokenBalances", []), wallet)
    after = owned_balances(meta.get("postTokenBalances", []), wallet)
    deltas = {mint: after[mint] - before[mint] for mint in set(before) | set(after)}
    native_change = native_sol_change(meta, message, wallet, instructions)
    if native_change is not None:
        deltas[SOL_MINT] = deltas.get(SOL_MINT, Decimal(0)) + native_change
    deltas = {
        mint: amount
        for mint, amount in deltas.items()
        if abs(amount) > Decimal("0.00000001")
    }
    record = {
        "id": signature,
        "signature": signature,
        "time": tx["blockTime"],
        "type": "unclassified",
        "asset": None,
        "amount": None,
        "price": None,
        "value": None,
        "funding": "unknown",
        "feeSol": meta.get("fee", 0) / 1e9,
        "protocol": "Solana",
        "deltas": {
            KNOWN_MINTS.get(mint, mint): float(value) for mint, value in deltas.items()
        },
    }
    if KAMINO_PROGRAM in programs:
        # Composite Kamino instructions may include a swap. Do not label their aggregate balance delta an execution.
        record.update(type="kamino", protocol="Kamino")
        return record
    swaps, composite = swap_evidence(instructions)
    if not swaps or composite or native_change is None:
        return record
    routed = any(swap["protocol"] == "Jupiter" for swap in swaps)
    # Do not assign a separate top-level SOL transfer to the swap. For direct
    # DEX swaps an explicit external outflow can be isolated exactly; its purpose
    # is unknown. Incoming transfers and unowned wrapping funding stay ambiguous.
    owned_accounts = {
        entry.get("accountIndex")
        for entry in [
            *meta.get("preTokenBalances", []),
            *meta.get("postTokenBalances", []),
        ]
        if entry.get("owner") == wallet
    }
    keys = [
        entry["pubkey"] if isinstance(entry, dict) else entry
        for entry in message.get("accountKeys", [])
    ]
    owned_addresses = {
        keys[index]
        for index in owned_accounts
        if index is not None and index < len(keys)
    }
    token_addresses = {
        keys[entry["accountIndex"]]
        for entry in [
            *meta.get("preTokenBalances", []),
            *meta.get("postTokenBalances", []),
        ]
        if isinstance(entry.get("accountIndex"), int)
        and 0 <= entry["accountIndex"] < len(keys)
    }
    for instruction in instructions:
        parsed = instruction.get("parsed", {})
        if isinstance(parsed, dict) and parsed.get("type") in (
            "initializeAccount",
            "initializeAccount2",
            "initializeAccount3",
        ):
            token_addresses.add(parsed["info"].get("account"))
            if parsed.get("info", {}).get("owner") == wallet:
                owned_addresses.add(parsed["info"].get("account"))
    external_lamports = 0
    for instruction in message.get("instructions", []):
        parsed = instruction.get("parsed", {})
        if (
            isinstance(parsed, dict)
            and instruction.get("program") == "system"
            and parsed.get("type") == "transfer"
        ):
            info = parsed.get("info", {})
            if info.get("destination") == wallet:
                return record
            if (
                info.get("source") == wallet
                and info.get("destination") not in owned_addresses
            ):
                lamports = info.get("lamports")
                if (
                    routed
                    or not info.get("destination")
                    or info["destination"] in token_addresses
                    or type(lamports) is not int
                    or lamports <= 0
                ):
                    return record
                external_lamports += lamports
    if external_lamports:
        deltas[SOL_MINT] = deltas.get(SOL_MINT, Decimal(0)) + Decimal(
            external_lamports
        ) / Decimal(10**9)
        deltas = {
            mint: amount
            for mint, amount in deltas.items()
            if abs(amount) > Decimal("0.00000001")
        }
        record.update(
            externalSolOutflowSol=external_lamports / 1e9,
            externalSolOutflowSource="Explicit System transfers · purpose unknown",
            deltas={
                KNOWN_MINTS.get(mint, mint): float(value)
                for mint, value in deltas.items()
            },
        )
    if len(deltas) != 2:
        return record
    mints = sorted(deltas)
    if deltas[mints[0]] * deltas[mints[1]] >= 0:
        return record
    protocols = sorted({swap["protocol"] for swap in swaps})
    venues = [protocol for protocol in protocols if protocol != "Jupiter"]
    protocol = "Jupiter" if "Jupiter" in protocols else " + ".join(protocols)
    legs = []
    for mint in mints:
        opposite = next(other for other in mints if other != mint)
        legs.append(
            {
                "mint": mint,
                "asset": KNOWN_MINTS.get(mint, mint[:8]),
                "side": "buy" if deltas[mint] > 0 else "sell",
                "signedAmount": float(deltas[mint]),
                "amount": float(abs(deltas[mint])),
                "price": float(abs(deltas[opposite] / deltas[mint])),
                "quote": KNOWN_MINTS.get(opposite, opposite[:8]),
                "quoteMint": opposite,
                "value": float(abs(deltas[opposite])),
            }
        )
    record.update(
        protocol=protocol, venues=venues, swapInstructions=swaps, executionLegs=legs
    )
    stable = [mint for mint in mints if mint in STABLE_MINTS]
    # Prefer a stable quote, then SOL. This retains SOL as the base for SOL/USDC
    # and gives token/SOL executions their actual SOL-denominated price.
    quote = stable[0] if len(stable) == 1 else SOL_MINT if SOL_MINT in mints else None
    if quote is None:
        record.update(type="swap", quote=None)
        return record
    mint = next(other for other in mints if other != quote)
    amount, quote_amount = abs(deltas[mint]), abs(deltas[quote])
    record.update(
        type="buy" if deltas[mint] > 0 else "sell",
        asset=KNOWN_MINTS.get(mint, mint[:8]),
        mint=mint,
        amount=float(amount),
        price=float(quote_amount / amount),
        value=float(quote_amount),
        quote=KNOWN_MINTS[quote],
        quoteMint=quote,
    )
    # Price is in executed quote units; never assert dollar parity or infer loan linkage.
    return record
