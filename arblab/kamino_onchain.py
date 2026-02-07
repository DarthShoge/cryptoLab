import base64
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List
from urllib import request

from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition

# Kamino uses 128-bit fixed-point numbers with a scale factor of 2^60.
SF_SCALE = Decimal(2**60)

# Well-known Solana token mints to human-readable symbols (fallback if API unavailable).
KNOWN_MINTS: Dict[str, str] = {
    "So11111111111111111111111111111111111111112": "SOL",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
    "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs": "ETH",
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "mSOL",
    "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn": "JitoSOL",
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": "stSOL",
    "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1": "bSOL",
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "BONK",
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": "PYTH",
    "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL": "JTO",
    "27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4": "JLP",
    "2u1tszSeqZ3qBWF3uNGPFc8TzMk2tdiwknnRMWGWjGWH": "USDG",
    "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv": "PENGU",
}

# Module-level cache for Jupiter token list symbols.
_jupiter_symbols: Dict[str, str] | None = None


def _fetch_jupiter_symbols() -> Dict[str, str]:
    """Fetch mint→symbol mapping from Jupiter's token list API (cached)."""
    global _jupiter_symbols
    if _jupiter_symbols is not None:
        return _jupiter_symbols

    # Try multiple endpoints in order of preference.
    endpoints = [
        "https://lite-api.jup.ag/tokens/v2/tag?query=verified",
    ]

    for url in endpoints:
        try:
            req = request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "KaminoRiskSim/1.0",
            })
            with request.urlopen(req, timeout=30) as resp:
                tokens = json.load(resp)
            # Jupiter V1 uses "address", V2 uses "id" for the mint field.
            result: Dict[str, str] = {}
            for t in tokens:
                mint = t.get("address") or t.get("id")
                symbol = t.get("symbol")
                if mint and symbol:
                    result[mint] = symbol
            if result:
                _jupiter_symbols = result
                return _jupiter_symbols
        except Exception:
            continue

    _jupiter_symbols = {}
    return _jupiter_symbols


def _resolve_symbol(mint: str) -> str:
    """Resolve a mint address to a human-readable token symbol."""
    symbol = KNOWN_MINTS.get(mint)
    if symbol:
        return symbol
    jupiter = _fetch_jupiter_symbols()
    return jupiter.get(mint, mint[:8])


@dataclass(frozen=True)
class ReserveConfig:
    symbol: str
    price: Decimal
    ltv: Decimal
    liquidation_threshold: Decimal
    decimals: int


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=30) as response:
        return json.load(response)


def _rpc_call(rpc_url: str, method: str, params: Iterable[Any]) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)}
    response = _post_json(rpc_url, payload)
    if "error" in response:
        raise RuntimeError(f"RPC error for {method}: {response['error']}")
    return response["result"]


def _get_account(rpc_url: str, address: str) -> Dict[str, Any]:
    result = _rpc_call(rpc_url, "getAccountInfo", [address, {"encoding": "base64"}])
    value = result.get("value")
    if not value or not value.get("data"):
        raise ValueError(f"No account data for {address}")
    data = base64.b64decode(value["data"][0])
    return {"data": data, "owner": value.get("owner")}


OBLIGATION_DISCRIMINATOR = hashlib.sha256(b"account:Obligation").digest()[:8]
RESERVE_DISCRIMINATOR = hashlib.sha256(b"account:Reserve").digest()[:8]
# Obligation owner field offset: 8 (discriminator) + 8 (tag) + 16 (LastUpdate) + 32 (lendingMarket)
_OBLIGATION_OWNER_OFFSET = 64
# Reserve lendingMarket offset: 8 (discriminator) + 8 (version) + 16 (LastUpdate)
_RESERVE_MARKET_OFFSET = 32


def find_wallet_obligations(
    wallet: str,
    program_id: str,
    rpc_url: str,
) -> List[str]:
    """Discover all Kamino obligation accounts owned by a wallet address."""
    try:
        import base58
    except ImportError:
        raise ImportError("base58 is required for wallet obligation discovery.")

    disc_b64 = base64.b64encode(OBLIGATION_DISCRIMINATOR).decode()
    wallet_bytes = base58.b58decode(wallet)
    wallet_b64 = base64.b64encode(wallet_bytes).decode()

    result = _rpc_call(rpc_url, "getProgramAccounts", [
        program_id,
        {
            "encoding": "base64",
            "dataSlice": {"offset": 0, "length": 0},
            "filters": [
                {"memcmp": {"offset": 0, "bytes": disc_b64, "encoding": "base64"}},
                {"memcmp": {"offset": _OBLIGATION_OWNER_OFFSET, "bytes": wallet_b64, "encoding": "base64"}},
            ],
        },
    ])
    return [account["pubkey"] for account in result]


def load_market_reserves(
    lending_market: str,
    program_id: str,
    rpc_url: str,
    idl_path: str,
) -> List[ReserveConfig]:
    """Fetch all reserve configs for a Kamino lending market."""
    try:
        import base58
    except ImportError:
        raise ImportError("base58 is required for market reserve discovery.")

    disc_b64 = base64.b64encode(RESERVE_DISCRIMINATOR).decode()
    market_bytes = base58.b58decode(lending_market)
    market_b64 = base64.b64encode(market_bytes).decode()

    idl = _load_idl(idl_path)

    result = _rpc_call(rpc_url, "getProgramAccounts", [
        program_id,
        {
            "encoding": "base64",
            "filters": [
                {"memcmp": {"offset": 0, "bytes": disc_b64, "encoding": "base64"}},
                {"memcmp": {"offset": _RESERVE_MARKET_OFFSET, "bytes": market_b64, "encoding": "base64"}},
            ],
        },
    ])

    reserves: List[ReserveConfig] = []
    for account in result:
        data = base64.b64decode(account["account"]["data"][0])
        decoded = _decode_account(idl, "Reserve", data)
        config = _parse_reserve_config(decoded)
        # Skip reserves with zero price (disabled/deprecated)
        if config.price > 0:
            reserves.append(config)

    reserves.sort(key=lambda r: r.symbol)
    return reserves


def _sf_to_decimal(sf_value: int) -> Decimal:
    """Convert a Scale-Factor (Sf) 128-bit fixed-point integer to a Decimal."""
    return Decimal(int(sf_value)) / SF_SCALE


def _normalize_amount(raw_amount: int, decimals: int) -> Decimal:
    return Decimal(raw_amount) / (Decimal(10) ** decimals)


def _compute_collateral_exchange_rate(reserve: Any) -> Decimal:
    """Compute the cToken-to-underlying exchange rate for a reserve.

    deposited_amount on an obligation is in cToken units.  Multiplying by
    this rate converts to underlying token units (before decimal normalization).
    """
    liquidity = reserve.liquidity
    collateral = reserve.collateral

    mint_total_supply = int(collateral.mint_total_supply)
    if mint_total_supply == 0:
        return Decimal(1)

    available = Decimal(int(liquidity.available_amount))
    borrowed = _sf_to_decimal(liquidity.borrowed_amount_sf)
    protocol_fees = _sf_to_decimal(liquidity.accumulated_protocol_fees_sf)
    referrer_fees = _sf_to_decimal(liquidity.accumulated_referrer_fees_sf)
    pending_fees = _sf_to_decimal(liquidity.pending_referrer_fees_sf)

    total_underlying = available + borrowed - protocol_fees - referrer_fees - pending_fees
    return total_underlying / Decimal(mint_total_supply)


def _load_idl(idl_path: str) -> Any:
    try:
        from anchorpy import Idl
    except ImportError as exc:
        raise ImportError("anchorpy is required for on-chain decoding.") from exc
    with open(idl_path, "r", encoding="utf-8") as handle:
        idl_raw = handle.read()
    return Idl.from_json(idl_raw)


def _decode_account(idl: Any, account_name: str, data: bytes) -> Any:
    try:
        from anchorpy.coder.accounts import AccountsCoder
    except ImportError as exc:
        raise ImportError("anchorpy is required for on-chain decoding.") from exc
    coder = AccountsCoder(idl)
    return coder.decode(data)


def _parse_reserve_config(reserve: Any) -> ReserveConfig:
    config = reserve.config
    liquidity = reserve.liquidity

    # Token symbol: Kamino doesn't store a symbol on-chain; resolve via Jupiter API.
    mint = str(liquidity.mint_pubkey)
    symbol = _resolve_symbol(mint)

    # Market price is stored as Sf (128-bit fixed-point, scale 2^60).
    price = _sf_to_decimal(liquidity.market_price_sf)

    # LTV and liquidation threshold are stored as u8 percentages.
    ltv = Decimal(int(config.loan_to_value_pct)) / Decimal(100)
    liquidation_threshold = Decimal(int(config.liquidation_threshold_pct)) / Decimal(100)

    decimals = int(liquidity.mint_decimals)

    return ReserveConfig(
        symbol=symbol,
        price=price,
        ltv=ltv,
        liquidation_threshold=liquidation_threshold,
        decimals=decimals,
    )


def get_obligation_market(
    obligation_address: str,
    program_id: str,
    rpc_url: str,
    idl_path: str,
) -> str:
    """Return the lending market address for an obligation."""
    idl = _load_idl(idl_path)
    account = _get_account(rpc_url, obligation_address)
    if account["owner"] != program_id:
        raise ValueError("Obligation account owner does not match the provided program id.")
    obligation = _decode_account(idl, "Obligation", account["data"])
    return str(obligation.lending_market)


def load_onchain_snapshot(
    obligation_address: str,
    program_id: str,
    rpc_url: str,
    idl_path: str,
    obligation_account_name: str = "Obligation",
    reserve_account_name: str = "Reserve",
) -> AccountSnapshot:
    idl = _load_idl(idl_path)

    obligation_account = _get_account(rpc_url, obligation_address)
    if obligation_account["owner"] != program_id:
        raise ValueError("Obligation account owner does not match the provided program id.")

    obligation = _decode_account(idl, obligation_account_name, obligation_account["data"])
    lending_market = str(obligation.lending_market)

    # Filter out empty slots (Kamino uses fixed-size arrays; empty entries have zero amounts).
    active_deposits = [
        entry for entry in obligation.deposits
        if int(entry.deposited_amount) > 0
    ]
    active_borrows = [
        entry for entry in obligation.borrows
        if int(entry.borrowed_amount_sf) > 0
    ]

    reserve_addresses = {
        str(entry.deposit_reserve) for entry in active_deposits
    }.union({
        str(entry.borrow_reserve) for entry in active_borrows
    })

    reserve_map: Dict[str, ReserveConfig] = {}
    exchange_rates: Dict[str, Decimal] = {}
    for reserve_address in reserve_addresses:
        reserve_account = _get_account(rpc_url, reserve_address)
        if reserve_account["owner"] != program_id:
            raise ValueError(f"Reserve {reserve_address} is not owned by {program_id}.")
        reserve = _decode_account(idl, reserve_account_name, reserve_account["data"])
        reserve_map[reserve_address] = _parse_reserve_config(reserve)
        exchange_rates[reserve_address] = _compute_collateral_exchange_rate(reserve)

    collateral_positions: List[CollateralPosition] = []
    for entry in active_deposits:
        reserve_address = str(entry.deposit_reserve)
        reserve = reserve_map[reserve_address]
        # deposited_amount is in cToken units; multiply by exchange rate to get underlying.
        ctoken_amount = Decimal(int(entry.deposited_amount))
        underlying_raw = ctoken_amount * exchange_rates[reserve_address]
        amount = underlying_raw / (Decimal(10) ** reserve.decimals)
        collateral_positions.append(
            CollateralPosition(
                symbol=reserve.symbol,
                amount=float(amount),
                price=float(reserve.price),
                ltv=float(reserve.ltv),
                liquidation_threshold=float(reserve.liquidation_threshold),
            )
        )

    debt_positions: List[DebtPosition] = []
    for entry in active_borrows:
        reserve_address = str(entry.borrow_reserve)
        reserve = reserve_map[reserve_address]
        # borrowed_amount_sf is a 128-bit fixed-point value (scale 2^60), in raw token units.
        raw_amount = _sf_to_decimal(entry.borrowed_amount_sf)
        amount = raw_amount / (Decimal(10) ** reserve.decimals)
        debt_positions.append(
            DebtPosition(
                symbol=reserve.symbol,
                amount=float(amount),
                price=float(reserve.price),
            )
        )

    return AccountSnapshot(collateral=collateral_positions, debt=debt_positions)
