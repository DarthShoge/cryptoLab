import base64
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List
from urllib import request

from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition


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


def _load_idl(idl_path: str) -> Any:
    try:
        from anchorpy import Idl
    except ImportError as exc:
        raise ImportError("anchorpy is required for on-chain decoding.") from exc
    with open(idl_path, "r", encoding="utf-8") as handle:
        idl_json = json.load(handle)
    return Idl.from_json(idl_json)


def _decode_account(idl: Any, account_name: str, data: bytes) -> Dict[str, Any]:
    try:
        from anchorpy.coder.accounts import AccountsCoder
    except ImportError as exc:
        raise ImportError("anchorpy is required for on-chain decoding.") from exc
    coder = AccountsCoder(idl)
    return coder.decode(account_name, data)


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _normalize_amount(raw_amount: int, decimals: int) -> Decimal:
    return Decimal(raw_amount) / (Decimal(10) ** decimals)


def _parse_reserve_config(reserve: Dict[str, Any]) -> ReserveConfig:
    config = reserve["config"]
    liquidity = reserve["liquidity"]
    symbol = liquidity.get("symbol", liquidity.get("mint_pubkey", "UNKNOWN"))
    price = _to_decimal(liquidity.get("market_price", 0))
    ltv = _to_decimal(config.get("loan_to_value_ratio", 0)) / Decimal(100)
    liquidation_threshold = _to_decimal(config.get("liquidation_threshold", 0)) / Decimal(100)
    decimals = int(liquidity.get("mint_decimals", 0))
    return ReserveConfig(
        symbol=symbol,
        price=price,
        ltv=ltv,
        liquidation_threshold=liquidation_threshold,
        decimals=decimals,
    )


def load_onchain_snapshot(
    obligation_address: str,
    program_id: str,
    rpc_url: str,
    idl_path: str,
    obligation_account_name: str = "obligation",
    reserve_account_name: str = "reserve",
) -> AccountSnapshot:
    idl = _load_idl(idl_path)
    obligation_account = _get_account(rpc_url, obligation_address)
    if obligation_account["owner"] != program_id:
        raise ValueError("Obligation account owner does not match the provided program id.")
    obligation_raw = obligation_account["data"]
    obligation = _decode_account(idl, obligation_account_name, obligation_raw)

    deposits = obligation.get("deposits", [])
    borrows = obligation.get("borrows", [])
    reserve_addresses = {
        str(entry["deposit_reserve"]) for entry in deposits
    }.union({str(entry["borrow_reserve"]) for entry in borrows})

    reserve_map: Dict[str, ReserveConfig] = {}
    for reserve_address in reserve_addresses:
        reserve_account = _get_account(rpc_url, reserve_address)
        if reserve_account["owner"] != program_id:
            raise ValueError(f"Reserve {reserve_address} is not owned by {program_id}.")
        reserve_raw = reserve_account["data"]
        reserve = _decode_account(idl, reserve_account_name, reserve_raw)
        reserve_map[reserve_address] = _parse_reserve_config(reserve)

    collateral_positions: List[CollateralPosition] = []
    for entry in deposits:
        reserve_address = str(entry["deposit_reserve"])
        reserve = reserve_map[reserve_address]
        deposited_amount = int(entry.get("deposited_amount", 0))
        amount = _normalize_amount(deposited_amount, reserve.decimals)
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
    for entry in borrows:
        reserve_address = str(entry["borrow_reserve"])
        reserve = reserve_map[reserve_address]
        borrowed_amount = entry.get("borrowed_amount")
        borrowed_amount_wads = entry.get("borrowed_amount_wads")
        if borrowed_amount is not None:
            amount = _normalize_amount(int(borrowed_amount), reserve.decimals)
        else:
            amount = _to_decimal(borrowed_amount_wads or 0) / Decimal(10**18)
        debt_positions.append(
            DebtPosition(
                symbol=reserve.symbol,
                amount=float(amount),
                price=float(reserve.price),
            )
        )

    return AccountSnapshot(collateral=collateral_positions, debt=debt_positions)
