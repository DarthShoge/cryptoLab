"""Bounded read-only provider calls. Credentials and private wallet stay server-side."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request, parse
from urllib.error import HTTPError

import base58
from dotenv import load_dotenv

from arblab.kamino_onchain import (
    KNOWN_MINTS,
    find_wallet_obligations,
    load_onchain_snapshot,
)
from arblab.paths import fixture_path

from .accounting import account_summary
from .kamino import load_recorded_risk
from .transactions import KAMINO_PROGRAM, SOL_MINT, parse_transaction
from .timeframes import INTERVALS, aggregate_candles, candle_start, candle_end

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
TOKEN_PROGRAMS = (
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
)
ASSETS = ("SOL", "ETH", "BTC")
_transaction_lock = threading.Lock()
_last_transaction_at = 0.0


def validate_wallet(wallet):
    if not isinstance(wallet, str) or not 32 <= len(wallet) <= 44:
        raise ValueError("Enter a valid Solana wallet address.")
    try:
        if len(base58.b58decode(wallet)) != 32:
            raise ValueError("Enter a valid Solana wallet address.")
    except (ValueError, TypeError):
        raise ValueError("Enter a valid Solana wallet address.") from None
    return wallet


def default_wallet():
    path = ROOT / "private" / "address.txt"
    return validate_wallet(path.read_text().strip()) if path.exists() else ""


def json_request(url, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "CryptoLab-Portfolio/1.0",
        },
    )
    for attempt in range(2):
        try:
            with request.urlopen(req, timeout=15) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 or attempt:
                raise
            time.sleep(2)


def rpc(method, params):
    global _last_transaction_at
    if (
        method == "getTransaction"
        and RPC_URL.rstrip("/") == "https://api.mainnet-beta.solana.com"
    ):
        # The public endpoint cannot sustain a transaction-history burst.
        with _transaction_lock:
            delay = 1.05 - (time.monotonic() - _last_transaction_at)
            if delay > 0:
                time.sleep(delay)
            _last_transaction_at = time.monotonic()
    response = json_request(
        RPC_URL, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    )
    if response.get("error"):
        raise RuntimeError(f"RPC rejected {method}.")
    return response["result"]


def safe_error(error):
    # Provider exception strings can contain RPC API keys. Return only a class/status.
    code = getattr(error, "code", None)
    return f"HTTP {code}" if code else type(error).__name__


def market_candles(asset="SOL", interval="1d", days=90, end_time=None, start_time=None):
    if (
        asset not in ASSETS
        or interval not in INTERVALS
        or days not in (0, 7, 30, 90, 365)
    ):
        raise ValueError("Unsupported chart asset, interval or range.")
    source_interval = "1h" if interval in ("1h", "4h") else "1d"
    granularity = INTERVALS[source_interval]
    end = (
        min(int(time.time()), int(end_time)) // granularity * granularity
        if end_time is not None
        else int(time.time()) // granularity * granularity
    )
    if days == 0:
        if start_time is None or not 0 < int(start_time) < end:
            raise ValueError("Inception charts need a valid start timestamp.")
        start = candle_start(int(start_time), interval)
    else:
        start = candle_start(end - days * 86400, interval)
    # Solana launched in 2020; reject unbounded user-supplied ranges.
    if start < 1577836800 or end - start > 3660 * 86400:
        raise ValueError("Chart range must be within ten years from 2020.")
    count = (end - start) // granularity
    rows = []
    for offset in range(0, count, 300):
        page_end = end - offset * granularity
        page_start = page_end - min(300, count - offset) * granularity
        query = parse.urlencode(
            {"granularity": granularity, "start": page_start, "end": page_end}
        )
        rows.extend(
            json_request(
                f"https://api.exchange.coinbase.com/products/{asset}-USD/candles?{query}"
            )
        )
    unique = {int(row[0]): row for row in rows if start <= int(row[0]) < end}
    candles = [
        {
            "time": timestamp,
            "endTime": candle_end(timestamp, source_interval),
            "low": float(row[1]),
            "high": float(row[2]),
            "open": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for timestamp, row in sorted(unique.items())
    ]
    if interval != source_interval:
        candles = aggregate_candles(candles, interval, end, source_interval)
    return candles


def prices():
    result = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {
            pool.submit(
                json_request,
                f"https://api.exchange.coinbase.com/products/{symbol}-USD/ticker",
            ): symbol
            for symbol in (*ASSETS, "USDC", "USDT")
        }
        for future in as_completed(jobs):
            try:
                result[jobs[future]] = float(future.result()["price"])
            except Exception:
                result[jobs[future]] = None
    if result.get("USDC") is None:
        try:
            result["USDC"] = float(
                json_request("https://api.coinbase.com/v2/prices/USDC-USD/spot")[
                    "data"
                ]["amount"]
            )
        except Exception:
            pass
    return result


def wallet_tokens(wallet, marks):
    native = rpc("getBalance", [wallet, {"commitment": "confirmed"}])["value"] / 1e9
    by_mint = {
        SOL_MINT: {
            "symbol": "SOL",
            "mint": SOL_MINT,
            "amount": native,
            "price": marks.get("SOL"),
        }
    }
    for program in TOKEN_PROGRAMS:
        accounts = rpc(
            "getTokenAccountsByOwner",
            [
                wallet,
                {"programId": program},
                {"encoding": "jsonParsed", "commitment": "confirmed"},
            ],
        )["value"]
        for account in accounts:
            info = account["account"]["data"]["parsed"]["info"]
            mint, token = info["mint"], info["tokenAmount"]
            amount = int(token["amount"]) / 10 ** token["decimals"]
            if not amount:
                continue
            symbol = KNOWN_MINTS.get(mint, mint[:8])
            if mint in by_mint:
                by_mint[mint]["amount"] += amount
            else:
                by_mint[mint] = {
                    "symbol": symbol,
                    "mint": mint,
                    "amount": amount,
                    "price": marks.get(symbol),
                }
    return list(by_mint.values())


def recent_transactions(wallet):
    # Compatibility wrapper; the index keeps all signatures and resumes across restarts.
    from .history_index import read_history

    result = read_history(wallet)
    return result["records"], result["status"]["missing"]


def merge_activity(records, protocol_records):
    signatures = {t.get("signature") for t in protocol_records if t.get("signature")}
    # Protocol actions can contain multiple legitimate legs per signature.
    # Replace the generic RPC Kamino row, retaining every sourced action.
    merged = [
        t
        for t in records
        if not (t.get("type") == "kamino" and t.get("signature") in signatures)
    ]
    merged.extend(protocol_records)
    return sorted(merged, key=lambda t: (t["time"], t["id"]), reverse=True)


def persist_observation(wallet, summary, sol_price=None):
    directory = ROOT / "private" / "solana-portfolio"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"mainnet-{wallet}.json"
    try:
        history = json.loads(path.read_text()) if path.exists() else []
    except (ValueError, OSError):
        history = []
    now = int(time.time())
    if not history or now - history[-1]["time"] >= 60:
        history.append(
            {
                "time": now,
                "equity": summary["knownEquity"],
                "externalFlow": None,
                "solPrice": sol_price,
                "valuationComplete": summary["netEquity"] is not None,
                "unpricedCount": summary["unpricedCount"],
            }
        )
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(history[-10000:]))
        temp.replace(path)
    return history


def live_portfolio(wallet):
    validate_wallet(wallet)
    warnings, snapshots, wallet_positions, protocol_risk = [], [], [], {}
    positions_complete = True
    marks = prices()
    try:
        addresses = find_wallet_obligations(wallet, KAMINO_PROGRAM, RPC_URL)
        if len(addresses) > 8:
            positions_complete = False
            warnings.append(
                "Only the first 8 Kamino obligations were loaded; account equity is incomplete."
            )
        for address in addresses[:8]:
            try:
                snapshot = load_onchain_snapshot(
                    address,
                    KAMINO_PROGRAM,
                    RPC_URL,
                    str(fixture_path("kamino_idl.json")),
                )
                snapshots.append((address, snapshot))
                try:
                    protocol_risk[address] = load_recorded_risk(
                        address, KAMINO_PROGRAM, RPC_URL
                    )
                except Exception as error:
                    warnings.append(
                        f"Protocol risk totals unavailable ({safe_error(error)}); health for this obligation is hidden."
                    )
            except Exception as error:
                positions_complete = False
                warnings.append(
                    f"A Kamino obligation could not be loaded ({safe_error(error)})."
                )
    except Exception as error:
        positions_complete = False
        warnings.append(
            f"Kamino discovery unavailable ({safe_error(error)}). Check your server SOLANA_RPC_URL."
        )
    try:
        wallet_positions = wallet_tokens(wallet, marks)
    except Exception as error:
        positions_complete = False
        warnings.append(f"Wallet balances unavailable ({safe_error(error)}).")
    summary = account_summary(snapshots, wallet_positions)
    for loan in summary["loans"]:
        if loan["address"] in protocol_risk:
            loan.update(protocol_risk[loan["address"]])
        else:
            loan.update(
                health=None,
                borrowHealth=None,
                ltv=None,
                liquidationLtv=None,
                liquidationBuffer=None,
            )
    summary["health"] = min(
        (loan["health"] for loan in summary["loans"] if loan["health"] is not None),
        default=None,
    )
    if not positions_complete:
        summary["netEquity"] = None
    if summary["unpricedCount"]:
        warnings.append(
            "Some wallet tokens have no supported price (including possible collateral receipt tokens); total equity is unavailable."
        )
    try:
        from .history_index import read_history

        indexed = read_history(wallet)
        trades, history_status = indexed["records"], indexed["status"]
        if history_status["missing"]:
            warnings.append(
                f"{history_status['missing']} historical transactions could not be retrieved; the index will retry."
            )
    except Exception as error:
        trades, history_status = [], None
        warnings.append(f"Transaction history unavailable ({safe_error(error)}).")
    try:
        from .kamino_history import load_kamino_history

        kamino_history = load_kamino_history(wallet)
        trades = merge_activity(trades, kamino_history["trades"])
    except Exception as error:
        kamino_history = {
            "history": [],
            "series": [],
            "warnings": [f"Kamino history unavailable ({safe_error(error)})."],
        }
    history = (
        persist_observation(wallet, summary, marks.get("SOL"))
        if positions_complete
        else []
    )
    warnings.append(
        "Wallet signatures are paginated to inception and transactions are cached in a resumable background index. See backfill progress for decoded coverage. Verified Jupiter, Orca and Raydium swaps preserve actual token execution legs and native/wrapped SOL with fee and token-account rent normalization. Composite actions and unrelated transfers remain unclassified."
    )
    warnings.append(
        "Live observations track the priced portion of equity from first load. Unpriced tokens can change coverage; external cash flows and opening cost basis are unverified. Historical returns and realized P&L are unavailable."
    )
    warnings.append(
        "Loan health uses protocol-recorded obligation risk totals, including elevation-group thresholds. Holdings use stored reserve prices; values may be stale between protocol refreshes. Verify current state in Kamino before acting."
    )
    return {
        "mode": "live",
        "wallet": wallet,
        "asOf": int(time.time()),
        "summary": summary,
        "history": history,
        "trades": trades,
        "historyComplete": False,
        "historyStatus": history_status,
        "healthHistory": kamino_history.get("healthHistory", []),
        "kaminoHistory": kamino_history["history"],
        "kaminoSeries": kamino_history["series"],
        "kaminoHistoryWarnings": kamino_history["warnings"],
        "warnings": warnings,
        "prices": marks,
        "source": "Solana mainnet RPC · Coinbase USD marks",
    }
