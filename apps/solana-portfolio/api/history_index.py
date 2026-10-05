"""Resumable, wallet-specific RPC activity index; this is not a P&L ledger.

Signature discovery and transaction decoding have separate coverage flags. A
null/pruned transaction stays missing and is retried on later refreshes. Failed
signatures are retained and counted, but never become successful executions.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

import base58

from .transactions import PARSER_VERSION

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[3] / "private" / "solana-portfolio"
_workers = {}
_workers_lock = threading.Lock()


def _wallet(value):
    try:
        if (
            not isinstance(value, str)
            or not 32 <= len(value) <= 44
            or len(base58.b58decode(value)) != 32
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("Enter a valid Solana wallet address.") from None
    return value


class _Store:
    def __init__(self, wallet, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"mainnet-{_wallet(wallet)}-history.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS signatures (
                signature TEXT PRIMARY KEY, slot INTEGER, block_time INTEGER,
                failed INTEGER NOT NULL DEFAULT 0, processed INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0, raw TEXT, record TEXT)""")
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=20)

    def meta(self, key, default=None):
        with self.connect() as db:
            row = db.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, **values):
        with self.connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in values.items()],
            )

    def add(self, entries, cursor=None):
        with self.connect() as db:
            db.executemany(
                """INSERT OR IGNORE INTO signatures
                (signature, slot, block_time, failed, processed) VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        item["signature"],
                        item.get("slot", 0),
                        item.get("blockTime"),
                        int(item.get("err") is not None),
                        int(item.get("err") is not None),
                    )
                    for item in entries
                ],
            )
            if cursor is not None:
                db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES ('cursor', ?)",
                    (json.dumps(cursor),),
                )

    def read(self):
        with self.connect() as db:
            total, processed, missing, failed, oldest = db.execute("""SELECT COUNT(*),
                COALESCE(SUM(processed), 0),
                COALESCE(SUM(CASE WHEN processed = 0 AND attempts > 0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(failed), 0), MIN(block_time) FROM signatures""").fetchone()
            records = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT record FROM signatures WHERE record IS NOT NULL ORDER BY block_time DESC, slot DESC"
                )
            ]
        discovery = self.meta("discoveryComplete", False) and not self.meta(
            "headAnchor"
        )
        decoding = processed == total
        return {
            "records": records,
            "status": {
                "discovered": total,
                "processed": processed,
                "missing": missing,
                "pending": total - processed,
                "failed": failed,
                "oldest": oldest,
                "discoveryComplete": discovery,
                "decodingComplete": decoding,
                "complete": discovery and decoding,
                "error": self.meta("error"),
                "updatedAt": self.meta("updatedAt"),
                "running": False,
            },
        }


def _safe_error(error):
    code = getattr(error, "code", None)
    return f"HTTP {code}" if isinstance(code, int) else type(error).__name__


def _page(wallet, before, rpc_call):
    options = {"limit": 1000, "commitment": "confirmed"}
    if before:
        options["before"] = before
    page = rpc_call("getSignaturesForAddress", [wallet, options])
    if not isinstance(page, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("signature"), str)
        for item in page
    ):
        raise ValueError("Invalid signature response.")
    # An RPC that ignores the pagination cursor must not spin forever.
    if page and page[-1]["signature"] == before:
        raise ValueError("Signature pagination did not advance.")
    return page


def _discover(store, wallet, rpc_call):
    with store.connect() as db:
        newest = db.execute(
            "SELECT signature FROM signatures ORDER BY slot DESC LIMIT 1"
        ).fetchone()
    complete = store.meta("discoveryComplete", False)
    cursor = store.meta("cursor")
    if newest:
        # Finish any interrupted multi-page head refresh before treating its
        # newly cached first page as an overlap with the old complete history.
        anchor = store.meta("headAnchor")
        if anchor:
            _refresh_head(store, wallet, rpc_call, anchor, store.meta("headCursor"))
        with store.connect() as db:
            newest = db.execute(
                "SELECT signature FROM signatures ORDER BY slot DESC LIMIT 1"
            ).fetchone()
        _refresh_head(store, wallet, rpc_call, newest[0], None)
        if complete:
            return
    # A previous interrupted historical scan resumes behind its durable cursor.
    if cursor is None:
        with store.connect() as db:
            oldest = db.execute(
                "SELECT signature FROM signatures ORDER BY slot ASC LIMIT 1"
            ).fetchone()
        cursor = oldest[0] if oldest else None
    while True:
        page = _page(wallet, cursor, rpc_call)
        if not page:
            store.set_meta(discoveryComplete=True)
            return
        cursor = page[-1]["signature"]
        store.add(page, cursor=cursor)


def _refresh_head(store, wallet, rpc_call, anchor, before):
    store.set_meta(headAnchor=anchor, headCursor=before)
    while True:
        page = _page(wallet, before, rpc_call)
        if not page:
            store.set_meta(headAnchor=None, headCursor=None)
            return
        overlap = next(
            (i for i, item in enumerate(page) if item["signature"] == anchor), None
        )
        store.add(page if overlap is None else page[:overlap])
        if overlap is not None:
            store.set_meta(headAnchor=None, headCursor=None)
            return
        before = page[-1]["signature"]
        store.set_meta(headCursor=before)


def _decode(store, wallet, rpc_call, parser, max_transactions):
    with store.connect() as db:
        pending = db.execute(
            "SELECT signature, raw FROM signatures WHERE processed = 0 ORDER BY slot DESC"
        ).fetchall()
    if max_transactions is not None:
        pending = pending[:max_transactions]
    for signature, cached_raw in pending:
        error = None
        raw = cached_raw
        record = None
        processed = failed = 0
        try:
            tx = (
                json.loads(cached_raw)
                if cached_raw is not None
                else rpc_call(
                    "getTransaction",
                    [
                        signature,
                        {
                            "encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0,
                            "commitment": "confirmed",
                        },
                    ],
                )
            )
            if tx is not None:
                raw = json.dumps(tx, allow_nan=False)
                failed = int(tx.get("meta", {}).get("err") is not None)
                record = None if failed else parser(tx, wallet, signature)
                processed = 1
        except Exception as exc:
            error = _safe_error(exc)
        with store.connect() as db:
            db.execute(
                """UPDATE signatures SET attempts = attempts + 1, raw = ?,
                processed = ?, failed = ?, record = ? WHERE signature = ?""",
                (
                    raw,
                    processed,
                    failed,
                    json.dumps(record, allow_nan=False) if record is not None else None,
                    signature,
                ),
            )
        updates = {"updatedAt": int(time.time())}
        if error:
            updates["error"] = error
        store.set_meta(**updates)


def _dependencies(rpc_call, parser):
    if rpc_call is None:
        from .providers import rpc

        rpc_call = rpc
    if parser is None:
        from .transactions import parse_transaction

        parser = parse_transaction
    return rpc_call, parser


def sync_history(
    wallet,
    *,
    directory=DEFAULT_DIRECTORY,
    rpc_call=None,
    parser=None,
    max_transactions=None,
):
    """Discover all signatures, then decode pending transactions once each.

    Optional max_transactions bounds one decoding pass without marking incomplete
    work complete. Network/parser failures are durable and safely reported.
    """
    wallet = _wallet(wallet)
    if max_transactions is not None and (
        type(max_transactions) is not int or max_transactions < 0
    ):
        raise ValueError("max_transactions must be a nonnegative integer.")
    rpc_call, parser = _dependencies(rpc_call, parser)
    store = _Store(wallet, directory)
    with store.connect() as db:
        saved = db.execute(
            "SELECT value FROM metadata WHERE key='parserVersion'"
        ).fetchone()
        if saved is None or json.loads(saved[0]) != PARSER_VERSION:
            # Retain records while the worker reclassifies cached raw transactions.
            db.execute(
                "UPDATE signatures SET processed=0 WHERE raw IS NOT NULL AND failed=0"
            )
            db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('parserVersion', ?)",
                (json.dumps(PARSER_VERSION),),
            )
    store.set_meta(error=None, lastAttempt=time.time())
    try:
        _discover(store, wallet, rpc_call)
    except Exception as error:
        store.set_meta(error=_safe_error(error))
    _decode(store, wallet, rpc_call, parser, max_transactions)
    store.set_meta(updatedAt=int(time.time()))
    return store.read()


def seed_signatures(
    wallet, entries, *, directory=DEFAULT_DIRECTORY, discovery_complete=False
):
    """Explicitly associate a verified signature export with one wallet.

    No shared signatures.json file is read automatically. An incomplete seed
    resumes discovery behind its oldest signature; a complete seed still refreshes
    the head on the next pass.
    """
    store = _Store(_wallet(wallet), directory)
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("signature"), str)
        for item in entries
    ):
        raise ValueError("Provide a signature array for the explicitly named wallet.")
    ordered = sorted(entries, key=lambda item: item.get("slot", 0), reverse=True)
    store.add(ordered)
    if ordered and not store.meta("cursor"):
        store.set_meta(cursor=ordered[-1]["signature"])
    if discovery_complete:
        store.set_meta(discoveryComplete=True)
    return store.read()


def read_history(wallet, *, directory=DEFAULT_DIRECTORY, rpc_call=None, parser=None):
    """Return durable progress immediately and start at most one wallet worker.

    Subsequent requests reuse the worker. Finished passes refresh after a minute;
    missing/pruned records remain visibly incomplete and are retried then.
    """
    wallet = _wallet(wallet)
    store = _Store(wallet, directory)
    key = str(store.path.resolve())
    with _workers_lock:
        worker = _workers.get(key)
        running = worker is not None and worker.is_alive()
        if not running and time.time() - store.meta("lastAttempt", 0) >= 60:

            def work():
                try:
                    sync_history(
                        wallet, directory=directory, rpc_call=rpc_call, parser=parser
                    )
                except Exception as error:
                    store.set_meta(error=_safe_error(error))

            worker = threading.Thread(
                target=work, name="portfolio-history", daemon=True
            )
            _workers[key] = worker
            worker.start()
            running = True
    result = store.read()
    result["status"]["running"] = running
    return result
