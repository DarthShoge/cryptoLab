"""Full-fidelity archive parser.

Format/prefix reference: bond-labs-dev/hyperliquid-data 0.1.0 (MIT).
https://github.com/bond-labs-dev/hyperliquid-data
This independent parser retains position and fee fields omitted by FillRow.
"""
from __future__ import annotations

import json
from datetime import date, datetime

from .contracts import FillEvent, ParseIssue, ParseResult, UTC, address, finite, semantic_hash, symbol, utc


def archive_keys(day: str) -> tuple[str, ...]:
    parsed = date.fromisoformat(day)
    if parsed < date(2025, 5, 25):
        raise ValueError("fills unavailable before 2025-05-25")
    prefix = "node_fills" if parsed < date(2025, 7, 27) else "node_fills_by_block"
    return tuple(f"{prefix}/hourly/{parsed:%Y%m%d}/{hour}.lz4" for hour in range(24))


def _integer(value):
    if isinstance(value, bool) or str(int(value)) != str(value):
        raise ValueError("integer required")
    return int(value)


def _event(pair, envelope, key, line, index, ingested):
    user, raw = pair
    user = address(user)
    coin = symbol(raw["coin"])
    side = raw["side"]
    if side not in ("B", "A") or not isinstance(raw["crossed"], bool):
        raise ValueError("invalid side/crossed")
    px, sz, start = (finite(raw[k]) for k in ("px", "sz", "startPosition"))
    if px <= 0 or sz <= 0:
        raise ValueError("price and size must be positive")
    tid, oid = _integer(raw["tid"]), _integer(raw["oid"])
    block = envelope.get("block_number")
    block = _integer(block) if block is not None else None
    block_time = envelope.get("block_time")
    block_time = utc(datetime.fromisoformat(block_time.replace("Z", "+00:00"))) if block_time else None
    identity = dict(schema="archive_fill_v1", source_key=key, block_number=block,
                    source_line=line, event_index=index, user=user, coin=coin, tid=tid)
    return FillEvent(
        semantic_hash(identity), datetime.fromtimestamp(_integer(raw["time"]) / 1000, UTC),
        block_time, block, line, index, user, coin, px, sz, side, start,
        start + sz if side == "B" else start - sz, str(raw["dir"]),
        finite(raw["closedPnl"]), finite(raw["fee"]), raw.get("feeToken"),
        raw["crossed"], tid, str(raw["hash"]), oid, bool(raw.get("liquidation")), key, ingested,
    )


def parse_archive_line(raw: bytes, source_key: str, line_index: int, *, ingested_at=None, coins=None) -> ParseResult:
    events, issues = [], []
    ingested = utc(ingested_at) if ingested_at else datetime.now(UTC)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            envelope, pairs = obj, obj["events"]
        elif isinstance(obj, list) and len(obj) == 2 and isinstance(obj[1], dict):
            envelope, pairs = {}, [obj]
        else:
            raise ValueError("unknown archive envelope")
        if not isinstance(pairs, list):
            raise ValueError("events must be a list")
    except (KeyError, TypeError, ValueError) as exc:
        return ParseResult((), (ParseIssue(source_key, line_index, -1, str(exc)),))
    for index, pair in enumerate(pairs):
        try:
            if coins is not None and isinstance(pair, list) and len(pair) == 2 and isinstance(pair[1], dict):
                raw_coin = pair[1].get("coin")
                if isinstance(raw_coin, str) and raw_coin not in coins:
                    continue
            events.append(_event(pair, envelope, source_key, line_index, index, ingested))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            issues.append(ParseIssue(source_key, line_index, index, str(exc)))
    return ParseResult(tuple(events), tuple(issues))
