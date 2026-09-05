"""Validated source records and byte-stable semantic identifiers."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal

UTC = timezone.utc


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return value.astimezone(UTC)


def address(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
        raise ValueError("invalid public address")
    return value.lower()


def symbol(value: str) -> str:
    if not isinstance(value, str) or len(value) > 80 or not re.fullmatch(r"(?:[A-Za-z][A-Za-z0-9_-]*:)?[A-Za-z][A-Za-z0-9_-]*", value):
        raise ValueError("invalid perp instrument ID")
    return value


def finite(value) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite number")
    return result


def _canonical(value):
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, datetime):
        return utc(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(value, (float, Decimal)):
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("non-finite number")
        text = format(number, "f")
        text = text.rstrip("0").rstrip(".") if "." in text else text
        return "0" if number == 0 else text
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("canonical keys must be strings")
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def canonical_json(value) -> bytes:
    return json.dumps(_canonical(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_hash(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class FillEvent:
    event_id: str
    exchange_time: datetime
    block_time: datetime | None
    block_number: int | None
    source_line: int
    event_index: int
    user: str
    coin: str
    px: float
    sz: float
    side: str
    start_position: float
    post_position: float
    direction: str
    closed_pnl: float
    fee: float
    fee_token: str | None
    crossed: bool
    tid: int
    tx_hash: str
    oid: int
    liquidation: bool
    source_key: str
    ingested_at: datetime

    @property
    def order_key(self):
        return (self.exchange_time, self.block_number if self.block_number is not None else -1,
                self.source_key, self.source_line, self.event_index, self.event_id)


@dataclass(frozen=True)
class ParseIssue:
    source_key: str
    source_line: int
    event_index: int
    reason: str


@dataclass(frozen=True)
class ParseResult:
    events: tuple[FillEvent, ...]
    issues: tuple[ParseIssue, ...]
