"""Explicit research proxy assignments, not historical exchange specifications."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .contracts import symbol, utc
from .proxy_sessions import CALENDARS


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Expected mapping date YYYY-MM-DD")
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class ProxyMapping:
    instrument_id: str
    provider: str
    ticker: str
    asset_class: str
    quote_currency: str
    calendar: str
    unit: str
    adjustment: str
    valid_from: str
    valid_to: str
    description: str
    provenance: str

    def __post_init__(self):
        symbol(self.instrument_id)
        if self.provider not in ("binance", "yahoo"):
            raise ValueError("Unsupported price provider")
        if not isinstance(self.ticker, str) or not re.fullmatch(
            r"[A-Za-z0-9^][A-Za-z0-9.^=_-]{0,39}", self.ticker
        ):
            raise ValueError("Invalid source ticker")
        if self.asset_class not in ("crypto", "commodities", "equities", "indices"):
            raise ValueError("Unsupported proxy asset class")
        if self.quote_currency != "USD" or self.adjustment != "raw":
            raise ValueError(
                "Only explicitly unadjusted USD research proxies supported"
            )
        if self.calendar not in CALENDARS:
            raise ValueError("Unsupported proxy calendar")
        if self.provider == "binance" and (
            self.calendar != "24/7"
            or self.asset_class != "crypto"
            or not re.fullmatch(r"[A-Z0-9]+USDT", self.ticker)
        ):
            raise ValueError("Binance proxy requires 24/7 USDT crypto contract")
        for name in ("unit", "description", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 1000:
                raise ValueError(f"Explicit {name} required")
        if _date(self.valid_from) >= _date(self.valid_to):
            raise ValueError("Invalid mapping research window")

    def active(self, at):
        return _date(self.valid_from) <= utc(at) < _date(self.valid_to)


class ProxyMappings:
    def __init__(self, records):
        self.records = tuple(records)
        self._by_id = {}
        for record in self.records:
            self._by_id.setdefault(record.instrument_id, []).append(record)
        for rows in self._by_id.values():
            rows.sort(key=lambda row: row.valid_from)
            for before, after in zip(rows, rows[1:]):
                if before.valid_to > after.valid_from:
                    raise ValueError("overlapping proxy mappings")

    def at(self, instrument_id, at):
        return next(
            (row for row in self._by_id.get(instrument_id, ()) if row.active(at)), None
        )

    def exclusions(self, instrument_ids, at):
        return {
            coin: "missing_proxy_mapping"
            for coin in sorted(instrument_ids)
            if self.at(coin, at) is None
        }
