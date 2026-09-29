"""Historical instrument identity, classification and supported lifetime contract."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from .contracts import symbol, utc, finite
from .lab_config_v2 import CLASSES


def timestamp(value):
    return utc(datetime.fromisoformat(value) if isinstance(value, str) else value)


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    display_name: str
    venue: str
    asset_class: str
    base: str
    quote: str
    settlement: str
    multiplier: float
    model: str
    known_at: datetime
    effective_from: datetime
    effective_to: datetime | None
    listed_at: datetime
    delisted_at: datetime | None

    def __post_init__(self):
        symbol(self.instrument_id)
        for name in (
            "display_name",
            "venue",
            "base",
            "quote",
            "settlement",
            "model",
            "asset_class",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not 1 <= len(value) <= 120:
                raise ValueError("Invalid instrument metadata")
        if self.venue != (
            self.instrument_id.split(":")[0] if ":" in self.instrument_id else "core"
        ):
            raise ValueError("Instrument namespace does not match venue")
        if finite(self.multiplier) <= 0:
            raise ValueError("Invalid contract multiplier")
        for name in (
            "known_at",
            "effective_from",
            "effective_to",
            "listed_at",
            "delisted_at",
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, timestamp(getattr(self, name)))
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("Invalid classification interval")
        if self.delisted_at is not None and self.delisted_at <= self.listed_at:
            raise ValueError("Invalid instrument lifetime")

    @property
    def supported(self):
        return (
            self.model == "linear_usd_continuous_v1"
            and self.multiplier == 1
            and self.quote == "USD"
            and self.settlement in ("USD", "USDC")
        )

    def active(self, at):
        return self.listed_at <= at and (
            self.delisted_at is None or at < self.delisted_at
        )


class Catalogue:
    def __init__(self, rows):
        self.records = [Instrument(**r) if isinstance(r, dict) else r for r in rows]
        self.by_id = {}
        for row in self.records:
            self.by_id.setdefault(row.instrument_id, []).append(row)
        for versions in self.by_id.values():
            versions.sort(key=lambda r: r.effective_from)
            for before, after in zip(versions, versions[1:]):
                if (
                    before.effective_to is None
                    or before.effective_to > after.effective_from
                ):
                    raise ValueError("Overlapping instrument catalogue versions")
                if (before.listed_at, before.delisted_at) != (
                    after.listed_at,
                    after.delisted_at,
                ):
                    raise ValueError("Conflicting instrument lifetimes")
                execution_fields = (
                    "base",
                    "quote",
                    "settlement",
                    "multiplier",
                    "model",
                )
                if any(
                    getattr(before, field) != getattr(after, field)
                    for field in execution_fields
                ):
                    raise ValueError(
                        "Changing execution specifications are unsupported"
                    )

    def known_ids(self, at):
        return sorted(
            k for k, v in self.by_id.items() if any(r.known_at < at for r in v)
        )

    def at(self, identifier, at):
        return next(
            (
                r
                for r in self.by_id.get(identifier, [])
                if r.known_at < at
                and r.effective_from <= at
                and (r.effective_to is None or at < r.effective_to)
            ),
            None,
        )

    def lifetimes(self, ids):
        return {identifier: self.by_id[identifier][0] for identifier in ids}

    def public_rows(self):
        return [
            asdict(r) | {"supported": r.supported}
            for r in sorted(
                self.records, key=lambda r: (r.instrument_id, r.effective_from)
            )
        ]
