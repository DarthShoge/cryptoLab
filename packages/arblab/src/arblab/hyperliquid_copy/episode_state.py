"""Lossless bounded active-episode state; not a source or cache certificate.

The caller owns canonical ordering, source/engine pins and accounted publication.
No arithmetic, file IO, source coverage inference or cache allocation occurs here.
"""

from datetime import datetime
from math import isfinite

from .contracts import address, symbol, utc
from .derived_publication import _encode
from .episodes import PositionEpisode

SCHEMA = "active_episode_state_v1"
MAX_BYTES = 64 * 1024
FIELDS = {
    "coin",
    "opened_at",
    "pnl",
    "fill_count",
    "taker_count",
    "fees",
    "peak_notional",
    "left_censored",
}


def _timestamp(value):
    if type(value) is not datetime:
        raise ValueError("Expected aware episode timestamp")
    try:
        return utc(value)
    except OverflowError as exc:
        raise ValueError("Episode timestamp outside UTC range") from exc


def _context(user, cutoff, semantics):
    if type(user) is not str or address(user) != user:
        raise ValueError("Expected canonical checkpoint wallet")
    cutoff = _timestamp(cutoff)
    if any((cutoff.hour, cutoff.minute, cutoff.second, cutoff.microsecond)):
        raise ValueError("Expected UTC midnight checkpoint cutoff")
    if type(semantics) is not str or semantics not in {
        "gross_excludes_fee",
        "net_includes_fee",
    }:
        raise ValueError("Expected resolved checkpoint fee semantics")
    return dict(
        schema=SCHEMA, user=user, cutoff=cutoff.isoformat(), fee_semantics=semantics
    ), cutoff


def _opened(value, cutoff):
    if type(value) is not str or not 1 <= len(value) <= 40:
        raise ValueError("Invalid checkpoint opening timestamp")
    try:
        opened = _timestamp(datetime.fromisoformat(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid checkpoint opening timestamp") from exc
    if opened.isoformat() != value or opened >= cutoff:
        raise ValueError("Checkpoint opening must precede cutoff in canonical UTC")
    return opened


def _number(value):
    if type(value) is int:
        valid = value.bit_length() <= 4096
    else:
        valid = type(value) is float and isfinite(value)
    if not valid:
        raise ValueError("Invalid or oversized episode accumulator")


def decode_episode_state(payload, *, user, cutoff, semantics):
    """Validate the entire snapshot, then restore fresh mutable episode objects."""
    header, cutoff = _context(user, cutoff, semantics)
    if (
        type(payload) is not dict
        or len(payload) != 5
        or set(payload) != set(header) | {"episodes"}
    ):
        raise ValueError("Invalid episode checkpoint schema")
    if any(
        type(payload[k]) is not str or payload[k] != value
        for k, value in header.items()
    ):
        raise ValueError("Episode checkpoint context mismatch")
    rows = payload["episodes"]
    if type(rows) is not list or len(rows) > 50:
        raise ValueError("Episode checkpoint asset bound exceeded")
    prepared, previous = [], None
    for row in rows:
        if type(row) is not dict or len(row) != len(FIELDS) or set(row) != FIELDS:
            raise ValueError("Invalid active episode row")
        coin = row["coin"]
        if (
            type(coin) is not str
            or symbol(coin) != coin
            or previous is not None
            and coin <= previous
        ):
            raise ValueError("Expected distinct sorted episode markets")
        opened = _opened(row["opened_at"], cutoff)
        for field in ("pnl", "fees", "peak_notional"):
            _number(row[field])
        if (
            row["peak_notional"] < 0
            or type(row["fill_count"]) is not int
            or not 1 <= row["fill_count"] < 2**63
            or type(row["taker_count"]) is not int
            or not 0 <= row["taker_count"] <= row["fill_count"]
            or type(row["left_censored"]) is not bool
        ):
            raise ValueError("Invalid active episode counters/state")
        prepared.append(dict(row, user=user, opened_at=opened))
        previous = coin
    # All structures are now shallow and scalar-bounded, before JSON encoding.
    if len(_encode(payload)) > MAX_BYTES:
        raise ValueError("Episode checkpoint byte limit exceeded")
    return {row["coin"]: PositionEpisode(**row) for row in prepared}


def encode_episode_state(active, *, user, cutoff, semantics):
    """Snapshot without aliasing, rounding, decimalizing or regrouping state."""
    header, cutoff = _context(user, cutoff, semantics)
    if type(active) is not dict or len(active) > 50:
        raise ValueError("Expected bounded active episode dictionary")
    if any(type(coin) is not str for coin in active):
        raise ValueError("Invalid episode market key")
    rows = []
    for coin in sorted(active):
        episode = active[coin]
        if (
            not isinstance(episode, PositionEpisode)
            or episode.user != user
            or episode.coin != coin
            or episode.closed_at is not None
        ):
            raise ValueError("Expected matching wallet/market active episode")
        row = {field: getattr(episode, field) for field in FIELDS}
        row["opened_at"] = _timestamp(episode.opened_at).isoformat()
        rows.append(row)
    payload = dict(header, episodes=rows)
    decode_episode_state(payload, user=user, cutoff=cutoff, semantics=semantics)
    return payload
