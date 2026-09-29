"""Explicit lossless feature storage records; not source-completeness evidence."""

from dataclasses import fields
import json

import pyarrow as pa

from .contracts import address, symbol
from .derived_publication import _encode
from .episode_state import _number, _timestamp, decode_episode_state
from .wallet_day_features import FillObservation, EpisodeObservation

MAX_PAYLOAD = 64 * 1024
MAX_KEY_BYTES = 16 * 1024
KEY_FIELDS = (
    "exchange_time",
    "block_number",
    "source_key",
    "source_line",
    "event_index",
    "event_id",
)
OBSERVATION_SCHEMA = pa.schema(
    [
        ("kind", pa.string()),
        ("user", pa.string()),
        ("coin", pa.string()),
        ("exchange_time", pa.timestamp("us", tz="UTC")),
        ("block_number", pa.int64()),
        ("source_key", pa.string()),
        ("source_line", pa.int64()),
        ("event_index", pa.int64()),
        ("event_id", pa.string()),
        ("payload", pa.binary()),
    ]
)
CHECKPOINT_SCHEMA = pa.schema([("user", pa.string()), ("payload", pa.binary())])
CLASSES = {"fill": FillObservation, "episode": EpisodeObservation}


def _wallet(user):
    if type(user) is not str or address(user) != user:
        raise ValueError("Expected canonical feature wallet")


def _key(key):
    if type(key) is not tuple or len(key) != 6:
        raise ValueError("Invalid feature native key")
    at = _timestamp(key[0])
    for i in (1, 3, 4):
        if type(key[i]) is not int or not -(2**63) <= key[i] < 2**63:
            raise ValueError("Feature native key integer outside int64")
    for i in (2, 5):
        if type(key[i]) is not str or len(key[i]) > MAX_KEY_BYTES:
            raise ValueError("Invalid or oversized feature key string")
    if sum(len(key[i].encode()) for i in (2, 5)) > MAX_KEY_BYTES:
        raise ValueError("Feature native key byte limit exceeded")
    return (at, *key[1:])


def _payload(kind, payload):
    expected = {field.name for field in fields(CLASSES[kind])} - {
        "user",
        "coin",
        "order_key",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("Invalid feature payload fields")
    for name, value in payload.items():
        if name not in ("side", "crossed"):
            _number(value)
    if kind == "fill":
        if (
            type(payload["side"]) is not str
            or payload["side"] not in ("B", "A")
            or type(payload["crossed"]) is not bool
        ):
            raise ValueError("Invalid feature fill side/crossed flag")
    elif (
        payload["minutes"] < 0
        or type(payload["fragments"]) is not int
        or not 0 < payload["fragments"] < 2**63
    ):
        raise ValueError("Invalid feature episode duration/count")
    encoded = _encode(payload)
    if len(encoded) > MAX_PAYLOAD:
        raise ValueError("Feature payload byte limit exceeded")
    return encoded


def _load(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PAYLOAD:
        raise ValueError("Invalid feature payload bytes")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid feature payload JSON") from exc


def encode_observation(observation):
    kind = next(
        (kind for kind, cls in CLASSES.items() if type(observation) is cls), None
    )
    if kind is None:
        raise ValueError("Expected immutable feature observation")
    _wallet(observation.user)
    coin = symbol(observation.coin)
    key = _key(observation.order_key)
    payload = {
        f.name: getattr(observation, f.name)
        for f in fields(observation)
        if f.name not in ("user", "coin", "order_key")
    }
    return dict(
        kind=kind,
        user=observation.user,
        coin=coin,
        **dict(zip(KEY_FIELDS, key)),
        payload=_payload(kind, payload),
    )


def decode_observation(row):
    if type(row) is not dict or set(row) != set(OBSERVATION_SCHEMA.names):
        raise ValueError("Invalid observation row fields")
    kind = row["kind"]
    if type(kind) is not str or kind not in CLASSES:
        raise ValueError("Invalid feature observation kind")
    _wallet(row["user"])
    coin = symbol(row["coin"])
    key = _key(tuple(row[name] for name in KEY_FIELDS))
    payload = _load(row["payload"])
    if _payload(kind, payload) != row["payload"]:
        raise ValueError("Expected canonical typed feature payload")
    return CLASSES[kind](user=row["user"], coin=coin, order_key=key, **payload)


def encode_checkpoint(payload, *, user, cutoff, semantics):
    decode_episode_state(payload, user=user, cutoff=cutoff, semantics=semantics)
    return dict(user=user, payload=_encode(payload))


def decode_checkpoint(row, *, user, cutoff, semantics):
    if (
        type(row) is not dict
        or set(row) != set(CHECKPOINT_SCHEMA.names)
        or row["user"] != user
    ):
        raise ValueError("Invalid checkpoint row/header")
    payload = _load(row["payload"])
    if (
        encode_checkpoint(payload, user=user, cutoff=cutoff, semantics=semantics)[
            "payload"
        ]
        != row["payload"]
    ):
        raise ValueError("Expected canonical checkpoint payload")
    return payload
