"""Bounded ownership declarations; decoding alone never authorizes cleanup."""

import json
import re

from .derived_publication import _encode, _inputs

MAX_BYTES = 64 * 1024
ROLES = {
    "manifest": ("staging", "", MAX_BYTES, "payload"),
    "scratch": ("scratch", "", 3 * 1024**3, "scratch"),
    "metrics": ("staging", ".parquet", 512 * 1024**2, "payload"),
    "scores": ("staging", ".parquet", 512 * 1024**2, "payload"),
    "ranking": ("artifacts", ".parquet", 512 * 1024**2, "payload"),
}


def encode_manifest(value):
    data = _inputs("ranking_staging_manifest", value)
    if (
        set(data) != {"schema", "context", "allocations"}
        or type(data["schema"]) is not int
        or data["schema"] != 1
        or type(data["context"]) is not dict
        or not data["context"]
        or type(data["allocations"]) is not dict
        or set(data["allocations"]) != set(ROLES)
    ):
        raise ValueError("Invalid ranking staging manifest")
    tokens, paths = set(), set()
    for role, (directory, suffix, maximum, purpose) in ROLES.items():
        row = data["allocations"][role]
        if (
            type(row) is not dict
            or set(row) != {"token", "path", "maximum", "purpose"}
            or type(row["token"]) is not str
            or not re.fullmatch(r"[a-f0-9]{32}", row["token"])
            or type(row["path"]) is not str
            or not re.fullmatch(
                directory + r"/[a-f0-9]{32}" + re.escape(suffix), row["path"]
            )
            or type(row["maximum"]) is not int
            or row["maximum"] != maximum
            or row["purpose"] != purpose
            or row["token"] in tokens
            or row["path"] in paths
        ):
            raise ValueError("Invalid ranking staging allocation ownership")
        tokens.add(row["token"])
        paths.add(row["path"])
    encoded = _encode(data)
    if len(encoded) > MAX_BYTES:
        raise ValueError("Ranking staging manifest byte limit exceeded")
    return encoded


def decode_manifest(encoded):
    if type(encoded) is not bytes or not 0 < len(encoded) <= MAX_BYTES:
        raise ValueError("Invalid ranking staging manifest bytes")
    try:
        value = json.loads(encoded)
        if encode_manifest(value) != encoded:
            raise ValueError("Noncanonical ranking staging manifest")
        return value
    except (UnicodeError, RecursionError, TypeError) as exc:
        raise ValueError("Invalid ranking staging manifest encoding") from exc
