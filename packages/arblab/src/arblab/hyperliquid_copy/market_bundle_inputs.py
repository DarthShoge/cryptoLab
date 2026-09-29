"""Bounded, hash-pinned local acquisition inputs for market bundles."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from .download import file_hash

MAX_ROWS = 250_000
MAX_NORMALIZED_BYTES = 128 * 1024**2
MAX_RAW_BYTES = 512 * 1024**2


def safe_file(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Expected regular input file without symlinks")
    return path


def pin_file(path, pins, *, digest=None, size=None):
    path = safe_file(path)
    actual_size, actual_hash = path.stat().st_size, file_hash(path)
    if (size is not None and size != actual_size) or (
        digest is not None and digest != actual_hash
    ):
        raise ValueError("Input hash/length identity mismatch")
    identity = (actual_size, actual_hash)
    if path in pins and pins[path] != identity:
        raise ValueError("Input identity changed")
    pins[path] = identity
    return path


def recheck(pins):
    for path, (size, digest) in pins.items():
        pin_file(path, {}, digest=digest, size=size)


def load_inputs(kind, manifest_paths):
    paths = tuple(manifest_paths)
    if not 1 <= len(paths) <= 64:
        raise ValueError("Expected 1..64 input manifests")
    pins, inputs = {}, []
    normalized_bytes = raw_bytes = rows = 0
    for path in paths:
        path = safe_file(path)
        if path.stat().st_size > 1024**2:
            raise ValueError("Manifest exceeds byte bound")
        pin_file(path, pins)
        data = json.loads(path.read_bytes())
        if data.get("schema") != f"hyperliquid_proxy_{kind}_v1":
            raise ValueError("Unexpected input schema")
        if type(data.get("complete")) is not bool:
            raise ValueError("Expected boolean complete flag")
        if kind == "funding" and not data["complete"]:
            raise ValueError("Incomplete funding coverage")
        sources = data.get("sources")
        if not isinstance(sources, list) or not sources or len(sources) > 10_000:
            raise ValueError("Expected bounded raw source evidence")
        for source in sources:
            name = source["file"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or name in (".", "..")
            ):
                raise ValueError("Expected flat source filename")
            raw = safe_file(path.parent / name)
            raw_bytes += raw.stat().st_size
            if raw_bytes > MAX_RAW_BYTES:
                raise ValueError("Raw evidence exceeds byte bound")
            if type(source["bytes"]) is not int or not isinstance(
                source["sha256"], str
            ):
                raise ValueError("Invalid source identity")
            pin_file(raw, pins, digest=source["sha256"], size=source["bytes"])
        filename = "bars.parquet" if kind == "prices" else "funding.parquet"
        parquet = safe_file(path.parent / filename)
        normalized_bytes += parquet.stat().st_size
        if normalized_bytes > MAX_NORMALIZED_BYTES:
            raise ValueError("Normalized input exceeds byte bound")
        pin_file(
            parquet,
            pins,
            digest=data["bars_sha256" if kind == "prices" else "funding_sha256"],
        )
        rows += pq.ParquetFile(parquet).metadata.num_rows
        if rows > MAX_ROWS:
            raise ValueError("Normalized input exceeds row bound")
        if kind == "prices":
            sessions = safe_file(path.parent / "sessions.json")
            if sessions.stat().st_size > 16 * 1024**2:
                raise ValueError("Sessions exceed byte bound")
            pin_file(sessions, pins, digest=data["sessions_sha256"])
        inputs.append((path, data, parquet))
    recheck(pins)
    return inputs, pins
