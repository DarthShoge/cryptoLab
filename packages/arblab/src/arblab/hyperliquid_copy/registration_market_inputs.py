"""Reconstruct market bundles from retained evidence before dataset publication."""

from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .market_bundle_inputs import pin_file, recheck, safe_file
from .proxy_market_bundle import bundle_market_inputs


def verified_market_bundle(pin, *, kind, temp_root):
    """Verify output identity and repeat bounded source/coverage validation offline."""
    if (
        kind not in ("prices", "funding")
        or not isinstance(pin, dict)
        or set(pin) != {"path", "sha256"}
    ):
        raise ValueError("Expected market bundle kind and manifest pin")
    path = safe_file(pin["path"])
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Market bundle manifest byte bound exceeded")
    pins = {}
    pin_file(path, pins, digest=pin["sha256"])
    data = json.loads(path.read_bytes())
    if (
        data.get("schema") != "hyperliquid_proxy_market_bundle_v1"
        or data.get("kind") != kind
    ):
        raise ValueError("Unexpected market bundle schema/kind")
    if (
        data.get("research_eligible") is not False
        or data.get("native_availability_claim") is not False
    ):
        raise ValueError("Market bundle cannot assert research/native availability")
    filename = "bars.parquet" if kind == "prices" else "funding.parquet"
    entry = data["file"]
    if (
        entry.get("name") != filename
        or type(entry.get("bytes")) is not int
        or not 0 < entry["bytes"] <= 64 * 1024**2
    ):
        raise ValueError("Invalid market bundle output identity")
    output = pin_file(
        path.parent / filename, pins, digest=entry["sha256"], size=entry["bytes"]
    )
    intervals = {
        coin: tuple(datetime.fromisoformat(t) for t in dates)
        for coin, dates in data["intervals"].items()
    }
    sources = data["source_manifests"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 64:
        raise ValueError("Market bundle source count bound exceeded")
    temp_root = Path(temp_root).absolute()
    if not temp_root.is_dir() or any(
        p.is_symlink() for p in (temp_root, *temp_root.parents)
    ):
        raise ValueError("Expected existing safe temporary directory")
    with TemporaryDirectory(prefix="registration_bundle_", dir=temp_root) as scratch:
        rebuilt = bundle_market_inputs(
            kind, [s["path"] for s in sources], intervals, Path(scratch)
        )
        if json.loads(rebuilt.read_bytes()) != data:
            raise ValueError(
                "Market bundle reconstruction differs from pinned evidence"
            )
        recheck(pins)
    return dict(manifest=data, path=path, file=output)
