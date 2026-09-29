"""Public Hyperliquid funding acquisition, with strict coverage and raw evidence."""

import hashlib
import json
import tempfile
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from .contracts import symbol, utc
from .proxy_funding import funding_history


ENDPOINT = "https://api.hyperliquid.xyz/info"
MAX_PAGE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024


def public_funding_page(payload):
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read(MAX_PAGE_BYTES + 1)
    if len(raw) > MAX_PAGE_BYTES:
        raise ValueError("Funding page exceeds byte ceiling")
    return raw


def download_funding(
    instrument_ids, start, end, output_root, *, fetch=public_funding_page
):
    start, end = utc(start), utc(end)
    cutoff = datetime.now(timezone.utc)
    if end > cutoff or not 0 < (end - start).total_seconds() <= 93 * 86400:
        raise ValueError("Invalid or future funding range")
    if any(t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError("Funding range must use whole UTC hours")
    coins = tuple(instrument_ids)
    if not 0 < len(coins) <= 50 or len(set(coins)) != len(coins):
        raise ValueError("Expected 1 to 50 distinct funding instruments")
    for coin in coins:
        symbol(coin)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="proxy_funding_", dir=output_root))
    total = 0
    sources, coverage, events = [], [], []

    def page(payload):
        nonlocal total
        if total + MAX_PAGE_BYTES > MAX_TOTAL_BYTES:
            raise ValueError("Funding download exceeds total byte ceiling")
        raw = fetch(payload)
        if len(raw) > MAX_PAGE_BYTES:
            raise ValueError("Funding page exceeds byte ceiling")
        total += len(raw)
        name = f"source_{len(sources):04d}.json"
        (root / name).write_bytes(raw)
        sources.append(
            dict(
                endpoint=ENDPOINT,
                request=payload,
                file=name,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
        return json.loads(raw)

    for coin in coins:
        result = funding_history(coin, start, end, page)
        coverage.append(
            dict(
                instrument_id=coin,
                events=len(result.events),
                missing_hours=[t.isoformat() for t in result.missing_hours],
            )
        )
        if result.missing_hours:
            (root / "coverage_failure.json").write_text(json.dumps(coverage, indent=2))
            raise ValueError(f"Missing historical funding: {coin}; see {root}")
        events.extend(asdict(event) for event in result.events)
    with duckdb.connect() as db:
        db.register("funding_rows", pd.DataFrame(events))
        db.execute(
            "COPY funding_rows TO ? (FORMAT PARQUET)", [str(root / "funding.parquet")]
        )
    manifest = dict(
        schema="hyperliquid_proxy_funding_v1",
        complete=True,
        acquisition_cutoff=cutoff.isoformat(),
        start=start.isoformat(),
        end=end.isoformat(),
        sources=sources,
        coverage=coverage,
        total_bytes=total,
        semantics="Native historical rates; actual settlement times retained; no funding-notional conversion performed",
        funding_sha256=hashlib.sha256(
            (root / "funding.parquet").read_bytes()
        ).hexdigest(),
    )
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
