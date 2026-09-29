"""Bounded public price acquisition; no credentials, paid requests or orders."""

import hashlib
import io
import json
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

from .contracts import utc
from .proxy_mapping import ProxyMappings
from .proxy_sessions import calendar_provenance, hourly_windows
from .proxy_sources import binance_bars, yahoo_bars


MAX_OBJECT_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024


def public_fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read(MAX_OBJECT_BYTES + 1)
    if len(raw) > MAX_OBJECT_BYTES:
        raise ValueError("Public source object exceeds byte ceiling")
    return raw


def download_prices(mappings, start, end, output_root, *, fetch=public_fetch):
    start, end = utc(start), utc(end)
    acquisition_cutoff = datetime.now(timezone.utc)
    if end > acquisition_cutoff:
        raise ValueError("Cannot acquire future or unfinished coverage")
    mappings = tuple(mappings)
    ProxyMappings(mappings)
    if (
        not 0 < (end - start).total_seconds() <= 93 * 86400
        or not 0 < len(mappings) <= 50
    ):
        raise ValueError("Acquisition limited to 93 days and 50 mappings")
    if len({m.instrument_id for m in mappings}) != len(mappings):
        raise ValueError("One mapping per instrument required for acquisition")
    for mapping in mappings:
        if not mapping.active(start) or not mapping.active(
            end - timedelta(microseconds=1)
        ):
            raise ValueError("Requested range outside mapping window")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="proxy_prices_", dir=output_root))
    sources, coverage, all_bars, sessions = [], [], [], {}
    total = 0

    def acquire(url):
        nonlocal total
        raw = fetch(url)
        if len(raw) > MAX_OBJECT_BYTES or total + len(raw) > MAX_TOTAL_BYTES:
            raise ValueError("Price acquisition exceeds byte ceiling")
        total += len(raw)
        name = f"source_{len(sources):04d}.raw"
        (root / name).write_bytes(raw)
        sources.append(
            dict(
                url=url,
                file=name,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
        return raw

    for mapping in mappings:
        windows = hourly_windows(mapping.calendar, start, end)
        sessions[mapping.instrument_id] = {
            a.isoformat(): b.isoformat() for a, b in windows.items()
        }
        if mapping.provider == "binance":
            chunks = []
            date = start.date()
            while date <= (end - timedelta(microseconds=1)).date():
                name = f"{mapping.ticker}-1h-{date.isoformat()}"
                url = f"https://data.binance.vision/data/futures/um/daily/klines/{mapping.ticker}/1h/{name}.zip"
                raw = acquire(url)
                checksum = acquire(url + ".CHECKSUM").decode().split()[0]
                if hashlib.sha256(raw).hexdigest() != checksum:
                    raise ValueError("Binance archive checksum mismatch")
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    entries = archive.infolist()
                    if (
                        len(entries) != 1
                        or entries[0].filename != name + ".csv"
                        or entries[0].file_size > MAX_OBJECT_BYTES
                    ):
                        raise ValueError("Unexpected Binance archive contents")
                    chunks.append(archive.read(entries[0]).decode())
                date += timedelta(days=1)
            normalized = binance_bars(
                "\n".join(chunks), mapping.instrument_id, start=start, end=end
            )
        else:
            query = urllib.parse.urlencode(
                dict(
                    period1=int(start.timestamp()),
                    period2=int(end.timestamp()),
                    interval="1h",
                    includePrePost="false",
                    events="div,splits,capitalGains",
                )
            )
            url = (
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                + urllib.parse.quote(mapping.ticker, safe="")
                + "?"
                + query
            )
            normalized = yahoo_bars(
                json.loads(acquire(url)),
                mapping.instrument_id,
                start=start,
                end=end,
                windows=windows,
            )
        all_bars.extend(asdict(bar) for bar in normalized.bars)
        coverage.append(
            dict(
                instrument_id=mapping.instrument_id,
                bars=len(normalized.bars),
                scheduled_bars=sum(finish <= end for finish in windows.values()),
                session_closed_entire_range=not windows,
                missing_starts=[t.isoformat() for t in normalized.missing_starts],
            )
        )
    if not all_bars:
        raise ValueError("No usable proxy bars acquired")
    with duckdb.connect() as db:
        db.register("bar_rows", pd.DataFrame(all_bars))
        db.execute("COPY bar_rows TO ? (FORMAT PARQUET)", [str(root / "bars.parquet")])
    (root / "sessions.json").write_text(json.dumps(sessions, indent=2))
    manifest = dict(
        schema="hyperliquid_proxy_prices_v1",
        mode="approximate_proxy_prices",
        price_policy=dict(
            adjustment="raw", corporate_actions="none_detected", rolls="not_applicable"
        ),
        price_policy_basis="Yahoo requested event payload contains no in-window actions; instrument type restricted to EQUITY/ETF/INDEX. Binance source is USD-M perpetual klines. No total-return adjustment or roll model applied.",
        captured_at=datetime.now(timezone.utc).isoformat(),
        acquisition_cutoff=acquisition_cutoff.isoformat(),
        start=start.isoformat(),
        end=end.isoformat(),
        mappings=[asdict(m) for m in mappings],
        sources=sources,
        coverage=coverage,
        calendar=calendar_provenance(),
        complete=all(
            not c["missing_starts"] and c["bars"] == c["scheduled_bars"]
            for c in coverage
        ),
        numeraire_assumption="USD; Binance USDT treated at USD parity",
        total_bytes=total,
        funding_included=False,
        bars_sha256=hashlib.sha256((root / "bars.parquet").read_bytes()).hexdigest(),
        sessions_sha256=hashlib.sha256(
            (root / "sessions.json").read_bytes()
        ).hexdigest(),
    )
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
