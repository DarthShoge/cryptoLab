"""Offline market coverage bundles; never evidence of native listing dates."""

from dataclasses import asdict
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import symbol, utc
from .download import file_hash
from .market_bundle_inputs import load_inputs, recheck
from .proxy_bars import ProxyBar, ProxyBars
from .proxy_mapping import ProxyMapping
from .proxy_sessions import calendar_provenance, hourly_windows

POLICY = dict(
    adjustment="raw", corporate_actions="none_detected", rolls="not_applicable"
)


def hour(value):
    if not isinstance(value, datetime):
        raise ValueError("Expected datetime")
    value = utc(value)
    if value.minute or value.second or value.microsecond:
        raise ValueError("Expected aligned whole hour")
    return value


def windows(calendar, start, end):
    result = {}
    while start < end:
        stop = min(end, start + timedelta(days=365))
        result.update(hourly_windows(calendar, start, stop))
        start = stop
    return result


def source_interval(data):
    start, end = (hour(datetime.fromisoformat(data[k])) for k in ("start", "end"))
    if not start < end or end - start > timedelta(days=93):
        raise ValueError("Invalid bounded source interval")
    return start, end


def price_metadata(inputs, requested):
    mappings = {}
    for path, data, _ in inputs:
        start, end = source_interval(data)
        if data.get("price_policy") != POLICY:
            raise ValueError("Unsupported price policy")
        if data.get("calendar") != calendar_provenance():
            raise ValueError("Inconsistent calendar version")
        local = {}
        if not isinstance(data.get("mappings"), list) or not 1 <= len(
            data["mappings"]
        ) <= len(requested):
            raise ValueError("Unexpected mapping markets")
        for record in data["mappings"]:
            mapping = ProxyMapping(**record)
            coin = mapping.instrument_id
            if coin not in requested:
                raise ValueError("Unexpected mapping markets")
            if coin in local or (coin in mappings and mappings[coin] != mapping):
                raise ValueError("Inconsistent proxy mapping")
            local[coin] = mapping
            mappings[coin] = mapping
        sessions = json.loads((path.parent / "sessions.json").read_bytes())
        expected = {
            coin: {
                a.isoformat(): b.isoformat()
                for a, b in windows(m.calendar, start, end).items()
            }
            for coin, m in local.items()
        }
        if sessions != expected:
            raise ValueError("Session coverage disagrees with calendar")
    return mappings


def validated_rows(kind, inputs, requested, mappings):
    unique = {}
    time_key = "start" if kind == "prices" else "hour"
    fields = (
        {"instrument_id", "start", "end", "open", "high", "low", "close"}
        if kind == "prices"
        else {"instrument_id", "hour", "time", "rate", "premium"}
    )
    for _, data, parquet in inputs:
        begin, end = source_interval(data)
        coverage = data.get("coverage")
        if not isinstance(coverage, list) or not coverage:
            raise ValueError("Missing coverage evidence")
        coins = {c["instrument_id"] for c in coverage}
        if not coins <= requested or len(coins) != len(coverage):
            raise ValueError("Unexpected market coverage")
        scheduled = (
            {coin: windows(mappings[coin].calendar, begin, end) for coin in coins}
            if kind == "prices"
            else {}
        )
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=4096):
            if set(batch.schema.names) != fields or len(batch.schema.names) != len(
                fields
            ):
                raise ValueError("Unexpected row schema")
            for field in batch.schema:
                if field.name in ("start", "end", "hour", "time") and (
                    not pa.types.is_timestamp(field.type)
                    or field.type.unit not in ("s", "ms", "us")
                    or field.type.tz is None
                ):
                    raise ValueError("Unsupported timestamp precision or timezone")
            for row in batch.to_pylist():
                coin = row["instrument_id"]
                symbol(coin)
                if coin not in coins:
                    raise ValueError("Unexpected row instrument")
                for key, value in row.items():
                    if key in ("open", "high", "low", "close", "rate", "premium"):
                        if type(value) not in (int, float) or not math.isfinite(value):
                            raise ValueError("Expected finite numeric row value")
                    elif key != "instrument_id":
                        if not isinstance(value, datetime):
                            raise ValueError("Expected timestamp row value")
                        row[key] = utc(value)
                at = row[time_key]
                if not begin <= at < end:
                    raise ValueError("Row outside source interval")
                if kind == "prices":
                    ProxyBar(**row)
                    mapping = mappings.get(coin)
                    if mapping is None or not mapping.active(at):
                        raise ValueError("Missing active mapping")
                    if scheduled[coin].get(at) != row["end"]:
                        raise ValueError("Price row disagrees with source calendar")
                else:
                    hour(at)
                    if not at <= row["time"] < at + timedelta(hours=1):
                        raise ValueError("Settlement outside funding hour")
                key = (coin, at)
                if key in unique and unique[key] != row:
                    raise ValueError("conflicting market records")
                unique[key] = row
    if {coin for coin, _ in unique} != requested:
        raise ValueError("Missing requested market coverage")
    if kind == "prices":
        ProxyBars(ProxyBar(**r) for r in unique.values())
    return unique


def sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def bundle_market_inputs(kind, manifest_paths, intervals, output_root):
    """Verify and bundle exact requested intervals without downloading or filling gaps."""
    if kind not in ("prices", "funding") or not intervals or len(intervals) > 64:
        raise ValueError("Expected bounded prices/funding intervals")
    intervals = {symbol(coin): (hour(a), hour(b)) for coin, (a, b) in intervals.items()}
    if any(not a < b or b - a > timedelta(days=732) for a, b in intervals.values()):
        raise ValueError("Invalid requested interval")
    dependencies = (
        "proxy_market_bundle.py",
        "market_bundle_inputs.py",
        "proxy_bars.py",
        "proxy_mapping.py",
        "proxy_sessions.py",
        "contracts.py",
        "download.py",
    )
    engine = {
        Path(__file__).parent / name: file_hash(Path(__file__).parent / name)
        for name in dependencies
    }
    inputs, pins = load_inputs(kind, manifest_paths)
    mappings = price_metadata(inputs, set(intervals)) if kind == "prices" else {}
    if kind == "prices" and set(mappings) != set(intervals):
        raise ValueError("Unexpected mapping markets")
    unique = validated_rows(kind, inputs, set(intervals), mappings)
    selected = []
    for coin, (begin, end) in sorted(intervals.items()):
        expected = windows(
            mappings[coin].calendar if kind == "prices" else "24/7", begin, end
        )
        actual = {
            at: row
            for (c, at), row in unique.items()
            if c == coin and begin <= at < end
        }
        if actual.keys() != expected.keys():
            raise ValueError(f"Missing or unexpected market coverage: {coin}")
        if kind == "prices" and any(
            actual[at]["end"] != finish for at, finish in expected.items()
        ):
            raise ValueError("Price bar end disagrees with calendar coverage")
        selected.extend(actual[at] for at in sorted(actual))
    if not selected:
        raise ValueError("Empty requested market coverage")
    output_root = Path(output_root).absolute()
    if any(p.is_symlink() for p in (output_root, *output_root.parents)):
        raise ValueError("Output symlinks are not supported")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".market_bundle_", dir=output_root) as tmp:
        stage = Path(tmp)
        filename = "bars.parquet" if kind == "prices" else "funding.parquet"
        output = stage / filename
        pq.write_table(pa.Table.from_pylist(selected), output, compression="zstd")
        if output.stat().st_size > 64 * 1024**2:
            raise ValueError("Bundle output exceeds byte bound")
        report = dict(
            schema="hyperliquid_proxy_market_bundle_v1",
            kind=kind,
            research_eligible=False,
            native_availability_claim=False,
            intervals={
                coin: [a.isoformat(), b.isoformat()]
                for coin, (a, b) in intervals.items()
            },
            source_manifests=[
                dict(
                    path=str(path),
                    sha256=pins[path][1],
                    complete=data["complete"],
                    coverage=data["coverage"],
                )
                for path, data, _ in inputs
            ],
            file=dict(
                name=filename,
                sha256=file_hash(output),
                bytes=output.stat().st_size,
                rows=len(selected),
            ),
            engine={str(p): digest for p, digest in engine.items()},
        )
        if kind == "prices":
            report.update(
                mappings=[asdict(m) for m in mappings.values()],
                price_policy=POLICY,
                calendar=calendar_provenance(),
            )
        encoded = json.dumps(report, indent=2, sort_keys=True).encode()
        if len(encoded) > 16 * 1024**2:
            raise ValueError("Bundle manifest exceeds byte bound")
        manifest = stage / "manifest.json"
        manifest.write_bytes(encoded)
        recheck(pins)
        if any(file_hash(p) != digest for p, digest in engine.items()):
            raise ValueError("Bundle engine changed")
        sync(output)
        sync(manifest)
        sync(stage)
        destination = output_root / f"market_bundle_{uuid.uuid4().hex}"
        os.rename(stage, destination)
        sync(output_root)
    return destination / "manifest.json"
