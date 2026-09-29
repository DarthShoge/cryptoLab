"""Exact source membership and trade bounds for canonical prefix validation."""

from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from .archive import archive_keys
from .archive_cache import _safe
from .archive_plan import object_index
from .compact_catalog import _validated
from .contracts import symbol
from .download import BUCKET, file_hash
from .lab_config import day


def read_json(path, limit=1_000_000):
    path = Path(path).absolute()
    _safe(path)
    if path.name != "manifest.json" or path.stat().st_size > limit:
        raise ValueError("Invalid qualification manifest path/size")
    return json.loads(path.read_text())


def raw_source(evidence):
    path = Path(evidence["source_manifest"]).absolute()
    identity = file_hash(path)
    data = read_json(path)
    if identity != evidence["source_manifest_sha256"]:
        raise ValueError("Raw provenance manifest identity mismatch")
    if (
        data.get("schema") != "hyperliquid_proxy_archive_v1"
        or data.get("complete") is not True
        or data.get("bucket") != BUCKET
    ):
        raise ValueError("Incomplete raw source provenance")
    begin, finish = day(data["start"]), day(data["end"])
    if not 0 < (finish - begin).days <= 7 or (data["start"], data["end"]) != (
        evidence["start"],
        evidence["end"],
    ):
        raise ValueError("Raw source dates do not match compact evidence")
    expected = sorted(
        key
        for n in range((finish - begin).days)
        for key in archive_keys((begin + timedelta(days=n)).date().isoformat())
    )
    objects = data["objects"]
    scope = object_index([{k: o[k] for k in ("key", "bytes", "etag")} for o in objects])
    if sorted(scope) != expected or evidence["source_keys"] != expected:
        raise ValueError("Incomplete source-key membership")
    if (
        type(data.get("expected_bytes")) is not int
        or data["expected_bytes"] != sum(o["bytes"] for o in objects)
        or data["expected_bytes"] > 6 * 1024**3
    ):
        raise ValueError("Invalid raw source byte totals")
    saved = []
    for o in objects:
        if (
            o.get("status") != "downloaded"
            or not isinstance(o.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", o["sha256"])
            or Path(o["file"]).name != o["file"]
            or not o["file"].endswith(".lz4")
        ):
            raise ValueError("Invalid original raw content provenance")
        saved.append({k: o[k] for k in ("key", "etag", "bytes", "sha256", "file")})
    if file_hash(path) != identity:
        raise ValueError("Raw provenance changed during qualification")
    return dict(
        path=str(path),
        sha256=identity,
        start=data["start"],
        end=data["end"],
        objects=saved,
        raw_bytes_reverified=False,
    )


def collect(manifest_paths):
    paths = [Path(p).absolute() for p in manifest_paths]
    if not 1 <= len(paths) <= 732 or len(set(paths)) != len(paths):
        raise ValueError("Invalid canonical prefix manifest count")
    identities = [file_hash(p) for p in paths]
    data = [read_json(p) for p in paths]
    if (
        sum(d["output_bytes"] for d in data) > 64 * 1024**3
        or sum(len(d["files"]) for d in data) > 5000
    ):
        raise ValueError("Canonical prefix input ceiling exceeded")
    manifests, files, sources, coins, finish = [], [], [], None, None
    schema = None
    for path, d, frozen_identity in zip(paths, data, identities):
        evidence = d["source_evidence"]
        current = evidence["coins"]
        if (
            not isinstance(current, list)
            or not 1 <= len(current) <= 50
            or current != sorted({symbol(c) for c in current})
            or (coins is not None and coins != current)
        ):
            raise ValueError("Canonical prefix market scope changed")
        coins = current
        if (
            evidence.get("scope") != "all_wallets_for_declared_markets"
            or evidence.get("boundary_spill_retained") is not True
            or d.get("partitioning") != "source_day"
            or d.get("validation") != "exact_canonical_projection_only"
        ):
            raise ValueError(
                "Canonical prefix requires boundary-safe all-wallet projection"
            )
        source = raw_source(evidence)
        if finish is not None and source["start"] != finish:
            raise ValueError("Source prefix is not contiguous")
        finish = source["end"]
        days = [
            (day(source["start"]) + timedelta(days=n)).date().isoformat()
            for n in range((day(finish) - day(source["start"])).days)
        ]
        if [f.get("source_day") for f in d["files"]] != days:
            raise ValueError("Incomplete source-day compact layout")
        identity, _, records = _validated(path)
        if identity != frozen_identity:
            raise ValueError("Compact manifest identity changed during qualification")
        manifests.append(dict(path=str(path), sha256=identity))
        sources.append(source)
        for _, name, digest, size, rows, low, high in records:
            fingerprint = hashlib.sha256(
                pq.read_schema(name).serialize().to_pybytes()
            ).hexdigest()
            if schema is not None and fingerprint != schema:
                raise ValueError(
                    "Canonical prefix schema changed; full new version required"
                )
            schema = fingerprint
            files.append(
                dict(
                    path=name,
                    sha256=digest,
                    bytes=size,
                    rows=rows,
                    min_time=low,
                    max_time=high,
                    schema_sha256=fingerprint,
                )
            )
    if len({e["path"] for e in files}) != len(files):
        raise ValueError("Duplicate canonical prefix files")
    return manifests, files, sources, coins


def trade_bounds(entry, coins):
    bounds = {}
    with pq.ParquetFile(entry["path"]) as reader:
        for batch in reader.iter_batches(
            batch_size=4096, columns=["coin", "tid"], use_threads=False
        ):
            if batch.nbytes > 64 * 1024**2:
                raise ValueError("Trade bounds decoded batch limit exceeded")
            for coin, tid in zip(
                batch.column(0).to_pylist(), batch.column(1).to_pylist()
            ):
                if coin not in coins or type(tid) is not int:
                    raise ValueError("Invalid canonical trade bound identity")
                low, high = bounds.get(coin, (tid, tid))
                bounds[coin] = [min(low, tid), max(high, tid)]
    return bounds


def overlaps(a, b):
    return any(
        max(a[c][0], b[c][0]) <= min(a[c][1], b[c][1]) for c in a.keys() & b.keys()
    )
