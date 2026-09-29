"""Offline, bounded normalization of a fully downloaded native fill batch."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

import lz4.frame
import pyarrow.parquet as pq

from .archive import archive_keys, parse_archive_line
from .contracts import symbol
from .download import BUCKET, _table, file_hash
from .lab_config import day


def import_archive(
    manifest_path,
    coins,
    output_root,
    *,
    max_line_bytes=16 * 1024**2,
    max_batch_bytes=16 * 1024**2,
    retain_boundary_spill=False,
    progress=None,
):
    source = Path(manifest_path).resolve(strict=True)
    if source.stat().st_size > 1_000_000:
        raise ValueError("Archive manifest exceeds size limit")
    source_hash = file_hash(source)
    if type(retain_boundary_spill) is not bool:
        raise ValueError("Invalid boundary retention flag")
    data = json.loads(source.read_text())
    if (
        data.get("schema") != "hyperliquid_proxy_archive_v1"
        or data.get("complete") is not True
        or data.get("bucket") != BUCKET
    ):
        raise ValueError("Archive acquisition is not complete/qualified")
    start, end = day(data["start"]), day(data["end"])
    if not 0 < (end - start).days <= 7:
        raise ValueError("Import limited to seven days")
    coins = tuple(sorted(symbol(c) for c in coins))
    if not coins or len(coins) > 50 or len(set(coins)) != len(coins):
        raise ValueError("Invalid import markets")
    if not 0 < max_line_bytes <= 16 * 1024**2:
        raise ValueError("Invalid decoded line limit")
    if not 0 < max_batch_bytes <= 16 * 1024**2:
        raise ValueError("Invalid batch byte limit")
    expected = {
        key
        for offset in range((end - start).days)
        for key in archive_keys(str((start + timedelta(days=offset)).date()))
    }
    objects = data["objects"]
    if len(objects) != len(expected) or {o["key"] for o in objects} != expected:
        raise ValueError("Missing or duplicate hourly archive objects")
    inputs = []
    for obj in objects:
        path = (source.parent / obj["file"]).resolve(strict=True)
        if (
            not path.is_relative_to(source.parent)
            or path in inputs
            or obj.get("status") != "downloaded"
        ):
            raise ValueError("Invalid archive object path/status")
        if path.stat().st_size != obj["bytes"] or file_hash(path) != obj["sha256"]:
            raise ValueError("Archive object identity changed")
        inputs.append(path)
    if sum(p.stat().st_size for p in inputs) > 6 * 1024**3:
        raise ValueError("Archive input exceeds byte ceiling")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="proxy_activity_import_", dir=output_root))
    ingested = datetime.now(timezone.utc)
    files, evidence = [], []
    total_rows = decoded_total = output_bytes = 0
    for index, (obj, path) in enumerate(zip(objects, inputs)):
        target = root / f"fills-{index:04d}.parquet"
        batch, writer, rows, outside, decoded_bytes = [], None, 0, 0, 0
        batch_bytes = 0
        first = last = None

        def flush():
            nonlocal writer, rows, batch_bytes
            if not batch:
                return
            table = _table(batch)
            writer = writer or pq.ParquetWriter(
                target, table.schema, compression="zstd"
            )
            writer.write_table(table)
            rows += len(batch)
            batch.clear()
            batch_bytes = 0
            if output_bytes + target.stat().st_size > 8 * 1024**3:
                raise ValueError("Normalized output exceeds disk ceiling")

        try:
            with lz4.frame.open(path, "rb") as decoded:
                line = 0
                while raw := decoded.readline(max_line_bytes + 1):
                    if len(raw) > max_line_bytes:
                        raise ValueError("Archive decoded line exceeds limit")
                    decoded_bytes += len(raw)
                    decoded_total += len(raw)
                    if decoded_bytes > 2 * 1024**3 or decoded_total > 96 * 1024**3:
                        raise ValueError("Archive decoded byte ceiling exceeded")
                    parsed = parse_archive_line(
                        raw, obj["key"], line, coins=coins, ingested_at=ingested
                    )
                    if parsed.issues:
                        raise ValueError(
                            f"Invalid archive {obj['key']}:{line}: {parsed.issues[0].reason}"
                        )
                    for event in parsed.events:
                        first = (
                            event.exchange_time
                            if first is None
                            else min(first, event.exchange_time)
                        )
                        last = (
                            event.exchange_time
                            if last is None
                            else max(last, event.exchange_time)
                        )
                        if not start <= event.exchange_time < end:
                            outside += 1
                            if not retain_boundary_spill:
                                continue
                        batch.append(asdict(event))
                        batch_bytes += (
                            len(event.raw_details_json.encode("utf-8")) + 2048
                        )
                        if len(batch) >= 5000 or batch_bytes >= max_batch_bytes:
                            flush()
                    line += 1
                flush()
        finally:
            if writer is not None:
                writer.close()
        if rows:
            size = target.stat().st_size
            output_bytes += size
            if output_bytes > 8 * 1024**3:
                raise ValueError("Normalized output exceeds disk ceiling")
            files.append(
                dict(name=target.name, rows=rows, bytes=size, sha256=file_hash(target))
            )
        total_rows += rows
        evidence.append(
            dict(
                key=obj["key"],
                rows=rows,
                outside_window=outside,
                decoded_bytes=decoded_bytes,
                first_event=first.isoformat() if first else None,
                last_event=last.isoformat() if last else None,
            )
        )
        if progress:
            progress(dict(imported=index + 1, objects=len(objects), rows=total_rows))
    if not total_rows:
        raise ValueError("No mapped native fills in archive")
    # A manifest written after normalization must describe the source that was
    # actually parsed, not a replacement installed during a long import.
    if file_hash(source) != source_hash:
        raise ValueError("Archive manifest changed during import")
    for obj, path in zip(objects, inputs):
        if (
            path.is_symlink()
            or path.stat().st_size != obj["bytes"]
            or file_hash(path) != obj["sha256"]
        ):
            raise ValueError("Archive object changed during import")
    manifest = dict(
        schema="hyperliquid_proxy_activity_v1",
        complete=True,
        start=data["start"],
        end=data["end"],
        coins=coins,
        scope="all_wallets_for_declared_markets",
        source_keys=sorted(expected),
        source_manifest_sha256=source_hash,
        source_manifest=str(source),
        objects=evidence,
        rows=total_rows,
        files=files,
        decoded_bytes=decoded_total,
        output_bytes=output_bytes,
        coverage_basis="complete_source_partitions_not_exchange_time_boundaries",
        boundary_spill_retained=retain_boundary_spill,
        time_policy=(
            "Retain all mapped fills including exchange-time spill; start/end identify source partitions, not event-time coverage"
            if retain_boundary_spill
            else "Source hourly partitions; filter fill timestamps to declared half-open interval; boundary spill counts retained"
        ),
    )
    output = root / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2))
    return output
