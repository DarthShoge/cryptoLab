"""Bounded-memory, atomic full-fidelity S3 fill partitions.

Archive layout reference: hyperliquid-data 0.1.0, MIT, bond-labs-dev.
The implementation uses boto3/LZ4 public interfaces, never upstream internals.
"""
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import lz4.frame
import pyarrow as pa
import pyarrow.parquet as pq

from .archive import archive_keys, parse_archive_line
from .contracts import symbol
from .data_paths import fill_partition

BUCKET = "hl-mainnet-node-data"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pull_fill_day(s3, root, day, *, coins=("BTC", "ETH", "SOL"), accepted_cost=False):
    if not accepted_cost:
        raise ValueError("explicit estimated cost acceptance required")
    coins = tuple(sorted(symbol(c) for c in coins))
    target = fill_partition(Path(root), day)
    manifest_path = target.with_name("manifest.json")
    if target.exists():
        if not manifest_path.exists():
            raise ValueError("uncommitted partition; manifest missing")
        manifest = json.loads(manifest_path.read_text())
        if manifest["coins"] != list(coins) or manifest["sha256"] != file_hash(target):
            raise ValueError("partition configuration/checksum mismatch")
        return target
    keys = archive_keys(day)
    for key in keys:
        s3.head_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix="download-") as temp:
        temporary = Path(temp) / "fills.parquet"
        writer, batch, row_count = None, [], 0
        try:
            for key in keys:
                with closing(s3.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")["Body"]) as body:
                    with lz4.frame.open(body, "rb") as decoded:
                        for line_no, raw in enumerate(decoded):
                            # Skip other markets before parsing core-perp-specific fields.
                            result = parse_archive_line(raw, key, line_no, coins=coins)
                            if result.issues:
                                raise ValueError(f"invalid archive: {key}:{line_no}: {result.issues[0].reason}")
                            batch.extend(asdict(f) for f in result.events if f.coin in coins)
                            if len(batch) >= 10000:
                                table = _table(batch)
                                writer = writer or pq.ParquetWriter(temporary, table.schema)
                                writer.write_table(table)
                                row_count += len(batch)
                                batch.clear()
            if batch:
                table = _table(batch)
                writer = writer or pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table)
                row_count += len(batch)
            if writer is None:
                raise ValueError("no core fills in requested partition")
        finally:
            if writer:
                writer.close()
        manifest = dict(schema="fill_partition_v1", date=day, coins=list(coins),
                        source_keys=list(keys), row_count=row_count, sha256=file_hash(temporary))
        metadata = Path(temp) / "manifest.json"
        metadata.write_text(json.dumps(manifest, sort_keys=True))
        for path in (temporary, metadata):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.replace(metadata, manifest_path)
    return target


def _table(rows):
    table = pa.Table.from_pylist(rows)
    # Nullable fields must not infer null types in legacy-only first batches.
    for name, kind in (("block_number", pa.int64()), ("block_time", pa.timestamp("us", tz="UTC")), ("fee_token", pa.string()), ("raw_details_json", pa.string())):
        index = table.schema.get_field_index(name)
        if index < 0:  # Older callers/partitions do not contain raw details.
            continue
        table = table.set_column(index, name, table.column(name).cast(kind))
    return table
