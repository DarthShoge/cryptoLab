"""Manifest-first local loading with an explicit, non-sampling resource ceiling."""
from datetime import datetime, timedelta
import json
from pathlib import Path

import pyarrow.parquet as pq

from .archive import archive_keys
from .contracts import FillEvent, UTC
from .data_paths import fill_partition, market_partition
from .data_quality import dataset_hash
from .download import file_hash
from .market_data import MarketData, read_parquet


def midnight(day):
    return datetime.strptime(day,"%Y-%m-%d").replace(tzinfo=UTC)


def days(start,end):
    at = start
    while at < end:
        yield at.strftime("%Y-%m-%d")
        at += timedelta(days=1)


def load_dataset(root,start,end,coins, *, market_start=None,max_rows=1000000):
    root = Path(root)
    fill_paths = [fill_partition(root,day) for day in days(start,end)]
    book_paths = [market_partition(root,coin,day) for day in days(market_start or start,end) for coin in coins]
    funding_paths = [root/"funding"/"hyperliquid"/f"{coin.lower()}_perp"/"funding.parquet" for coin in coins]
    manifests, row_count = [], 0
    for path in fill_paths+book_paths+funding_paths:
        if not path.exists():
            raise ValueError(f"missing partition: {path.relative_to(root)}")
        sha = file_hash(path)
        metadata = pq.ParquetFile(path).metadata
        row_count += metadata.num_rows
        if path in fill_paths:
            manifest_path = path.with_name("manifest.json")
            if not manifest_path.exists():
                raise ValueError("missing fill manifest")
            manifest = json.loads(manifest_path.read_text())
            if manifest["sha256"] != sha or set(manifest["coins"]) != set(coins):
                raise ValueError("fill checksum/scope mismatch")
            if manifest["source_keys"] != list(archive_keys(manifest["date"])):
                raise ValueError("missing archive hours")
        manifests.append(dict(source_key=path.relative_to(root).as_posix(),sha256=sha,size=path.stat().st_size,row_count=metadata.num_rows))
    if row_count > max_rows:
        raise ValueError(f"resource ceiling: {row_count:,} rows exceeds {max_rows:,}; no wallet sampling performed. Profile a larger memory budget before raising --max-rows.")
    fills = [FillEvent(**r) for r in read_parquet(fill_paths,"exchange_time",start,end) if r["exchange_time"] < end]
    market = MarketData.from_parquet(book_paths,funding_paths,market_start or start,end)
    return fills,market,dict(dataset_hash=dataset_hash(manifests),files=manifests,start=start,end=end)
