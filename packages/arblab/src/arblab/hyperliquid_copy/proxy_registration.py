"""Offline publication of a hash-verified, padded interior research dataset."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import tempfile

import duckdb
import pyarrow.parquet as pq

from .archive import archive_keys
from .download import file_hash
from .lab_config import day
from .proxy_dataset import ProxyDatasetManifest, SCHEMA
from .proxy_compact import MAX_PARTITIONS


def _manifest(path, schema):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 1_000_000:
        raise ValueError("Source manifest size limit exceeded")
    data = json.loads(path.read_text())
    if data.get("schema") != schema or data.get("complete") is not True:
        raise ValueError("Incomplete source manifest")
    return path, data


def _verified(root, name, sha256, size=None):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Source path escapes manifest directory")
    if (size is not None and path.stat().st_size != size) or file_hash(path) != sha256:
        raise ValueError("Source identity changed")
    return path


def register_dataset(
    activity_manifest,
    price_manifest,
    funding_manifest,
    target,
    *,
    start,
    end,
    name,
    default_config=None,
):
    target = Path(target).resolve()
    if target.exists():
        raise ValueError(
            "Dataset target already exists; never overwrite saved evidence"
        )
    activity_path, activity = _manifest(
        activity_manifest, "hyperliquid_proxy_activity_v1"
    )
    price_path, prices = _manifest(price_manifest, "hyperliquid_proxy_prices_v1")
    funding_path, funding = _manifest(funding_manifest, "hyperliquid_proxy_funding_v1")
    begin, finish = day(start), day(end)
    if not (
        day(activity["start"]) + timedelta(days=1)
        <= begin
        < finish
        <= day(activity["end"]) - timedelta(days=1)
    ):
        raise ValueError("Require one full source day of padding on both boundaries")
    for data in (prices, funding):
        if (
            datetime.fromisoformat(data["start"]) > begin
            or datetime.fromisoformat(data["end"]) < finish
        ):
            raise ValueError("Price/funding coverage does not span dataset")
    if activity.get("scope") != "all_wallets_for_declared_markets":
        raise ValueError("Native activity must retain all wallets")
    expected = {
        key
        for d in range((day(activity["end"]) - day(activity["start"])).days)
        for key in archive_keys(
            str((day(activity["start"]) + timedelta(days=d)).date())
        )
    }
    if set(activity["source_keys"]) != expected or len(activity["source_keys"]) != len(
        expected
    ):
        raise ValueError("Missing padded source partitions")
    source_path = Path(activity["source_manifest"]).resolve(strict=True)
    if file_hash(source_path) != activity["source_manifest_sha256"]:
        raise ValueError("Archive provenance identity changed")
    _, source = _manifest(source_path, "hyperliquid_proxy_archive_v1")
    if {o["key"] for o in source["objects"]} != expected:
        raise ValueError("Archive provenance keys differ")
    fill_paths = [
        _verified(activity_path.parent, e["name"], e["sha256"], e["bytes"])
        for e in activity["files"]
    ]
    if (
        not fill_paths
        or len(fill_paths) > MAX_PARTITIONS
        or sum(p.stat().st_size for p in fill_paths) > 8 * 1024**3
    ):
        raise ValueError("Activity input bounds exceeded")
    for manifest, data in ((price_path, prices), (funding_path, funding)):
        for obj in data["sources"]:
            _verified(manifest.parent, obj["file"], obj["sha256"], obj["bytes"])
    bars = _verified(price_path.parent, "bars.parquet", prices["bars_sha256"])
    rates = _verified(funding_path.parent, "funding.parquet", funding["funding_sha256"])
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = target.parent.parent / "dataset_staging"
    staging_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="proxy_registration_", dir=staging_root
    ) as scratch:
        stage = Path(scratch) / "dataset"
        stage.mkdir()
        files = []
        total = 0
        with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as db:
            db.execute("SET TimeZone='UTC'")
            db.execute("SET temp_directory=?", [str(Path(scratch) / "spill")])
            db.execute("SET max_temp_directory_size='2GB'")
            bounds = db.execute(
                "SELECT coin,min(exchange_time),max(exchange_time) FROM read_parquet(?,union_by_name=true) GROUP BY coin",
                [[str(p) for p in fill_paths]],
            ).fetchall()
            if {row[0] for row in bounds} != set(activity["coins"]) or any(
                first >= begin or last < finish for _, first, last in bounds
            ):
                raise ValueError(
                    "Native timestamps do not bracket retained coverage for every market"
                )
            for index, path in enumerate(fill_paths):
                output = stage / f"fills-{index:04d}.parquet"
                db.execute(
                    "COPY (SELECT * FROM read_parquet($source) WHERE exchange_time >= $begin AND exchange_time < $finish) TO $output (FORMAT PARQUET, COMPRESSION ZSTD)",
                    dict(
                        source=str(path), begin=begin, finish=finish, output=str(output)
                    ),
                )
                total += output.stat().st_size
                if total > 8 * 1024**3:
                    raise ValueError("Registered dataset disk bound exceeded")
                files.append(output)
        shutil.copyfile(bars, stage / "bars.parquet")
        shutil.copyfile(rates, stage / "funding.parquet")
        retained = {
            key
            for d in range((finish - begin).days)
            for key in archive_keys(str((begin + timedelta(days=d)).date()))
        }
        metadata = dict(
            schema=SCHEMA,
            name=name,
            synthetic=False,
            coverage_start=start,
            coverage_end=end,
            coins=activity["coins"],
            fee_semantics="gross_excludes_fee",
            mappings=prices["mappings"],
            coverage_note="All-wallet native activity for four research mappings (or the explicitly declared subset); hourly proxy-priced follower. Interior exchange-time window with full adjacent source days retained; not an all-market historical census or exact execution replay.",
            activity_provenance=dict(
                scope=activity["scope"],
                complete=True,
                source_keys=sorted(retained),
                padding_source_keys=sorted(expected - retained),
                source_manifest_hash=file_hash(activity_path),
                boundary_policy="One complete source day each side; observed native timestamps bracket coverage for each mapped market",
            ),
            price_source_manifest_hash=file_hash(price_path),
            funding_source_manifest_hash=file_hash(funding_path),
            price_policy=prices["price_policy"],
            files=[
                dict(
                    name=p.name,
                    bytes=p.stat().st_size,
                    rows=pq.ParquetFile(p).metadata.num_rows,
                    sha256=file_hash(p),
                )
                for p in [*files, stage / "bars.parquet", stage / "funding.parquet"]
            ],
        )
        if default_config is not None:
            metadata["default_config"] = default_config.to_dict()
        for original, filename in (
            (activity_path, "activity_source.json"),
            (price_path, "price_source.json"),
            (funding_path, "funding_source.json"),
        ):
            shutil.copyfile(original, stage / filename)
        (stage / "manifest.json").write_text(json.dumps(metadata, indent=2))
        registered = ProxyDatasetManifest(stage)
        with registered.load(temp_root=scratch, expected_hash=registered.identity):
            pass
        stage.rename(target)
    return target / "manifest.json"
