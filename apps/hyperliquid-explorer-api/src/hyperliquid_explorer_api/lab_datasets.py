"""Operator-registered, manifest-first bounded local datasets. No remote IO."""

import json
from pathlib import Path
import re
from datetime import timedelta

import pyarrow.parquet as pq
from arblab.hyperliquid_copy.contracts import FillEvent, semantic_hash
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_ranking import validate_ranking_bound
from arblab.hyperliquid_copy.market_data import MarketData

FILES = {"fills.parquet", "books.parquet", "funding.parquet"}


class DatasetCatalog:
    def __init__(self, root):
        self.root = (Path(root) / "datasets").resolve()

    def directory(self, identifier):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier):
            raise ValueError("Unknown dataset")
        path = (self.root / identifier).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise ValueError("Unknown dataset")
        return path

    def manifest(self, identifier):
        directory = self.directory(identifier)
        path = (directory / "manifest.json").resolve()
        if not path.is_relative_to(directory) or path.stat().st_size > 1000000:
            raise ValueError("Invalid dataset manifest")
        metadata = json.loads(path.read_text())
        if (
            metadata.get("schema") != "hyperliquid_lab_dataset_v1"
            or type(metadata.get("synthetic")) is not bool
        ):
            raise ValueError("Unsupported dataset manifest")
        if not set(metadata["coins"]) <= {"BTC", "ETH", "SOL"} or metadata[
            "fee_semantics"
        ] not in {"unknown", "gross_excludes_fee", "net_includes_fee"}:
            raise ValueError("Unsupported dataset scope or fee semantics")
        if {r["name"] for r in metadata["files"]} != FILES or len(
            metadata["files"]
        ) != len(FILES):
            raise ValueError("Dataset requires fills, books and funding files")
        for row in metadata["files"]:
            path = (directory / row["name"]).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ValueError("Dataset file unavailable")
        return metadata

    def list(self):
        output = []
        if not self.root.is_dir():
            return output
        for path in sorted(self.root.iterdir()):
            try:
                data = self.manifest(path.name)
                output.append(
                    dict(
                        id=path.name,
                        available=True,
                        name=data["name"],
                        synthetic=data["synthetic"],
                        coverage_start=data["coverage_start"],
                        coverage_end=data["coverage_end"],
                        coins=data["coins"],
                        rows=sum(r["rows"] for r in data["files"]),
                        dataset_hash=semantic_hash(data),
                        fee_semantics=data["fee_semantics"],
                        coverage_note=data["coverage_note"],
                        default_config=data.get("default_config"),
                    )
                )
            except (ValueError, KeyError, TypeError, OSError):
                output.append(
                    dict(
                        id=path.name,
                        available=False,
                        name=path.name,
                        synthetic=False,
                        coverage_note="Dataset unavailable or malformed",
                    )
                )
        return output

    def preflight(self, identifier, config):
        data = self.manifest(identifier)
        warmup = day(config.start) - timedelta(
            days=max(
                config.lookback_days,
                config.scale_lookback_days
                if config.aggregation == "conviction_trimmed"
                else 1,
            )
        )
        if day(data["coverage_start"]) > warmup or day(data["coverage_end"]) < day(
            config.end
        ):
            raise ValueError("Dataset lacks full lookback/warmup or requested dates")
        if not set(config.coins + ["BTC"]) <= set(data["coins"]):
            raise ValueError("Dataset lacks copied-market or BTC benchmark coverage")
        rows = 0
        directory = self.directory(identifier)
        for entry in data["files"]:
            path = directory / entry["name"]
            # Inspect metadata before any row loading, then verify immutable bytes.
            actual = pq.ParquetFile(path).metadata.num_rows
            if entry["name"] == "fills.parquet":
                validate_ranking_bound(actual, config)
            rows += actual
            if actual != entry["rows"] or file_hash(path) != entry["sha256"]:
                raise ValueError("Dataset checksum or row count changed")
        minutes = int((day(config.end) - day(config.start)).total_seconds() / 60)
        if (
            rows > 1000000
            or minutes * len(config.coins) > 250000
            or (minutes // config.update_minutes + 1)
            * len(config.coins)
            * config.max_cohort
            > 1000000
        ):
            raise ValueError(
                "Dataset/run exceeds bounded resource ceiling; no wallet sampling"
            )
        return dict(
            dataset_hash=semantic_hash(data),
            manifest=data,
            input_rows=rows,
            warmup_start=warmup.isoformat(),
        )

    def load(self, identifier, config, frozen):
        current = self.preflight(identifier, config)
        if current["dataset_hash"] != frozen["dataset_hash"]:
            raise ValueError("Dataset changed since submission")
        directory = self.directory(identifier)
        fills = [
            FillEvent(**r)
            for r in pq.read_table(directory / "fills.parquet").to_pylist()
        ]
        market = MarketData(
            pq.read_table(directory / "books.parquet").to_pylist(),
            pq.read_table(directory / "funding.parquet").to_pylist(),
        )
        return fills, market, current["manifest"]
