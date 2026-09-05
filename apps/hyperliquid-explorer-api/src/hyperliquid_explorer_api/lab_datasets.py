"""Operator-registered, manifest-first bounded local datasets. No remote IO."""

import json
from dataclasses import asdict
from pathlib import Path
import re
from datetime import timedelta

import pyarrow.parquet as pq
from arblab.hyperliquid_copy.contracts import FillEvent, semantic_hash
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_validation import LabValidationError, ValidationIssue
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

    def _inspect(self, identifier, config):
        data = self.manifest(identifier)
        coverage_start, coverage_end = (
            day(data["coverage_start"]),
            day(data["coverage_end"]),
        )
        start, end = day(config.start), day(config.end)
        warmup = day(config.start) - timedelta(
            days=max(
                config.lookback_days,
                config.scale_lookback_days
                if config.aggregation == "conviction_trimmed"
                else 1,
            )
        )
        issues = []
        if coverage_start > warmup:
            issues.append(
                ValidationIssue(
                    "insufficient_warmup",
                    "config.lookback_days",
                    f"The {config.lookback_days}-day trader lookback and applicable normalization warmup require data from {warmup.date()}; this dataset starts {coverage_start.date()}. Load the synthetic preset or choose a dataset with enough history.",
                    str(warmup.date()),
                    str(coverage_start.date()),
                )
            )
        if start < coverage_start:
            issues.append(
                ValidationIssue(
                    "start_before_coverage",
                    "config.start",
                    f"Backtest start {start.date()} precedes dataset coverage beginning {coverage_start.date()}.",
                    str(start.date()),
                    str(coverage_start.date()),
                )
            )
        if end > coverage_end:
            issues.append(
                ValidationIssue(
                    "end_after_coverage",
                    "config.end",
                    f"Backtest end {end.date()} exceeds dataset coverage ending {coverage_end.date()}.",
                    str(end.date()),
                    str(coverage_end.date()),
                )
            )
        if not set(config.coins + ["BTC"]) <= set(data["coins"]):
            issues.append(
                ValidationIssue(
                    "missing_market_coverage",
                    "config.coins",
                    "Dataset lacks a copied market or the independent BTC benchmark.",
                )
            )
        rows, fill_rows = 0, 0
        directory = self.directory(identifier)
        for entry in data["files"]:
            path = directory / entry["name"]
            # Advisory checks inspect bounded metadata, never complete row data.
            actual = pq.ParquetFile(path).metadata.num_rows
            if entry["name"] == "fills.parquet":
                fill_rows = actual
            rows += actual
            if actual != entry["rows"]:
                issues.append(
                    ValidationIssue(
                        "dataset_row_count_changed",
                        "dataset_id",
                        "Dataset row counts differ from the registered manifest; ask the operator to check the dataset.",
                    )
                )
        minutes = int((end - start).total_seconds() / 60)
        estimates = dict(
            input_rows=rows,
            asset_minutes=minutes * len(config.coins),
            contribution_rows=(minutes // config.update_minutes + 1)
            * len(config.coins)
            * config.max_cohort,
            ranking_rows=fill_rows * (end - start).days,
        )
        for key, limit in dict(
            input_rows=1000000,
            asset_minutes=250000,
            contribution_rows=1000000,
            ranking_rows=1000000,
        ).items():
            if estimates[key] > limit:
                issues.append(
                    ValidationIssue(
                        "resource_ceiling",
                        "config",
                        f"Estimated {key.replace('_', ' ')} ({estimates[key]}) exceeds the development limit ({limit}); no wallet sampling is performed.",
                        str(estimates[key]),
                        str(limit),
                    )
                )
        return data, dict(
            ready=not issues,
            issues=[asdict(i) for i in issues],
            config_hash=semantic_hash(config.to_dict()),
            required_start=str(warmup.date()),
            required_end=str(end.date()),
            estimates=estimates,
        )

    def inspect(self, identifier, config):
        return self._inspect(identifier, config)[1]

    def preflight(self, identifier, config):
        data, inspection = self._inspect(identifier, config)
        if not inspection["ready"]:
            raise LabValidationError(
                [ValidationIssue(**i) for i in inspection["issues"]]
            )
        directory = self.directory(identifier)
        for entry in data["files"]:
            if file_hash(directory / entry["name"]) != entry["sha256"]:
                raise LabValidationError(
                    [
                        ValidationIssue(
                            "dataset_checksum_changed",
                            "dataset_id",
                            "Dataset checksum changed since registration; ask the operator to check the dataset.",
                        )
                    ]
                )
        return dict(
            dataset_hash=semantic_hash(data),
            manifest=data,
            input_rows=inspection["estimates"]["input_rows"],
            warmup_start=day(inspection["required_start"]).isoformat(),
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
