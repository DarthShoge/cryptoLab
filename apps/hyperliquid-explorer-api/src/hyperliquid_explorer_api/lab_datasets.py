"""Operator-registered, manifest-first bounded local datasets. No remote IO."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from datetime import timedelta

import pyarrow.parquet as pq
from arblab.hyperliquid_copy.contracts import FillEvent, semantic_hash, symbol, utc
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_validation import LabValidationError, ValidationIssue
from arblab.hyperliquid_copy.market_data import MarketData
from arblab.hyperliquid_copy.lab_config_v2 import (
    LabConfigV2,
    ExplicitUniverse,
    LiquidityUniverse,
    CLASSES,
)
from arblab.hyperliquid_copy.lab_instruments import Catalogue, timestamp
from arblab.hyperliquid_copy.lab_volume import MarketVolume

FILES = {"fills.parquet", "books.parquet", "funding.parquet"}


@dataclass
class LoadedDataset:
    fills: list
    market: MarketData
    manifest: dict
    catalogue: Catalogue
    volume: MarketVolume | None = None


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
            metadata.get("schema")
            not in ("hyperliquid_lab_dataset_v1", "hyperliquid_lab_dataset_v2")
            or type(metadata.get("synthetic")) is not bool
        ):
            raise ValueError("Unsupported dataset manifest")
        if not isinstance(metadata["coins"], list) or len(
            set(metadata["coins"])
        ) != len(metadata["coins"]):
            raise ValueError("Invalid dataset instruments")
        for coin in metadata["coins"]:
            symbol(coin)
        if metadata["fee_semantics"] not in {
            "unknown",
            "gross_excludes_fee",
            "net_includes_fee",
        }:
            raise ValueError("Unsupported dataset scope or fee semantics")
        names = {r["name"] for r in metadata["files"]}
        required = FILES | (
            {"instruments.parquet"}
            if metadata["schema"] == "hyperliquid_lab_dataset_v2"
            else set()
        )
        allowed = required | (
            {"market_volume.parquet"}
            if metadata["schema"] == "hyperliquid_lab_dataset_v2"
            else set()
        )
        if not required <= names <= allowed or len(metadata["files"]) != len(names):
            raise ValueError("Dataset requires fills, books and funding files")
        if metadata["schema"] == "hyperliquid_lab_dataset_v2":
            timestamp(metadata["snapshot_at"])
            if metadata["catalogue_hash"] != next(
                r["sha256"]
                for r in metadata["files"]
                if r["name"] == "instruments.parquet"
            ):
                raise ValueError("Catalogue identity mismatch")
            if "market_volume.parquet" in names:
                provenance = metadata["volume_provenance"]
                if (
                    provenance["currency"] != "USD"
                    or provenance["counting"] != "market_once"
                    or provenance["interval"] != "utc_day"
                    or not provenance["source"]
                    or not provenance["conversion"]
                ):
                    raise ValueError("Unsupported market-volume convention")
        for row in metadata["files"]:
            path = (directory / row["name"]).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ValueError("Dataset file unavailable")
        return metadata

    def catalogue(self, identifier, data=None):
        data = data or self.manifest(identifier)
        if data["schema"] == "hyperliquid_lab_dataset_v1":
            start = day(data["coverage_start"])
            return Catalogue(
                [
                    dict(
                        instrument_id=coin,
                        display_name=coin,
                        venue="core",
                        asset_class="crypto",
                        base=coin,
                        quote="USD",
                        settlement="USDC",
                        multiplier=1.0,
                        model="linear_usd_continuous_v1",
                        known_at=start,
                        effective_from=start,
                        effective_to=None,
                        listed_at=start,
                        delisted_at=None,
                    )
                    for coin in data["coins"]
                ]
            )
        path = self.directory(identifier) / "instruments.parquet"
        if pq.ParquetFile(path).metadata.num_rows > 5000:
            raise ValueError("Catalogue exceeds bounded metadata ceiling")
        catalogue = Catalogue(pq.read_table(path).to_pylist())
        if set(catalogue.by_id) != set(data["coins"]):
            raise ValueError("Catalogue and dataset instruments disagree")
        if any(r.known_at > timestamp(data["snapshot_at"]) for r in catalogue.records):
            raise ValueError("Catalogue published after snapshot")
        return catalogue

    def instruments(
        self, identifier, *, page=1, page_size=50, search="", asset_class=None
    ):
        rows = self.catalogue(identifier).public_rows()
        rows = [
            r
            for r in rows
            if (asset_class is None or r["asset_class"] == asset_class)
            and search.lower() in (r["instrument_id"] + " " + r["display_name"]).lower()
        ]
        return dict(
            rows=rows[(page - 1) * page_size : page * page_size], total=len(rows)
        )

    def candidate_ids(self, identifier, config, data=None):
        if not isinstance(config, LabConfigV2):
            return config.coins
        universe = config.market_universe
        catalogue = self.catalogue(identifier, data)
        if isinstance(universe, ExplicitUniverse):
            if not set(universe.instrument_ids) <= set(catalogue.by_id):
                raise LabValidationError(
                    [
                        ValidationIssue(
                            "unknown_instrument",
                            "market_universe.instrument_ids",
                            "An explicit instrument is absent from this dataset catalogue.",
                        )
                    ]
                )
            return universe.instrument_ids
        return sorted(
            {
                r.instrument_id
                for r in catalogue.records
                if r.known_at < day(config.end)
                and r.effective_from < day(config.end)
                and r.asset_class in (CLASSES if universe.general else universe.classes)
            }
        )

    def list(self):
        output = []
        if not self.root.is_dir():
            return output
        for path in sorted(self.root.iterdir()):
            try:
                data = self.manifest(path.name)
                catalogue = self.catalogue(path.name, data)
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
                        supported_classes=sorted(
                            {
                                r.asset_class
                                for r in catalogue.records
                                if r.supported and r.asset_class in CLASSES
                            }
                        ),
                        liquidity_available=any(
                            r["name"] == "market_volume.parquet" for r in data["files"]
                        ),
                        catalogue_hash=data.get("catalogue_hash"),
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
        submitted = config
        v2 = isinstance(config, LabConfigV2)
        volume_days = 0
        if v2:
            ids = self.candidate_ids(identifier, config, data)
            config = config.effective(ids, {c: 1 / len(ids) for c in ids})
            if isinstance(submitted.market_universe, LiquidityUniverse):
                volume_days = (
                    submitted.market_universe.lookback_days
                    + submitted.market_universe.publication_lag_days
                )
        coverage_start, coverage_end = (
            day(data["coverage_start"]),
            day(data["coverage_end"]),
        )
        start, end = day(config.start), day(config.end)
        warmup = day(config.start) - timedelta(
            days=max(
                config.lookback_days,
                volume_days,
                config.scale_lookback_days
                if config.aggregation == "conviction_trimmed"
                else 1,
            )
        )
        issues = []
        if volume_days and not any(
            r["name"] == "market_volume.parquet" for r in data["files"]
        ):
            issues.append(
                ValidationIssue(
                    "missing_market_volume",
                    "market_universe",
                    "Automatic market selection requires historical market-wide daily volume; this dataset has none.",
                )
            )
        if v2:
            catalogue = self.catalogue(identifier, data)
            for coin in config.coins:
                records = catalogue.by_id[coin]
                if any(
                    r.delisted_at is not None
                    and day(config.start) <= r.delisted_at <= day(config.end)
                    for r in records
                ):
                    issues.append(
                        ValidationIssue(
                            "unsupported_settlement",
                            "market_universe",
                            "A requested market delists during this run; settlement modelling is not supported.",
                        )
                    )
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
            market_history_rows=(
                len(catalogue.known_ids(end))
                if isinstance(submitted.market_universe, LiquidityUniverse)
                else len(config.coins)
            )
            * (end - start).days
            if v2
            else 0,
        )
        for key, limit in dict(
            input_rows=1000000,
            asset_minutes=250000,
            contribution_rows=1000000,
            ranking_rows=1000000,
            market_history_rows=1000000,
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
            config_hash=semantic_hash(submitted.to_dict()),
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
        catalogue = self.catalogue(identifier, current["manifest"])
        volume = None
        if any(
            r["name"] == "market_volume.parquet" for r in current["manifest"]["files"]
        ):
            volume = MarketVolume(
                pq.read_table(directory / "market_volume.parquet").to_pylist(),
                snapshot_at=current["manifest"]["snapshot_at"],
                instrument_ids=catalogue.by_id,
            )
        return LoadedDataset(fills, market, current["manifest"], catalogue, volume)
