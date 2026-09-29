"""Operator-registered proxy datasets; no remote IO and no in-memory fill archive.

Coverage provenance is an operator assertion checked structurally here. The
acquisition/registration tool must establish it from complete source partitions;
a hash alone does not establish economic correctness or historical coverage.
"""

from dataclasses import dataclass, field
from datetime import timedelta
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from .archive import archive_keys
from .contracts import semantic_hash, symbol
from .download import file_hash
from .lab_config import day
from .feature_history_policy import checked_feature_policy
from .annual_execution_policy import checked_execution_policy
from .ranking_staging_policy import validate_policy as checked_staging_policy
from .lab_config_proxy import LabConfigProxyScheduled
from .registered_activity import load_registered_activity
from .sharded_validation import MAX_CORPUS_BYTES
from .scheduled_activity import ScheduledActivity
from .proxy_activity import ProxyActivity
from .proxy_bars import ProxyBar, ProxyBars
from .proxy_funding import FundingEvent
from .proxy_mapping import ProxyMapping, ProxyMappings
from .proxy_availability import native_history_starts, validate_native_fills
from .native_history_evidence import verify_native_history_evidence
from .registration_provenance import verify_registration_provenance
from .qualified_registration import (
    verify_qualified_registration,
    cache_reference,
    POLICY,
)
from .qualified_registered_activity import load_qualified_activity

SCHEMA = "hyperliquid_lab_proxy_dataset_v1"
MAX_BYTES = 8 * 1024**3


@dataclass
class LoadedProxyDataset:
    activity: ProxyActivity | ScheduledActivity
    bars: tuple
    funding: tuple
    mappings: ProxyMappings
    manifest: dict
    coverage_start: object
    coverage_end: object
    native_starts: dict = field(default_factory=dict)
    funding_starts: dict = field(default_factory=dict)

    def close(self):
        self.activity.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ProxyDatasetManifest:
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        path = (self.directory / "manifest.json").resolve(strict=True)
        if not path.is_relative_to(self.directory) or path.stat().st_size > 1_000_000:
            raise ValueError("Invalid proxy manifest path/size")
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA or type(data.get("synthetic")) is not bool:
            raise ValueError("Invalid proxy dataset schema")
        self.validation_mode = data.get("validation_mode", "full_v1")
        if self.validation_mode not in ("full_v1", "sharded_v1", "qualified_v1"):
            raise ValueError("Invalid proxy validation mode")
        feature_policy = checked_feature_policy(data.get("feature_history_policy"))
        staging_policy = checked_staging_policy(data.get("ranking_staging_policy"))
        if staging_policy is not None and self.validation_mode != "qualified_v1":
            raise ValueError("Ranking staging policy requires qualified registration")
        if feature_policy is not None and self.validation_mode != "qualified_v1":
            raise ValueError("Feature history policy requires qualified registration")
        if self.validation_mode == "qualified_v1":
            if data.get("history_policy") != POLICY:
                raise ValueError("Explicit qualified source-seed policy required")
            reference = cache_reference(data.get("derived_cache"))
            checked_execution_policy(
                data.get("execution_policy"),
                feature_history_policy=feature_policy,
                ranking_staging_policy=staging_policy,
                registration_policy=data.get("history_policy"),
                cache_reference=reference,
            )
        elif data.get("execution_policy") is not None:
            raise ValueError("Annual execution policy requires qualified registration")
        start, end = day(data["coverage_start"]), day(data["coverage_end"])
        # Calendar span is separate from input volume. Permit annual evaluation
        # plus warmup; mode-specific byte and unchanged file/market-row ceilings
        # still bound loading.
        if not 0 < (end - start).days <= 732:
            raise ValueError("Invalid proxy dataset coverage window")
        coins = data["coins"]
        if (
            not isinstance(coins, list)
            or not 1 <= len(coins) <= 50
            or len(set(coins)) != len(coins)
        ):
            raise ValueError("Invalid proxy dataset markets")
        for coin in coins:
            symbol(coin)
        if "BTC" not in coins or data["fee_semantics"] not in (
            "gross_excludes_fee",
            "net_includes_fee",
            "unknown",
        ):
            raise ValueError("Proxy benchmark or fee semantics unavailable")
        for name in ("name", "coverage_note"):
            if not isinstance(data.get(name), str) or not data[name].strip():
                raise ValueError("Proxy dataset description required")
        self.mappings = ProxyMappings(ProxyMapping(**row) for row in data["mappings"])
        if not self.mappings.records or not {
            m.instrument_id for m in self.mappings.records
        } <= set(coins):
            raise ValueError("Mapping/dataset instrument mismatch")
        for name in ("price_source_manifest_hash", "funding_source_manifest_hash"):
            if not re.fullmatch(r"[a-f0-9]{64}", data.get(name, "")):
                raise ValueError("Source manifest identity required")
        if data.get("price_policy") != dict(
            adjustment="raw", corporate_actions="none_detected", rolls="not_applicable"
        ):
            raise ValueError("Proxy corporate-action/roll policy is not qualified")
        provenance = data["activity_provenance"]
        if not data["synthetic"]:
            expected = {
                key
                for d in range((end - start).days)
                for key in archive_keys((start + timedelta(days=d)).date().isoformat())
            }
            keys = provenance.get("source_keys", [])
            if (
                provenance.get("scope") != "all_wallets_for_declared_markets"
                or provenance.get("complete") is not True
                or not isinstance(keys, list)
                or len(keys) != len(expected)
                or set(keys) != expected
            ):
                raise ValueError("Incomplete all-wallet hourly archive coverage")
        entries = data["files"]
        if not isinstance(entries, list) or not 3 <= len(entries) <= 5000:
            raise ValueError("Invalid proxy dataset files")
        self.paths, self.fill_paths = {}, []
        total = 0
        for entry in entries:
            name = entry["name"]
            if name in self.paths or not re.fullmatch(
                r"(?:bars|funding|fills-[A-Za-z0-9_-]+)\.parquet", name
            ):
                raise ValueError("Invalid or duplicate proxy dataset file")
            path = (self.directory / name).resolve(strict=True)
            if not path.is_relative_to(self.directory) or not path.is_file():
                raise ValueError("Proxy dataset file escapes directory")
            if (
                type(entry.get("bytes")) is not int
                or entry["bytes"] <= 0
                or type(entry.get("rows")) is not int
                or entry["rows"] < 0
                or not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", ""))
            ):
                raise ValueError("Invalid proxy file identity")
            total += path.stat().st_size
            self.paths[name] = path
            if name.startswith("fills-"):
                self.fill_paths.append(path)
            elif pq.ParquetFile(path).metadata.num_rows > 250_000:
                raise ValueError("Proxy market rows exceed memory ceiling")
        limit = (
            MAX_CORPUS_BYTES
            if self.validation_mode in ("sharded_v1", "qualified_v1")
            else MAX_BYTES
        )
        if total > limit:
            raise ValueError("Proxy dataset input byte ceiling exceeded")
        if (
            not self.fill_paths
            or not {"bars.parquet", "funding.parquet"} <= self.paths.keys()
        ):
            raise ValueError("Proxy dataset requires fills, bars and funding files")
        self.metadata = data
        self.native_starts = native_history_starts(data, coins, start, end)
        self.funding_starts = verify_native_history_evidence(
            self.directory, data, self.native_starts
        )
        verify_registration_provenance(self.directory, data)
        from .registration_capacity import load_registration_capacity

        load_registration_capacity(self.directory, data)
        self.identity = semantic_hash(data)
        self.coverage_start, self.coverage_end = start, end

    def verify(self):
        # Re-read registration before checks: a held manifest object cannot hide
        # an operator edit made after a job was submitted.
        current = ProxyDatasetManifest(self.directory)
        if current.identity != self.identity:
            raise ValueError("Proxy dataset identity changed")
        if self.validation_mode == "qualified_v1":
            if semantic_hash(self.metadata) != current.identity:
                raise ValueError("Qualified dataset metadata changed")
            verify_qualified_registration(current)
        for entry in self.metadata["files"]:
            if self.validation_mode == "qualified_v1" and entry["name"].startswith(
                "fills-"
            ):
                continue  # Already verified against both original and retained pins.
            path = current.paths[entry["name"]]
            if (
                path.stat().st_size != entry["bytes"]
                or pq.ParquetFile(path).metadata.num_rows != entry["rows"]
                or file_hash(path) != entry["sha256"]
            ):
                raise ValueError("Proxy dataset file changed since registration")

    def load(self, *, temp_root, expected_hash, config=None):
        if expected_hash != self.identity:
            raise ValueError("Proxy dataset identity changed since submission")
        if self.validation_mode in ("sharded_v1", "qualified_v1") and not isinstance(
            config, LabConfigProxyScheduled
        ):
            raise ValueError(
                "Disk-backed history requires a scheduled v2 configuration"
            )
        self.verify()
        bars = tuple(
            ProxyBar(**r) for r in pq.read_table(self.paths["bars.parquet"]).to_pylist()
        )
        ProxyBars(
            bars
        )  # Reject malformed/conflicting bars before opening query resources.
        funding = tuple(
            FundingEvent(**r)
            for r in pq.read_table(self.paths["funding.parquet"]).to_pylist()
        )
        coins = set(self.metadata["coins"])
        if any(b.instrument_id not in coins for b in bars) or any(
            f.instrument_id not in coins for f in funding
        ):
            raise ValueError("Proxy market file contains undeclared instrument")
        if self.validation_mode == "qualified_v1":
            activity = load_qualified_activity(self, config)
        elif isinstance(config, LabConfigProxyScheduled):
            activity = load_registered_activity(self, config, temp_root=temp_root)
        else:
            activity = ProxyActivity(
                self.fill_paths, temp_root=temp_root, max_input_bytes=MAX_BYTES
            )
            try:
                activity.validate_registered_scope(
                    coins, self.coverage_start, self.coverage_end
                )
                validate_native_fills(activity, self.native_starts)
            except BaseException:
                activity.close()
                raise
        return LoadedProxyDataset(
            activity,
            bars,
            funding,
            self.mappings,
            self.metadata,
            self.coverage_start,
            self.coverage_end,
            self.native_starts,
            self.funding_starts,
        )
