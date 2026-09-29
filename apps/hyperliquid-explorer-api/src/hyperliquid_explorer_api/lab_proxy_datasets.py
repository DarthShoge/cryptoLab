"""Proxy-specific catalogue/preflight; never synthesize legacy book metadata."""

from dataclasses import asdict
from datetime import timedelta

import pyarrow.parquet as pq
from arblab.hyperliquid_copy.contracts import semantic_hash
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_config_proxy import (
    LabConfigProxy,
    LabConfigProxyScheduled,
)
from arblab.hyperliquid_copy.ranking_artifact import MAX_RANKING_ROWS
from arblab.hyperliquid_copy.lab_config_v2 import ExplicitUniverse
from arblab.hyperliquid_copy.lab_validation import ValidationIssue
from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
from arblab.hyperliquid_copy.proxy_selection import CLASSES
from arblab.hyperliquid_copy.proxy_schedule import decision_times
from arblab.hyperliquid_copy.capacity_schedule import ranking_upper_bound
from arblab.hyperliquid_copy.registration_capacity import load_registration_capacity


def instruments(manifest):
    return [
        dict(
            instrument_id=m.instrument_id,
            display_name=m.instrument_id,
            venue=m.instrument_id.split(":")[0]
            if ":" in m.instrument_id
            else "hyperliquid",
            asset_class=CLASSES[m.asset_class],
            supported=True,
            listed_at=None,
            delisted_at=None,
            known_at=None,
            effective_from=day(m.valid_from),
            effective_to=day(m.valid_to),
            availability_basis="observed_only_research_mapping",
            proxy_ticker=m.ticker,
            proxy_unit=m.unit,
            calendar=m.calendar,
        )
        for m in sorted(
            manifest.mappings.records, key=lambda m: (m.instrument_id, m.valid_from)
        )
    ]


def summary(identifier, directory):
    manifest = ProxyDatasetManifest(directory)
    data = manifest.metadata
    return dict(
        id=identifier,
        available=True,
        name=data["name"],
        synthetic=data["synthetic"],
        coverage_start=data["coverage_start"],
        coverage_end=data["coverage_end"],
        coins=data["coins"],
        rows=sum(r["rows"] for r in data["files"]),
        dataset_hash=manifest.identity,
        fee_semantics=data["fee_semantics"],
        coverage_note=data["coverage_note"],
        default_config=data.get("default_config"),
        supported_classes=sorted({r["asset_class"] for r in instruments(manifest)}),
        liquidity_available=True,
        catalogue_hash=semantic_hash(data["mappings"]),
        pricing_mode="hourly_proxy",
        proxy_mappings=data["mappings"],
    )


def candidate_ids(manifest, config):
    universe = config.market_universe
    if isinstance(universe, ExplicitUniverse):
        return universe.instrument_ids
    return sorted(
        {
            m.instrument_id
            for m in manifest.mappings.records
            if day(m.valid_from) < day(config.end)
            and day(m.valid_to) > day(config.start)
            and (universe.general or CLASSES[m.asset_class] in universe.classes)
        }
    )


def inspect(directory, config):
    manifest = ProxyDatasetManifest(directory)
    data = manifest.metadata
    if not isinstance(config, LabConfigProxy):
        issue = ValidationIssue(
            "pricing_mode_mismatch",
            "config.schema_version",
            "Select the hourly proxy configuration for this dataset; it contains no order books.",
        )
        return data, dict(
            ready=False,
            issues=[asdict(issue)],
            config_hash=semantic_hash(config.to_dict()),
            required_start=config.start,
            required_end=config.end,
            estimates={},
        )
    start, end = day(config.start), day(config.end)
    f, u = config.follower, config.market_universe
    days = max(
        config.trader.lookback_days,
        f.scale_lookback_days if f.aggregation == "conviction_trimmed" else 0,
        0
        if isinstance(u, ExplicitUniverse)
        else u.lookback_days + u.publication_lag_days,
    )
    required = start - timedelta(days=days)
    ids = candidate_ids(manifest, config)
    issues = []
    if manifest.validation_mode in ("sharded_v1", "qualified_v1") and not isinstance(
        config, LabConfigProxyScheduled
    ):
        issues.append(
            ValidationIssue(
                "scheduled_config_required",
                "config.schema_version",
                "This disk-backed history dataset requires weekly or daily scheduled proxy configuration.",
            )
        )
    if required < manifest.coverage_start:
        issues.append(
            ValidationIssue(
                "missing_warmup",
                "config.start",
                "Dataset lacks required trader/market warmup.",
                str(required.date()),
                data["coverage_start"],
            )
        )
    if end > manifest.coverage_end:
        issues.append(
            ValidationIssue(
                "end_after_coverage",
                "config.end",
                "Backtest end exceeds proxy dataset coverage.",
                config.end,
                data["coverage_end"],
            )
        )
    if not set(ids) <= set(data["coins"]):
        issues.append(
            ValidationIssue(
                "unknown_instrument",
                "market_universe.instrument_ids",
                "A selected market is absent from the declared dataset scope.",
            )
        )
    rows, fills = 0, 0
    for entry in data["files"]:
        path = manifest.paths[entry["name"]]
        count = pq.ParquetFile(path).metadata.num_rows
        rows += count
        if entry["name"].startswith("fills-"):
            fills += count
        if count != entry["rows"] or path.stat().st_size != entry["bytes"]:
            issues.append(
                ValidationIssue(
                    "dataset_file_changed",
                    "dataset_id",
                    "Registered dataset size or row count changed.",
                )
            )
    hours = int((end - start).total_seconds() / 3600)
    estimates = dict(
        input_rows=rows,
        asset_hours=hours * len(set(ids) | {"BTC"}),
        contribution_rows=hours * len(ids) * config.trader.max_cohort,
        market_history_rows=len(data["coins"]) * (end - start).days,
    )
    if hasattr(config, "rebalance"):
        decisions = (
            sum(1 for _ in decision_times(start, end, config.rebalance))
            if (end - start).days <= 732
            else hours
        )
        estimates.update(
            contribution_rows=decisions * len(ids) * config.trader.max_cohort,
            market_history_rows=len(data["coins"]) * decisions,
        )
    notes = [
        "Ranking rows are a conservative upper bound across potential trader/market selection events, not an exact forecast.",
        "Row capacity does not establish ranking-file byte, shared-cache or total disk capacity; runtime limits still apply.",
    ]
    if 0 < (end - start).days <= 732 and set(ids) <= set(data["coins"]):
        evidence = load_registration_capacity(directory, data)
        if not ids:
            estimates["ranking_rows"] = 0
        elif evidence is not None:
            if evidence.source.origin <= start < end <= evidence.source.finish:
                estimates["ranking_rows"] = evidence.ranking_rows(config, ids)
                notes.append(
                    "Basis: verified complete source-origin candidate history, including dormant and ineligible wallets."
                )
            else:
                issues.append(
                    ValidationIssue(
                        "capacity_evidence_required",
                        "dataset_id",
                        "Registered candidate-count history does not cover this run; select a covered interval or re-register complete capacity evidence.",
                    )
                )
                notes.append(
                    "Ranking row capacity is not estimated outside the verified count-history interval."
                )
        elif manifest.validation_mode == "qualified_v1":
            issues.append(
                ValidationIssue(
                    "capacity_evidence_required",
                    "dataset_id",
                    "This qualified dataset needs registered candidate-count evidence before running; re-register it with capacity evidence.",
                )
            )
        else:
            estimates["ranking_rows"] = ranking_upper_bound(
                config, ids, lambda *_: fills
            )
            notes.append(
                "Basis: conservative fill-count upper bound; no distinct-wallet evidence is registered and no 100,000-trader clamp is applied."
            )
    ranking_limit = (
        MAX_RANKING_ROWS if isinstance(config, LabConfigProxyScheduled) else 1_000_000
    )
    if estimates.get("ranking_rows", 0) > ranking_limit:
        issues.append(
            ValidationIssue(
                "capacity_unproven",
                "config",
                f"Ranking-row upper bound {estimates['ranking_rows']:,} exceeds the {ranking_limit:,}-row limit. This does not prove actual overflow; tighter verified capacity evidence is required. No wallet sampling.",
            )
        )
    for name, maximum in {
        "asset_hours": 250_000,
        "contribution_rows": 1_000_000,
        "market_history_rows": 1_000_000,
    }.items():
        if estimates[name] > maximum:
            issues.append(
                ValidationIssue(
                    "resource_ceiling",
                    "config",
                    f"Estimated {name} exceeds proxy replay limit; no wallet sampling.",
                )
            )
    maximum = 366 if isinstance(config, LabConfigProxyScheduled) else 93
    if hours > maximum * 24:
        issues.append(
            ValidationIssue(
                "resource_ceiling",
                "config.end",
                f"Proxy runs are limited to {maximum} days.",
            )
        )
    return data, dict(
        ready=not issues,
        issues=[asdict(i) for i in issues],
        config_hash=semantic_hash(config.to_dict()),
        required_start=str(required.date()),
        required_end=config.end,
        estimates=estimates,
        estimate_notes=notes,
    )
