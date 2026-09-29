#!/usr/bin/env python3
"""Offline, fail-closed admission audit for a qualified annual copy-trader run."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from arblab.hyperliquid_copy.annual_execution_capacity import (
    CacheState,
    ExecutionRun,
    bounded_metadata_growth,
    calculate_envelope,
)
from arblab.hyperliquid_copy.candidate_capacity import (
    CandidateCapacity,
    MAX_BYTES as COUNT_BYTES,
    SPILL_BYTES as COUNT_SPILL_BYTES,
    build_candidate_capacity,
)
from arblab.hyperliquid_copy.capacity_schedule import (
    ranking_upper_bound,
    selection_ticks,
)
from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import METADATA_BYTES
from arblab.hyperliquid_copy.derived_publication import _encode
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config_codec import parse_lab_config
from arblab.hyperliquid_copy.qualified_registered_activity import _open_resources
from arblab.hyperliquid_copy.qualified_registration import cache_reference
from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
from arblab.hyperliquid_copy.ranking_artifact import (
    MAX_RANKING_BYTES,
    MAX_RANKING_ROWS,
)

REPORT_COPIES = 5
FEATURE_SCRATCH_BYTES = 3 * 1024**3
DISK_SAFETY_BYTES = 64 * 1024**2
APPROVED_WEEKLY_SHA256 = (
    "ab92c74c2e8ef8a2ba229ceb0a8ffc8f3d5cb1fe79ea1ac58a3f7711da45a441"
)
APPROVED_DAILY_SHA256 = (
    "9a610d8f166a321ec881c96d284ec892025d184416eca16557674e60f042e3fb"
)
APPROVED_SOURCE_SHA256 = (
    "8672ce51736b76157592ff2b9ed2253a74af043a681b0006ba62987d88389368"
)
APPROVED_SOURCE_PATH = "/home/lshoge/code/cryptoLab/.worktrees/hyperliquid-trader-ensemble/.hyperliquid_cache/annual_job_20250901_20260901_300gib/batches/0066/qualified/qualification_6kv0n3kc/manifest.json"


def _json(path):
    return json.loads(Path(path).read_text())


def _pin(path):
    value = _json(path)
    if set(value) != {"path", "sha256"}:
        raise ValueError("Expected exact qualification pin JSON")
    return value


def _config(path):
    value = _json(path)
    config = parse_lab_config(value)
    if config.to_dict() != value:
        raise ValueError("Configuration does not roundtrip")
    return config


def _approved_workload(pin, weekly_path, daily_path, weekly, daily):
    hashes = {
        "weekly": file_hash(Path(weekly_path)),
        "daily": file_hash(Path(daily_path)),
    }
    if hashes != {
        "weekly": APPROVED_WEEKLY_SHA256,
        "daily": APPROVED_DAILY_SHA256,
    }:
        raise ValueError("Annual configuration is not the frozen approved workload")
    if pin != {"path": APPROVED_SOURCE_PATH, "sha256": APPROVED_SOURCE_SHA256}:
        raise ValueError("Qualification pin is not the frozen approved source")
    weekly_value, daily_value = weekly.to_dict(), daily.to_dict()
    for value, cadence in ((weekly_value, "weekly"), (daily_value, "daily")):
        if (
            value["start"] != "2025-09-01"
            or value["end"] != "2026-09-01"
            or value["rebalance"] != cadence
            or value["trader"]["reselection"] != cadence
            or value["market_universe"]["reselection"] != cadence
            or value["trader"]["lookback_days"] != 90
            or value["trader"]["scope"] != "per_asset"
            or value["market_universe"]["mode"] != "explicit"
            or value["market_universe"]["instrument_ids"]
            != ["BTC", "xyz:GOLD", "xyz:SP500", "xyz:TSLA"]
        ):
            raise ValueError("Annual workload scope or cadence changed")
    for value in (weekly_value, daily_value):
        value["rebalance"] = "cadence"
        value["trader"]["reselection"] = "cadence"
        value["market_universe"]["reselection"] = "cadence"
    if weekly_value != daily_value:
        raise ValueError("Annual configs differ outside cadence fields")
    return hashes


def _project_catalogue_pages(resources, state, terms, source_days):
    """Insert worst-sized annual rows into a scratch backup of the real catalogue."""
    if not hasattr(resources, "_connect"):
        return None
    allocation_count = terms["projected_allocation_count"] - state.allocation_count
    descriptor_groups = (
        (source_days, 2 * 1024),
        (terms["feature_peak_days"], 8 * 1024),
        (terms["anchor_count"], 4 * 1024),
        (terms["ranking_retirement_receipt_count"] + 1, 1024),
        (terms["feature_retirement_receipt_count"], 2 * 1024),
        (3, 64 * 1024),
    )
    publication_count = sum(count for count, _ in descriptor_groups)
    expected_publications = (
        terms["projected_publication_count"] - state.publication_count
    )
    if allocation_count < 0 or publication_count != expected_publications:
        raise ValueError("Invalid annual catalogue projection counts")
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as temporary:
        target = sqlite3.connect(temporary.name)
        try:
            with resources._connect() as source:
                source.backup(target)
            target.execute("PRAGMA max_page_count=1073741823")
            with target:
                for index in range(allocation_count):
                    token = f"{index + 1:032x}"
                    target.execute(
                        "INSERT INTO allocations VALUES (?,?,?,?,?,?,?)",
                        (
                            token,
                            f"artifacts/{token}.parquet",
                            256 * 1024**2,
                            "payload",
                            "retained",
                            1,
                            "f" * 64,
                        ),
                    )
                index = 0
                for count, descriptor_bytes in descriptor_groups:
                    for _ in range(count):
                        index += 1
                        key = f"{index:064x}"
                        descriptor = "x" * descriptor_bytes
                        target.execute(
                            "INSERT INTO publications VALUES (?,?,?)",
                            (
                                key,
                                descriptor,
                                hashlib.sha256(descriptor.encode()).hexdigest(),
                            ),
                        )
            return target.execute("PRAGMA page_count").fetchone()[0]
        finally:
            target.close()


def _result(
    source,
    resources,
    reference,
    weekly,
    daily,
    count_inputs,
    counts,
    config_hashes=None,
):
    coins = list(source.coins)
    source_days = len(getattr(source, "_files", (None,)))
    rows = {
        "weekly": ranking_upper_bound(weekly, coins, counts.upper_bound),
        "daily": ranking_upper_bound(daily, coins, counts.upper_bound),
    }
    decisions = {
        "weekly": len(selection_ticks(weekly)),
        "daily": len(selection_ticks(daily)),
    }
    cache = resources.audit()
    free = shutil.disk_usage(resources.root).free
    ranking_disk = REPORT_COPIES * MAX_RANKING_BYTES
    peak_transient = max(COUNT_SPILL_BYTES, FEATURE_SCRATCH_BYTES)
    if hasattr(resources, "_connect"):
        with resources._connect() as db:
            catalogue = dict(
                allocation_count=db.execute(
                    "SELECT count(*) FROM allocations"
                ).fetchone()[0],
                publication_count=db.execute(
                    "SELECT count(*) FROM publications"
                ).fetchone()[0],
                database_pages=db.execute("PRAGMA page_count").fetchone()[0],
                pending_allocations=db.execute(
                    "SELECT count(*) FROM allocations WHERE state='pending'"
                ).fetchone()[0],
            )
    else:
        catalogue = getattr(
            resources,
            "catalogue_state",
            dict(
                allocation_count=0,
                publication_count=0,
                database_pages=1,
                pending_allocations=int(bool(cache["reserved_bytes"])),
            ),
        )
    state = CacheState(
        retained_bytes=cache["retained_bytes"],
        reserved_bytes=cache["reserved_bytes"],
        allocation_count=catalogue["allocation_count"],
        publication_count=catalogue["publication_count"],
        database_pages=catalogue["database_pages"],
        unknown_objects=False,
        pending_allocations=catalogue["pending_allocations"],
    )
    runs = (
        ExecutionRun(
            "weekly",
            tuple(selection_ticks(weekly)),
            len(coins) if weekly.trader.scope == "per_asset" else 1,
        ),
        ExecutionRun(
            "daily",
            tuple(selection_ticks(daily)),
            len(coins) if daily.trader.scope == "per_asset" else 1,
        ),
    )
    preliminary = calculate_envelope(
        cache=state,
        runs=runs,
        source_days=source_days,
        lookback_days=max(weekly.trader.lookback_days, daily.trader.lookback_days),
        free_disk_bytes=free,
        projected_metadata_bytes=None,
        cache_limit_bytes=resources.limit,
    )
    terms = preliminary.terms
    metadata_growth = bounded_metadata_growth(
        new_allocations=terms["projected_allocation_count"] - state.allocation_count,
        source_days=source_days,
        feature_peak_days=terms["feature_peak_days"],
        anchor_count=terms["anchor_count"],
        ranking_retirements=terms["ranking_retirement_receipt_count"],
        feature_retirements=terms["feature_retirement_receipt_count"],
    )
    projected_pages = _project_catalogue_pages(resources, state, terms, source_days)
    if projected_pages is not None:
        metadata_growth = max(0, projected_pages - state.database_pages) * 4096
    envelope = calculate_envelope(
        cache=state,
        runs=runs,
        source_days=source_days,
        lookback_days=max(weekly.trader.lookback_days, daily.trader.lookback_days),
        free_disk_bytes=free,
        projected_metadata_bytes=metadata_growth,
        cache_limit_bytes=resources.limit,
    )
    blockers = list(envelope.blocking_bounds)
    for cadence, value in rows.items():
        if value > MAX_RANKING_ROWS:
            blockers.append(f"{cadence}_ranking_rows_exceed_{MAX_RANKING_ROWS}")
    incremental_cache = max(
        0, envelope.terms["cache_peak_bytes"] - cache["total_bytes"]
    )
    combined_required = (
        incremental_cache + ranking_disk + peak_transient + DISK_SAFETY_BYTES
    )
    if free < combined_required:
        blockers.append("combined_filesystem_peak")
    return {
        "schema": "hyperliquid_annual_capacity_audit_v2",
        "source": source.inputs(),
        "source_pin_sha256": source.inputs()["pin"]["sha256"],
        "cache_reference_sha256": __import__("hashlib")
        .sha256(_encode(reference))
        .hexdigest(),
        "configs": {
            "weekly_sha256": __import__("hashlib")
            .sha256(_encode(weekly.to_dict()))
            .hexdigest(),
            "weekly_file_sha256": None
            if config_hashes is None
            else config_hashes["weekly"],
            "daily_file_sha256": None
            if config_hashes is None
            else config_hashes["daily"],
            "daily_sha256": __import__("hashlib")
            .sha256(_encode(daily.to_dict()))
            .hexdigest(),
        },
        "decision_counts": decisions,
        "ranking_rows": rows,
        "ranking_row_limit": MAX_RANKING_ROWS,
        "cache": {
            **cache,
            "limit_bytes": resources.limit,
            "candidate_count_output_bound": COUNT_BYTES,
            "bounded_new_retained": max(
                0, envelope.terms["cache_peak_bytes"] - cache["total_bytes"]
            ),
            "metadata_allowance": METADATA_BYTES,
        },
        "transient": {
            "candidate_spill_bound": COUNT_SPILL_BYTES,
            "feature_spill_bound": FEATURE_SCRATCH_BYTES,
            "peak_bound": peak_transient,
        },
        "output_disk": {
            "free_bytes": free,
            "ranking_file_bound": MAX_RANKING_BYTES,
            "ranking_copy_count": REPORT_COPIES,
            "ranking_copy_bound": ranking_disk,
            "safety_bytes": DISK_SAFETY_BYTES,
            "incremental_cache_bound": incremental_cache,
            "combined_required_bytes": combined_required,
        },
        "candidate_capacity_inputs": count_inputs,
        "execution_envelope": {
            "terms": envelope.terms,
            "formula": envelope.formula,
            "assumptions": list(envelope.assumptions),
            "metadata_projection": (
                "scratch_backup_worst_case_rows"
                if projected_pages is not None
                else "conservative_encoded_row_allowance"
            ),
        },
        "admitted": not blockers,
        "blocking_bounds": blockers,
        "notes": [
            "Candidate counts bound rows, not encoded feature bytes.",
            "Five ranking files cover two retained copies per completed run plus one active scratch copy.",
            "No compression estimate is treated as a hard admission bound.",
        ],
    }


def audit(args):
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("Capacity output already exists")
    pin, reference = (
        _pin(args.qualification_pin),
        cache_reference(_json(args.cache_reference)),
    )
    weekly, daily = _config(args.weekly_config), _config(args.daily_config)
    config_hashes = _approved_workload(
        pin, args.weekly_config, args.daily_config, weekly, daily
    )
    source = QualifiedSourceSession(pin)
    with CacheLease(reference["path"]) as lease:
        resources = _open_resources(lease, reference)
        inputs = build_candidate_capacity(resources, source)
        counts = CandidateCapacity(resources, source, inputs)
        result = _result(
            source,
            resources,
            reference,
            weekly,
            daily,
            inputs,
            counts,
            config_hashes,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(_encode(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-pin", required=True)
    parser.add_argument("--cache-reference", required=True)
    parser.add_argument("--weekly-config", required=True)
    parser.add_argument("--daily-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args)
    print(
        json.dumps(
            {
                "output": args.output,
                "admitted": result["admitted"],
                "blocking_bounds": result["blocking_bounds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
