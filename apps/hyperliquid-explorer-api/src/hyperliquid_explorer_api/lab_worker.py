"""Owned subprocess entrypoint. Writes staging only; parent publishes success."""

import argparse
import json
import traceback
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config import LabConfig, day
from arblab.hyperliquid_copy.lab_config_codec import parse_lab_config
from arblab.hyperliquid_copy.lab_config_v2 import LabConfigV2
from arblab.hyperliquid_copy.lab_config_proxy import (
    LabConfigProxy,
    LabConfigProxyScheduled,
)
from arblab.hyperliquid_copy.ranking_artifact import RankingFile
from arblab.hyperliquid_copy.lab_pipeline_proxy import (
    run_configured_proxy,
    ProxySelectionState,
)
from arblab.hyperliquid_copy.lab_pipeline_v2 import run_configured_v2
from arblab.hyperliquid_copy.lab_schedule import preview_selection, SelectionState
from arblab.hyperliquid_copy.disk_preview import preview_selection_disk
from arblab.hyperliquid_copy.lab_market_evidence import (
    MARKET_RANKINGS,
    MARKET_COHORTS,
    TRADER_COHORTS,
    EMPTY_RANKINGS,
    EMPTY_CONTRIBUTIONS,
)
from arblab.hyperliquid_copy.lab_pipeline import run_configured
from arblab.hyperliquid_copy.lab_ranking import rank_universe
from arblab.hyperliquid_copy.report import write_report
from .lab_datasets import DatasetCatalog
from .lab_store import ExperimentStore
from .lab_scratch import scratch_path, clean_scratch
from .lab_verification_session import verification_session


def write_rows(path, rows, schema=None):
    if isinstance(rows, RankingFile):
        rows.copy_to(path)
        return
    cleaned = [{k: None if v == {} else v for k, v in row.items()} for row in rows]
    table = (
        pa.Table.from_pylist(cleaned, schema=schema)
        if cleaned
        else pa.Table.from_pylist([], schema=schema or EMPTY_RANKINGS)
    )
    pq.write_table(table, path, compression="zstd")


def execute(root, identifier):
    item = ExperimentStore(root).get(identifier)
    policy = item["provenance"].get("verification_policy", "full")
    with verification_session(policy) as session:
        print(
            json.dumps(dict(event="verification_session_started", policy=policy)),
            flush=True,
        )
        try:
            _execute(root, identifier)
        finally:
            if session is not None:
                print(
                    json.dumps(
                        dict(
                            event="verification_session",
                            policy=policy,
                            hits=session.hits,
                            misses=session.misses,
                            bytes_read=session.bytes_read,
                        )
                    ),
                    flush=True,
                )


def _execute(root, identifier):
    store = ExperimentStore(root)
    item = store.get(identifier)
    config = parse_lab_config(item["config"])
    scratch = scratch_path(root, identifier)
    scratch.mkdir(parents=True, exist_ok=False)
    loaded = None
    try:
        loaded = DatasetCatalog(root).load(
            item["dataset_id"], config, item["provenance"], temp_root=scratch
        )
        _execute_loaded(root, identifier, item, config, loaded)
    finally:
        if isinstance(config, LabConfigProxy):
            if loaded is not None:
                loaded.close()
        clean_scratch(root, identifier)


def _execute_loaded(root, identifier, item, config, loaded):
    metadata = loaded.manifest
    proxy = isinstance(config, LabConfigProxy)
    if not proxy:
        fills, market = loaded.fills, loaded.market
    staging = Path(root) / "staging" / identifier
    staging.mkdir(parents=True, exist_ok=False)
    v2 = isinstance(config, (LabConfigV2, LabConfigProxy))
    if item["kind"] == "cohort_preview":
        if v2:
            scheduled = isinstance(config, LabConfigProxyScheduled)
            preview = preview_selection_disk if scheduled else preview_selection
            options = (
                {
                    "rankings_path": scratch_path(root, identifier)
                    / "preview-rankings.parquet"
                }
                if scheduled
                else {}
            )
            rows, state, hypothetical = preview(
                loaded,
                config,
                day(item["preview_date"]),
                item["preview_scope"],
                state_type=ProxySelectionState if proxy else SelectionState,
                **options,
            )
            write_rows(
                staging / "market_rankings.parquet",
                state.market_rankings,
                MARKET_RANKINGS,
            )
            write_rows(
                staging / "market_cohorts.parquet", state.market_cohorts, MARKET_COHORTS
            )
            (staging / "preview.json").write_text(
                json.dumps({"hypothetical": hypothetical})
            )
        else:
            rows = rank_universe(
                fills,
                day(item["preview_date"]),
                config,
                item["preview_scope"],
                metadata["fee_semantics"],
                smoke=True,
            )
        write_rows(staging / "rankings.parquet", rows)
    else:
        if v2:
            output = (
                run_configured_proxy(
                    loaded,
                    config,
                    rankings_path=scratch_path(root, identifier) / "rankings.parquet"
                    if isinstance(config, LabConfigProxyScheduled)
                    else None,
                )
                if proxy
                else run_configured_v2(loaded, config)
            )
            result, contributions = output.result, output.contributions
            write_rows(
                staging / "market_rankings.parquet",
                output.market_rankings,
                MARKET_RANKINGS,
            )
            write_rows(
                staging / "market_cohorts.parquet",
                output.market_cohorts,
                MARKET_COHORTS,
            )
        else:
            result, contributions = run_configured(fills, market, config, metadata)
        if not isinstance(result.scores, RankingFile):
            result.scores = [
                {k: None if v == {} else v for k, v in row.items()}
                for row in result.scores
            ]
        report_config = config.to_dict() | {
            "mode": "smoke_only",
            "research_eligible": False,
        }
        report = staging / "report"
        write_report(
            report,
            result,
            report_config,
            metadata | {"dataset_hash": item["provenance"]["dataset_hash"]},
            {
                "split": "development",
                "config_hash": item["config_hash"],
                "engine": config.schema_version.removeprefix("hyperliquid_")
                if proxy
                else "copy_lab_v2"
                if v2
                else "copy_lab_v1",
            },
        )
        write_rows(staging / "rankings.parquet", result.scores)
        write_rows(
            staging / "cohorts.parquet", result.cohorts, TRADER_COHORTS if v2 else None
        )
        write_rows(
            staging / "contributions.parquet",
            contributions,
            None if contributions else EMPTY_CONTRIBUTIONS,
        )
        # Override only this new unpublished report's legacy fixed-sweep prose.
        if not proxy:
            _write_legacy_prose(report, config, metadata)
    hashes = {
        p.relative_to(staging).as_posix(): file_hash(p)
        for p in sorted(staging.rglob("*"))
        if p.is_file()
    }
    (staging / "completion.json").write_text(
        json.dumps({"hashes": hashes}, allow_nan=False)
    )


def _write_legacy_prose(report, config, metadata):
    (report / "report.md").write_text(
        "# Hyperliquid copy-strategy experiment\n\n"
        + config.summary()
        + "\n\nDEVELOPMENT ONLY. "
        + (
            "SYNTHETIC DATA; not historical performance. "
            if metadata["synthetic"]
            else ""
        )
        + "One configured scenario; BTC perpetual benchmark and cash. Net costs and funding included.\n\n"
        "See summary.json for reconciled equity, collateral, unrealized PnL and native residual positions; "
        "inspect the saved cohort and contribution tables for selection evidence.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    try:
        execute(args.root, args.id)
    except Exception:
        # Source parser exceptions can carry local paths; logs are not an HTTP API.
        traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
