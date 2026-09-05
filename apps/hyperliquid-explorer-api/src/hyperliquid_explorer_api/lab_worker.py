"""Owned subprocess entrypoint. Writes staging only; parent publishes success."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config import LabConfig, day
from arblab.hyperliquid_copy.lab_pipeline import run_configured
from arblab.hyperliquid_copy.lab_ranking import rank_universe
from arblab.hyperliquid_copy.report import write_report
from .lab_datasets import DatasetCatalog
from .lab_store import ExperimentStore


def write_rows(path, rows):
    cleaned = [{k: None if v == {} else v for k, v in row.items()} for row in rows]
    table = (
        pa.Table.from_pylist(cleaned)
        if cleaned
        else pa.table({"user": pa.array([], type=pa.string())})
    )
    pq.write_table(table, path, compression="zstd")


def execute(root, identifier):
    store = ExperimentStore(root)
    item = store.get(identifier)
    config = LabConfig.from_dict(item["config"])
    fills, market, metadata = DatasetCatalog(root).load(
        item["dataset_id"], config, item["provenance"]
    )
    staging = Path(root) / "staging" / identifier
    staging.mkdir(parents=True, exist_ok=False)
    if item["kind"] == "cohort_preview":
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
        result, contributions = run_configured(fills, market, config, metadata)
        result.scores = [
            {k: None if v == {} else v for k, v in row.items()} for row in result.scores
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
                "engine": "copy_lab_v1",
            },
        )
        write_rows(staging / "rankings.parquet", result.scores)
        write_rows(staging / "cohorts.parquet", result.cohorts)
        write_rows(staging / "contributions.parquet", contributions)
        # Override only this new unpublished report's legacy fixed-sweep prose.
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
    hashes = {
        p.relative_to(staging).as_posix(): file_hash(p)
        for p in sorted(staging.rglob("*"))
        if p.is_file()
    }
    (staging / "completion.json").write_text(
        json.dumps({"hashes": hashes}, allow_nan=False)
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
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
