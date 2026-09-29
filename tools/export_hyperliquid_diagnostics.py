"""Export saved-run diagnostic evidence without starting a coordinator or worker."""

import argparse
import json
from pathlib import Path
import re
import sqlite3

from hyperliquid_explorer_api.diagnostic_service import build_diagnostics
from hyperliquid_explorer_api.repository import Repository, ReportError, public_json


def export(repo, item, output):
    if item["status"] != "completed" or item["kind"] != "backtest":
        raise ValueError("Only completed backtests can be exported")
    data = build_diagnostics(repo, item)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostics.json").write_text(data.model_dump_json(indent=2) + "\n")
    (output / "experiment.json").write_text(json.dumps(public_json(item), indent=2, allow_nan=False) + "\n")
    try:
        detail = repo.detail(item["run_id"])
    except ReportError:
        detail = None
    if detail is not None:
        (output / "report.json").write_text(detail.model_dump_json(indent=2) + "\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{32}", args.experiment):
        parser.error("Expected a saved experiment identifier")
    root = args.lab_root.resolve()
    db = sqlite3.connect(f"file:{root / 'experiments.sqlite3'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        row = db.execute("SELECT * FROM experiments WHERE id=?", [args.experiment]).fetchone()
        if row is None:
            parser.error("Unknown experiment")
        item = dict(row)
        item.update(json.loads(item.pop("payload")))
        item["artifact_hashes"] = json.loads(item["artifact_hashes"])
        item["needs_resume"] = bool(item["needs_resume"])
    finally:
        db.close()
    data = export(Repository(root / "reports"), item, args.output)
    print(json.dumps(dict(output=str(args.output), experiment=data.experiment_id,
                          complete_weeks=data.statistics["count"].value,
                          accounting_passed=all(c.passed is True for c in data.checks))))


if __name__ == "__main__":
    main()
