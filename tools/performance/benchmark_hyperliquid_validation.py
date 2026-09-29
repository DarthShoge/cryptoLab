#!/usr/bin/env python3
"""Benchmark real cache audits on disposable copies of quarantined files only."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy import derived_cache_resources as module
from arblab.hyperliquid_copy.download import file_hash
from hyperliquid_hash_session import FileHashSession


def timed(function, repeats):
    start = time.perf_counter()
    values = [function() for _ in range(repeats)]
    return time.perf_counter() - start, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("Expected 1 to 20 repetitions")
    # The completed recovery manifest defines the allowed source inventory.
    json.loads((args.quarantine / "complete.json").read_text())
    plan = json.loads((args.quarantine / "plan.json").read_text())
    samples = [row for row in plan["allocations"] if row[4] == "retained"][:6]
    if not samples:
        raise ValueError("No completed quarantined samples")
    with tempfile.TemporaryDirectory(prefix="hyperliquid-validation-bench-") as directory:
        root = Path(directory)
        with CacheLease(root) as lease:
            resources = module.CacheResources.create(lease, "isolated-performance-probe")
            for index, row in enumerate(samples):
                source = args.quarantine / Path(row[1]).name
                if source.is_symlink() or file_hash(source) != row[6]:
                    raise ValueError("Quarantined sample identity changed")
                relative = f"artifacts/{index:032x}.parquet"
                token = resources.reserve(relative, row[5], "payload")
                shutil.copyfile(source, root / relative)
                resources.settle(token)
            # Model retained, immutable history rather than just-written files.
            time.sleep(1.05)
            baseline_seconds, baseline = timed(resources.audit, args.repeats)
            session = FileHashSession()
            with patch.object(module, "file_hash", session):
                cold_seconds, cold = timed(resources.audit, 1)
                warm_seconds, warm = timed(resources.audit, args.repeats)
            assert all(value == baseline[0] for value in baseline + cold + warm)
            sample_bytes = sum(row[5] for row in samples)
            result = dict(
                samples=len(samples), sample_bytes=sample_bytes, repeats=args.repeats,
                baseline_seconds=baseline_seconds, first_session_audit_seconds=cold_seconds,
                repeated_session_audits_seconds=warm_seconds,
                repeated_audit_speedup=baseline_seconds / warm_seconds,
                baseline_repeated_hash_bytes=sample_bytes * args.repeats,
                session_total_hash_bytes=session.bytes_read,
                session_hits=session.hits, session_misses=session.misses,
                ledger_totals_equal=True,
                scope="Isolated audit microbenchmark; not end-to-end backtest throughput",
                source="Preserved failed-write quarantine; no live cache opened or locked",
            )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
