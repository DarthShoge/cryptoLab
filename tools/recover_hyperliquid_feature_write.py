#!/usr/bin/env python3
"""Move explicitly named failed-write allocations to a recoverable quarantine."""

import argparse
import json
from pathlib import Path

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_policy_64 import open_expanded_64_cache
from arblab.hyperliquid_copy.failed_feature_write_recovery import quarantine
from hyperliquid_explorer_api.lab_jobs import LabJobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-root", required=True, type=Path)
    parser.add_argument("--failed-id", required=True)
    parser.add_argument("--token", required=True, action="append")
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--accept-recovery", action="store_true")
    args = parser.parse_args()
    if not args.accept_recovery:
        parser.error("Explicit --accept-recovery required")
    # Take the normal coordinator lock, but start no worker or recovery loop.
    import fcntl

    jobs = LabJobs(args.lab_root)
    with (jobs.root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        item = jobs.store.get(args.failed_id)
        if item["status"] != "failed":
            raise ValueError("Expected a failed experiment")
        manifest = jobs.root / "datasets" / item["dataset_id"] / "manifest.json"
        reference = json.loads(manifest.read_text())["derived_cache"]
        with CacheLease(reference["path"]) as lease:
            resources = open_expanded_64_cache(
                lease,
                reference["identity"],
                reference["expansion_receipt"],
                reference["expansion_64gib_receipt"],
            )
            print(
                json.dumps(quarantine(resources, args.token, args.quarantine)),
                flush=True,
            )


if __name__ == "__main__":
    main()
