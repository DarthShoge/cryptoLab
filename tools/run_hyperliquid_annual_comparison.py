#!/usr/bin/env python3
"""Submit and monitor the frozen annual comparison through the normal coordinator."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config_codec import parse_lab_config
from hyperliquid_explorer_api.lab_jobs import LabJobs
from hyperliquid_explorer_api.lab_models import Submission
from hyperliquid_explorer_api.lab_verification_session import POLICIES


def _config(path):
    raw = json.loads(path.read_text())
    config = parse_lab_config(raw)
    if config.to_dict() != raw:
        raise ValueError("Annual runner configuration does not roundtrip")
    return config


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _checkpoint(path, args, hashes):
    if path.exists():
        value = json.loads(path.read_text())
        if (
            value.get("schema") != "hyperliquid_annual_comparison_v1"
            or value.get("dataset_id") != args.dataset_id
            or value.get("config_file_sha256") != hashes
            or value.get("verification_policy", "full") != args.verification_policy
        ):
            raise ValueError("Annual comparison checkpoint context changed")
        return value
    value = dict(
        schema="hyperliquid_annual_comparison_v1",
        dataset_id=args.dataset_id,
        config_file_sha256=hashes,
        verification_policy=args.verification_policy,
        weekly_id=None,
        daily_id=None,
        status="new",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    _write(path, value)
    return value


def _save(path, value, **changes):
    value.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
    _write(path, value)


def _await(jobs, identifier, label, checkpoint_path, checkpoint):
    last = None
    heartbeat = time.monotonic()
    while True:
        item = jobs.store.get(identifier)
        if item["status"] != last or time.monotonic() - heartbeat >= 60:
            print(
                json.dumps(
                    dict(
                        event="status",
                        cadence=label,
                        id=identifier,
                        status=item["status"],
                        updated_at=item["updated_at"],
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
            last, heartbeat = item["status"], time.monotonic()
        _save(checkpoint_path, checkpoint, status=f"{label}_{item['status']}")
        if item["status"] in ("completed", "failed", "cancelled"):
            return item
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-root", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--weekly-config", required=True, type=Path)
    parser.add_argument("--daily-config", required=True, type=Path)
    parser.add_argument("--acceptance-dir", required=True, type=Path)
    parser.add_argument("--verification-policy", choices=POLICIES, default="full")
    args = parser.parse_args()
    args.acceptance_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.acceptance_dir / "annual_comparison.json"
    hashes = {
        "weekly": file_hash(args.weekly_config),
        "daily": file_hash(args.daily_config),
    }
    weekly, daily = _config(args.weekly_config), _config(args.daily_config)
    checkpoint = _checkpoint(checkpoint_path, args, hashes)
    jobs = LabJobs(args.lab_root, verification_policy=args.verification_policy)
    jobs.start()
    try:
        if checkpoint["weekly_id"] is None:
            item = jobs.submit(
                Submission(
                    name="Real annual | BTC+TSLA+GOLD+SP500 | top5% cap25 | 90d | weekly",
                    dataset_id=args.dataset_id,
                    config=weekly.to_dict(),
                )
            )
            _save(
                checkpoint_path,
                checkpoint,
                weekly_id=item["id"],
                status="weekly_submitted",
            )
        weekly_item = jobs.store.get(checkpoint["weekly_id"])
        if weekly_item["status"] == "queued" and weekly_item["needs_resume"]:
            jobs.store.resume(weekly_item["id"])
        weekly_item = _await(
            jobs,
            checkpoint["weekly_id"],
            "weekly",
            checkpoint_path,
            checkpoint,
        )
        if weekly_item["status"] != "completed":
            raise ValueError(f"Weekly annual run ended {weekly_item['status']}")

        if checkpoint["daily_id"] is None:
            item = jobs.submit(
                Submission(
                    name="Real annual | BTC+TSLA+GOLD+SP500 | top5% cap25 | 90d | daily",
                    dataset_id=args.dataset_id,
                    config=daily.to_dict(),
                    parent_id=checkpoint["weekly_id"],
                )
            )
            _save(
                checkpoint_path,
                checkpoint,
                daily_id=item["id"],
                status="daily_submitted",
            )
        daily_item = jobs.store.get(checkpoint["daily_id"])
        if daily_item["status"] == "queued" and daily_item["needs_resume"]:
            jobs.store.resume(daily_item["id"])
        daily_item = _await(
            jobs,
            checkpoint["daily_id"],
            "daily",
            checkpoint_path,
            checkpoint,
        )
        if daily_item["status"] != "completed":
            raise ValueError(f"Daily annual run ended {daily_item['status']}")
        _save(checkpoint_path, checkpoint, status="completed")
        print(json.dumps(checkpoint, sort_keys=True), flush=True)
    finally:
        jobs.close()


if __name__ == "__main__":
    main()
