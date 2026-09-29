"""Create a local frozen job, or advance one batch. Network is opt-in; no orders."""

import argparse
import csv
import json
from pathlib import Path

from arblab.hyperliquid_copy.archive_job import ArchiveJob
from arblab.hyperliquid_copy.proxy_archive_download import MAX_BYTES


def reader_source(path):
    # Called only after explicit CLI consent. No implicit credential discovery.
    import boto3
    from botocore.config import Config

    if path.name == "rootkey.csv" or path.stat().st_size > 32 * 1024:
        raise ValueError("Use a bounded IAM reader credential CSV, not root keys")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if (
        len(rows) != 1
        or not rows[0].get("Access key ID")
        or not rows[0].get("Secret access key")
    ):
        raise ValueError("Expected one IAM reader credential")
    return boto3.client(
        "s3",
        aws_access_key_id=rows[0]["Access key ID"],
        aws_secret_access_key=rows[0]["Secret access key"],
        region_name="us-east-1",
        config=Config(
            connect_timeout=15, read_timeout=60, retries={"total_max_attempts": 1}
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", help="Freeze local metadata; does not download"
    )
    create.add_argument("--root", required=True, type=Path)
    create.add_argument("--inventory", required=True, type=Path)
    create.add_argument("--coins", required=True, nargs="+")
    create.add_argument("--cache-manifest", action="append", default=[], type=Path)
    create.add_argument("--max-download-bytes", required=True, type=int)
    create.add_argument("--max-batch-bytes", type=int, default=MAX_BYTES)
    step = commands.add_parser(
        "step", help="Advance one batch; offline unless explicitly approved"
    )
    step.add_argument("--root", required=True, type=Path)
    step.add_argument("--accept-approved-download", action="store_true")
    step.add_argument("--credentials-file", type=Path)
    recover = commands.add_parser(
        "recover-request", help="Offline preparation of one explicitly approved retry"
    )
    recover.add_argument("--root", required=True, type=Path)
    recover.add_argument("--batch", required=True, type=int)
    recover.add_argument("--object-index", required=True, type=int)
    recover.add_argument("--accept-additional-bytes", required=True, type=int)
    recover.add_argument("--accept-approved-download", action="store_true")
    recover.add_argument("--credentials-file", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        job = ArchiveJob.create(
            args.inventory,
            args.root,
            args.coins,
            max_download_bytes=args.max_download_bytes,
            cache_manifests=args.cache_manifest,
            max_batch_bytes=args.max_batch_bytes,
        )
        print(
            json.dumps(
                dict(
                    phase="created",
                    root=str(job.store.root),
                    batch_count=len(job.batches),
                    network_authorization="not_granted_by_creation",
                )
            )
        )
        return
    if args.command == "recover-request":
        if not args.accept_approved_download or args.credentials_file is None:
            parser.error("Approved retry requires network approval and credentials")
        from arblab.hyperliquid_copy.archive_retry_recovery import (
            recover_job_archive_request,
        )

        result = recover_job_archive_request(
            args.root,
            batch=args.batch,
            object_index=args.object_index,
            approved_bytes=args.accept_additional_bytes,
            source=reader_source(args.credentials_file),
        )
        print(json.dumps(result), flush=True)
        return
    if args.credentials_file is not None and not args.accept_approved_download:
        parser.error("Network credential use requires --accept-approved-download")
    if args.accept_approved_download and args.credentials_file is None:
        parser.error("Approved network acquisition requires --credentials-file")
    job = ArchiveJob(args.root)
    if args.accept_approved_download and job.budget is None:
        parser.error("Network acquisition is forbidden for a zero-budget job")
    source = (
        reader_source(args.credentials_file) if args.accept_approved_download else None
    )
    result = job.run_next(
        source=source, progress=lambda item: print(json.dumps(item), flush=True)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
