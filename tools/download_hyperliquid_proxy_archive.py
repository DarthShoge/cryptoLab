"""Download the explicitly approved seven-day native fill batch; no orders."""

import argparse
import csv
import json
from pathlib import Path

import boto3
from botocore.config import Config
from arblab.hyperliquid_copy.proxy_archive_download import (
    download_archive,
    resume_archive,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-approved-download", action="store_true")
    parser.add_argument(
        "--resume-manifest",
        type=Path,
        help="Resume unrequested objects within an existing approved batch budget; no retries",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(".hyperliquid_cache/proxy_archives")
    )
    args = parser.parse_args()
    if not args.accept_approved_download:
        parser.error(
            "Explicit approval required: original batch budget for resume, or August 1–7 batch (6 GiB maximum) for new acquisition"
        )
    credential = Path(
        "/home/lshoge/code/cryptoLab/private/hyperliquid-reader_accessKeys.csv"
    )
    with credential.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("Expected one IAM reader credential")
    row = rows[0]
    s3 = boto3.client(
        "s3",
        aws_access_key_id=row["Access key ID"],
        aws_secret_access_key=row["Secret access key"],
        region_name="us-east-1",
        config=Config(
            connect_timeout=15, read_timeout=60, retries={"total_max_attempts": 1}
        ),
    )
    progress = lambda item: print(json.dumps(item), flush=True)
    if args.resume_manifest is not None:
        path = resume_archive(s3, args.resume_manifest, progress=progress)
    else:
        path = download_archive(
            s3, "2026-08-01", "2026-08-08", args.output_root, progress=progress
        )
    print(path, flush=True)


if __name__ == "__main__":
    main()
