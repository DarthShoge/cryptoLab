"""Read-only archive planning and local content verification; no AWS requests."""

import argparse
import json
from pathlib import Path

from arblab.hyperliquid_copy.archive_cache import VerifiedArchiveCache
from arblab.hyperliquid_copy.archive_plan import plan_archive
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_archive_download import MAX_BYTES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--cache-manifest", action="append", default=[], type=Path)
    parser.add_argument("--max-batch-bytes", type=int, default=MAX_BYTES)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Omit individual planned and cached objects",
    )
    args = parser.parse_args()
    plan = plan_archive(args.inventory, max_batch_bytes=args.max_batch_bytes)
    objects = [obj for batch in plan["batches"] for obj in batch["objects"]]
    cache = VerifiedArchiveCache(args.cache_manifest, objects)
    if file_hash(Path(plan["inventory"])) != plan["inventory_sha256"]:
        raise ValueError("Inventory changed during cache verification")
    plan.update(
        cache_manifests=cache.evidence,
        reused_objects=len(cache.entries),
        reused_bytes=cache.total_bytes,
        remaining_download_bytes=plan["total_bytes"] - cache.total_bytes,
        batch_count=len(plan["batches"]),
        max_planned_batch_bytes=max(b["bytes"] for b in plan["batches"]),
    )
    if args.summary:
        del plan["batches"]
    else:
        plan["cache_objects"] = [
            dict(key=key, **entry) for key, entry in cache.entries.items()
        ]
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
