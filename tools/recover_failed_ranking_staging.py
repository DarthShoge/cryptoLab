#!/usr/bin/env python3
"""Explicitly recover one failed, manifest-bound ranking invocation."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from arblab.hyperliquid_copy.derived_cache_expansion_64 import open_expanded_64_cache
from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_publication import _encode
from arblab.hyperliquid_copy.failed_ranking_staging_recovery import recover
from arblab.hyperliquid_copy.qualified_registration import cache_reference
from arblab.hyperliquid_copy.ranking_staging_manifest import decode_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-reference", required=True, type=Path)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-context-sha256", required=True)
    parser.add_argument("--accept-recovery", action="store_true")
    args = parser.parse_args()
    if not args.accept_recovery:
        raise SystemExit("Recovery requires --accept-recovery")
    reference = cache_reference(json.loads(args.cache_reference.read_text()))
    manifest_path = Path(reference["path"]) / args.manifest
    body = decode_manifest(manifest_path.read_bytes())
    actual = hashlib.sha256(_encode(body["context"])).hexdigest()
    if actual != args.expected_context_sha256:
        raise SystemExit(
            "Manifest context does not match the independently expected hash"
        )
    with CacheLease(reference["path"]) as lease:
        resources = open_expanded_64_cache(
            lease,
            reference["identity"],
            reference["expansion_receipt"],
            reference["expansion_64gib_receipt"],
        )
        result = recover(resources, args.manifest, body["context"])
        print(
            json.dumps({**result, "receipt": asdict(result["receipt"])}, sort_keys=True)
        )


if __name__ == "__main__":
    main()
