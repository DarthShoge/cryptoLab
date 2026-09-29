#!/usr/bin/env python3
"""Explicitly resume one exact journaled annual retirement operation."""

import argparse
import json
from pathlib import Path

from arblab.hyperliquid_copy.annual_execution_recovery import recover
from arblab.hyperliquid_copy.derived_cache_expansion_64 import open_expanded_64_cache
from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.qualified_registration import cache_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-reference", required=True, type=Path)
    parser.add_argument("--operation", required=True, type=Path)
    parser.add_argument("--accept-recovery", action="store_true")
    args = parser.parse_args()
    if not args.accept_recovery:
        raise SystemExit("Recovery requires --accept-recovery")
    reference = cache_reference(json.loads(args.cache_reference.read_text()))
    operation = json.loads(args.operation.read_text())
    with CacheLease(reference["path"]) as lease:
        resources = open_expanded_64_cache(
            lease,
            reference["identity"],
            reference["expansion_receipt"],
            reference["expansion_64gib_receipt"],
        )
        print(json.dumps(recover(resources, operation), sort_keys=True))


if __name__ == "__main__":
    main()
