"""Normalize a completed local archive without any network requests."""

import argparse
import json
from pathlib import Path

from arblab.hyperliquid_copy.proxy_archive_import import import_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--retain-boundary-spill",
        action="store_true",
        help="Keep exchange-time spill for multi-batch history assembly; does not grant coverage",
    )
    parser.add_argument(
        "--coins", nargs="+", default=["BTC", "xyz:TSLA", "xyz:GOLD", "xyz:SP500"]
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(".hyperliquid_cache/proxy_activity")
    )
    args = parser.parse_args()
    print(
        import_archive(
            args.manifest,
            args.coins,
            args.output_root,
            retain_boundary_spill=args.retain_boundary_spill,
            progress=lambda item: print(json.dumps(item), flush=True),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
