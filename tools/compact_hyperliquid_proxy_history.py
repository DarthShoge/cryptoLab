"""Measure lossless local compaction. Does not download or delete source data."""

import argparse
import json
from pathlib import Path

from arblab.hyperliquid_copy.proxy_compact import compact_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--partitioning", choices=("source_file", "source_day"), default="source_file"
    )
    args = parser.parse_args()
    result = compact_history(
        args.manifest, args.output_root, partitioning=args.partitioning
    )
    data = json.loads(result.read_text())
    print(
        json.dumps(
            dict(
                manifest=str(result),
                input_bytes=data["input_bytes"],
                output_bytes=data["output_bytes"],
                output_input_ratio=data["output_input_ratio"],
            )
        )
    )


if __name__ == "__main__":
    main()
