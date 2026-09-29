"""Download mapped hourly research proxies, without any AWS credentials."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from arblab.hyperliquid_copy.proxy_download import download_prices
from arblab.hyperliquid_copy.proxy_mapping import ProxyMapping
from arblab.hyperliquid_copy.proxy_funding_download import download_funding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mappings",
        type=Path,
        required=True,
        help="JSON array of explicit ProxyMapping records",
    )
    parser.add_argument("--start", required=True, help="UTC date, inclusive")
    parser.add_argument("--end", required=True, help="UTC date, exclusive")
    parser.add_argument(
        "--with-funding",
        action="store_true",
        help="Also acquire native Hyperliquid hourly funding with strict coverage",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(".hyperliquid_cache/proxy_prices")
    )
    args = parser.parse_args()
    mappings = [ProxyMapping(**row) for row in json.loads(args.mappings.read_text())]
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(download_prices(mappings, start, end, args.output_root))
    if args.with_funding:
        print(
            download_funding(
                [m.instrument_id for m in mappings], start, end, args.output_root
            )
        )


if __name__ == "__main__":
    main()
