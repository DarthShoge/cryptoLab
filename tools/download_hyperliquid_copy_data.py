"""Read-only cost previews and explicitly authorized data acquisition."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys

from arblab.hyperliquid_copy.archive import archive_keys
from arblab.hyperliquid_copy.contracts import symbol
from arblab.hyperliquid_copy.download import pull_fill_day
from arblab.hyperliquid_copy.storage import days, load_dataset, midnight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["estimate","pull-fills","pull-market","validate"])
    parser.add_argument("--dataset",choices=["fills","l2book"],default="fills")
    parser.add_argument("--coins",nargs="+",default=["BTC","ETH","SOL"])
    parser.add_argument("--start",required=True,help="First UTC day, inclusive")
    parser.add_argument("--end",required=True,help="Last UTC day, inclusive")
    parser.add_argument("--cache-root",type=Path,default=Path(".hyperliquid_cache"))
    parser.add_argument("--accept-estimated-cost",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--max-rows",type=int,default=1000000)
    args = parser.parse_args(argv)
    try:
        start,end = midnight(args.start),midnight(args.end)+timedelta(days=1)
        coins = [symbol(c) for c in args.coins]
        if start >= end or not coins or len(set(coins)) != len(coins):
            raise ValueError("invalid date range/coins")
        commands = [[sys.executable,"-m","hyperliquid_data.cli","l2book","pull","--coins",*coins,
                     "--start",args.start,"--end",args.end,"--root",str(args.cache_root)],
                    [sys.executable,"-m","hyperliquid_data.cli","funding","pull","--coins",*coins,
                     "--start-ms",str(int(start.timestamp()*1000)),"--root",str(args.cache_root),
                     "--raw-dir",str(args.cache_root/"raw"/"funding")]]
        if args.dry_run:
            print(json.dumps(dict(command=args.command,days=list(days(start,end)),coins=coins,
                                  market_commands=commands,fill_objects=24*(end-start).days,
                                  warning="Cost estimates perform billed S3 LIST requests; this dry run does not."),indent=2))
            return 0
        if args.command == "estimate":
            from hyperliquid_data import estimate
            print(json.dumps(estimate(args.dataset,args.start,args.end,coins=coins).to_dict(),indent=2))
        elif args.command.startswith("pull"):
            if not args.accept_estimated_cost:
                raise ValueError("run estimate and explicitly --accept-estimated-cost before download")
            if args.command == "pull-fills":
                import boto3
                s3 = boto3.client("s3")
                for day in days(start,end):
                    print(pull_fill_day(s3,args.cache_root,day,coins=coins,accepted_cost=True))
            else:
                for command in commands:
                    subprocess.run(command,check=True)
        else:
            from arblab.hyperliquid_copy.data_quality import validate_fills, validate_market
            fills,market,manifest = load_dataset(args.cache_root,start,end,coins,max_rows=args.max_rows)
            validate_fills(fills,start,end).assert_accepted()
            validate_market(market,coins,start,end,research=True).assert_accepted()
            print(json.dumps({"accepted":True,"dataset_hash":manifest["dataset_hash"]}))
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
