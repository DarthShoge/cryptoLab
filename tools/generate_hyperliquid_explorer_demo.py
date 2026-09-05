"""Generate fabricated integration artifacts locally, never exchange data.

Development utility: reuses the strategy's tested fixture and report pipeline.
Requires the repository checkout and development dependencies.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New hyperliquid_trader_ensemble_* report directory",
    )
    args = parser.parse_args()
    if not args.output.name.startswith("hyperliquid_trader_ensemble_"):
        parser.error("output directory must start with hyperliquid_trader_ensemble_")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "packages/arblab/tests"))
    from hyperliquid_copy.test_pipeline import fixture
    from arblab.hyperliquid_copy.pipeline import run_offline
    from arblab.hyperliquid_copy.ranking import RankingConfig
    from arblab.hyperliquid_copy.report import write_report

    fills, market, start, end, warmup = fixture()
    config = dict(
        mode="smoke_only",
        research_eligible=False,
        coins=["BTC"],
        closed_pnl_fee_semantics="unknown",
        ranking=asdict(
            RankingConfig(
                min_active_days=1, min_episodes=1, min_notional=0, min_minutes=0
            )
        ),
    )
    result = run_offline(
        fills, market, start, end, warmup, config, dataset_sha256="synthetic-fixture"
    )
    write_report(
        args.output, result, config, dict(synthetic=True, start=start, end=end), {}
    )
    print(f"SYNTHETIC integration demo: {args.output.resolve()}")


if __name__ == "__main__":
    main()
