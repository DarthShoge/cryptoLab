"""Register a verified interior-window dataset; no network access or trading."""

import argparse
from pathlib import Path

from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxy
from arblab.hyperliquid_copy.lab_config_v2 import (
    ExplicitUniverse,
    TraderSettings,
    FollowerSettings,
)
from arblab.hyperliquid_copy.proxy_registration import register_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-manifest", type=Path, required=True)
    parser.add_argument("--price-manifest", type=Path, required=True)
    parser.add_argument("--funding-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = LabConfigProxy(
        market_universe=ExplicitUniverse(
            classes=["crypto", "commodity", "equity", "index"],
            instrument_ids=["BTC", "xyz:GOLD", "xyz:SP500", "xyz:TSLA"],
        ),
        trader=TraderSettings(
            lookback_days=2,
            min_active_days=1,
            min_episodes=2,
            min_notional=10000,
            min_volume=10000,
            min_minutes=0,
            selection="n",
            top_n=5,
            top_fraction=None,
            min_cohort=5,
            max_cohort=5,
        ),
        follower=FollowerSettings(update_minutes=60),
        start="2026-08-04",
        end="2026-08-06",
    )
    print(
        register_dataset(
            args.activity_manifest,
            args.price_manifest,
            args.funding_manifest,
            args.output,
            start="2026-08-02",
            end="2026-08-07",
            name="Real cross-class hourly proxies · Aug 2026 · short research window",
            default_config=config,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
