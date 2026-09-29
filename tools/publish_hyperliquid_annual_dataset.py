#!/usr/bin/env python3
"""Publish a pinned annual dataset without discovery, downloads or overwrite."""

import argparse
import json
from pathlib import Path

from arblab.hyperliquid_copy.annual_registration import register_annual_dataset
from arblab.hyperliquid_copy.annual_execution_policy import POLICY
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.feature_history_policy import ROLLING_POLICY
from arblab.hyperliquid_copy.lab_config_codec import parse_lab_config
from arblab.hyperliquid_copy.ranking_staging_policy import (
    POLICY as RANKING_STAGING_POLICY,
)


def _json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--price-pin", required=True)
    parser.add_argument("--funding-pin", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--cache-reference", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    config_data = _json(args.config)
    config = parse_lab_config(config_data)
    if config.to_dict() != config_data:
        raise ValueError("Configuration does not roundtrip")
    path = register_annual_dataset(
        args.job_root,
        _json(args.price_pin),
        _json(args.funding_pin),
        args.target,
        config=config,
        name=args.name,
        cache_reference=_json(args.cache_reference),
        feature_history_policy=ROLLING_POLICY,
        ranking_staging_policy=RANKING_STAGING_POLICY,
        execution_policy_name=POLICY,
    )
    print(json.dumps({"path": str(path), "sha256": file_hash(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
