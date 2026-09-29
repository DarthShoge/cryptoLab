from dataclasses import replace
import json

import lz4.frame
import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.qualified_scheduled_activity import (
    QualifiedScheduledActivity,
)
from .test_candidate_day import resources
from .test_archive_job import inputs
from .test_prefix_qualification import qualify
from .test_qualified_scheduled_activity import scheduled_config
from .test_scheduled_feature_history import rows


@pytest.mark.parametrize("staging_policy", [None, "bounded_ranking_staging_v1"])
def test_historical_active_markets_have_distinct_complete_receipts(
    resources, tmp_path, monkeypatch, staging_policy
):
    from arblab.hyperliquid_copy import candidate_metric_producer as raw
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from arblab.hyperliquid_copy.rolling_feature_history import RollingFeatureHistory
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

    coins = ["BTC", "xyz:GOLD"]
    _, archive, _, _ = inputs(tmp_path, days=3)
    for key, body in archive.bodies.items():
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        payload = json.loads(lz4.frame.decompress(body))
        for event in payload["events"]:
            event[1]["coin"] = coins[hour % 2]
        archive.bodies[key] = lz4.frame.compress(json.dumps(payload).encode() + b"\n")
    downloaded = download_archive(archive, "2026-08-01", "2026-08-04", tmp_path / "raw")
    normalized = import_archive(
        downloaded, coins, tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    pin = qualify([compact], tmp_path)
    base = scheduled_config()
    config = replace(
        base,
        start="2026-08-02",
        market_universe=replace(
            base.market_universe, instrument_ids=coins, classes=["crypto", "commodity"]
        ),
    )

    def reader():
        return QualifiedScheduledActivity(
            resources,
            pin,
            config,
            coverage_start="2026-08-01",
            coverage_end="2026-08-04",
            semantics="gross_excludes_fee",
            feature_history_policy="rolling_feature_anchor_v1",
            ranking_staging_policy=staging_policy,
        )

    with reader() as actual:
        at = day("2026-08-03")
        actual.prepare(at)
        actual.rank(
            at,
            config.effective(coins, {c: 0.5 for c in coins}),
            None,
            "gross_excludes_fee",
            smoke=True,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("historical market change reconstructed features")

    monkeypatch.setattr(RollingFeatureHistory, "__init__", forbidden)
    publications = set()
    with reader() as actual:
        at = day("2026-08-02")
        actual.prepare(at)
        for active, scope in [
            (["BTC"], "BTC"),
            (coins, None),
            (["xyz:GOLD"], "xyz:GOLD"),
        ]:
            effective = config.effective(active, {c: 1 / len(active) for c in active})
            reference_root = tmp_path / f"reference-{len(publications)}"
            reference_root.mkdir()
            with CacheLease(reference_root) as lease:
                reference_cache = CacheResources.create(lease, "market-reference")
                reference = raw.build_and_score_candidates(
                    reference_cache,
                    pin,
                    "2026-08-01",
                    at,
                    effective,
                    scope,
                    "gross_excludes_fee",
                )
                expected = rows(reference)
            result = actual.rank(at, effective, scope, "gross_excludes_fee", smoke=True)
            assert rows(result) == expected
            assert result.selected == reference.selected
            assert result.candidate_count == reference.candidate_count
            assert result.eligible_count == reference.eligible_count
            assert result.bound_scope == scope and result.bound_decision == at
            publications.add(result.publication.key)
        assert actual._features is None
    assert len(publications) == 3
