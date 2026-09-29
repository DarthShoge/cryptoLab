"""Composed mixed-market whale acceptance using qualified archive fixtures."""

from datetime import timedelta
import json
import uuid

import lz4.frame

from arblab.hyperliquid_copy.feature_day_builder import build_feature_day
from arblab.hyperliquid_copy.feature_metric_producer import build_and_score_features
from arblab.hyperliquid_copy.candidate_metric_producer import build_and_score_candidates
from arblab.hyperliquid_copy.feature_window import FeatureWindow
from arblab.hyperliquid_copy.ordered_feature_partitions import OrderedFeaturePartitions
from arblab.hyperliquid_copy.proxy_archive_download import download_archive
from arblab.hyperliquid_copy.proxy_archive_import import import_archive
from arblab.hyperliquid_copy.proxy_compact import compact_history
from arblab.hyperliquid_copy.lab_config_v2 import (
    LabConfigV2,
    ExplicitUniverse,
    TraderSettings,
)
from .test_archive_job import inputs
from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS
from .test_prefix_qualification import qualify


def test_mixed_market_whale_query_rankings_and_strict_intraday_cutoff(
    tmp_path, resources
):
    _, archive, _, _ = inputs(tmp_path, days=1)
    count = 100_002
    coins = ["BTC", "xyz:GOLD"]
    config = LabConfigV2(
        market_universe=ExplicitUniverse(
            classes=["crypto", "commodity"], instrument_ids=coins
        ),
        trader=TraderSettings(
            lookback_days=1,
            min_active_days=1,
            min_episodes=0,
            min_notional=0,
            min_volume=0,
            min_minutes=0,
            metric_weights={"gross_volume": 1},
            selection="n",
            top_n=1,
            top_fraction=None,
            min_cohort=1,
        ),
    ).effective(coins, {c: 0.5 for c in coins})
    for key, body in list(archive.bodies.items()):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        data = json.loads(lz4.frame.decompress(body))
        user, template = data["events"][0]
        data["events"] = []
        for i in range(hour * 4167, min((hour + 1) * 4167, count)):
            opening = (i // 2) % 2 == 0
            fill = dict(
                template,
                tid=i,
                coin=coins[i % 2],
                side="B" if opening else "A",
                sz="1",
                startPosition="0" if opening else "1",
                closedPnl="0" if opening else "1",
                fee="0.01",
                dir="Open Long" if opening else "Close Long",
            )
            data["events"].append([user, fill])
        archive.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    raw = download_archive(archive, "2026-08-01", "2026-08-02", tmp_path / "raw")
    normalized = import_archive(
        raw, coins, tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    pin = qualify([compact], tmp_path)
    day = build_feature_day(resources, pin, DAY, coins, SEMANTICS)
    fills = episodes = 0
    for row in day.observations():
        if hasattr(row, "net_pnl"):
            fills += 1
        else:
            episodes += 1
    assert fills == count and episodes > 0
    decision = DAY + timedelta(days=1)
    for scope in (None, *coins):
        feature = build_and_score_features(
            resources,
            pin,
            "2026-08-01",
            [day],
            decision,
            config,
            scope,
            SEMANTICS,
            max_partition_rows=4,
        )
        reference = build_and_score_candidates(
            resources,
            pin,
            "2026-08-01",
            decision,
            config,
            scope,
            SEMANTICS,
            max_partition_rows=4,
        )
        assert feature.candidate_count == reference.candidate_count == 1
        assert [r for batch in feature.iter_batches() for r in batch] == [
            r for batch in reference.iter_batches() for r in batch
        ]
        assert feature.selected[0]["user"] == user
    # Later observations exist in this publication. Only the exact first twelve
    # hours may reach this query, even though it reads the same physical shards.
    cutoff = DAY + timedelta(hours=12)
    expected = sum(
        1
        for r in day.observations()
        if r.coin == "BTC" and r.order_key[0] < cutoff and hasattr(r, "net_pnl")
    )
    assert 0 < expected < count // 2
    relative = "scratch/" + uuid.uuid4().hex
    resources.reserve(relative, 3 * 1024**3, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    (scratch / "spill").mkdir()
    window = FeatureWindow(resources, pin, [day], DAY, cutoff, ["BTC"], SEMANTICS)
    reader = OrderedFeaturePartitions(window, scratch)
    actual = 0
    with reader.verified_batch():
        parts = [p for p in reader.plan(max_rows=4) if p.physical_rows]
        assert len(parts) == 1 and parts[0].single_wallet == user
        artifact = reader.write(parts[0], scratch / "intraday.parquet")
        for row in reader.read(artifact):
            assert row.user == user and row.coin == "BTC" and row.order_key[0] < cutoff
            actual += hasattr(row, "net_pnl")
    assert actual == expected
