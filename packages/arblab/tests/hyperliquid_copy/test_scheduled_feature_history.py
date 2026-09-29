from dataclasses import replace
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy import feature_metric_producer as producer_module
from .test_candidate_day import resources
from .test_feature_day_builder import publication_count
from .test_qualified_day import qualified
from .test_qualified_scheduled_activity import reader, scheduled_config


def rows(result):
    return [row for batch in result.iter_batches() for row in batch]


def test_scheduled_feature_route_matches_raw_ranking_without_raw_metric_replay(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_metric_producer as raw

    at = day("2026-08-03")
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    expected = raw.build_and_score_candidates(
        resources, qualified, "2026-08-01", at, config, "BTC", "gross_excludes_fee"
    )
    # Reference both per-asset and pooled scopes; forbid the raw entry point below
    # so an already-cached raw result cannot mask the wrong scheduled route.
    pooled = raw.build_and_score_candidates(
        resources, qualified, "2026-08-01", at, config, None, "gross_excludes_fee"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("scheduled ranking used raw full-window metric producer")

    from arblab.hyperliquid_copy import qualified_scheduled_activity as module

    monkeypatch.setattr(module, "build_and_score_candidates", forbidden, raising=False)
    monkeypatch.setattr(raw, "merge_metric_rows", forbidden)
    with reader(resources, qualified) as actual:
        actual.prepare(at)
        for scope, reference in (("BTC", expected), (None, pooled)):
            result = actual.rank(at, config, scope, "gross_excludes_fee", smoke=True)
            assert rows(result) == rows(reference)
            assert result.selected == reference.selected
            assert result.candidate_count == reference.candidate_count
            assert result.eligible_count == reference.eligible_count
        assert publication_count(resources) == 2


def test_weekly_daily_same_monday_reuse_does_not_sort_or_reduce(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy.qualified_scheduled_activity import (
        QualifiedScheduledActivity,
    )
    from arblab.hyperliquid_copy import feature_metric_producer as producer
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    at = day("2026-08-03")
    weekly = scheduled_config()
    with reader(resources, qualified) as actual:
        actual.prepare(at)
        first = actual.rank(
            at,
            weekly.effective(["BTC"], {"BTC": 1}),
            "BTC",
            "gross_excludes_fee",
            smoke=True,
        )
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("identical Monday inputs were recomputed")

    monkeypatch.setattr(producer, "partition_metric_rows", forbidden)
    monkeypatch.setattr(OrderedWalletPartitions, "plan", forbidden)
    daily = replace(
        weekly,
        rebalance="daily",
        trader=replace(weekly.trader, reselection="daily"),
        market_universe=replace(weekly.market_universe, reselection="daily"),
    )
    with QualifiedScheduledActivity(
        resources,
        qualified,
        daily,
        coverage_start="2026-08-01",
        coverage_end="2026-08-04",
        semantics="gross_excludes_fee",
    ) as actual:
        actual.prepare(at)
        result = actual.rank(
            at,
            daily.effective(["BTC"], {"BTC": 1}),
            "BTC",
            "gross_excludes_fee",
            smoke=True,
        )
        assert result == first
        assert publication_count(resources) == 2
    assert resources.audit() == before


def test_intraday_advancement_reuses_calendar_day_and_exact_cutoffs(
    resources, qualified
):
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_and_score_candidates,
    )

    at = day("2026-08-03")
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    with reader(resources, qualified) as actual:
        for hour, count in ((0, 2), (1, 3), (2, 3)):
            decision = at + timedelta(hours=hour)
            actual.prepare(decision)
            result = actual.rank(
                decision, config, "BTC", "gross_excludes_fee", smoke=True
            )
            expected = build_and_score_candidates(
                resources,
                qualified,
                "2026-08-01",
                decision,
                config,
                "BTC",
                "gross_excludes_fee",
            )
            assert rows(result) == rows(expected)
            assert publication_count(resources) == count


@pytest.mark.parametrize("target", ["effective", "facade", "pin", "engine", "decision"])
def test_changed_context_after_history_build_rejects_before_metric_producer(
    resources, qualified, monkeypatch, target
):
    from arblab.hyperliquid_copy import qualified_scheduled_activity as module
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    at = day("2026-08-03")
    actual = reader(resources, qualified)
    actual.prepare(at)
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    original = FeatureHistory.window
    calls = []

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "effective":
            config.min_volume += 1
        elif target == "facade":
            actual.config.trader.metric_weights["gross_volume"] = 0.5
        elif target == "pin":
            qualified["sha256"] = "0" * 64
        elif target == "decision":
            actual.last_decision += timedelta(hours=1)
        else:
            monkeypatch.setattr(module, "_engine", lambda: {"changed": True})
        return result

    def producer(*args, **kwargs):
        calls.append(True)
        raise ValueError("producer must not run for changed context")

    monkeypatch.setattr(FeatureHistory, "window", changed)
    monkeypatch.setattr(producer_module, "build_and_score_features", producer)
    with pytest.raises(ValueError):
        actual.rank(at, config, "BTC", "gross_excludes_fee", smoke=True)
    assert calls == []
    assert publication_count(resources) == 2
    with resources._connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM publications WHERE json_extract(descriptor,'$.kind')='candidate_metrics'"
            ).fetchone()[0]
            == 0
        )
    actual.close()


def test_expanding_and_contracting_active_markets_reuse_full_source_features(
    resources, tmp_path
):
    import json
    import lz4.frame
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_and_score_candidates,
    )
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from arblab.hyperliquid_copy.qualified_scheduled_activity import (
        QualifiedScheduledActivity,
    )
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    coins = ["BTC", "xyz:GOLD"]
    _, archive, _, _ = inputs(tmp_path, days=3)
    for key, body in archive.bodies.items():
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        payload = json.loads(lz4.frame.decompress(body))
        for event in payload["events"]:
            event[1]["coin"] = coins[hour % 2]
        archive.bodies[key] = lz4.frame.compress(json.dumps(payload).encode() + b"\n")
    raw = download_archive(archive, "2026-08-01", "2026-08-04", tmp_path / "raw")
    normalized = import_archive(
        raw, coins, tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    pin = qualify([compact], tmp_path)
    base = scheduled_config()
    config = replace(
        base,
        market_universe=replace(
            base.market_universe, instrument_ids=coins, classes=["crypto", "commodity"]
        ),
    )
    at = day("2026-08-03")
    with QualifiedScheduledActivity(
        resources,
        pin,
        config,
        coverage_start="2026-08-01",
        coverage_end="2026-08-04",
        semantics="gross_excludes_fee",
    ) as actual:
        actual.prepare(at)
        for active, scope in (
            (["BTC"], "BTC"),
            (coins, None),
            (["xyz:GOLD"], "xyz:GOLD"),
        ):
            effective = config.effective(
                active, {coin: 1 / len(active) for coin in active}
            )
            result = actual.rank(at, effective, scope, "gross_excludes_fee", smoke=True)
            reference = build_and_score_candidates(
                resources, pin, "2026-08-01", at, effective, scope, "gross_excludes_fee"
            )
            assert rows(result) == rows(reference)
            assert publication_count(resources) == 2
        with resources._connect() as db:
            scopes = db.execute(
                "SELECT json_extract(descriptor,'$.inputs.coins') FROM publications "
                "WHERE json_extract(descriptor,'$.kind')='qualified_features_day'"
            ).fetchall()
        assert len(scopes) == 2 and all(json.loads(row[0]) == coins for row in scopes)


@pytest.mark.parametrize("scenario", ["future", "invalid_scope", "empty"])
def test_invalid_or_empty_rank_never_derives_feature_days(
    resources, qualified, monkeypatch, scenario
):
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    at = day("2026-08-03")
    actual = reader(resources, qualified)
    actual.prepare(at)
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid/empty request derived feature history")

    monkeypatch.setattr(FeatureHistory, "window", forbidden)
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    if scenario == "empty":
        assert (
            actual.rank(
                at,
                scheduled_config().effective([], {}),
                None,
                "gross_excludes_fee",
                smoke=True,
            )
            == []
        )
    else:
        with pytest.raises(ValueError):
            actual.rank(
                at + timedelta(hours=scenario == "future"),
                config,
                "ETH" if scenario == "invalid_scope" else "BTC",
                "gross_excludes_fee",
                smoke=True,
            )
    assert resources.audit() == before and publication_count(resources) == 0
    actual.close()
