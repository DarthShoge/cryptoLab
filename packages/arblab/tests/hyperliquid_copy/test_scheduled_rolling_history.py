from dataclasses import replace
from datetime import timedelta
import json

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.qualified_scheduled_activity import (
    QualifiedScheduledActivity,
)
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_qualified_scheduled_activity import reader, scheduled_config
from .test_scheduled_feature_history import rows

POLICY = "rolling_feature_anchor_v1"


@pytest.mark.parametrize("scope", ["BTC", None])
@pytest.mark.parametrize("staging_policy", [None, "bounded_ranking_staging_v1"])
def test_new_historical_hypothesis_does_not_resurrect_retired_features(
    resources, qualified, monkeypatch, tmp_path, scope, staging_policy
):
    from arblab.hyperliquid_copy import candidate_metric_producer as raw
    from arblab.hyperliquid_copy import feature_metric_producer as features
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions
    from arblab.hyperliquid_copy.rolling_feature_history import RollingFeatureHistory

    config = replace(scheduled_config(), start="2026-08-02")
    first, later = day("2026-08-02"), day("2026-08-03")

    def open_reader(configuration, cache=resources):
        return QualifiedScheduledActivity(
            cache,
            qualified,
            configuration,
            coverage_start="2026-08-01",
            coverage_end="2026-08-04",
            semantics="gross_excludes_fee",
            feature_history_policy=POLICY,
            ranking_staging_policy=staging_policy,
        )

    def feature_catalogue():
        with resources._connect() as db:
            return sorted(
                row[0]
                for row in db.execute("SELECT descriptor FROM publications")
                if json.loads(row[0])["kind"]
                in ("qualified_features_day", "qualified_feature_anchor")
            )

    with open_reader(config) as actual:
        actual.prepare(first)
        effective = config.effective(["BTC"], {"BTC": 1})
        actual.rank(first, effective, "BTC", "gross_excludes_fee", smoke=True)
        old_paths = [
            resources.root / artifact["path"]
            for encoded in feature_catalogue()
            for entry in [json.loads(encoded)]
            if entry["kind"] == "qualified_features_day"
            for artifact in entry["artifacts"]
        ]
        actual.prepare(later)
        actual.rank(later, effective, "BTC", "gross_excludes_fee", smoke=True)
    assert old_paths and all(not path.exists() for path in old_paths)
    hypothesis = replace(
        config,
        trader=replace(
            config.trader, metric_weights={"pnl_efficiency": 1}, metric_directions={}
        ),
    )
    effective = hypothesis.effective(["BTC"], {"BTC": 1})
    reference_root = tmp_path / "independent_reference"
    reference_root.mkdir()
    with CacheLease(reference_root) as lease:
        reference_cache = CacheResources.create(lease, "historical-reference")
        reference = raw.build_and_score_candidates(
            reference_cache,
            qualified,
            "2026-08-01",
            first,
            effective,
            scope,
            "gross_excludes_fee",
        )
        expected, selected = rows(reference), reference.selected
        users = [row["user"] for row in expected]
        expected_positions = native_positions(
            reference_cache, qualified, first, "BTC", users
        )
    before = feature_catalogue()

    def forbidden(*args, **kwargs):
        pytest.fail("historical miss attempted feature reconstruction")

    with monkeypatch.context() as patch:
        patch.setattr(RollingFeatureHistory, "__init__", forbidden)
        with open_reader(hypothesis) as actual:
            actual.prepare(first)
            result = actual.rank(
                first, effective, scope, "gross_excludes_fee", smoke=True
            )
            assert rows(result) == expected and result.selected == selected
            assert result.bound_decision == first and result.bound_scope == scope
            assert actual.positions(users, "BTC", first) == expected_positions
            assert actual._features is None
    assert feature_catalogue() == before
    assert all(not path.exists() for path in old_paths)
    # A new forward query must still use the retained chronological history.
    with open_reader(hypothesis) as actual:
        forward = later + timedelta(hours=1)
        actual.prepare(forward)
        actual.rank(forward, effective, scope, "gross_excludes_fee", smoke=True)
        assert isinstance(actual._features, RollingFeatureHistory)
    before_reopen = resources.audit()
    resources.lease.__exit__(None, None, None)
    daily = replace(
        hypothesis,
        rebalance="daily",
        trader=replace(hypothesis.trader, reselection="daily"),
        market_universe=replace(hypothesis.market_universe, reselection="daily"),
    )
    with monkeypatch.context() as patch:
        patch.setattr(raw, "build_and_score_candidates", forbidden)
        patch.setattr(features, "build_and_score_features", forbidden)
        patch.setattr(RollingFeatureHistory, "__init__", forbidden)
        with CacheLease(resources.root) as lease:
            reopened = CacheResources(lease, resources.identity)
            for configuration in (hypothesis, daily):
                with open_reader(configuration, reopened) as actual:
                    actual.prepare(first)
                    effective = configuration.effective(["BTC"], {"BTC": 1})
                    result = actual.rank(
                        first, effective, scope, "gross_excludes_fee", smoke=True
                    )
                    assert rows(result) == expected and result.selected == selected
                    assert actual.positions(users, "BTC", first) == expected_positions
            assert reopened.audit() == before_reopen


@pytest.mark.parametrize("fault", ["policy", "config", "effective"])
def test_changed_rank_context_prevents_physical_unlink(
    resources, qualified, monkeypatch, fault
):
    from arblab.hyperliquid_copy import cache_retirement as transaction

    actual = reader(resources, qualified, feature_history_policy=POLICY)
    at = day("2026-08-03")
    actual.prepare(at)
    effective = scheduled_config().effective(["BTC"], {"BTC": 1})
    original = transaction._unlink_owned
    attempted = []

    def changed(resources, directory_fd, row, identity, guard):
        attempted.append(resources._path(row[1]))
        if fault == "policy":
            actual.feature_history_policy = None
        elif fault == "config":
            actual.config.trader.metric_weights["gross_volume"] = 0.5
        else:
            effective.min_volume += 1
        return original(resources, directory_fd, row, identity, guard)

    monkeypatch.setattr(transaction, "_unlink_owned", changed)
    with pytest.raises(ValueError):
        actual.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
    assert attempted and all(path.exists() for path in attempted)
    actual.close()


@pytest.mark.parametrize("policy", ["unknown", True, 1, {}, []])
def test_unknown_feature_policy_rejects_without_allocations(
    resources, qualified, policy
):
    before = resources.audit()
    with pytest.raises(ValueError, match="feature.*policy"):
        reader(resources, qualified, feature_history_policy=policy)
    assert resources.audit() == before


@pytest.mark.parametrize("policy", [None, POLICY])
def test_scheduled_feature_policy_is_explicit_and_frozen(resources, qualified, policy):
    from arblab.hyperliquid_copy.rolling_feature_history import RollingFeatureHistory
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    at = day("2026-08-03")
    actual = reader(resources, qualified, feature_history_policy=policy)
    actual.prepare(at)
    effective = scheduled_config().effective(["BTC"], {"BTC": 1})
    actual.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
    assert isinstance(
        actual._features, FeatureHistory if policy is None else RollingFeatureHistory
    )
    actual.feature_history_policy = POLICY if policy is None else None
    with pytest.raises(ValueError, match="changed"):
        actual.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
    actual.close()


@pytest.mark.parametrize("staging_policy", [None, "bounded_ranking_staging_v1"])
def test_rolling_scheduled_receipts_survive_chronological_retirement(
    resources, qualified, monkeypatch, staging_policy
):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.rolling_feature_history import RollingFeatureHistory
    from arblab.hyperliquid_copy import feature_metric_producer as producer

    weekly = replace(scheduled_config(), start="2026-08-02")
    decisions = [day("2026-08-02"), day("2026-08-03")]
    captured = {}
    with QualifiedScheduledActivity(
        resources,
        qualified,
        weekly,
        coverage_start="2026-08-01",
        coverage_end="2026-08-04",
        semantics="gross_excludes_fee",
        feature_history_policy=POLICY,
        ranking_staging_policy=staging_policy,
    ) as actual:
        for at in decisions:
            actual.prepare(at)
            effective = weekly.effective(["BTC"], {"BTC": 1})
            for scope in ("BTC", None):
                result = actual.rank(
                    at, effective, scope, "gross_excludes_fee", smoke=True
                )
                users = [r["user"] for r in rows(result)]
                captured[at, scope] = (
                    rows(result),
                    result.selected,
                    actual.positions(users, "BTC", at),
                )
            if at == decisions[0]:
                with resources._connect() as db:
                    entries = [
                        json.loads(row[0])
                        for row in db.execute("SELECT descriptor FROM publications")
                    ]
                old_paths = [
                    resources.root / p["path"]
                    for entry in entries
                    if entry["kind"] == "qualified_features_day"
                    for p in entry["artifacts"]
                ]
    assert old_paths and all(not p.exists() for p in old_paths)
    before = resources.audit()
    resources.lease.__exit__(None, None, None)

    def forbidden(*args, **kwargs):
        pytest.fail("saved ranking attempted expired feature reconstruction or sorting")

    monkeypatch.setattr(RollingFeatureHistory, "__init__", forbidden)
    monkeypatch.setattr(producer, "build_and_score_features", forbidden)
    daily = replace(
        weekly,
        rebalance="daily",
        trader=replace(weekly.trader, reselection="daily"),
        market_universe=replace(weekly.market_universe, reselection="daily"),
    )
    with CacheLease(resources.root) as lease:
        reopened = CacheResources(lease, resources.identity)
        for config in (weekly, daily):
            with QualifiedScheduledActivity(
                reopened,
                qualified,
                config,
                coverage_start="2026-08-01",
                coverage_end="2026-08-04",
                semantics="gross_excludes_fee",
                feature_history_policy=POLICY,
                ranking_staging_policy=staging_policy,
            ) as actual:
                for at in decisions:
                    actual.prepare(at)
                    for scope in ("BTC", None):
                        effective = config.effective(["BTC"], {"BTC": 1})
                        result = actual.rank(
                            at, effective, scope, "gross_excludes_fee", smoke=True
                        )
                        expected_rows, selected, positions = captured[at, scope]
                        assert (
                            rows(result) == expected_rows
                            and result.selected == selected
                        )
                        assert (
                            result.bound_decision == at and result.bound_scope == scope
                        )
                        assert actual.positions(list(positions), "BTC", at) == positions
                        assert actual._features is None
            assert reopened.audit() == before
