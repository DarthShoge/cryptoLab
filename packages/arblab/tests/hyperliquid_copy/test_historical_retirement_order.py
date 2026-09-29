from dataclasses import replace
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.qualified_scheduled_activity import (
    QualifiedScheduledActivity,
)
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_qualified_scheduled_activity import scheduled_config


@pytest.mark.parametrize("stage", ["begin_retirement", "finish_retirement"])
def test_historical_producer_starts_only_after_pending_retirement(
    resources, qualified, monkeypatch, stage
):
    from arblab.hyperliquid_copy import rolling_feature_history as rolling
    from arblab.hyperliquid_copy import candidate_metric_producer as raw

    origin = day("2026-08-01")
    history = rolling.RollingFeatureHistory(
        resources, qualified, ["BTC"], "gross_excludes_fee"
    )
    first = history.window(origin, origin + timedelta(days=2))
    expired = [resources.root / p.path for p in first.days[0].publication.artifacts]

    def interrupted(*args, **kwargs):
        raise RuntimeError("fixture interruption")

    with monkeypatch.context() as patch:
        patch.setattr(rolling, stage, interrupted)
        with pytest.raises(RuntimeError, match="fixture interruption"):
            history.window(origin + timedelta(days=1), origin + timedelta(days=2))
    assert all(path.exists() for path in expired)
    original = raw.build_and_score_candidates
    called = []

    def checked(*args, **kwargs):
        assert all(not path.exists() for path in expired)
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(raw, "build_and_score_candidates", checked)
    config = replace(scheduled_config(), start="2026-08-02")
    with QualifiedScheduledActivity(
        resources,
        qualified,
        config,
        coverage_start="2026-08-01",
        coverage_end="2026-08-04",
        semantics="gross_excludes_fee",
        feature_history_policy="rolling_feature_anchor_v1",
    ) as actual:
        at = day("2026-08-02")
        actual.prepare(at)
        result = actual.rank(
            at,
            config.effective(["BTC"], {"BTC": 1}),
            "BTC",
            "gross_excludes_fee",
            smoke=True,
        )
        assert result.candidate_count > 0 and actual._features is None
    assert called == [True]


@pytest.mark.parametrize(
    "fault", ["foreign_owner", "stale_preparation", "corrupt_anchor"]
)
def test_invalid_retirement_state_blocks_historical_producer(
    resources, qualified, monkeypatch, fault
):
    from arblab.hyperliquid_copy import candidate_metric_producer as raw
    from arblab.hyperliquid_copy.cache_retirement import prepare_retirement
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.feature_history import FeatureHistory
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor

    semantics = "gross_excludes_fee"
    origin = day("2026-08-01")
    days = (
        FeatureHistory(resources, qualified, ["BTC"], semantics)
        .window(origin, origin + timedelta(days=3))
        .days
    )
    older = publish_feature_anchor(resources, qualified, days[1:2], ["BTC"], semantics)
    latest = publish_feature_anchor(resources, qualified, days[2:], ["BTC"], semantics)
    if fault == "corrupt_anchor":
        with (resources.root / latest.publication.artifacts[0].path).open(
            "ab"
        ) as stream:
            stream.write(b"corruption")
    else:
        owner = older if fault == "foreign_owner" else latest
        prepare_retirement(
            resources,
            "qualified_features_day",
            days[0].inputs,
            owner=owner.publication.key,
            reason="fixture pending",
        )
        if fault == "stale_preparation":
            PublishedArtifacts(resources).publish(
                "intervening_fixture_publication",
                {},
                [p.token for p in days[-1].publication.artifacts],
            )

    # The corrupt payload deliberately cannot pass a resource audit. Observe
    # ledger state directly to prove rejection made no allocation/deletion.
    def ledger():
        with resources._connect() as db:
            return (
                db.execute("SELECT * FROM allocations ORDER BY token").fetchall(),
                db.execute("SELECT * FROM publications ORDER BY key").fetchall(),
            )

    before = ledger()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid retirement state reached raw producer")

    monkeypatch.setattr(raw, "build_and_score_candidates", forbidden)
    config = replace(scheduled_config(), start="2026-08-02")
    with QualifiedScheduledActivity(
        resources,
        qualified,
        config,
        coverage_start="2026-08-01",
        coverage_end="2026-08-04",
        semantics=semantics,
        feature_history_policy="rolling_feature_anchor_v1",
    ) as actual:
        at = day("2026-08-02")
        actual.prepare(at)
        with pytest.raises(ValueError):
            actual.rank(
                at, config.effective(["BTC"], {"BTC": 1}), "BTC", semantics, smoke=True
            )
    assert ledger() == before


def test_historical_capacity_failure_preserves_saved_rankings(resources, qualified):
    from .test_scheduled_feature_history import rows

    config = replace(scheduled_config(), start="2026-08-02")

    def reader():
        return QualifiedScheduledActivity(
            resources,
            qualified,
            config,
            coverage_start="2026-08-01",
            coverage_end="2026-08-04",
            semantics="gross_excludes_fee",
            feature_history_policy="rolling_feature_anchor_v1",
        )

    effective = config.effective(["BTC"], {"BTC": 1})
    later = day("2026-08-03")
    with reader() as actual:
        actual.prepare(later)
        saved = actual.rank(later, effective, "BTC", "gross_excludes_fee", smoke=True)
        expected = rows(saved)
    remaining = resources.limit - resources.audit()["total_bytes"]
    resources.reserve("staging/" + "e" * 32, remaining, "payload")
    before = resources.audit()
    with reader() as actual:
        earlier = day("2026-08-02")
        actual.prepare(earlier)
        with pytest.raises(ValueError, match="budget"):
            actual.rank(earlier, effective, "BTC", "gross_excludes_fee", smoke=True)
        assert actual._features is None
    assert resources.audit() == before
    with reader() as actual:
        actual.prepare(later)
        reopened = actual.rank(
            later, effective, "BTC", "gross_excludes_fee", smoke=True
        )
        assert rows(reopened) == expected
        assert reopened.publication == saved.publication
    assert resources.audit() == before
