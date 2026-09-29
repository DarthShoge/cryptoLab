# ruff: noqa: F401,F811

from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
from arblab.hyperliquid_copy.rolling_feature_history import RollingFeatureHistory
from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS
from .test_qualified_day import qualified


def route(resources, qualified, start, end, **kwargs):
    from arblab.hyperliquid_copy.rolling_history_route import prepare_history_route

    return prepare_history_route(
        resources,
        QualifiedSourceSession(qualified),
        ["BTC"],
        SEMANTICS,
        start,
        end,
        **kwargs,
    )


def test_cold_route_is_nonallocating_features(resources, qualified):
    before = resources.audit()
    assert route(resources, qualified, DAY, DAY + timedelta(days=1)) == "features"
    assert resources.audit() == before


def test_route_frontier_is_isolated_by_execution_policy(resources, qualified):
    from arblab.hyperliquid_copy.annual_execution_policy import POLICY

    rolling = RollingFeatureHistory(
        resources,
        qualified,
        ["BTC"],
        SEMANTICS,
        execution_policy_name=POLICY,
    )
    rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))

    assert route(resources, qualified, DAY, DAY + timedelta(days=1)) == "features"
    assert (
        route(
            resources,
            qualified,
            DAY,
            DAY + timedelta(days=1),
            execution_policy_name=POLICY,
        )
        == "raw"
    )


def test_cold_route_final_callback_cannot_change_catalogue(resources, qualified):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from .test_feature_history import history

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=1)).days
    calls = 0

    def count():
        nonlocal calls
        calls += 1

    route(resources, qualified, DAY, DAY + timedelta(days=1), validation=count)
    final_call, calls = calls, 0
    changed = False

    def validation():
        nonlocal calls, changed
        calls += 1
        if calls == final_call:
            PublishedArtifacts(resources).publish(
                "route_callback_probe",
                {},
                [pin.token for pin in days[0].publication.artifacts],
            )
            changed = True

    with pytest.raises(ValueError, match="catalogue|changed"):
        route(resources, qualified, DAY, DAY + timedelta(days=1), validation=validation)
    assert changed


def test_foreign_prepared_owner_blocks_route_without_new_allocations(
    resources, qualified
):
    from arblab.hyperliquid_copy.cache_retirement import prepare_retirement
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from .test_feature_history import history

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days
    owner = publish_feature_anchor(resources, qualified, days[1:2], ["BTC"], SEMANTICS)
    publish_feature_anchor(resources, qualified, days[2:], ["BTC"], SEMANTICS)
    prepare_retirement(
        resources,
        "qualified_features_day",
        days[0].inputs,
        owner=owner.publication.key,
        reason="Foreign unfinished owner",
    )
    before = resources.audit()
    with pytest.raises(ValueError, match="owner"):
        route(resources, qualified, DAY, DAY + timedelta(days=1))
    assert resources.audit() == before


@pytest.mark.parametrize("fault", ["scope", "semantics", "interval", "validation"])
def test_invalid_route_rejects_without_allocations(resources, qualified, fault):
    from arblab.hyperliquid_copy.rolling_history_route import prepare_history_route

    source = QualifiedSourceSession(qualified)
    coins = [] if fault == "scope" else ["BTC"]
    semantics = "unknown" if fault == "semantics" else SEMANTICS
    end = DAY if fault == "interval" else DAY + timedelta(days=1)
    validation = 1 if fault == "validation" else None
    before = resources.audit()
    with pytest.raises(ValueError):
        prepare_history_route(
            resources, source, coins, semantics, DAY, end, validation=validation
        )
    assert resources.audit() == before


@pytest.mark.parametrize(
    "retained_end,start,end,expected",
    [
        (2, 1, 2, "features"),
        (2, 1, 3, "features"),
        (2, 0, 2, "raw"),
        (3, 1, 2, "raw"),
        (3, 1.25, 2.5, "features"),
    ],
)
def test_route_uses_verified_latest_calendar_frontier(
    resources, qualified, retained_end, start, end, expected
):
    history = RollingFeatureHistory(resources, qualified, ["BTC"], SEMANTICS)
    history.window(DAY + timedelta(days=1), DAY + timedelta(days=retained_end))
    before = resources.audit()
    assert (
        route(
            resources, qualified, DAY + timedelta(days=start), DAY + timedelta(days=end)
        )
        == expected
    )
    assert resources.audit() == before


def test_annual_cadence_namespace_starts_without_adopting_base_frontier(
    resources, qualified
):
    from arblab.hyperliquid_copy.annual_execution_policy import (
        POLICY,
        feature_execution_policy_name,
    )

    history = RollingFeatureHistory(
        resources,
        qualified,
        ["BTC"],
        SEMANTICS,
        execution_policy_name=POLICY,
    )
    history.window(DAY + timedelta(days=1), DAY + timedelta(days=3))

    weekly = feature_execution_policy_name(POLICY, "weekly")
    assert (
        route(
            resources,
            qualified,
            DAY,
            DAY + timedelta(days=1),
            execution_policy_name=weekly,
        )
        == "features"
    )


def test_corrupt_latest_anchor_is_not_raw_fallback(resources, qualified):
    from arblab.hyperliquid_copy.feature_resume_anchor import FeatureAnchor

    history = RollingFeatureHistory(resources, qualified, ["BTC"], SEMANTICS)
    history.window(DAY + timedelta(days=1), DAY + timedelta(days=3))
    anchor = FeatureAnchor(resources, qualified, history.anchor_inputs)
    with (resources.root / anchor.publication.artifacts[0].path).open("ab") as stream:
        stream.write(b"corrupt anchor")
    with pytest.raises(ValueError):
        route(resources, qualified, DAY, DAY + timedelta(days=1))


def test_route_keeps_anchor_protected_through_final_callback(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import rolling_history_route as module
    from arblab.hyperliquid_copy.feature_resume_anchor import FeatureAnchor

    history = RollingFeatureHistory(resources, qualified, ["BTC"], SEMANTICS)
    history.window(DAY + timedelta(days=1), DAY + timedelta(days=3))
    anchor = FeatureAnchor(resources, qualified, history.anchor_inputs)
    path = resources.root / anchor.publication.artifacts[0].path
    original = module.recover_feature_retirements
    armed = changed = False

    def recover(*args, **kwargs):
        nonlocal armed
        result = original(*args, **kwargs)
        armed = True
        return result

    def validation():
        nonlocal changed
        if armed and not changed:
            changed = True
            with path.open("ab") as stream:
                stream.write(b"changed after recovery")

    monkeypatch.setattr(module, "recover_feature_retirements", recover)
    with pytest.raises(ValueError):
        route(resources, qualified, DAY, DAY + timedelta(days=1), validation=validation)
    assert changed


@pytest.mark.parametrize("stage", ["begin_retirement", "finish_retirement"])
def test_route_recovers_existing_obligation_before_allowing_raw(
    resources, qualified, monkeypatch, stage
):
    from arblab.hyperliquid_copy import rolling_feature_history as module

    history = RollingFeatureHistory(resources, qualified, ["BTC"], SEMANTICS)
    first = history.window(DAY, DAY + timedelta(days=2))
    original = getattr(module, stage)

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted retirement")

    monkeypatch.setattr(module, stage, interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        history.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    monkeypatch.setattr(module, stage, original)
    assert route(resources, qualified, DAY, DAY + timedelta(days=1)) == "raw"
    assert all(
        not (resources.root / pin.path).exists()
        for pin in first.days[0].publication.artifacts
    )
    assert resources.audit()["reserved_bytes"] == 0
