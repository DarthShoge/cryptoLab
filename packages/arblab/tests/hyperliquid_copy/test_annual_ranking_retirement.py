# ruff: noqa: F811

import pytest

from arblab.hyperliquid_copy.annual_execution_policy import POLICY
from arblab.hyperliquid_copy.annual_ranking_consumption import RankingConsumption
from arblab.hyperliquid_copy.annual_ranking_retirement import discover_targets
from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
from arblab.hyperliquid_copy.saved_feature_rankings import SavedFeatureRankings
from .test_candidate_day import resources  # noqa: F401
from .test_feature_metric_producer import args
from .test_feature_window import days  # noqa: F401
from .test_qualified_day import qualified  # noqa: F401

STAGING = "bounded_ranking_staging_v1"


def _captured(resources, qualified, days):
    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources,
        QualifiedSourceSession(qualified),
        staging_policy=STAGING,
        execution_policy_name=POLICY,
    )
    inputs = receipts.capture_staged(
        values[4], values[5], values[6], values[7], days=values[3]
    )
    result = receipts.load(inputs, values[4], values[5], values[6], values[7])
    pin = result.publication.artifacts[0]
    consumption = RankingConsumption(
        execution_policy=POLICY,
        source_path=qualified["path"],
        source_sha256=qualified["sha256"],
        config_sha256="b" * 64,
        decision=result.bound_decision,
        scope=result.bound_scope,
        ranking_key=result.publication.key,
        artifact_tokens=(pin.token,),
        artifact_sha256=pin.sha256,
        candidate_count=result.candidate_count,
        eligible_count=result.eligible_count,
        selected_count=len(result.selected),
    )
    return receipts, inputs, result, consumption


def test_discovers_exact_candidate_saved_and_staged_targets(resources, qualified, days):
    _, inputs, result, consumption = _captured(resources, qualified, days)

    targets = discover_targets(resources, inputs, consumption)

    assert [target.kind for target in targets.ordered] == [
        "candidate_history",
        "saved_feature_ranking",
        "staged_cohort_rankings",
    ]
    assert targets.saved.key == result.publication.key
    assert targets.saved.tokens == targets.staged.tokens == consumption.artifact_tokens
    assert set(targets.candidate.tokens).isdisjoint(targets.saved.tokens)


def test_unrelated_reference_to_owned_token_rejects_without_mutation(
    resources, qualified, days
):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    _, inputs, result, consumption = _captured(resources, qualified, days)
    token = result.publication.artifacts[0].token
    PublishedArtifacts(resources).publish("unrelated_reference", {}, [token])
    before = resources.audit()

    try:
        discover_targets(resources, inputs, consumption)
    except ValueError as exc:
        assert "reference" in str(exc)
    else:
        raise AssertionError("unrelated token reference was accepted")
    assert resources.audit() == before


def test_single_operation_retires_in_order_and_preserves_candidate_days(
    resources, qualified, days
):
    from arblab.hyperliquid_copy.annual_ranking_retirement import (
        prepare_operation,
        execute_operation,
    )
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    receipts, inputs, result, consumption = _captured(resources, qualified, days)
    targets = discover_targets(resources, inputs, consumption)
    candidate_day_keys = []
    with resources._connect() as db:
        candidate_day_keys = [
            row[0]
            for row in db.execute(
                "SELECT key FROM publications "
                "WHERE json_extract(descriptor,'$.kind')='candidate_day'"
            )
        ]
    operation = prepare_operation(resources, inputs, consumption)
    observed = []

    def reload_saved():
        loaded = receipts.load(
            inputs,
            result.bound_decision,
            receipts._source and args(resources, qualified, days)[5],
            result.bound_scope,
            "gross_excludes_fee",
        )
        observed.append((loaded.candidate_count, loaded.artifact.sha256))

    outcome = execute_operation(resources, operation, verify_saved=reload_saved)

    assert observed == [(result.candidate_count, result.artifact.sha256)]
    assert outcome["retired_targets"] == [target.key for target in targets.ordered]
    assert all(
        PublishedArtifacts(resources).lookup(target.kind, target.inputs) is None
        for target in targets.ordered
    )
    with resources._connect() as db:
        remaining = {
            row[0]
            for row in db.execute(
                "SELECT key FROM publications "
                "WHERE json_extract(descriptor,'$.kind')='candidate_day'"
            )
        }
        operation_kinds = [
            row[0]
            for row in db.execute(
                "SELECT json_extract(descriptor,'$.kind') FROM publications "
                "WHERE json_extract(descriptor,'$.kind') LIKE 'annual_ranking_retirement_%'"
            )
        ]
    assert remaining == set(candidate_day_keys)
    assert operation_kinds == ["annual_ranking_retirement_committed"]
    assert not (resources.root / result.publication.artifacts[0].path).exists()
    assert resources.audit()["reserved_bytes"] == 0


def test_explicit_recovery_resumes_after_saved_detach(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import annual_ranking_retirement as module

    receipts, inputs, result, consumption = _captured(resources, qualified, days)
    operation = module.prepare_operation(resources, inputs, consumption)
    original = module._detach
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("crash before staged detach")
        return original(*args, **kwargs)

    def reload_saved():
        receipts.load(
            inputs,
            result.bound_decision,
            args(resources, qualified, days)[5],
            result.bound_scope,
            "gross_excludes_fee",
        )

    with monkeypatch.context() as faults:
        faults.setattr(module, "_detach", interrupt)
        try:
            module.execute_operation(resources, operation, verify_saved=reload_saved)
        except RuntimeError as exc:
            assert "staged detach" in str(exc)
        else:
            raise AssertionError("failure injection did not interrupt retirement")

    result = module.recover_operation(resources, operation)
    assert len(result["retired_targets"]) == 3
    assert resources.audit()["reserved_bytes"] == 0


def test_retry_disposes_candidate_pending_after_crash_before_unlink(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import annual_ranking_retirement as module

    receipts, inputs, result, consumption = _captured(resources, qualified, days)
    operation = module.prepare_operation(resources, inputs, consumption)
    original = module._dispose
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("crash before candidate disposal")
        return original(*args, **kwargs)

    with monkeypatch.context() as faults:
        faults.setattr(module, "_dispose", interrupt)
        with pytest.raises(RuntimeError, match="candidate disposal"):
            module.execute_operation(resources, operation, verify_saved=lambda: None)

    def reload_saved():
        receipts.load(
            inputs,
            result.bound_decision,
            args(resources, qualified, days)[5],
            result.bound_scope,
            "gross_excludes_fee",
        )

    module.execute_operation(resources, operation, verify_saved=reload_saved)
    assert resources.audit()["reserved_bytes"] == 0


def test_recovery_disposes_staged_pending_after_crash_before_unlink(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import annual_ranking_retirement as module

    _, inputs, _, consumption = _captured(resources, qualified, days)
    operation = module.prepare_operation(resources, inputs, consumption)
    original = module._dispose
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("crash before staged disposal")
        return original(*args, **kwargs)

    with monkeypatch.context() as faults:
        faults.setattr(module, "_dispose", interrupt)
        with pytest.raises(RuntimeError, match="staged disposal"):
            module.execute_operation(resources, operation, verify_saved=lambda: None)

    module.recover_operation(resources, operation)
    assert resources.audit()["reserved_bytes"] == 0
