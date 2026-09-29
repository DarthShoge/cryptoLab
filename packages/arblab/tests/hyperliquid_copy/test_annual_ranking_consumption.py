# ruff: noqa: F401,F811

from dataclasses import replace

import pytest

from arblab.hyperliquid_copy.annual_execution_policy import POLICY
from arblab.hyperliquid_copy.annual_ranking_consumption import (
    acknowledge_consumption,
)
from arblab.hyperliquid_copy.bound_scoring_context import BoundScoredCohort
from arblab.hyperliquid_copy.derived_publication import ArtifactPin, Publication
from arblab.hyperliquid_copy.ranking_artifact import RankingSink
from arblab.hyperliquid_copy.selection_disk_evidence import append_scored_cohort
from .test_selection_disk_evidence import cohort


def _bound(result, decision, scope):
    publication = Publication(
        "c" * 64,
        (
            ArtifactPin(
                "d" * 32,
                "artifacts/" + "d" * 32 + ".parquet",
                result.artifact.bytes,
                result.artifact.sha256,
            ),
        ),
    )
    return BoundScoredCohort(
        publication=publication,
        artifact=result.artifact,
        candidate_count=result.candidate_count,
        eligible_count=result.eligible_count,
        requested_count=result.requested_count,
        _selected_json=result._selected_json,
        bound_decision=decision,
        bound_scope=scope,
    )


def test_acknowledgement_is_emitted_only_after_complete_append(tmp_path, cohort):
    result, _, decision, scope = cohort
    result = _bound(result, decision, scope)
    seen = []

    def acknowledge(current, snapshot, selected):
        assert sink.rows == current.candidate_count
        seen.append(
            acknowledge_consumption(
                current,
                snapshot,
                selected,
                execution_policy=POLICY,
                source_pin={"path": "/qualified", "sha256": "a" * 64},
                config_sha256="b" * 64,
            )
        )

    with RankingSink(tmp_path / "complete.parquet") as sink:
        append_scored_cohort(
            result,
            sink,
            decision,
            scope,
            None,
            market_time=decision,
            trigger="scheduled",
            acknowledge=acknowledge,
        )

    assert len(seen) == 1
    assert seen[0].candidate_count == result.candidate_count
    assert seen[0].ranking_key == result.publication.key
    assert seen[0].artifact_sha256 == result.artifact.sha256


def test_failed_or_partial_append_never_acknowledges(tmp_path, cohort):
    result, _, decision, scope = cohort
    result = _bound(result, decision, scope)
    seen = []
    invalid = replace(result, candidate_count=result.candidate_count + 1)

    with pytest.raises(ValueError, match="count mismatch"):
        with RankingSink(tmp_path / "failed.parquet") as sink:
            append_scored_cohort(
                invalid,
                sink,
                decision,
                scope,
                None,
                market_time=decision,
                trigger="scheduled",
                acknowledge=lambda *args: seen.append(args),
            )
    assert seen == []


def test_callback_failure_prevents_sink_finalization(tmp_path, cohort):
    result, _, decision, scope = cohort
    result = _bound(result, decision, scope)

    def fail(*_):
        raise RuntimeError("retirement not committed")

    with pytest.raises(RuntimeError, match="retirement not committed"):
        with RankingSink(tmp_path / "callback-failed.parquet") as sink:
            append_scored_cohort(
                result,
                sink,
                decision,
                scope,
                None,
                market_time=decision,
                trigger="scheduled",
                acknowledge=fail,
            )
    with pytest.raises(ValueError, match="not successfully finalized"):
        sink.artifact


def test_caught_callback_failure_poisons_sink_and_prevents_duplicate_retry(
    tmp_path, cohort
):
    result, _, decision, scope = cohort
    result = _bound(result, decision, scope)

    with RankingSink(tmp_path / "caught-callback-failed.parquet") as sink:
        with pytest.raises(RuntimeError, match="retirement not committed"):
            append_scored_cohort(
                result,
                sink,
                decision,
                scope,
                None,
                market_time=decision,
                trigger="scheduled",
                acknowledge=lambda *_: (_ for _ in ()).throw(
                    RuntimeError("retirement not committed")
                ),
            )
        with pytest.raises(ValueError, match="not open"):
            append_scored_cohort(
                result,
                sink,
                decision,
                scope,
                None,
                market_time=decision,
                trigger="scheduled",
            )

    with pytest.raises(ValueError, match="not successfully finalized"):
        sink.artifact


def test_unbound_or_wrong_policy_cannot_acknowledge(cohort):
    result, rows, decision, scope = cohort
    snapshot = {
        "candidate_count": result.candidate_count,
        "eligible_count": result.eligible_count,
    }
    with pytest.raises(ValueError, match="bound"):
        acknowledge_consumption(
            result,
            snapshot,
            rows,
            execution_policy=POLICY,
            source_pin={"path": "/qualified", "sha256": "a" * 64},
            config_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="annual execution policy"):
        acknowledge_consumption(
            _bound(result, decision, scope),
            snapshot,
            rows,
            execution_policy="wrong",
            source_pin={"path": "/qualified", "sha256": "a" * 64},
            config_sha256="b" * 64,
        )
