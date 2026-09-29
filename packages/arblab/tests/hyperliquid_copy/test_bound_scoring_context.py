from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
import json

import pytest

from arblab.hyperliquid_copy.ranking_artifact import RankingSink
from arblab.hyperliquid_copy.selection_disk_evidence import append_scored_cohort
from .test_disk_score_query import resources
from .test_disk_cohort_scoring import fixture, run


def bind(
    resources, result, config, decision, scope="BTC", semantics="gross_excludes_fee"
):
    from arblab.hyperliquid_copy.bound_scoring_context import bind_scored_context

    return bind_scored_context(resources, result, config, decision, scope, semantics)


@pytest.mark.parametrize("empty", [True, False])
def test_real_publication_chain_binds_without_writes(resources, empty):
    config, at, inputs = fixture(resources, **({"records": []} if empty else {}))
    result = run(resources, config, at, inputs)
    before = resources.audit()
    bound = bind(resources, result, config, at)
    assert bound.bound_decision == at and bound.bound_scope == "BTC"
    assert bound.publication == result.publication and bound.selected == result.selected
    assert sum(len(batch) for batch in bound.iter_batches()) == result.candidate_count
    assert bind(resources, result, config, at) == bound
    with pytest.raises(FrozenInstanceError):
        bound.bound_scope = "ETH"
    assert resources.audit() == before


def test_empty_bound_result_records_cash_and_previous_exits(resources, tmp_path):
    config, at, inputs = fixture(resources, records=[])
    original = run(resources, config, at, inputs)
    result = bind(resources, original, config, at)
    with RankingSink(tmp_path / "empty.parquet") as sink:
        snapshot, selected = append_scored_cohort(
            result, sink, at, "BTC", ["old"], market_time=at, trigger="scheduled"
        )
    assert snapshot["candidate_count"] == snapshot["eligible_count"] == 0
    assert snapshot["members"] == [] and snapshot["exits"] == ["old"]
    assert snapshot["membership_turnover"] == 1 and snapshot["cutoff_address"] is None
    assert selected == [] and sink.artifact.rows == 0
    assert original.selected == ()


@pytest.mark.parametrize(
    "change", ["decision", "scope", "lookback", "selection", "semantics"]
)
def test_wrong_requested_context_cannot_bind_empty_cohort(resources, change):
    config, at, inputs = fixture(resources, records=[])
    result = run(resources, config, at, inputs)
    scope, semantics = "BTC", "gross_excludes_fee"
    if change == "decision":
        at += timedelta(days=1)
    elif change == "scope":
        scope = None
    elif change == "lookback":
        config = replace(config, lookback_days=config.lookback_days + 1)
    elif change == "selection":
        config = replace(config, top_n=config.top_n + 1)
    else:
        semantics = "net_includes_fee"
    before = resources.audit()
    with pytest.raises(ValueError):
        bind(resources, result, config, at, scope, semantics)
    assert resources.audit() == before


@pytest.mark.parametrize("scope_change", [False, True])
def test_bound_empty_result_rejects_wrong_consumer_context(
    resources, tmp_path, scope_change
):
    config, at, inputs = fixture(resources, records=[])
    bound = bind(resources, run(resources, config, at, inputs), config, at)
    with pytest.raises(ValueError, match="context"):
        with RankingSink(tmp_path / "wrong.parquet") as sink:
            append_scored_cohort(
                bound,
                sink,
                at if scope_change else at + timedelta(days=1),
                None if scope_change else "BTC",
                None,
                market_time=at,
                trigger="scheduled",
            )


@pytest.mark.parametrize("fault", ["missing", "json", "oversized", "rekey"])
def test_bad_dependency_descriptor_rejects(resources, fault):
    config, at, inputs = fixture(resources, records=[])
    result = run(resources, config, at, inputs)
    with resources._connect() as db, db:
        records = db.execute("SELECT key,descriptor FROM publications").fetchall()
        key, raw = next(
            (key, raw)
            for key, raw in records
            if json.loads(raw)["kind"] == "candidate_scores"
        )
        if fault == "missing":
            db.execute("DELETE FROM publications WHERE key=?", (key,))
        elif fault == "rekey":
            db.execute("UPDATE publications SET key=? WHERE key=?", ("f" * 64, key))
        else:
            db.execute(
                "UPDATE publications SET descriptor=? WHERE key=?",
                ("{" if fault == "json" else "x" * (1024**2 + 1), key),
            )
    with pytest.raises(ValueError):
        bind(resources, result, config, at)


def test_unrelated_result_artifact_cannot_use_valid_publication_context(resources):
    config, at, inputs = fixture(resources)
    result = run(resources, config, at, inputs)
    changed = replace(result, artifact=replace(result.artifact, sha256="0" * 64))
    with pytest.raises(ValueError):
        bind(resources, changed, config, at)


def test_final_dependency_recheck_catches_payload_mutation(resources, monkeypatch):
    from arblab.hyperliquid_copy import bound_scoring_context as module

    config, at, inputs = fixture(resources)
    result = run(resources, config, at, inputs)
    original = module._read_publication

    def changed(resources, key, kind):
        publication, inputs = original(resources, key, kind)
        if kind == "candidate_metrics":
            with (resources.root / publication.artifacts[0].path).open("ab") as payload:
                payload.write(b"changed")
        return publication, inputs

    monkeypatch.setattr(module, "_read_publication", changed)
    with pytest.raises(ValueError):
        bind(resources, result, config, at)


def test_final_request_recheck_catches_mutation(resources, monkeypatch):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    config, at, inputs = fixture(resources)
    result = run(resources, config, at, inputs)
    original = PublishedArtifacts.lookup
    calls = []

    def changed(publications, kind, inputs):
        answer = original(publications, kind, inputs)
        calls.append(kind)
        if len(calls) == 6:
            config.metric_weights[next(iter(config.metric_weights))] = 0.123
        return answer

    monkeypatch.setattr(PublishedArtifacts, "lookup", changed)
    with pytest.raises(ValueError):
        bind(resources, result, config, at)
