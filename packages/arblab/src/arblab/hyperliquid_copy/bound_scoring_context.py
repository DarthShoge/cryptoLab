"""Bind internal scored results to verified publication decision/market context."""

from dataclasses import dataclass, fields
from datetime import datetime
import json
from pathlib import Path
import re

from .contracts import utc
from .derived_publication import MAX_DESCRIPTOR_BYTES, PublishedArtifacts, _encode
from .disk_cohort_scoring import _engine, _selection, metric_context
from .disk_score_result import ScoredCohort
from .download import file_hash


@dataclass(frozen=True)
class BoundScoredCohort(ScoredCohort):
    bound_decision: datetime
    bound_scope: str | None


def _read_publication(resources, key, kind):
    resources.lease.check()
    if type(key) is not str or not re.fullmatch("[a-f0-9]{64}", key):
        raise ValueError("Invalid scoring publication key")
    with resources._connect() as db:
        row = db.execute(
            "SELECT CASE WHEN length(CAST(descriptor AS BLOB)) BETWEEN 1 AND ? "
            "THEN descriptor END FROM publications WHERE key=?",
            (MAX_DESCRIPTOR_BYTES, key),
        ).fetchone()
    if row is None or type(row[0]) is not str:
        raise ValueError("Missing or oversized scoring publication descriptor")
    try:
        descriptor = json.loads(row[0])
    except (ValueError, RecursionError) as exc:
        raise ValueError("Invalid scoring publication descriptor") from exc
    if (
        type(descriptor) is not dict
        or descriptor.get("kind") != kind
        or type(descriptor.get("inputs")) is not dict
    ):
        raise ValueError("Invalid scoring publication context")
    inputs = descriptor["inputs"]
    publication = PublishedArtifacts(resources).lookup(kind, inputs)
    if publication is None or publication.key != key:
        raise ValueError("Scoring publication identity mismatch")
    return publication, inputs


def bind_scored_context(resources, result, config, decision, scope, semantics):
    """Verify provenance links, not source qualification or client authorization."""
    resources.lease.check()
    if not isinstance(result, ScoredCohort):
        raise ValueError("Expected internal scored cohort")
    decision = utc(decision)
    context = _encode(metric_context(config, decision, scope, semantics))
    selection = _encode(_selection(resources.root, config, decision, scope))
    engine, helper = _engine(), file_hash(Path(__file__))
    final, final_inputs = _read_publication(
        resources, result.publication.key, "cohort_rankings"
    )
    if (
        final != result.publication
        or set(final_inputs) != {"scores", "selection", "engine", "max_bytes"}
        or final_inputs["engine"] != engine
        or _encode(final_inputs["selection"]) != selection
        or len(final.artifacts) != 1
    ):
        raise ValueError("Scored result publication context mismatch")
    pin = final.artifacts[0]
    if (
        result.artifact.path != resources.root / pin.path
        or result.artifact.bytes != pin.bytes
        or result.artifact.sha256 != pin.sha256
        or result.artifact.rows != result.candidate_count
    ):
        raise ValueError("Scored result artifact mismatch")
    scores, score_inputs = _read_publication(
        resources, final_inputs["scores"], "candidate_scores"
    )
    if (
        set(score_inputs) != {"source", "engine", "context", "max_bytes"}
        or score_inputs["engine"] != engine
        or _encode(score_inputs["context"]) != context
        or score_inputs["max_bytes"] != final_inputs["max_bytes"]
    ):
        raise ValueError("Score publication context mismatch")
    metrics, metric_inputs = _read_publication(
        resources, score_inputs["source"], "candidate_metrics"
    )
    if (
        set(metric_inputs) != {"context", "provenance"}
        or _encode(metric_inputs["context"]) != context
        or type(metric_inputs["provenance"]) is not dict
        or not metric_inputs["provenance"]
    ):
        raise ValueError("Metric publication context mismatch")
    publications = PublishedArtifacts(resources)
    for kind, inputs, original in (
        ("cohort_rankings", final_inputs, final),
        ("candidate_scores", score_inputs, scores),
        ("candidate_metrics", metric_inputs, metrics),
    ):
        if publications.lookup(kind, inputs) != original:
            raise ValueError("Scoring publication dependency changed")
    if (
        _encode(metric_context(config, decision, scope, semantics)) != context
        or _encode(_selection(resources.root, config, decision, scope)) != selection
        or _engine() != engine
        or file_hash(Path(__file__)) != helper
    ):
        raise ValueError("Scoring request or binding engine changed")
    resources.lease.check()
    return BoundScoredCohort(
        **{field.name: getattr(result, field.name) for field in fields(ScoredCohort)},
        bound_decision=decision,
        bound_scope=scope,
    )
