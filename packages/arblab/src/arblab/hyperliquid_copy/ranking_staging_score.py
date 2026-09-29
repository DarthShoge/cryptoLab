"""Unpublished score passes; caller establishes complete source provenance."""

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path

from .contracts import utc
from .derived_publication import _encode, _inputs
from .disk_cohort_scoring import metric_context, _selection, _engine as scoring_engine
from .disk_score_query import ScoreQuery
from .disk_cohort_query import CohortQuery
from .download import file_hash
from .ranking_staging_artifact import StagingArtifact
from .ranking_staging_owner import RankingStagingOwner


def _engine():
    root = Path(__file__).parent
    return _encode(
        dict(
            scoring=scoring_engine(),
            code={
                name: file_hash(root / name)
                for name in (
                    "ranking_staging_score.py",
                    "ranking_staging_artifact.py",
                    "ranking_staging_owner.py",
                )
            },
        )
    )


@dataclass(frozen=True)
class PendingScores:
    metrics: StagingArtifact
    scores: StagingArtifact
    ranking: StagingArtifact
    _summary: dict

    @property
    def summary(self):
        return deepcopy(self._summary)


def _write(owner, role, writer):
    held = None

    def capture(fd):
        nonlocal held
        owner.capture_created_fd(role, fd)
        held = os.dup(fd)

    try:
        result = writer(owner.path(role), on_created=capture)
        if held is None:
            raise ValueError("Missing staging writer ownership capture")
        os.fsync(held)
        artifact = StagingArtifact.capture(owner.path(role), held)
        return artifact, result
    finally:
        if held is not None:
            os.close(held)


def score_pending_metrics(
    owner, metrics, config, decision, scope, semantics, *, verify_source
):
    if (
        type(owner) is not RankingStagingOwner
        or type(metrics) is not StagingArtifact
        or not callable(verify_source)
    ):
        raise ValueError("Invalid pending scoring inputs/verifier")
    decision = utc(decision)
    if metrics.path != owner.path("metrics"):
        raise ValueError("Metric artifact is not owned by invocation")

    def context():
        return _encode(
            dict(
                config=vars(config),
                metrics=metric_context(config, decision, scope, semantics),
                selection=_selection(owner.path("scratch"), config, decision, scope),
            )
        )

    frozen = context()
    engine = _engine()
    provenance = _encode(_inputs("staging_provenance", verify_source()))
    if provenance == b"{}":
        raise ValueError("Empty pending scoring source provenance")
    pins = [metrics]

    def verify():
        owner.verify()
        if _encode(_inputs("staging_provenance", verify_source())) != provenance:
            raise ValueError("Pending scoring source provenance changed")
        owner.verify()
        if _engine() != engine or context() != frozen:
            raise ValueError("Pending scoring engine/configuration changed")
        for artifact in pins:
            artifact.verify()
        if context() != frozen:
            raise ValueError(
                "Pending scoring configuration changed during verification"
            )

    verify()
    with ScoreQuery(metrics.path, owner.path("scratch"), config) as query:
        scores, count = _write(owner, "scores", query.write_scores)
    pins.append(scores)
    verify()
    with CohortQuery(
        scores.path, owner.path("scratch"), config, decision, scope
    ) as query:
        ranking, summary = _write(owner, "ranking", query.write_rankings)
    pins.append(ranking)
    verify()
    if summary["candidate_count"] != count:
        raise ValueError("Pending score/ranking candidate counts differ")
    return PendingScores(metrics, scores, ranking, deepcopy(summary))
