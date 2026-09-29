"""Authenticated proof that a complete ranking entered aggregate run evidence."""

from dataclasses import dataclass
from datetime import datetime
import re

from .annual_execution_policy import execution_policy as checked_policy
from .bound_scoring_context import BoundScoredCohort
from .contracts import utc


def _digest(value):
    return type(value) is str and re.fullmatch("[a-f0-9]{64}", value) is not None


@dataclass(frozen=True)
class RankingConsumption:
    execution_policy: str
    source_path: str
    source_sha256: str
    config_sha256: str
    decision: datetime
    scope: str | None
    ranking_key: str
    artifact_tokens: tuple[str, ...]
    artifact_sha256: str
    candidate_count: int
    eligible_count: int
    selected_count: int


def acknowledge_consumption(
    result,
    snapshot,
    selected,
    *,
    execution_policy,
    source_pin,
    config_sha256,
):
    """Create an immutable acknowledgment after the caller exhausts the result."""
    policy = checked_policy(execution_policy)
    if not isinstance(result, BoundScoredCohort):
        raise ValueError("Annual consumption requires a bound scored cohort")
    if (
        type(snapshot) is not dict
        or type(selected) is not list
        or type(source_pin) is not dict
        or set(source_pin) != {"path", "sha256"}
        or type(source_pin["path"]) is not str
        or not source_pin["path"]
        or not _digest(source_pin["sha256"])
        or not _digest(config_sha256)
        or not _digest(result.publication.key)
        or len(result.publication.artifacts) != 1
    ):
        raise ValueError("Invalid annual ranking consumption context")
    pin = result.publication.artifacts[0]
    result.artifact.verify()
    if (
        snapshot.get("candidate_count") != result.candidate_count
        or snapshot.get("eligible_count") != result.eligible_count
        or snapshot.get("selected_count") != len(selected)
        or len(selected) != len(result.selected)
        or pin.bytes != result.artifact.bytes
        or pin.sha256 != result.artifact.sha256
    ):
        raise ValueError("Annual ranking consumption count/artifact mismatch")
    return RankingConsumption(
        execution_policy=policy.name,
        source_path=source_pin["path"],
        source_sha256=source_pin["sha256"],
        config_sha256=config_sha256,
        decision=utc(result.bound_decision),
        scope=result.bound_scope,
        ranking_key=result.publication.key,
        artifact_tokens=tuple(
            artifact.token for artifact in result.publication.artifacts
        ),
        artifact_sha256=result.artifact.sha256,
        candidate_count=result.candidate_count,
        eligible_count=result.eligible_count,
        selected_count=len(selected),
    )
