"""Explicit opt-in contract for bounded annual copy-trader execution."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace

from .feature_history_policy import ROLLING_POLICY
from .ranking_staging_policy import POLICY as BOUNDED_STAGING_POLICY

POLICY = "annual_bounded_rolling_v1"
QUALIFIED_REGISTRATION_POLICY = "qualified_source_seed_v1"
FEATURE_POLICY_NAMES = {
    cadence: f"{POLICY}:features:{cadence}:v1" for cadence in ("weekly", "daily")
}


@dataclass(frozen=True)
class AnnualExecutionPolicy:
    name: str = POLICY
    feature_day_bytes: int = 256 * 1024**2
    feature_day_artifacts: int = 16
    candidate_day_bytes: int = 8 * 1024**2
    candidate_history_bytes: int = 64 * 1024**2
    feature_anchor_bytes: int = 8 * 1024**2
    retirement_journal_bytes: int = 64 * 1024
    retirement_descriptor_bytes: int = 2 * 1024
    operation_descriptor_bytes: int = 1024
    candidate_day_descriptor_bytes: int = 2 * 1024
    feature_day_descriptor_bytes: int = 8 * 1024
    feature_anchor_descriptor_bytes: int = 4 * 1024
    active_ranking_descriptor_bytes: int = 64 * 1024


ANNUAL_BOUNDED_ROLLING = AnnualExecutionPolicy()


def require_descriptor_bound(resources, kind, inputs, tokens, maximum):
    """Preflight the exact publication encoding against an annual policy cap."""
    from .derived_publication import PublishedArtifacts, _encode

    with resources._connect() as db:
        pins = PublishedArtifacts(resources)._records(db, tokens)
    descriptor = _encode(
        dict(
            schema=1,
            kind=kind,
            inputs=inputs,
            artifacts=[asdict(pin) for pin in pins],
        )
    )
    if len(descriptor) > maximum:
        raise ValueError(f"{kind} publication descriptor limit exceeded")


def execution_policy(value):
    """Resolve only the scalar policy name at an already-authenticated boundary."""
    if value is None:
        return None
    if type(value) is not str or value not in {POLICY, *FEATURE_POLICY_NAMES.values()}:
        raise ValueError("Invalid annual execution policy")
    return (
        ANNUAL_BOUNDED_ROLLING
        if value == POLICY
        else replace(ANNUAL_BOUNDED_ROLLING, name=value)
    )


def feature_execution_policy_name(value, cadence):
    """Return the bounded cache namespace for one annual comparison cadence."""
    if value != POLICY or cadence not in FEATURE_POLICY_NAMES:
        raise ValueError("Invalid annual feature execution policy")
    return FEATURE_POLICY_NAMES[cadence]


def checked_execution_policy(
    value,
    *,
    feature_history_policy=None,
    ranking_staging_policy=None,
    registration_policy=None,
    cache_reference=None,
):
    """Validate the whole opt-in contract without opening the referenced cache."""
    if value is not None and value != POLICY:
        raise ValueError("Invalid annual execution policy")
    policy = execution_policy(value)
    if policy is None:
        return None
    if feature_history_policy != ROLLING_POLICY:
        raise ValueError("Annual policy requires rolling feature history")
    if ranking_staging_policy != BOUNDED_STAGING_POLICY:
        raise ValueError("Annual policy requires bounded ranking staging")
    if registration_policy != QUALIFIED_REGISTRATION_POLICY:
        raise ValueError("Annual policy requires qualified registration")
    if type(cache_reference) is not dict or not {
        "expansion_receipt",
        "expansion_64gib_receipt",
    }.issubset(cache_reference):
        raise ValueError("Annual policy requires both expanded cache receipts")
    return policy
