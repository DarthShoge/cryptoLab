"""Explicit opt-in; never enable disposal by a default or unknown policy."""

POLICY = "bounded_ranking_staging_v1"


def validate_policy(value):
    if value is not None and value != POLICY:
        raise ValueError("Unknown ranking staging policy")
    return value
