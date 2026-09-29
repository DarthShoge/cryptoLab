"""Explicit opt-in to physical retirement; absence preserves retained history."""

ROLLING_POLICY = "rolling_feature_anchor_v1"


def checked_feature_policy(value):
    if value is not None and (type(value) is not str or value != ROLLING_POLICY):
        raise ValueError("Invalid feature history policy")
    return value
