import pytest

from arblab.hyperliquid_copy.annual_execution_policy import (
    POLICY,
    checked_execution_policy,
    execution_policy,
    feature_execution_policy_name,
)


def _qualified_reference():
    return {
        "path": "/unused/by-structural-policy-check",
        "identity": "cache",
        "expansion_receipt": {},
        "expansion_64gib_receipt": {},
    }


def test_absent_policy_preserves_legacy_behavior():
    assert checked_execution_policy(None) is None


def test_policy_is_frozen_and_has_exact_runtime_caps():
    policy = checked_execution_policy(
        POLICY,
        feature_history_policy="rolling_feature_anchor_v1",
        ranking_staging_policy="bounded_ranking_staging_v1",
        registration_policy="qualified_source_seed_v1",
        cache_reference=_qualified_reference(),
    )

    assert policy.name == "annual_bounded_rolling_v1"
    assert policy.feature_day_bytes == 256 * 1024**2
    assert policy.candidate_day_bytes == 8 * 1024**2
    assert policy.candidate_history_bytes == 64 * 1024**2
    assert policy.feature_anchor_bytes == 8 * 1024**2
    assert policy.retirement_journal_bytes == 64 * 1024
    assert policy.retirement_descriptor_bytes == 2 * 1024
    assert policy.operation_descriptor_bytes == 1024


def test_weekly_and_daily_feature_histories_have_distinct_bounded_namespaces():
    weekly = feature_execution_policy_name(POLICY, "weekly")
    daily = feature_execution_policy_name(POLICY, "daily")

    assert weekly != daily
    assert execution_policy(weekly).name == weekly
    assert execution_policy(daily).name == daily
    with pytest.raises(ValueError, match="feature execution"):
        feature_execution_policy_name(POLICY, "monthly")


@pytest.mark.parametrize("value", [True, False, "unknown", 1, {}])
def test_unknown_or_non_string_policy_is_rejected(value):
    with pytest.raises(ValueError, match="annual execution policy"):
        checked_execution_policy(value)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"feature_history_policy": None}, "rolling feature"),
        ({"ranking_staging_policy": None}, "bounded ranking"),
        ({"registration_policy": None}, "qualified registration"),
        ({"cache_reference": None}, "expanded cache receipts"),
        (
            {
                "cache_reference": {
                    "path": "/unused",
                    "identity": "cache",
                    "expansion_receipt": {},
                }
            },
            "expanded cache receipts",
        ),
    ],
)
def test_policy_requires_its_complete_prerequisite_contract(override, message):
    arguments = dict(
        feature_history_policy="rolling_feature_anchor_v1",
        ranking_staging_policy="bounded_ranking_staging_v1",
        registration_policy="qualified_source_seed_v1",
        cache_reference=_qualified_reference(),
    )
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        checked_execution_policy(POLICY, **arguments)
