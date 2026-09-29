# ruff: noqa: F811

from datetime import datetime, timezone
import json

import pytest

from arblab.hyperliquid_copy.annual_execution_policy import POLICY
from arblab.hyperliquid_copy.candidate_day import build_candidate_day
from arblab.hyperliquid_copy.candidate_history import build_candidate_history
from .test_candidate_day import resources  # noqa: F401,F811
from .test_qualified_day import qualified  # noqa: F401,F811


def test_policy_candidate_day_has_exact_bound_and_identity(qualified, resources):
    publication = build_candidate_day(
        resources,
        qualified,
        "2026-08-01",
        max_bytes=8 * 1024**2,
        execution_policy_name=POLICY,
    )
    with resources._connect() as db:
        descriptor = json.loads(
            db.execute(
                "SELECT descriptor FROM publications WHERE key=?", (publication.key,)
            ).fetchone()[0]
        )
    assert descriptor["inputs"]["max_bytes"] == 8 * 1024**2
    assert descriptor["inputs"]["execution_policy"] == POLICY


def test_policy_rejects_arbitrary_candidate_limits(qualified, resources):
    with pytest.raises(ValueError, match="resource bounds"):
        build_candidate_day(
            resources,
            qualified,
            "2026-08-01",
            max_bytes=9 * 1024**2,
            execution_policy_name=POLICY,
        )

    with pytest.raises(ValueError, match="history byte limit"):
        build_candidate_history(
            resources,
            qualified,
            "2026-08-01",
            datetime(2026, 8, 2, tzinfo=timezone.utc),
            ["BTC"],
            max_bytes=65 * 1024**2,
            execution_policy_name=POLICY,
        )


def test_policy_history_keeps_complete_candidate_chain(qualified, resources):
    publication = build_candidate_history(
        resources,
        qualified,
        "2026-08-01",
        datetime(2026, 8, 2, tzinfo=timezone.utc),
        ["BTC"],
        max_bytes=64 * 1024**2,
        execution_policy_name=POLICY,
    )
    with resources._connect() as db:
        descriptor = json.loads(
            db.execute(
                "SELECT descriptor FROM publications WHERE key=?", (publication.key,)
            ).fetchone()[0]
        )
        daily = [
            json.loads(row[0])
            for row in db.execute(
                "SELECT descriptor FROM publications "
                "WHERE json_extract(descriptor,'$.kind')='candidate_day'"
            )
        ]
    assert descriptor["inputs"]["max_bytes"] == 64 * 1024**2
    assert descriptor["inputs"]["candidate_day_bytes"] == 8 * 1024**2
    assert descriptor["inputs"]["execution_policy"] == POLICY
    assert daily and all(row["inputs"]["max_bytes"] == 8 * 1024**2 for row in daily)
