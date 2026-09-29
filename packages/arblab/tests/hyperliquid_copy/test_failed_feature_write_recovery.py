import json

import pytest

from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from .test_candidate_day import resources  # noqa: F401


def allocations(resources):
    tokens = []
    for i in range(2):
        path = f"artifacts/{i:032x}.parquet"
        token = resources.reserve(path, 1024, "payload")
        (resources.root / path).write_bytes(b"partial-or-complete-output")
        if i == 0:
            resources.settle(token)
        tokens.append(token)
    return tokens


def test_quarantine_preserves_unpublished_files_and_removes_only_their_charges(
    resources, tmp_path
):
    from arblab.hyperliquid_copy.failed_feature_write_recovery import quarantine

    tokens = allocations(resources)
    destination = tmp_path / "recovery"
    result = quarantine(resources, tokens, destination)
    assert result["files"] == 2
    assert resources.audit()["reserved_bytes"] == 0
    assert resources.audit()["retained_bytes"] == 0
    plan = json.loads((destination / "plan.json").read_text())
    assert len(plan["allocations"]) == 2
    assert len(list(destination.glob("*.parquet"))) == 2


def test_published_artifact_is_rejected_before_any_move(resources, tmp_path):
    from arblab.hyperliquid_copy.failed_feature_write_recovery import quarantine

    tokens = allocations(resources)
    PublishedArtifacts(resources).publish(
        "test_output", {"day": "2025-10-10"}, [tokens[0]]
    )
    before = resources.audit()
    with pytest.raises(ValueError, match="referenced"):
        quarantine(resources, tokens, tmp_path / "recovery")
    assert resources.audit() == before
    assert not (tmp_path / "recovery").exists()


def test_failed_recovery_restores_files_and_ledger(resources, tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.failed_feature_write_recovery import quarantine

    tokens = allocations(resources)
    before = resources.audit()
    original = resources._audit

    def fail_after_detach(db):
        if db.execute("SELECT count(*) FROM allocations").fetchone()[0] == 0:
            raise ValueError("injected final audit failure")
        return original(db)

    monkeypatch.setattr(resources, "_audit", fail_after_detach)
    with pytest.raises(ValueError, match="injected"):
        quarantine(resources, tokens, tmp_path / "recovery")
    assert resources.audit() == before
