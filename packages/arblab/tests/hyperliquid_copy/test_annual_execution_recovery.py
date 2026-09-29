# ruff: noqa: F401,F811

import hashlib
import json

import pytest

from arblab.hyperliquid_copy import annual_ranking_retirement as retirement
from arblab.hyperliquid_copy.annual_execution_recovery import operation_status, recover
from .test_annual_ranking_retirement import _captured
from .test_candidate_day import resources
from .test_feature_window import days
from .test_qualified_day import qualified


def _operation(resources, qualified, days):
    _, inputs, _, consumption = _captured(resources, qualified, days)
    operation = retirement.prepare_operation(resources, inputs, consumption)
    _, publication, body = retirement._load_operation(resources, operation)
    return operation, publication, body


def test_cross_process_recovery_reloads_persisted_saved_ranking(
    resources, qualified, days
):
    operation, publication, body = _operation(resources, qualified, days)
    targets = body["targets"]
    retirement._detach(
        resources,
        operation,
        publication.key,
        targets[0],
        [row["key"] for row in targets[1:]],
    )
    assert operation_status(resources, operation)["state"] == "candidate_detached"

    result = recover(resources, operation)

    assert result["after"]["state"] == "targets_detached"
    assert resources.audit()["reserved_bytes"] == 0


def test_recovery_rejects_changed_live_target_receipt_without_mutation(
    resources, qualified, days
):
    operation, _, body = _operation(resources, qualified, days)
    target = body["targets"][0]
    with resources._connect() as db, db:
        raw = db.execute(
            "SELECT descriptor FROM publications WHERE key=?", (target["key"],)
        ).fetchone()[0]
        descriptor = json.loads(raw)
        descriptor["artifacts"][0]["sha256"] = "0" * 64
        changed = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        db.execute(
            "UPDATE publications SET descriptor=?,sha256=? WHERE key=?",
            (changed, hashlib.sha256(changed.encode()).hexdigest(), target["key"]),
        )

    with pytest.raises(ValueError, match="target changed"):
        recover(resources, operation)
    with resources._connect() as db:
        assert db.execute(
            "SELECT 1 FROM publications WHERE key=?", (target["key"],)
        ).fetchone()


def test_recovery_rejects_impossible_retained_state_after_detach(
    resources, qualified, days
):
    operation, publication, body = _operation(resources, qualified, days)
    target = body["targets"][0]
    retirement._detach(
        resources,
        operation,
        publication.key,
        target,
        [row["key"] for row in body["targets"][1:]],
    )
    allocation = target["allocations"][0]
    with resources._connect() as db, db:
        db.execute(
            "UPDATE allocations SET state='retained',bytes=?,sha256=? WHERE token=?",
            (allocation[5], allocation[6], allocation[0]),
        )

    with pytest.raises(ValueError, match="remained retained"):
        recover(resources, operation)


def test_changed_saved_receipt_rejects_before_candidate_mutation(
    resources, qualified, days
):
    operation, _, body = _operation(resources, qualified, days)
    candidate, saved = body["targets"][:2]
    with resources._connect() as db, db:
        raw = db.execute(
            "SELECT descriptor FROM publications WHERE key=?", (saved["key"],)
        ).fetchone()[0]
        descriptor = json.loads(raw)
        descriptor["artifacts"][0]["sha256"] = "0" * 64
        changed = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        db.execute(
            "UPDATE publications SET descriptor=?,sha256=? WHERE key=?",
            (changed, hashlib.sha256(changed.encode()).hexdigest(), saved["key"]),
        )

    with pytest.raises(ValueError, match="changed"):
        recover(resources, operation)
    with resources._connect() as db:
        assert db.execute(
            "SELECT 1 FROM publications WHERE key=?", (candidate["key"],)
        ).fetchone()
