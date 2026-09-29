import json
import os
import uuid
from copy import deepcopy

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts


def expanded_fixture(root, lease):
    from arblab.hyperliquid_copy.derived_cache_expansion import (
        apply_expansion,
        prepare_expansion,
    )

    old = CacheResources.create(lease, "expansion-64-test")
    relative = f"artifacts/{uuid.uuid4().hex}"
    token = old.reserve(relative, 100, "payload")
    (root / relative).write_bytes(b"preserve this published history")
    old.settle(token)
    publication = PublishedArtifacts(old).publish("history", {"day": 1}, [token])
    predecessor = prepare_expansion(
        old,
        approved_from=8 * 1024**3,
        approved_to=16 * 1024**3,
        approval="first explicit approval",
    )
    return (
        apply_expansion(lease, old.identity, predecessor),
        predecessor,
        publication,
    )


def prepare(resources, predecessor):
    from arblab.hyperliquid_copy.derived_cache_expansion_64 import prepare_expansion_64

    return prepare_expansion_64(
        resources,
        predecessor,
        approved_from=16 * 1024**3,
        approved_to=64 * 1024**3,
        approval="second explicit approval",
    )


def rows(resources):
    with resources._connect() as db:
        return (
            db.execute("SELECT * FROM allocations ORDER BY token").fetchall(),
            db.execute("SELECT * FROM publications ORDER BY key").fetchall(),
        )


def test_expansion_64_chains_receipts_preserves_payloads_and_reopens(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_expansion_64 import apply_expansion_64
    from arblab.hyperliquid_copy.derived_cache_policy import _receipt
    from arblab.hyperliquid_copy.derived_cache_policy_64 import open_expanded_64_cache

    with CacheLease(tmp_path) as lease:
        expanded, predecessor, history = expanded_fixture(tmp_path, lease)
        before = rows(expanded)
        predecessor_publication = _receipt(expanded, predecessor)[0]
        receipt = prepare(expanded, predecessor)
        resources = apply_expansion_64(
            lease, expanded.identity, predecessor, receipt
        )
        assert resources.limit == 64 * 1024**3
        assert resources.audit()["reserved_bytes"] == 0
        after = rows(resources)
        assert set(before[0]) <= set(after[0])
        assert set(before[1]) <= set(after[1])
        assert _receipt(resources, predecessor)[0] == predecessor_publication
        assert PublishedArtifacts(resources).lookup("history", {"day": 1}) == history
    with CacheLease(tmp_path) as lease:
        reopened = open_expanded_64_cache(
            lease, "expansion-64-test", predecessor, receipt
        )
        assert reopened.audit()["reserved_bytes"] == 0


def test_expansion_64_requires_exact_predecessor_and_no_pending_work(tmp_path):
    with CacheLease(tmp_path) as lease:
        expanded, predecessor, _ = expanded_fixture(tmp_path, lease)
        changed = json.loads(json.dumps(predecessor))
        changed["approval"] = "different"
        with pytest.raises(ValueError):
            prepare(expanded, changed)
        expanded.reserve(f"scratch/{uuid.uuid4().hex}", 1, "scratch")
        before = rows(expanded)
        with pytest.raises(ValueError, match="pending"):
            prepare(expanded, predecessor)
        assert rows(expanded) == before


@pytest.mark.parametrize("point", ["before_replace", "after_replace", "before_release"])
def test_expansion_64_interruption_recovers_only_with_exact_receipt(
    tmp_path, monkeypatch, point
):
    from arblab.hyperliquid_copy import derived_cache_expansion_64 as migration

    with CacheLease(tmp_path) as lease:
        expanded, predecessor, _ = expanded_fixture(tmp_path, lease)
        receipt = prepare(expanded, predecessor)
        replace, release = os.replace, CacheResources.release_missing

        def interrupted_replace(*args, **kwargs):
            if point == "before_replace":
                raise RuntimeError("interrupted")
            replace(*args, **kwargs)
            if point == "after_replace":
                raise RuntimeError("interrupted")

        def interrupted_release(*args, **kwargs):
            if point == "before_release":
                raise RuntimeError("interrupted")
            return release(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", interrupted_replace)
            patch.setattr(CacheResources, "release_missing", interrupted_release)
            with pytest.raises(RuntimeError):
                migration.apply_expansion_64(
                    lease, expanded.identity, predecessor, receipt
                )
    with CacheLease(tmp_path) as lease:
        resources = migration.apply_expansion_64(
            lease, "expansion-64-test", predecessor, receipt
        )
        assert resources.audit()["reserved_bytes"] == 0


def test_expansion_64_rejects_changed_baseline_without_mutating_marker(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_expansion_64 import apply_expansion_64

    with CacheLease(tmp_path) as lease:
        expanded, predecessor, history = expanded_fixture(tmp_path, lease)
        receipt = prepare(expanded, predecessor)
        PublishedArtifacts(expanded).publish("extra", {}, [history.artifacts[0].token])
        with pytest.raises(ValueError, match="baseline"):
            apply_expansion_64(lease, expanded.identity, predecessor, receipt)
        assert json.loads(expanded.marker.read_text())["limit_bytes"] == 16 * 1024**3


def test_expansion_64_rejects_unsupported_marker_pair(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_expansion_64 import apply_expansion_64

    with CacheLease(tmp_path) as lease:
        expanded, predecessor, _ = expanded_fixture(tmp_path, lease)
        receipt = prepare(expanded, predecessor)
        marker = json.loads(expanded.marker.read_text())
        marker["limit_bytes"] = 64 * 1024**3
        expanded.marker.write_text(json.dumps(marker, sort_keys=True))
        with pytest.raises(ValueError, match="metadata state"):
            apply_expansion_64(lease, expanded.identity, predecessor, receipt)


def test_interrupted_expansion_64_recovery_requires_both_exact_receipts(
    tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import derived_cache_expansion_64 as migration

    with CacheLease(tmp_path) as lease:
        expanded, predecessor, _ = expanded_fixture(tmp_path, lease)
        receipt = prepare(expanded, predecessor)
        with monkeypatch.context() as patch:
            patch.setattr(
                os,
                "replace",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("interrupted")
                ),
            )
            with pytest.raises(RuntimeError, match="interrupted"):
                migration.apply_expansion_64(
                    lease, expanded.identity, predecessor, receipt
                )
        changed_predecessor = deepcopy(predecessor)
        changed_predecessor["approval"] = "changed predecessor"
        with pytest.raises(ValueError):
            migration.apply_expansion_64(
                lease, expanded.identity, changed_predecessor, receipt
            )
        changed_receipt = deepcopy(receipt)
        changed_receipt["approval"] = "changed second receipt"
        with pytest.raises(ValueError):
            migration.apply_expansion_64(
                lease, expanded.identity, predecessor, changed_receipt
            )
        recovered = migration.apply_expansion_64(
            lease, expanded.identity, predecessor, receipt
        )
        assert recovered.limit == 64 * 1024**3


@pytest.mark.parametrize("when", ["before_apply", "after_apply"])
def test_expansion_64_rejects_changed_retained_payload(tmp_path, when):
    from arblab.hyperliquid_copy.derived_cache_expansion_64 import apply_expansion_64
    from arblab.hyperliquid_copy.derived_cache_policy_64 import open_expanded_64_cache

    with CacheLease(tmp_path) as lease:
        expanded, predecessor, history = expanded_fixture(tmp_path, lease)
        receipt = prepare(expanded, predecessor)
        if when == "after_apply":
            apply_expansion_64(lease, expanded.identity, predecessor, receipt)
        payload = tmp_path / history.artifacts[0].path
        payload.write_bytes(b"x" * payload.stat().st_size)
        with pytest.raises(ValueError, match="payload"):
            if when == "before_apply":
                apply_expansion_64(lease, expanded.identity, predecessor, receipt)
            else:
                open_expanded_64_cache(
                    lease, expanded.identity, predecessor, receipt
                )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_predecessor",
        "changed_predecessor",
        "changed_receipt",
        "old_reference",
        "unknown_key",
    ],
)
def test_registered_64_reference_rejects_incomplete_or_changed_chain(
    tmp_path, fault
):
    from arblab.hyperliquid_copy.derived_cache_expansion_64 import apply_expansion_64
    from arblab.hyperliquid_copy.qualified_registration import cache_reference
    from arblab.hyperliquid_copy.qualified_registered_activity import _open_resources

    with CacheLease(tmp_path) as lease:
        expanded, predecessor, _ = expanded_fixture(tmp_path, lease)
        receipt = prepare(expanded, predecessor)
        apply_expansion_64(lease, expanded.identity, predecessor, receipt)
        reference = {
            "path": str(tmp_path),
            "identity": expanded.identity,
            "expansion_receipt": deepcopy(predecessor),
            "expansion_64gib_receipt": deepcopy(receipt),
        }
        if fault == "missing_predecessor":
            del reference["expansion_receipt"]
        elif fault == "changed_predecessor":
            reference["expansion_receipt"]["approval"] = "changed"
        elif fault == "changed_receipt":
            reference["expansion_64gib_receipt"]["approval"] = "changed"
        elif fault == "old_reference":
            del reference["expansion_64gib_receipt"]
        else:
            reference["limit_bytes"] = 64 * 1024**3
        with pytest.raises(ValueError):
            _open_resources(lease, cache_reference(reference))
