from datetime import timedelta

import pytest

from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS
from .test_feature_history import history
from .test_qualified_day import qualified


@pytest.fixture
def rolling(resources, qualified):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days
    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    return resources, qualified, days, anchor


def intent(rolling, *, target=None, owner=None):
    from arblab.hyperliquid_copy.cache_retirement import prepare_retirement

    resources, _, days, anchor = rolling
    target = days[0] if target is None else target
    return prepare_retirement(
        resources,
        "qualified_features_day",
        target.inputs,
        owner=anchor.publication.key if owner is None else owner,
        reason="Fixture rolling feature retirement",
        protected_keys=[anchor.publication.key],
    )


@pytest.mark.parametrize("stage", ["prepared", "detached", "partial", "complete"])
def test_fresh_lease_recovery_preserves_anchor_and_source(rolling, monkeypatch, stage):
    from arblab.hyperliquid_copy import cache_retirement as transaction
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.feature_resume_anchor import FeatureAnchor
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        inspect_feature_retirements,
        recover_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    inputs = intent(rolling)
    original_unlink = transaction._unlink_owned
    if stage != "prepared":
        transaction.begin_retirement(resources, inputs)
    if stage == "complete":
        transaction.finish_retirement(resources, inputs)
    elif stage == "partial":

        def interrupted(*args):
            original_unlink(*args)
            raise RuntimeError("fixture after one unlink")

        monkeypatch.setattr(transaction, "_unlink_owned", interrupted)
        with pytest.raises(RuntimeError, match="after one unlink"):
            transaction.finish_retirement(resources, inputs)
        monkeypatch.setattr(transaction, "_unlink_owned", original_unlink)
    anchor_inputs = anchor.inputs
    source_paths = [entry.path for entry in anchor._window.source.entries]
    expected = [list(day.checkpoints()) for day in anchor.days]
    resources.lease.__exit__(None)
    with CacheLease(resources.root) as lease:
        current = CacheResources(lease, resources.identity)
        before = current.audit()
        records = inspect_feature_retirements(current, pin, anchor_inputs)
        assert len(records) == 1
        assert records[0].state == ("detached" if stage == "partial" else stage)
        assert current.audit() == before
        recovered = recover_feature_retirements(current, pin, anchor_inputs)
        assert recovered == (days[0].publication.key,)
        assert all(
            not (current.root / p.path).exists() for p in days[0].publication.artifacts
        )
        assert all(path.exists() for path in source_paths)
        reopened = FeatureAnchor(current, pin, anchor_inputs)
        assert [list(day.checkpoints()) for day in reopened.days] == expected
        assert current.audit()["reserved_bytes"] == 0
        after = current.audit()
        assert recover_feature_retirements(current, pin, anchor_inputs) == recovered
        assert current.audit() == after


def test_foreign_unresolved_owner_blocks_even_with_compatible_anchor(rolling):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    other = publish_feature_anchor(resources, pin, days[2:], ["BTC"], SEMANTICS)
    intent(rolling, owner=other.publication.key)
    before = resources.audit()
    with pytest.raises(ValueError, match="owner"):
        recover_feature_retirements(resources, pin, anchor.inputs)
    assert resources.audit() == before
    assert all(
        (resources.root / p.path).exists() for p in days[0].publication.artifacts
    )


def test_active_day_target_is_never_recovered(rolling):
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    intent(rolling, target=days[1])
    before = resources.audit()
    with pytest.raises(ValueError, match="boundary"):
        recover_feature_retirements(resources, pin, anchor.inputs)
    assert resources.audit() == before


def test_stale_prepared_baseline_is_not_replanned(rolling):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    intent(rolling)
    PublishedArtifacts(resources).publish(
        "fixture_later", {}, [days[1].publication.artifacts[0].token]
    )
    before = resources.audit()
    with pytest.raises(ValueError, match="catalogue changed"):
        recover_feature_retirements(resources, pin, anchor.inputs)
    assert resources.audit() == before


def test_duplicate_unresolved_intents_reject_without_mutation(rolling):
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, _, anchor = rolling
    intent(rolling)
    intent(rolling)
    before = resources.audit()
    with pytest.raises(ValueError, match="ambiguous|multiple"):
        recover_feature_retirements(resources, pin, anchor.inputs)
    assert resources.audit() == before


def test_republished_completed_target_is_not_adopted(rolling):
    from arblab.hyperliquid_copy.cache_retirement import (
        begin_retirement,
        finish_retirement,
    )
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    inputs = intent(rolling)
    begin_retirement(resources, inputs)
    finish_retirement(resources, inputs)
    PublishedArtifacts(resources).publish(
        "qualified_features_day",
        days[0].inputs,
        [days[1].publication.artifacts[0].token],
    )
    before = resources.audit()
    with pytest.raises(ValueError, match="republished"):
        recover_feature_retirements(resources, pin, anchor.inputs)
    assert resources.audit() == before


@pytest.mark.parametrize("stage", ["prepared", "detached"])
@pytest.mark.parametrize("fault", ["caller", "anchor_payload", "source"])
def test_context_change_during_target_hash_prevents_disposal(
    rolling, monkeypatch, stage, fault
):
    from pathlib import Path
    from arblab.hyperliquid_copy import cache_retirement as transaction
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    receipt = intent(rolling)
    if stage == "detached":
        transaction.begin_retirement(resources, receipt)
    inputs = anchor.inputs
    paths = {resources.root / p.path for p in days[0].publication.artifacts}
    original = transaction.file_hash
    changed = False

    def corrupt(path):
        nonlocal changed
        result = original(path)
        if Path(path) in paths and not changed:
            changed = True
            if fault == "caller":
                inputs["coins"].append("ETH")
            else:
                selected = (
                    resources.root / anchor.days[0].publication.artifacts[0].path
                    if fault == "anchor_payload"
                    else Path(pin["path"])
                )
                with selected.open("ab") as handle:
                    handle.write(b"corrupted before disposal")
        return result

    # The read-only inventory uses the same function. Arm only after it returns
    # to exercise the last destructive boundary, not an earlier rejection.
    from arblab.hyperliquid_copy import rolling_retirement_recovery as recovery

    inspect = recovery.inspect_owned

    def arm(*args):
        result = inspect(*args)
        monkeypatch.setattr(transaction, "file_hash", corrupt)
        return result

    monkeypatch.setattr(recovery, "inspect_owned", arm)
    with pytest.raises(ValueError):
        recover_feature_retirements(resources, pin, inputs)
    assert changed
    assert all(path.exists() for path in paths), "context changed before unlink"


@pytest.mark.parametrize(
    "fault", ["scope", "semantics", "report", "membership", "source_type"]
)
def test_incompatible_owner_target_metadata_is_rejected(rolling, fault):
    from types import SimpleNamespace
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    changed = days[0].inputs
    if fault == "scope":
        changed["coins"] = ["ETH"]
    elif fault == "semantics":
        changed["semantics"] = "net_includes_fee"
    elif fault == "report":
        changed["source"]["report_sha256"] = "0" * 64
    elif fault == "membership":
        changed["source"]["file_membership_sha256"] = "0" * 64
    else:
        changed["source"] = []
    PublishedArtifacts(resources).publish(
        "qualified_features_day",
        changed,
        [p.token for p in days[0].publication.artifacts],
    )
    intent(rolling, target=SimpleNamespace(inputs=changed))
    before = resources.audit()
    with pytest.raises(ValueError, match="context|source|membership"):
        recover_feature_retirements(resources, pin, anchor.inputs)
    assert resources.audit() == before


def test_completed_foreign_owner_is_not_reexecuted(rolling, monkeypatch):
    from arblab.hyperliquid_copy import rolling_retirement_recovery as module
    from arblab.hyperliquid_copy.cache_retirement import (
        begin_retirement,
        finish_retirement,
    )
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor

    resources, pin, days, anchor = rolling
    other = publish_feature_anchor(resources, pin, days[2:], ["BTC"], SEMANTICS)
    receipt = intent(rolling, owner=other.publication.key)
    begin_retirement(resources, receipt)
    finish_retirement(resources, receipt)
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("completed foreign operation was executed")

    monkeypatch.setattr(module, "begin_retirement", forbidden)
    monkeypatch.setattr(module, "finish_retirement", forbidden)
    assert module.recover_feature_retirements(resources, pin, anchor.inputs) == ()
    assert resources.audit() == before


@pytest.mark.parametrize("limit", ["MAX_OPERATIONS", "MAX_INPUT_BYTES"])
def test_inventory_limits_reject_without_mutation(rolling, monkeypatch, limit):
    from arblab.hyperliquid_copy import rolling_retirement_inventory as inventory
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, _, anchor = rolling
    intent(rolling)
    monkeypatch.setattr(inventory, limit, 0)
    before = resources.audit()
    with pytest.raises(ValueError, match="resource limit"):
        recover_feature_retirements(resources, pin, anchor.inputs)
    assert resources.audit() == before


def test_catalogue_mutation_during_inventory_is_rejected(rolling, monkeypatch):
    from arblab.hyperliquid_copy import rolling_retirement_inventory as inventory
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        inspect_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    intent(rolling)
    original = inventory.load_journal

    def changed(*args):
        result = original(*args)
        PublishedArtifacts(resources).publish(
            "fixture_concurrent", {}, [days[1].publication.artifacts[0].token]
        )
        return result

    monkeypatch.setattr(inventory, "load_journal", changed)
    with pytest.raises(ValueError, match="changed|catalog"):
        inspect_feature_retirements(resources, pin, anchor.inputs)
    assert all(
        (resources.root / p.path).exists() for p in days[0].publication.artifacts
    )


def test_expired_lease_rejects_recovery(rolling):
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, _, anchor = rolling
    intent(rolling)
    resources.lease.__exit__(None)
    with pytest.raises(ValueError, match="lease"):
        recover_feature_retirements(resources, pin, anchor.inputs)


def test_canonical_hardlinks_are_preserved_during_recovery(rolling, tmp_path):
    import os
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        recover_feature_retirements,
    )

    resources, pin, days, anchor = rolling
    source = anchor._window.source.witness.path
    alias = tmp_path / "registered-canonical-link.parquet"
    os.link(source, alias)
    intent(rolling)
    assert recover_feature_retirements(resources, pin, anchor.inputs) == (
        days[0].publication.key,
    )
    assert source.stat().st_ino == alias.stat().st_ino
    assert source.stat().st_nlink == 2


@pytest.mark.parametrize("boundary", ["anchor_context", "transaction_context"])
def test_mutation_during_last_guard_hash_prevents_unlink(
    rolling, monkeypatch, boundary
):
    from pathlib import Path
    from arblab.hyperliquid_copy import cache_retirement as transaction
    from arblab.hyperliquid_copy import rolling_retirement_recovery as recovery

    resources, pin, days, anchor = rolling
    receipt = intent(rolling)
    transaction.begin_retirement(resources, receipt)
    target_paths = {resources.root / p.path for p in days[0].publication.artifacts}
    hashing_target = False
    original_hash = transaction.file_hash

    def arm(path):
        nonlocal hashing_target
        result = original_hash(path)
        if Path(path) in target_paths:
            hashing_target = True
        return result

    monkeypatch.setattr(transaction, "file_hash", arm)
    changed = False
    checks = 0
    if boundary == "anchor_context":
        original = recovery._engine
    else:
        original = transaction.context

    def late_change(*args):
        nonlocal changed, checks
        result = original(*args)
        if hashing_target:
            checks += 1
        if (
            hashing_target
            and not changed
            and checks == (2 if boundary == "anchor_context" else 1)
        ):
            changed = True
            with Path(pin["path"]).open("ab") as handle:
                handle.write(b"late guard corruption")
        return result

    monkeypatch.setattr(
        recovery if boundary == "anchor_context" else transaction,
        "_engine" if boundary == "anchor_context" else "context",
        late_change,
    )
    with pytest.raises(ValueError):
        recovery.recover_feature_retirements(resources, pin, anchor.inputs)
    assert changed
    assert all(path.exists() for path in target_paths)


def test_unrelated_owner_journal_is_filtered_before_loading(rolling, monkeypatch):
    from arblab.hyperliquid_copy import rolling_retirement_inventory as inventory
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.feature_resume_anchor import KIND
    from arblab.hyperliquid_copy.rolling_retirement_recovery import (
        inspect_feature_retirements,
    )
    from arblab.hyperliquid_copy.cache_retirement_journal import INTENT

    resources, pin, days, anchor = rolling
    # Authenticated metadata for another source. Its old receipt engine and
    # payload must not be interpreted as this source's recovery operation.
    inputs = anchor.inputs
    inputs["report_sha256"] = "b" * 64
    other = PublishedArtifacts(resources).publish(
        KIND, inputs, [anchor.publication.artifacts[0].token]
    )
    receipt = dict(
        schema=1,
        owner=other.key,
        target="c" * 64,
        journal_sha256=anchor.publication.artifacts[0].sha256,
        engine={"old": 1},
    )
    PublishedArtifacts(resources).publish(
        INTENT, receipt, [anchor.publication.artifacts[0].token]
    )

    def forbidden(*args):
        pytest.fail("unrelated journal payload was opened")

    monkeypatch.setattr(inventory, "load_journal", forbidden)
    before = resources.audit()
    assert inspect_feature_retirements(resources, pin, anchor.inputs) == ()
    assert resources.audit() == before


@pytest.mark.parametrize("operation", ["inspect", "recover"])
def test_old_source_change_during_inventory_is_not_adopted(
    rolling, monkeypatch, operation
):
    import json
    from pathlib import Path
    from arblab.hyperliquid_copy import rolling_retirement_inventory as inventory
    from arblab.hyperliquid_copy import rolling_retirement_recovery as recovery

    resources, pin, days, anchor = rolling
    intent(rolling)
    active_paths = {entry.path for entry in anchor._window.source.entries}
    source = next(
        Path(entry["path"])
        for entry in json.loads(Path(pin["path"]).read_text())["files"]
        if Path(entry["path"]) not in active_paths
    )
    original = inventory._target_binding

    def corrupt(*args):
        result = original(*args)
        with source.open("ab") as handle:
            handle.write(b"changed after source verification")
        return result

    monkeypatch.setattr(inventory, "_target_binding", corrupt)
    function = (
        recovery.inspect_feature_retirements
        if operation == "inspect"
        else recovery.recover_feature_retirements
    )
    with pytest.raises(ValueError, match="identity|source|changed"):
        function(resources, pin, anchor.inputs)
    assert all(
        (resources.root / p.path).exists() for p in days[0].publication.artifacts
    )
