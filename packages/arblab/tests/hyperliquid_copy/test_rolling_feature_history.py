from datetime import timedelta

import pytest

from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS
from .test_qualified_day import qualified


def controller(resources, pin):
    from arblab.hyperliquid_copy.rolling_feature_history import RollingFeatureHistory

    return RollingFeatureHistory(resources, pin, ["BTC"], SEMANTICS)


def test_replacement_anchor_must_reopen_before_retirement(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import rolling_feature_history as module

    rolling = controller(resources, qualified)
    original = module.FeatureHistory

    def fail_reopen(*args, **kwargs):
        if kwargs.get("anchor_inputs") is not None:
            raise RuntimeError("replacement anchor reopen failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "FeatureHistory", fail_reopen)
    with pytest.raises(RuntimeError, match="replacement anchor reopen failed"):
        rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    assert rolling.anchor_inputs is None
    with resources._connect() as db:
        descriptors = [
            row[0] for row in db.execute("SELECT descriptor FROM publications")
        ]
    import json

    kinds = [json.loads(value)["kind"] for value in descriptors]
    assert kinds.count("qualified_features_day") == 2
    assert not any(kind.startswith("cache_retirement") for kind in kinds)


def test_advance_retires_only_old_day_and_reuses_current_window(resources, qualified):
    from arblab.hyperliquid_copy.feature_resume_anchor import FeatureAnchor

    rolling = controller(resources, qualified)
    first = rolling.window(DAY, DAY + timedelta(days=2))
    expected = list(first.days[1].checkpoints())
    second = rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    assert second.days == first.days[1:]
    assert list(second.days[0].checkpoints()) == expected
    assert all(
        not (resources.root / p.path).exists()
        for p in first.days[0].publication.artifacts
    )
    anchor = FeatureAnchor(resources, qualified, rolling.anchor_inputs)
    assert anchor.first == DAY + timedelta(days=1)
    assert anchor.inputs["origin"] == DAY.date().isoformat()
    before = resources.audit()
    assert rolling.window(second.start, second.end).inputs() == second.inputs()
    assert resources.audit() == before


def test_new_lease_builds_only_missing_day_after_retirement(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

    rolling = controller(resources, qualified)
    first = rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    expected = list(first.days[0].checkpoints())
    original = feature_history.build_feature_day
    calls = []

    def build(*args, **kwargs):
        calls.append(args[2])
        assert args[2] == DAY + timedelta(days=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_history, "build_feature_day", build)
    resources.lease.__exit__(None)
    with CacheLease(resources.root) as lease:
        current = CacheResources(lease, resources.identity)
        resumed = controller(current, qualified)
        result = resumed.window(DAY + timedelta(days=1), DAY + timedelta(days=3))
        assert calls == [DAY + timedelta(days=2)]
        assert list(result.days[0].checkpoints()) == expected
        assert list(result.days[-1].checkpoints())  # Dormant episodes survive.
        assert resumed.anchor_inputs["origin"] == DAY.date().isoformat()
        assert current.audit()["reserved_bytes"] == 0


@pytest.mark.parametrize("stage", ["prepared", "detached", "partial"])
def test_explicit_retry_recovers_before_new_publications(
    resources, qualified, monkeypatch, stage
):
    from arblab.hyperliquid_copy import rolling_feature_history as module
    from arblab.hyperliquid_copy import cache_retirement as transaction

    rolling = controller(resources, qualified)
    first = rolling.window(DAY, DAY + timedelta(days=2))
    if stage == "prepared":
        name, owner = "begin_retirement", module
    elif stage == "detached":
        name, owner = "finish_retirement", module
    else:
        name, owner = "_unlink_owned", transaction
    original = getattr(owner, name)

    def interrupted(*args, **kwargs):
        if stage == "partial":
            original(*args, **kwargs)
        raise RuntimeError("fixture interrupted retirement")

    monkeypatch.setattr(owner, name, interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    monkeypatch.setattr(owner, name, original)
    assert rolling.anchor_inputs["first"] == DAY.isoformat()
    # A fresh controller must discover the published replacement anchor, recover
    # its pending operation, then build day three. New writes before recovery
    # would invalidate the prepared operation's pinned catalogue baseline.
    resumed = controller(resources, qualified)
    result = resumed.window(DAY + timedelta(days=1), DAY + timedelta(days=3))
    assert len(result.days) == 2
    assert all(
        not (resources.root / p.path).exists()
        for p in first.days[0].publication.artifacts
    )
    assert resources.audit()["reserved_bytes"] == 0


def test_reversed_window_rejects_before_mutation(resources, qualified):
    rolling = controller(resources, qualified)
    rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=3))
    before = resources.audit()
    with pytest.raises(ValueError, match="reversed|lookback"):
        rolling.window(DAY, DAY + timedelta(days=3))
    assert resources.audit() == before


def test_constructor_rejects_pin_change_during_initial_engine_hash(
    resources, qualified, monkeypatch, tmp_path
):
    import shutil
    from arblab.hyperliquid_copy import rolling_feature_history as module

    alias = tmp_path / "other-report" / "manifest.json"
    alias.parent.mkdir()
    shutil.copyfile(qualified["path"], alias)
    original = module._engine

    def changed():
        result = original()
        qualified["path"] = str(alias)
        return result

    monkeypatch.setattr(module, "_engine", changed)
    before = resources.audit()
    with pytest.raises(ValueError, match="context|caller"):
        controller(resources, qualified)
    assert resources.audit() == before


def test_scalar_context_change_during_target_hash_prevents_retirement(
    resources, qualified, monkeypatch
):
    from pathlib import Path
    from arblab.hyperliquid_copy import cache_retirement as transaction

    rolling = controller(resources, qualified)
    first = rolling.window(DAY, DAY + timedelta(days=2))
    paths = {resources.root / p.path for p in first.days[0].publication.artifacts}
    original = transaction.file_hash
    changed = False

    def corrupt(path):
        nonlocal changed
        result = original(path)
        if Path(path) in paths and not changed:
            changed = True
            rolling._semantics = "net_includes_fee"
        return result

    monkeypatch.setattr(transaction, "file_hash", corrupt)
    with pytest.raises(ValueError, match="context"):
        rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    assert changed
    assert all(path.exists() for path in paths)
    assert rolling.anchor_inputs["first"] == DAY.isoformat()


def test_old_source_change_after_target_discovery_is_not_adopted(
    resources, qualified, monkeypatch
):
    import json
    from pathlib import Path
    from arblab.hyperliquid_copy import rolling_feature_history as module

    rolling = controller(resources, qualified)
    first = rolling.window(DAY, DAY + timedelta(days=2))
    original = module.expired_feature_days

    def changed(resources, pin, anchor, **kwargs):
        result = original(resources, pin, anchor, **kwargs)
        active = {entry.path for entry in anchor._window.source.entries}
        path = next(
            Path(entry["path"])
            for entry in json.loads(Path(pin["path"]).read_text())["files"]
            if Path(entry["path"]) not in active
        )
        with path.open("ab") as handle:
            handle.write(b"changed after target discovery")
        return result

    monkeypatch.setattr(module, "expired_feature_days", changed)
    with pytest.raises(ValueError, match="source|identity|changed"):
        rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    assert all(
        (resources.root / p.path).exists() for p in first.days[0].publication.artifacts
    )
    assert rolling.anchor_inputs["first"] == DAY.isoformat()


@pytest.mark.parametrize("payload", ["day", "anchor"])
def test_late_check_corruption_does_not_commit_cursor(
    resources, qualified, monkeypatch, payload
):
    from arblab.hyperliquid_copy.feature_resume_anchor import FeatureAnchor

    rolling = controller(resources, qualified)
    rolling.window(DAY, DAY + timedelta(days=2))
    original_advance = rolling._advance
    original_check = rolling._check
    armed = False
    changed = False

    def complete(*args):
        nonlocal armed
        result = original_advance(*args)
        armed = True
        return result

    def corrupt():
        nonlocal changed
        original_check()
        if armed and not changed:
            changed = True
            previous = FeatureAnchor(resources, qualified, rolling.anchor_inputs)
            pin = (
                previous.days[-1].publication.artifacts[0]
                if payload == "day"
                else previous.publication.artifacts[0]
            )
            with (resources.root / pin.path).open("ab") as handle:
                handle.write(b"late context corruption")

    monkeypatch.setattr(rolling, "_advance", complete)
    monkeypatch.setattr(rolling, "_check", corrupt)
    # Identical window keeps the same anchor, letting the fault target the actual
    # evidence about to be returned rather than a legitimately expired receipt.
    with pytest.raises(ValueError):
        rolling.window(DAY, DAY + timedelta(days=2))
    assert changed
    assert rolling._last_start == DAY and rolling._last_end == DAY + timedelta(days=2)


@pytest.mark.parametrize("compatible", [False, True])
def test_foreign_pending_owner_blocks_before_new_publication(
    resources, qualified, compatible
):
    from arblab.hyperliquid_copy.cache_retirement import prepare_retirement
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from .test_feature_history import history

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days
    if compatible:
        publish_feature_anchor(resources, qualified, days[1:2], ["BTC"], SEMANTICS)
    owner = publish_feature_anchor(resources, qualified, days[2:], ["BTC"], SEMANTICS)
    prepare_retirement(
        resources,
        "qualified_features_day",
        days[0].inputs,
        owner=owner.publication.key,
        reason="Fixture unresolved owner",
    )
    rolling = controller(resources, qualified)
    before = resources.audit()
    start = DAY + timedelta(days=1) if compatible else DAY
    with pytest.raises(ValueError, match="owner"):
        rolling.window(start, DAY + timedelta(days=2))
    assert resources.audit() == before
    assert rolling.anchor_inputs is None and rolling._last_end is None


def test_future_days_and_other_fee_semantics_are_preserved(resources, qualified):
    from arblab.hyperliquid_copy.feature_history import FeatureHistory
    from .test_feature_history import history

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days
    other = FeatureHistory(resources, qualified, ["BTC"], "net_includes_fee").window(
        DAY, DAY + timedelta(days=1)
    )
    rolling = controller(resources, qualified)
    rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    days[2].verify()
    other.verify()
    assert all(
        not (resources.root / p.path).exists() for p in days[0].publication.artifacts
    )


def test_intraday_window_keeps_exact_interval_and_detached_anchor_inputs(
    resources, qualified
):
    rolling = controller(resources, qualified)
    start, end = DAY + timedelta(days=1, hours=3), DAY + timedelta(days=1, hours=4)
    result = rolling.window(start, end)
    assert result.start == start and result.end == end
    assert len(result.days) == 1
    inputs = rolling.anchor_inputs
    inputs["coins"].append("ETH")
    assert rolling.anchor_inputs["coins"] == ["BTC"]
    before = resources.audit()
    assert rolling.window(start, end).inputs() == result.inputs()
    assert resources.audit() == before


def test_anchor_budget_failure_preserves_all_existing_days(resources, qualified):
    from arblab.hyperliquid_copy.feature_resume_anchor import MAX_BYTES
    from .test_feature_history import history

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=2)).days
    remaining = resources.limit - resources.audit()["total_bytes"]
    resources.reserve("staging/" + "e" * 32, remaining - MAX_BYTES + 1, "payload")
    rolling = controller(resources, qualified)
    before = resources.audit()
    with pytest.raises(ValueError, match="budget"):
        rolling.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    assert resources.audit() == before
    assert rolling.anchor_inputs is None
    for day in days:
        day.verify()


def test_reentrant_window_fails_and_releases_busy_flag(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import rolling_feature_history as module

    rolling = controller(resources, qualified)
    original = module.select_anchor

    def nested(*args, **kwargs):
        rolling.window(DAY, DAY + timedelta(days=1))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "select_anchor", nested)
    before = resources.audit()
    with pytest.raises(ValueError, match="busy"):
        rolling.window(DAY, DAY + timedelta(days=1))
    assert not rolling._busy and resources.audit() == before


def test_expired_lease_rejects_even_same_window(resources, qualified):
    rolling = controller(resources, qualified)
    rolling.window(DAY, DAY + timedelta(days=1))
    resources.lease.__exit__(None)
    with pytest.raises(ValueError, match="lease"):
        rolling.window(DAY, DAY + timedelta(days=1))
