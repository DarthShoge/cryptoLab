import json

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheBusyError, CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.lab_config import day
from .test_qualified_registration import registered_source, resources


def load(manifest, config):
    from arblab.hyperliquid_copy.qualified_registered_activity import (
        load_qualified_activity,
    )

    return load_qualified_activity(manifest, config)


def test_explicit_rolling_policy_reaches_owned_reader(registered_source, resources):
    from arblab.hyperliquid_copy.rolling_feature_history import RollingFeatureHistory

    manifest, config = registered_source
    manifest.metadata["feature_history_policy"] = "rolling_feature_anchor_v1"
    (manifest.directory / "manifest.json").write_text(json.dumps(manifest.metadata))
    resources.lease.__exit__(None, None, None)
    with load(manifest, config) as owned:
        at = day(config.start)
        owned.prepare(at)
        owned.rank(
            at,
            config.effective(["BTC"], {"BTC": 1}),
            "BTC",
            "gross_excludes_fee",
            smoke=True,
        )
        assert isinstance(owned._activity._features, RollingFeatureHistory)


def test_manifest_policy_change_prevents_physical_unlink(
    registered_source, resources, monkeypatch
):
    from arblab.hyperliquid_copy import cache_retirement as transaction

    manifest, config = registered_source
    manifest.metadata["feature_history_policy"] = "rolling_feature_anchor_v1"
    path = manifest.directory / "manifest.json"
    path.write_text(json.dumps(manifest.metadata))
    resources.lease.__exit__(None, None, None)
    owned = load(manifest, config)
    at = day(config.start)
    owned.prepare(at)
    original = transaction._unlink_owned
    attempted = []

    def changed(resources, directory_fd, row, identity, guard):
        attempted.append(resources._path(row[1]))
        metadata = json.loads(path.read_text())
        metadata.pop("feature_history_policy", None)
        path.write_text(json.dumps(metadata))
        return original(resources, directory_fd, row, identity, guard)

    monkeypatch.setattr(transaction, "_unlink_owned", changed)
    try:
        with pytest.raises(ValueError, match="changed"):
            owned.rank(
                at,
                config.effective(["BTC"], {"BTC": 1}),
                "BTC",
                "gross_excludes_fee",
                smoke=True,
            )
        assert attempted and all(item.exists() for item in attempted)
    finally:
        with pytest.raises(ValueError):
            owned.close()


@pytest.mark.parametrize("fault", ["disk", "metadata", "facade"])
def test_last_registration_hash_cannot_revoke_policy_before_unlink(
    registered_source, resources, monkeypatch, fault
):
    from arblab.hyperliquid_copy import cache_retirement as transaction
    from arblab.hyperliquid_copy import qualified_registered_activity as module

    manifest, config = registered_source
    manifest.metadata["feature_history_policy"] = "rolling_feature_anchor_v1"
    path = manifest.directory / "manifest.json"
    path.write_text(json.dumps(manifest.metadata))
    resources.lease.__exit__(None, None, None)
    owned = load(manifest, config)
    at = day(config.start)
    owned.prepare(at)
    original_unlink, original_engine = transaction._unlink_owned, module._engine
    attempted = []
    changed = False
    calls, final_call, counting = 0, None, False

    def arm(resources, directory_fd, row, identity, guard):
        nonlocal calls, final_call, counting
        attempted.append(resources._path(row[1]))
        counting = True
        guard()
        final_call, calls, counting = calls, 0, False
        return original_unlink(resources, directory_fd, row, identity, guard)

    def engine():
        nonlocal changed, calls
        result = original_engine()
        if attempted:
            calls += 1
        if attempted and not counting and calls == final_call and not changed:
            changed = True
            if fault == "disk":
                metadata = json.loads(path.read_text())
                metadata.pop("feature_history_policy")
                path.write_text(json.dumps(metadata))
            elif fault == "metadata":
                manifest.metadata.pop("feature_history_policy")
            else:
                owned._activity.feature_history_policy = None
        return result

    monkeypatch.setattr(transaction, "_unlink_owned", arm)
    monkeypatch.setattr(module, "_engine", engine)
    try:
        with pytest.raises(ValueError):
            owned.rank(
                at,
                config.effective(["BTC"], {"BTC": 1}),
                "BTC",
                "gross_excludes_fee",
                smoke=True,
            )
        assert changed and attempted and all(item.exists() for item in attempted)
    finally:
        with pytest.raises(ValueError):
            owned.close()


def test_busy_existing_cache_is_not_replaced(registered_source, resources):
    manifest, config = registered_source
    before = resources.audit()
    with pytest.raises(CacheBusyError):
        load(manifest, config)
    assert resources.audit() == before


def test_owned_loader_ranks_without_full_reader_and_reuses_same_cache(
    registered_source, resources, monkeypatch
):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from arblab.hyperliquid_copy import candidate_metric_producer as producer

    manifest, config = registered_source
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)
    monkeypatch.setattr(
        ProxyActivity, "__init__", lambda *a, **k: pytest.fail("full reader opened")
    )
    at = day(config.start)
    effective = config.effective(["BTC"], {"BTC": 1})
    activity = load(manifest, config)
    with pytest.raises(CacheBusyError):
        with CacheLease(root):
            pass
    activity.prepare(at)
    expected = activity.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
    assert expected.candidate_count > 0
    activity.close()
    activity.close()
    with pytest.raises(ValueError):
        activity.prepare(at)
    with CacheLease(root) as lease:
        before = CacheResources(lease, identity).audit()
    monkeypatch.setattr(
        producer,
        "merge_metric_rows",
        lambda *a, **k: pytest.fail("same decision recomputed metrics"),
    )
    reopened = load(manifest, config)
    reopened.prepare(at)
    assert (
        reopened.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
        == expected
    )
    reopened.close()
    with CacheLease(root) as lease:
        assert CacheResources(lease, identity).audit() == before


def test_constructor_failure_releases_only_owned_lease(
    registered_source, resources, monkeypatch
):
    from arblab.hyperliquid_copy.qualified_scheduled_activity import (
        QualifiedScheduledActivity,
    )

    manifest, config = registered_source
    root, identity, before = resources.root, resources.identity, resources.audit()
    resources.lease.__exit__(None, None, None)

    def fail(*a, **k):
        raise RuntimeError("constructor interrupted")

    monkeypatch.setattr(QualifiedScheduledActivity, "__init__", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        load(manifest, config)
    with CacheLease(root) as lease:
        assert CacheResources(lease, identity).audit() == before


def test_close_verification_failure_still_releases_lease(registered_source, resources):
    manifest, config = registered_source
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)
    activity = load(manifest, config)
    manifest.metadata["name"] = "changed during run"
    with pytest.raises(ValueError, match="changed"):
        activity.close()
    activity.close()
    with CacheLease(root) as lease:
        CacheResources(lease, identity).audit()


@pytest.mark.parametrize("when", ["before_load", "before_close"])
def test_disk_manifest_change_rejected_and_lease_released(
    registered_source, resources, when
):
    manifest, config = registered_source
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)
    activity = load(manifest, config) if when == "before_close" else None
    path = manifest.directory / "manifest.json"
    changed = json.loads(path.read_text())
    changed["name"] = "Different registration on disk"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed"):
        if activity is None:
            load(manifest, config)
        else:
            activity.close()
    if activity is not None:
        activity.close()
    with CacheLease(root) as lease:
        CacheResources(lease, identity).audit()


def test_reentrant_close_cannot_release_active_query_lease(
    registered_source, resources, monkeypatch
):
    from arblab.hyperliquid_copy import qualified_scheduled_activity as module

    manifest, config = registered_source
    root = resources.root
    resources.lease.__exit__(None, None, None)
    activity = load(manifest, config)
    original = module.observed_markets

    def observe(*args):
        with pytest.raises(ValueError, match="busy"):
            activity.close()
        with pytest.raises(CacheBusyError):
            with CacheLease(root):
                pass
        return original(*args)

    monkeypatch.setattr(module, "observed_markets", observe)
    at = day(config.start)
    activity.prepare(at)
    assert activity.observed(at) == ["BTC"]
    activity.close()
    with CacheLease(root):
        pass


def test_disk_manifest_change_during_final_scan_rejected(
    registered_source, resources, monkeypatch
):
    from arblab.hyperliquid_copy import qualified_registered_activity as module

    manifest, config = registered_source
    resources.lease.__exit__(None, None, None)
    activity = load(manifest, config)
    original = module.verify_qualified_registration

    def mutate_after_scan(value):
        result = original(value)
        path = value.directory / "manifest.json"
        data = json.loads(path.read_text())
        data["name"] = "Changed during final scan"
        path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(module, "verify_qualified_registration", mutate_after_scan)
    with pytest.raises(ValueError, match="changed"):
        activity.close()
    with CacheLease(resources.root) as lease:
        CacheResources(lease, resources.identity).audit()
