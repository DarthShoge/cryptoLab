from copy import deepcopy
import json

import pytest

from arblab.hyperliquid_copy.derived_cache_expansion import (
    prepare_expansion,
    apply_expansion,
)
from arblab.hyperliquid_copy.derived_cache_lease import CacheLease, CacheBusyError
from arblab.hyperliquid_copy.derived_cache_policy import open_expanded_cache
from arblab.hyperliquid_copy.lab_config import day
from .test_qualified_registration import registered_source, resources
from .test_qualified_registered_activity import load


@pytest.fixture
def expanded_source(registered_source, resources):
    manifest, config = registered_source
    receipt = prepare_expansion(
        resources,
        approved_from=8 * 1024**3,
        approved_to=16 * 1024**3,
        approval="explicit test approval",
    )
    expanded = apply_expansion(resources.lease, resources.identity, receipt)
    manifest.metadata["derived_cache"]["expansion_receipt"] = deepcopy(receipt)
    (manifest.directory / "manifest.json").write_text(json.dumps(manifest.metadata))
    return manifest, config, expanded, receipt


def test_expanded_owned_loader_ranks_and_reuses_same_cache(
    expanded_source, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as producer
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    manifest, config, resources, receipt = expanded_source
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)
    monkeypatch.setattr(
        ProxyActivity, "__init__", lambda *a, **k: pytest.fail("raw reader opened")
    )
    at = day(config.start)
    effective = config.effective(["BTC"], {"BTC": 1})
    with load(manifest, config) as activity:
        activity.prepare(at)
        expected = activity.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
        assert expected.candidate_count > 0
    with CacheLease(root) as lease:
        before = open_expanded_cache(lease, identity, receipt).audit()
    monkeypatch.setattr(
        producer.OrderedFeaturePartitions,
        "plan",
        lambda *a, **k: pytest.fail("ranking reuse repeated sorting"),
    )
    with load(manifest, config) as activity:
        activity.prepare(at)
        assert (
            activity.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
            == expected
        )
    with CacheLease(root) as lease:
        assert open_expanded_cache(lease, identity, receipt).audit() == before


def test_nested_reference_is_a_snapshot(expanded_source):
    from arblab.hyperliquid_copy.qualified_registration import cache_reference

    manifest, _, _, receipt = expanded_source
    reference = manifest.metadata["derived_cache"]
    checked = cache_reference(reference)
    checked["expansion_receipt"]["old_marker"]["nonce"] = "a" * 32
    assert reference["expansion_receipt"] == receipt


def test_nested_caller_mutation_during_precheck_rejected(expanded_source, monkeypatch):
    from arblab.hyperliquid_copy import qualified_registration as module

    manifest, _, _, _ = expanded_source
    reference = manifest.metadata["derived_cache"]
    original = module.expansion_engine

    def changed():
        engine = original()
        reference["expansion_receipt"]["approval"] = "changed by caller"
        return engine

    monkeypatch.setattr(module, "expansion_engine", changed)
    with pytest.raises(ValueError, match="changed"):
        module.cache_reference(reference)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "none",
        "wrong_nonce",
        "wrong_engine",
        "bool_schema",
        "larger_limit",
        "unknown_key",
    ],
)
def test_bad_expanded_reference_rejected_without_mutation(expanded_source, fault):
    manifest, config, resources, receipt = expanded_source
    before = resources.audit()
    reference = manifest.metadata["derived_cache"]
    if fault == "missing":
        del reference["expansion_receipt"]
    elif fault == "none":
        reference["expansion_receipt"] = None
    elif fault == "wrong_nonce":
        reference["expansion_receipt"]["old_marker"]["nonce"] = "f" * 32
    elif fault == "wrong_engine":
        reference["expansion_receipt"]["engine"]["derived_cache_policy"] = "a" * 64
    elif fault == "bool_schema":
        reference["expansion_receipt"]["schema"] = True
    elif fault == "larger_limit":
        reference["expansion_receipt"]["new_limit"] *= 2
    else:
        reference["limit_bytes"] = 32 * 1024**3
    resources.lease.__exit__(None, None, None)
    with pytest.raises(ValueError):
        load(manifest, config)
    with CacheLease(resources.root) as lease:
        assert open_expanded_cache(lease, resources.identity, receipt).audit() == before


def test_expanded_busy_cache_is_not_replaced(expanded_source):
    manifest, config, resources, _ = expanded_source
    before = resources.audit()
    with pytest.raises(CacheBusyError):
        load(manifest, config)
    assert resources.audit() == before


def test_expanded_constructor_failure_releases_lease(expanded_source, monkeypatch):
    from arblab.hyperliquid_copy.qualified_scheduled_activity import (
        QualifiedScheduledActivity,
    )

    manifest, config, resources, receipt = expanded_source
    before = resources.audit()
    resources.lease.__exit__(None, None, None)

    def fail(*args, **kwargs):
        raise RuntimeError("injected constructor failure")

    monkeypatch.setattr(QualifiedScheduledActivity, "__init__", fail)
    with pytest.raises(RuntimeError, match="injected"):
        load(manifest, config)
    with CacheLease(resources.root) as lease:
        assert open_expanded_cache(lease, resources.identity, receipt).audit() == before


@pytest.mark.parametrize("when", ["before_load", "before_close"])
def test_corrupt_receipt_never_repaired_and_releases_lease(expanded_source, when):
    from arblab.hyperliquid_copy.derived_cache_policy import _receipt

    manifest, config, resources, receipt = expanded_source
    publication, _ = _receipt(resources, receipt)
    path = resources.root / publication.artifacts[0].path
    with resources._connect() as db:
        before = db.execute("SELECT * FROM allocations ORDER BY token").fetchall()
    resources.lease.__exit__(None, None, None)
    activity = load(manifest, config) if when == "before_close" else None
    corrupted = b"x" * path.stat().st_size
    path.write_bytes(corrupted)
    with pytest.raises(ValueError):
        load(manifest, config) if activity is None else activity.close()
    if activity is not None:
        assert activity.closed
    with CacheLease(resources.root):
        assert path.read_bytes() == corrupted
        # The intentionally corrupt payload prevents audit; compare ledger rows
        # directly to prove the loader did not delete/refund/repair anything.
        import sqlite3

        with sqlite3.connect(resources.path) as db:
            assert (
                db.execute("SELECT * FROM allocations ORDER BY token").fetchall()
                == before
            )


def test_prepared_but_unapplied_receipt_does_not_migrate_on_load(
    registered_source, resources
):
    manifest, config = registered_source
    receipt = prepare_expansion(
        resources,
        approved_from=8 * 1024**3,
        approved_to=16 * 1024**3,
        approval="test approval",
    )
    manifest.metadata["derived_cache"]["expansion_receipt"] = receipt
    (manifest.directory / "manifest.json").write_text(json.dumps(manifest.metadata))
    before = resources.audit()
    resources.lease.__exit__(None, None, None)
    with pytest.raises(ValueError):
        load(manifest, config)
    with CacheLease(resources.root) as lease:
        from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

        assert CacheResources(lease, resources.identity).audit() == before


@pytest.mark.parametrize("fault", ["caller", "engine"])
def test_change_during_final_receipt_audit_rejected_and_lease_released(
    expanded_source, monkeypatch, fault
):
    from arblab.hyperliquid_copy import qualified_registered_activity as module

    manifest, config, resources, receipt = expanded_source
    resources.lease.__exit__(None, None, None)
    activity = load(manifest, config)
    original = module._open_resources

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "caller":
            manifest.metadata["derived_cache"]["expansion_receipt"]["approval"] = (
                "changed during final audit"
            )
        else:
            monkeypatch.setattr(module, "_engine", lambda: ("different code",))
        return result

    monkeypatch.setattr(module, "_open_resources", changed)
    with pytest.raises(ValueError, match="context changed"):
        activity.close()
    assert activity.closed
    with CacheLease(resources.root) as lease:
        assert (
            open_expanded_cache(lease, resources.identity, receipt).audit()[
                "reserved_bytes"
            ]
            == 0
        )
