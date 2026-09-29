import copy
import json
import uuid

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from .test_derived_cache_expansion import fixture_cache, prepare


@pytest.mark.parametrize(
    "fault",
    [
        "nonce",
        "root",
        "identity",
        "engine",
        "version_bool",
        "limit_bool",
        "limit_large",
        "approval",
        "receipt",
    ],
)
def test_open_rejects_changed_pins(tmp_path, fault):
    from arblab.hyperliquid_copy.derived_cache_expansion import apply_expansion
    from arblab.hyperliquid_copy.derived_cache_policy import (
        open_expanded_cache,
        _receipt,
    )

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
        resources = apply_expansion(lease, "expansion-test", inputs)
        changed = copy.deepcopy(inputs)
        if fault == "nonce":
            changed["old_marker"]["nonce"] = "a" * 32
        elif fault == "root":
            changed["old_marker"]["root"][1] += 1
        elif fault == "identity":
            changed["old_marker"]["identity"] = "other"
        elif fault == "engine":
            changed["engine"]["derived_cache_policy"] = "a" * 64
        elif fault == "version_bool":
            changed["schema"] = True
        elif fault == "limit_bool":
            changed["new_limit"] = True
        elif fault == "limit_large":
            changed["new_limit"] *= 2
        elif fault == "approval":
            changed["approval"] = ""
        else:
            pub, _ = _receipt(resources, inputs)
            path = tmp_path / pub.artifacts[0].path
            path.write_bytes(b"x" * path.stat().st_size)
        with pytest.raises(ValueError):
            open_expanded_cache(lease, "expansion-test", changed)


def test_expired_lease_rejected(tmp_path):
    from arblab.hyperliquid_copy.derived_cache_expansion import apply_expansion
    from arblab.hyperliquid_copy.derived_cache_policy import open_expanded_cache

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
        apply_expansion(lease, "expansion-test", inputs)
    with pytest.raises(ValueError, match="lease"):
        open_expanded_cache(lease, "expansion-test", inputs)


def test_caller_mutation_while_opening_rejected(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.derived_cache_expansion import apply_expansion
    from arblab.hyperliquid_copy import derived_cache_policy as policy

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
        apply_expansion(lease, "expansion-test", inputs)
        original = policy._receipt

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            inputs["approval"] = "substituted approval"
            return result

        monkeypatch.setattr(policy, "_receipt", changed)
        with pytest.raises(ValueError, match="Changed"):
            policy.open_expanded_cache(lease, "expansion-test", inputs)


def test_one_shared_envelope_accounts_retained_and_pending(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.derived_cache_expansion import apply_expansion
    from arblab.hyperliquid_copy import derived_cache_resources as core

    with CacheLease(tmp_path) as lease:
        old, _ = fixture_cache(tmp_path, lease)
        inputs = prepare(old)
        resources = apply_expansion(lease, "expansion-test", inputs)
        # Avoid depending on CI disk capacity; no payload bytes are written here.
        monkeypatch.setattr(
            core.shutil,
            "disk_usage",
            lambda _: type("Space", (), {"free": 64 * 1024**3})(),
        )
        totals = resources.audit()
        remaining = resources.limit - totals["total_bytes"]
        resources.reserve(f"scratch/{uuid.uuid4().hex}", remaining, "scratch")
        assert resources.audit()["total_bytes"] == 16 * 1024**3
        with pytest.raises(ValueError, match="budget"):
            resources.reserve(f"scratch/{uuid.uuid4().hex}", 1, "scratch")
        assert core.MAX_BYTES == 8 * 1024**3
