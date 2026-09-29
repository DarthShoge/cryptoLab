import hashlib
import json
import sqlite3

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from .test_derived_publication import artifact


@pytest.fixture
def cache(tmp_path):
    with CacheLease(tmp_path) as lease:
        resources = CacheResources.create(lease, "retirement-inventory")
        own = artifact(resources)
        shared = artifact(resources)
        publications = PublishedArtifacts(resources)
        target = publications.publish(
            "candidate_metrics", {"source": "fixture"}, [own[0], shared[0]]
        )
        other = publications.publish(
            "saved_feature_ranking", {"fixture": True}, [shared[0]]
        )
        yield resources, target, other


def test_inventory_preserves_shared_tokens_without_writes(cache):
    from arblab.hyperliquid_copy.cache_retirement_inventory import retirement_inventory

    resources, target, other = cache
    before = resources.audit()
    database = resources.path.read_bytes()
    result = retirement_inventory(
        resources,
        "candidate_metrics",
        {"source": "fixture"},
        protected_keys=[other.key],
    )
    assert result.target == target
    assert result.exclusive == (target.artifacts[0].token,)
    assert result.shared == (target.artifacts[1].token,)
    assert len(result.catalogue_digest) == 64
    assert resources.path.read_bytes() == database
    assert resources.audit() == before
    assert all((resources.root / pin.path).exists() for pin in target.artifacts)


@pytest.mark.parametrize("fault", ["protected", "missing", "forbidden", "expired"])
def test_inventory_rejects_invalid_target_without_writes(cache, fault):
    from arblab.hyperliquid_copy.cache_retirement_inventory import retirement_inventory

    resources, target, other = cache
    kind, inputs, protected = "candidate_metrics", {"source": "fixture"}, []
    if fault == "protected":
        protected = [target.key]
    elif fault == "missing":
        inputs = {"source": "missing"}
    elif fault == "forbidden":
        kind, inputs = "saved_feature_ranking", {"fixture": True}
    else:
        resources.lease.__exit__(None, None, None)
    database = resources.path.read_bytes()
    with pytest.raises(ValueError):
        retirement_inventory(resources, kind, inputs, protected_keys=protected)
    assert resources.path.read_bytes() == database


@pytest.mark.parametrize("fault", ["hash", "artifact", "allocation"])
def test_inventory_authenticates_unrelated_references(cache, fault):
    from arblab.hyperliquid_copy.cache_retirement_inventory import retirement_inventory
    from arblab.hyperliquid_copy.derived_publication import _encode

    resources, target, other = cache
    with resources._connect() as db, db:
        raw = db.execute(
            "SELECT descriptor FROM publications WHERE key=?", (other.key,)
        ).fetchone()[0]
        data = json.loads(raw)
        if fault == "hash":
            db.execute(
                "UPDATE publications SET sha256=? WHERE key=?", ("0" * 64, other.key)
            )
        elif fault == "artifact":
            data["artifacts"][0]["path"] = "../foreign"
            raw = _encode(data)
            db.execute(
                "UPDATE publications SET descriptor=?,sha256=? WHERE key=?",
                (raw.decode(), hashlib.sha256(raw).hexdigest(), other.key),
            )
        else:
            db.execute(
                "UPDATE allocations SET sha256=? WHERE token=?",
                ("0" * 64, other.artifacts[0].token),
            )
    database = resources.path.read_bytes()
    with pytest.raises(ValueError):
        retirement_inventory(resources, "candidate_metrics", {"source": "fixture"})
    assert resources.path.read_bytes() == database


def test_inventory_detects_reference_added_during_target_hash(cache, monkeypatch):
    from arblab.hyperliquid_copy import cache_retirement_inventory as module
    from arblab.hyperliquid_copy.derived_publication import _encode, publication_key

    resources, target, other = cache
    original = module.file_hash
    changed = False

    def concurrent_reference(path):
        nonlocal changed
        result = original(path)
        if path == resources.root / target.artifacts[0].path and not changed:
            changed = True
            with sqlite3.connect(resources.path) as db:
                data = json.loads(
                    db.execute(
                        "SELECT descriptor FROM publications WHERE key=?", (target.key,)
                    ).fetchone()[0]
                )
                data["kind"] = "concurrent_reference"
                raw = _encode(data)
                db.execute(
                    "INSERT INTO publications VALUES (?,?,?)",
                    (
                        publication_key(data["kind"], data["inputs"]),
                        raw.decode(),
                        hashlib.sha256(raw).hexdigest(),
                    ),
                )
        return result

    monkeypatch.setattr(module, "file_hash", concurrent_reference)
    with pytest.raises(ValueError):
        module.retirement_inventory(
            resources, "candidate_metrics", {"source": "fixture"}
        )
    assert changed


@pytest.mark.parametrize(
    "fault", ["input", "protected", "engine", "lease", "payload", "alias"]
)
def test_inventory_rejects_late_context_changes(cache, monkeypatch, fault):
    from arblab.hyperliquid_copy import cache_retirement_inventory as module
    import os

    resources, target, other = cache
    inputs, protected = {"source": "fixture"}, [other.key]
    path = resources.root / target.artifacts[0].path
    original = module.file_hash
    changed = False

    def mutate(at):
        nonlocal changed
        result = original(at)
        if at == path and not changed:
            changed = True
            if fault == "input":
                inputs["source"] = "changed"
            elif fault == "protected":
                protected.append(target.key)
            elif fault == "engine":
                monkeypatch.setattr(module, "_engine", lambda: {"changed": True})
            elif fault == "lease":
                resources.lease.__exit__(None, None, None)
            elif fault == "payload":
                path.write_bytes(b"changed data!!")
            else:
                os.link(path, resources.root.parent / (path.name + "-alias"))
        return result

    monkeypatch.setattr(module, "file_hash", mutate)
    with pytest.raises(ValueError):
        module.retirement_inventory(
            resources, "candidate_metrics", inputs, protected_keys=protected
        )
    assert changed


def test_inventory_rejects_oversized_metadata_before_decode(cache, monkeypatch):
    from arblab.hyperliquid_copy import cache_retirement_inventory as module

    resources, target, other = cache
    with resources._connect() as db, db:
        db.execute(
            "UPDATE publications SET descriptor=? WHERE key=?",
            ("x" * (module.MAX_DESCRIPTOR_BYTES + 1), other.key),
        )
    original = module.json.loads

    def bounded(raw, *args, **kwargs):
        assert len(raw) <= module.MAX_DESCRIPTOR_BYTES
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(module.json, "loads", bounded)
    with pytest.raises(ValueError):
        module.retirement_inventory(
            resources, "candidate_metrics", {"source": "fixture"}
        )


def test_inventory_bounds_protected_keys_before_traversal(cache, monkeypatch):
    from arblab.hyperliquid_copy import cache_retirement_inventory as module

    resources, target, other = cache
    monkeypatch.setattr(
        module, "_digest", lambda _: pytest.fail("oversized keys traversed")
    )
    with pytest.raises(ValueError):
        module.retirement_inventory(
            resources,
            "candidate_metrics",
            {"source": "fixture"},
            protected_keys=[other.key] * (module.MAX_PUBLICATIONS + 1),
        )


def test_inventory_detects_replaced_catalogue_with_new_reference(cache, monkeypatch):
    from arblab.hyperliquid_copy import cache_retirement_inventory as module
    from arblab.hyperliquid_copy.derived_publication import _encode, publication_key
    import os

    resources, target, other = cache
    replacement = resources.root.parent / (resources.root.name + "-replacement.sqlite3")
    with (
        sqlite3.connect(resources.path) as source,
        sqlite3.connect(replacement) as destination,
    ):
        source.backup(destination)
        data = json.loads(
            destination.execute(
                "SELECT descriptor FROM publications WHERE key=?", (target.key,)
            ).fetchone()[0]
        )
        data["kind"] = "new_survivor"
        raw = _encode(data)
        destination.execute(
            "INSERT INTO publications VALUES (?,?,?)",
            (
                publication_key(data["kind"], data["inputs"]),
                raw.decode(),
                hashlib.sha256(raw).hexdigest(),
            ),
        )
    original = module.file_hash
    changed = False

    def replace_catalogue(path):
        nonlocal changed
        result = original(path)
        if path == resources.root / target.artifacts[0].path and not changed:
            changed = True
            os.replace(replacement, resources.path)
        return result

    monkeypatch.setattr(module, "file_hash", replace_catalogue)
    with pytest.raises(ValueError):
        module.retirement_inventory(
            resources, "candidate_metrics", {"source": "fixture"}
        )
    assert changed
