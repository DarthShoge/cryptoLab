import json
import hashlib
import multiprocessing
import os
import sqlite3
import uuid

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources


def test_content_key_is_deterministic_and_parameter_sensitive():
    from arblab.hyperliquid_copy.derived_publication import publication_key

    base = dict(source="a" * 64, fee="gross", version=1)
    assert publication_key("day", base) == publication_key(
        "day", dict(reversed(list(base.items())))
    )
    for field in base:
        assert publication_key("day", base) != publication_key(
            "day", {**base, field: "changed"}
        )


@pytest.mark.parametrize(
    "kind,inputs",
    [
        ("../bad", {}),
        ("day", []),
        ("day", {"huge": "x" * 65537}),
        ("day", {"x": float("nan")}),
        ("day", {"x": list(range(9000))}),
    ],
)
def test_key_rejects_invalid_or_unbounded_input(kind, inputs):
    from arblab.hyperliquid_copy.derived_publication import publication_key

    with pytest.raises(ValueError):
        publication_key(kind, inputs)


def test_schema_v2_and_old_catalog_rejection(tmp_path):
    with CacheLease(tmp_path) as lease:
        resources = CacheResources.create(lease, "v2-test")
        db = sqlite3.connect(resources.path)
        try:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 2
            assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0
            db.execute("PRAGMA user_version=1")
        finally:
            db.close()
        with pytest.raises(ValueError):
            CacheResources(lease, "v2-test")


def artifact(resources, *, namespace="artifacts", purpose="payload", settle=True):
    relative = f"{namespace}/{uuid.uuid4().hex}"
    token = resources.reserve(relative, 100, purpose)
    (resources.root / relative).write_bytes(b"immutable data")
    if settle:
        resources.settle(token)
    return token, relative


def test_atomic_visibility_exact_reuse_and_single_payload_charge(tmp_path):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    with CacheLease(tmp_path) as lease:
        resources = CacheResources.create(lease, "publication-test")
        publications = PublishedArtifacts(resources)
        tokens = [artifact(resources)[0] for _ in range(2)]
        assert publications.lookup("day", {"source": "a"}) is None
        before = resources.audit()
        published = publications.publish("day", {"source": "a"}, tokens)
        assert len(published.artifacts) == 2
        assert all(not item.path.startswith("/") for item in published.artifacts)
        assert publications.lookup("day", {"source": "a"}) == published
        assert publications.publish("day", {"source": "a"}, tokens) == published
        assert publications.lookup("day", {"source": "b"}) is None
        publications.publish("other", {"source": "a"}, tokens)
        assert resources.audit() == before
        with pytest.raises(ValueError):
            publications.publish("day", {"source": "a"}, list(reversed(tokens)))


@pytest.mark.parametrize(
    "fault", ["pending", "scratch", "staging", "duplicate", "empty"]
)
def test_only_complete_distinct_artifact_sets_can_publish(tmp_path, fault):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    with CacheLease(tmp_path) as lease:
        resources = CacheResources.create(lease, "publication-test")
        publications = PublishedArtifacts(resources)
        token, _ = artifact(
            resources,
            namespace="staging" if fault == "staging" else "artifacts",
            purpose="scratch" if fault == "scratch" else "payload",
            settle=fault not in ("pending", "scratch"),
        )
        tokens = (
            []
            if fault == "empty"
            else [token, token]
            if fault == "duplicate"
            else [token]
        )
        with pytest.raises(ValueError):
            publications.publish("day", {}, tokens)
        assert publications.lookup("day", {}) is None


@pytest.mark.parametrize("fault", ["payload", "descriptor"])
def test_changed_published_content_fails_closed(tmp_path, fault):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    with CacheLease(tmp_path) as lease:
        resources = CacheResources.create(lease, "publication-test")
        publications = PublishedArtifacts(resources)
        token, relative = artifact(resources)
        publications.publish("day", {}, [token])
        if fault == "payload":
            (tmp_path / relative).write_bytes(b"changed data!!")
        else:
            db = sqlite3.connect(resources.path)
            try:
                db.execute("UPDATE publications SET descriptor='{}'")
                db.commit()
            finally:
                db.close()
        with pytest.raises(ValueError):
            publications.lookup("day", {})


def test_publication_bounds_and_closed_lease(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import derived_publication as module

    with CacheLease(tmp_path) as lease:
        resources = CacheResources.create(lease, "publication-test")
        publications = module.PublishedArtifacts(resources)
        token, _ = artifact(resources)
        with monkeypatch.context() as scoped:
            scoped.setattr(module, "MAX_DESCRIPTOR_BYTES", 10)
            with pytest.raises(ValueError, match="descriptor"):
                publications.publish("day", {}, [token])
        monkeypatch.setattr(module, "MAX_PUBLICATIONS", 1)
        publications.publish("day", {}, [token])
        with pytest.raises(ValueError, match="Publication.*limit"):
            publications.publish("other", {}, [token])
    for operation in [
        lambda: publications.lookup("day", {}),
        lambda: publications.publish("day", {}, [token]),
    ]:
        with pytest.raises(ValueError, match="held"):
            operation()


def test_key_does_not_confuse_numeric_and_string_parameters():
    from arblab.hyperliquid_copy.derived_publication import publication_key

    assert publication_key("day", {"weight": 1.0}) != publication_key(
        "day", {"weight": "1"}
    )


@pytest.mark.parametrize("fault", ["sync", "insert"])
def test_failed_publication_stays_invisible_and_charged(tmp_path, monkeypatch, fault):
    from arblab.hyperliquid_copy import derived_publication as module

    with CacheLease(tmp_path) as lease:
        resources = CacheResources.create(lease, "publication-test")
        publications = module.PublishedArtifacts(resources)
        token, _ = artifact(resources)
        before = resources.audit()
        if fault == "sync":

            def fail(_):
                raise OSError("sync failed")

            monkeypatch.setattr(module, "_sync", fail)
        else:
            db = sqlite3.connect(resources.path)
            try:
                db.execute(
                    "CREATE TRIGGER refuse_publication BEFORE INSERT ON publications BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
                )
            finally:
                db.close()
        with pytest.raises((OSError, ValueError)):
            publications.publish("day", {}, [token])
        assert publications.lookup("day", {}) is None
        assert resources.audit() == before
    with CacheLease(tmp_path) as lease:
        reopened = CacheResources(lease, "publication-test")
        assert reopened.audit() == before
        assert module.PublishedArtifacts(reopened).lookup("day", {}) is None


def publish_and_exit(root):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    with CacheLease(root) as lease:
        resources = CacheResources.create(lease, "publication-test")
        token, _ = artifact(resources)
        PublishedArtifacts(resources).publish("day", {}, [token])
        os._exit(17)


def test_committed_publication_survives_process_death(tmp_path):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    child = multiprocessing.get_context("spawn").Process(
        target=publish_and_exit, args=(tmp_path,)
    )
    child.start()
    child.join(10)
    try:
        assert child.exitcode == 17
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)
    with CacheLease(tmp_path) as lease:
        resources = CacheResources(lease, "publication-test")
        result = PublishedArtifacts(resources).lookup("day", {})
        assert len(result.artifacts) == 1
        assert (tmp_path / result.artifacts[0].path).read_bytes() == b"immutable data"


def test_rehashed_descriptor_cannot_change_parameter_types(tmp_path):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    with CacheLease(tmp_path) as lease:
        resources = CacheResources.create(lease, "publication-test")
        publications = PublishedArtifacts(resources)
        token, _ = artifact(resources)
        publications.publish("day", {"flag": True}, [token])
        db = sqlite3.connect(resources.path)
        try:
            data = json.loads(
                db.execute("SELECT descriptor FROM publications").fetchone()[0]
            )
            data["inputs"]["flag"] = 1
            raw = json.dumps(data)
            db.execute(
                "UPDATE publications SET descriptor=?,sha256=?",
                (raw, hashlib.sha256(raw.encode()).hexdigest()),
            )
            db.commit()
        finally:
            db.close()
        with pytest.raises(ValueError):
            publications.lookup("day", {"flag": True})
