# ruff: noqa: F401,F811,F841

import sqlite3

import pytest

from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from .test_cache_retirement_inventory import cache


def prepare(resources):
    from arblab.hyperliquid_copy.cache_retirement import prepare_retirement

    return prepare_retirement(
        resources,
        "candidate_metrics",
        {"source": "fixture"},
        protected_keys=(),
        reason="Fixture evidence retained independently",
    )


def test_prepare_retains_target_and_charges_journal(cache):
    from arblab.hyperliquid_copy.cache_retirement import INTENT

    resources, target, other = cache
    before = resources.audit()
    inputs = prepare(resources)
    receipt = PublishedArtifacts(resources).lookup(INTENT, inputs)
    assert receipt is not None and len(receipt.artifacts) == 1
    assert (
        PublishedArtifacts(resources).lookup("candidate_metrics", {"source": "fixture"})
        == target
    )
    assert (
        resources.audit()["retained_bytes"]
        == before["retained_bytes"] + receipt.artifacts[0].bytes
    )
    assert resources.audit()["reserved_bytes"] == 0


def test_begin_detaches_only_exclusive_payload_and_preserves_obligation(cache):
    from arblab.hyperliquid_copy.cache_retirement import begin_retirement, COMMITTED

    resources, target, other = cache
    inputs = prepare(resources)
    before = resources.audit()
    begin_retirement(resources, inputs)
    assert PublishedArtifacts(resources).lookup(COMMITTED, inputs) is not None
    assert (
        PublishedArtifacts(resources).lookup("candidate_metrics", {"source": "fixture"})
        is None
    )
    assert (
        PublishedArtifacts(resources).lookup("saved_feature_ranking", {"fixture": True})
        == other
    )
    assert all((resources.root / p.path).exists() for p in target.artifacts)
    after = resources.audit()
    assert after["total_bytes"] == before["total_bytes"]
    assert after["reserved_bytes"] == target.artifacts[0].bytes


def test_begin_rejects_changed_surviving_references(cache):
    from arblab.hyperliquid_copy.cache_retirement import begin_retirement

    resources, target, other = cache
    inputs = prepare(resources)
    PublishedArtifacts(resources).publish(
        "new_reference", {}, [target.artifacts[0].token]
    )
    before = resources.audit()
    with pytest.raises(ValueError):
        begin_retirement(resources, inputs)
    assert resources.audit() == before
    assert (
        PublishedArtifacts(resources).lookup("candidate_metrics", {"source": "fixture"})
        == target
    )


def test_begin_rolls_back_if_commit_receipt_insertion_fails(cache):
    from arblab.hyperliquid_copy.cache_retirement import begin_retirement

    resources, target, other = cache
    inputs = prepare(resources)
    with sqlite3.connect(resources.path) as db:
        db.execute(
            "CREATE TRIGGER fail_retirement BEFORE INSERT ON publications "
            "BEGIN SELECT RAISE(ABORT, 'fixture insertion failure'); END"
        )
    before = resources.audit()
    with pytest.raises(ValueError):
        begin_retirement(resources, inputs)
    assert resources.audit() == before
    assert (
        PublishedArtifacts(resources).lookup("candidate_metrics", {"source": "fixture"})
        == target
    )
    assert all((resources.root / p.path).exists() for p in target.artifacts)


def test_finish_only_removes_exclusive_owned_payload_and_is_idempotent(cache):
    from arblab.hyperliquid_copy.cache_retirement import (
        begin_retirement,
        finish_retirement,
    )

    resources, target, other = cache
    inputs = prepare(resources)
    begin_retirement(resources, inputs)
    before = resources.audit()
    result = finish_retirement(resources, inputs)
    assert result["retired_bytes"] == target.artifacts[0].bytes
    assert result["retired_files"] == 1
    assert not (resources.root / target.artifacts[0].path).exists()
    assert (resources.root / target.artifacts[1].path).exists()
    assert (
        resources.audit()["total_bytes"]
        == before["total_bytes"] - target.artifacts[0].bytes
    )
    assert resources.audit()["reserved_bytes"] == 0
    assert finish_retirement(resources, inputs) == result
    assert (
        PublishedArtifacts(resources).lookup("saved_feature_ranking", {"fixture": True})
        == other
    )


def test_finish_rejects_caller_change_before_any_unlink(cache, monkeypatch):
    from arblab.hyperliquid_copy import cache_retirement as module

    resources, target, other = cache
    inputs = prepare(resources)
    module.begin_retirement(resources, inputs)
    original = module.file_hash
    path = resources.root / target.artifacts[0].path

    def changed(at):
        digest = original(at)
        if at == path:
            inputs["schema"] = 2
        return digest

    monkeypatch.setattr(module, "file_hash", changed)
    with pytest.raises(ValueError):
        module.finish_retirement(resources, inputs)
    assert path.exists(), "caller mutation was detected only after deleting the file"


def test_prepare_rejects_late_caller_change(cache, monkeypatch):
    from arblab.hyperliquid_copy import cache_retirement_journal as module

    resources, target, other = cache
    inputs = {"source": "fixture"}
    original = module.load_journal

    def changed(*args):
        result = original(*args)
        inputs["source"] = "changed"
        return result

    monkeypatch.setattr(module, "load_journal", changed)
    with pytest.raises(ValueError):
        module.prepare_retirement(
            resources, "candidate_metrics", inputs, reason="fixture"
        )


@pytest.mark.parametrize("stage", ["after_unlink", "after_release"])
def test_interrupted_finish_reopens_with_missing_bytes_still_charged(
    cache, monkeypatch, stage
):
    from arblab.hyperliquid_copy import cache_retirement as module
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

    resources, target, other = cache
    inputs = prepare(resources)
    module.begin_retirement(resources, inputs)
    expected = resources.audit()["reserved_bytes"]
    with monkeypatch.context() as faults:
        if stage == "after_unlink":
            original = module.os.unlink

            def interrupted(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("fixture crash after unlink")

            faults.setattr(module.os, "unlink", interrupted)
        else:
            faults.setattr(
                resources,
                "_audit",
                lambda *_: (_ for _ in ()).throw(
                    RuntimeError("fixture crash before commit")
                ),
            )
        with pytest.raises(RuntimeError):
            module.finish_retirement(resources, inputs)
    assert not (resources.root / target.artifacts[0].path).exists()
    assert resources.audit()["reserved_bytes"] == expected
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)
    with CacheLease(root) as lease:
        reopened = CacheResources(lease, identity)
        result = module.finish_retirement(reopened, inputs)
        assert result["retired_files"] == 1
        assert reopened.audit()["reserved_bytes"] == 0


def test_begin_requires_intent_still_present_at_transaction_boundary(
    cache, monkeypatch
):
    from arblab.hyperliquid_copy import cache_retirement as module
    from arblab.hyperliquid_copy.derived_publication import publication_key

    resources, target, other = cache
    inputs = prepare(resources)
    original = module.load_journal

    def removed(*args):
        result = original(*args)
        with resources._connect() as db, db:
            db.execute(
                "DELETE FROM publications WHERE key=?",
                (publication_key(module.INTENT, inputs),),
            )
        return result

    monkeypatch.setattr(module, "load_journal", removed)
    with pytest.raises(ValueError):
        module.begin_retirement(resources, inputs)
    assert (
        PublishedArtifacts(resources).lookup("candidate_metrics", {"source": "fixture"})
        == target
    )


def rewrite_intent(resources, inputs, mutate):
    import json
    import hashlib
    from arblab.hyperliquid_copy.cache_retirement import INTENT
    from arblab.hyperliquid_copy.derived_publication import _encode, publication_key

    publication = PublishedArtifacts(resources).lookup(INTENT, inputs)
    pin = publication.artifacts[0]
    path = resources.root / pin.path
    body = json.loads(path.read_bytes())
    mutate(body)
    raw = _encode(body)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    changed = inputs | dict(journal_sha256=digest)
    with resources._connect() as db, db:
        descriptor = json.loads(
            db.execute(
                "SELECT descriptor FROM publications WHERE key=?", (publication.key,)
            ).fetchone()[0]
        )
        descriptor["inputs"] = changed
        descriptor["artifacts"][0].update(bytes=len(raw), sha256=digest)
        encoded = _encode(descriptor)
        db.execute(
            "UPDATE allocations SET bytes=?,sha256=? WHERE token=?",
            (len(raw), digest, pin.token),
        )
        db.execute("DELETE FROM publications WHERE key=?", (publication.key,))
        db.execute(
            "INSERT INTO publications VALUES (?,?,?)",
            (
                publication_key(INTENT, changed),
                encoded.decode(),
                hashlib.sha256(encoded).hexdigest(),
            ),
        )
    return changed


def test_begin_recomputes_shared_ownership_instead_of_trusting_journal(cache):
    from arblab.hyperliquid_copy.cache_retirement import begin_retirement

    resources, target, other = cache
    inputs = prepare(resources)

    def lie(body):
        body["inventory"]["exclusive"] += body["inventory"]["shared"]
        body["inventory"]["shared"] = []

    forged = rewrite_intent(resources, inputs, lie)
    before = resources.audit()
    with pytest.raises(ValueError):
        begin_retirement(resources, forged)
    assert resources.audit() == before
    assert (
        PublishedArtifacts(resources).lookup("saved_feature_ranking", {"fixture": True})
        == other
    )


def test_expanded_policy_reopens_between_detachment_and_disposal(cache):
    from arblab.hyperliquid_copy.cache_retirement import (
        begin_retirement,
        finish_retirement,
    )
    from arblab.hyperliquid_copy.derived_cache_expansion import (
        prepare_expansion,
        apply_expansion,
    )
    from arblab.hyperliquid_copy.derived_cache_policy import open_expanded_cache
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease

    resources, target, other = cache
    receipt = prepare_expansion(
        resources,
        approved_from=8 * 1024**3,
        approved_to=16 * 1024**3,
        approval="fixture",
    )
    apply_expansion(resources.lease, resources.identity, receipt)
    expanded = open_expanded_cache(resources.lease, resources.identity, receipt)
    inputs = prepare(expanded)
    begin_retirement(expanded, inputs)
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)
    with CacheLease(root) as lease:
        reopened = open_expanded_cache(lease, identity, receipt)
        finish_retirement(reopened, inputs)
        assert (
            open_expanded_cache(lease, identity, receipt).audit()["reserved_bytes"] == 0
        )


def test_full_sqlite_page_budget_rolls_back_before_disposal(cache, monkeypatch):
    from arblab.hyperliquid_copy import cache_retirement as module

    resources, target, other = cache
    inputs = prepare(resources)
    before = resources.audit()

    def out_of_pages(db, *_):
        pages = db.execute("PRAGMA page_count").fetchone()[0]
        db.execute(f"PRAGMA max_page_count={pages}")
        db.execute(
            "INSERT INTO publications VALUES ('fixture-full',zeroblob(1048576),'full')"
        )

    monkeypatch.setattr(module, "_insert_committed", out_of_pages)
    with pytest.raises(ValueError):
        module.begin_retirement(resources, inputs)
    assert resources.audit() == before
    assert (
        PublishedArtifacts(resources).lookup("candidate_metrics", {"source": "fixture"})
        == target
    )
    assert all((resources.root / p.path).exists() for p in target.artifacts)


def test_completion_needs_no_new_publication_slot_at_limit(cache):
    import hashlib
    from dataclasses import asdict
    from arblab.hyperliquid_copy.cache_retirement import (
        begin_retirement,
        finish_retirement,
    )
    from arblab.hyperliquid_copy.derived_publication import (
        _encode,
        publication_key,
        MAX_PUBLICATIONS,
    )

    resources, target, other = cache
    with resources._connect() as db, db:
        for index in range(MAX_PUBLICATIONS - 3):
            inputs = dict(index=index)
            raw = _encode(
                dict(
                    schema=1,
                    kind="fixture_filler",
                    inputs=inputs,
                    artifacts=[asdict(other.artifacts[0])],
                )
            )
            db.execute(
                "INSERT INTO publications VALUES (?,?,?)",
                (
                    publication_key("fixture_filler", inputs),
                    raw.decode(),
                    hashlib.sha256(raw).hexdigest(),
                ),
            )
    inputs = prepare(resources)
    with resources._connect() as db:
        assert (
            db.execute("SELECT count(*) FROM publications").fetchone()[0]
            == MAX_PUBLICATIONS
        )
    begin_retirement(resources, inputs)
    finish_retirement(resources, inputs)
    with resources._connect() as db:
        assert (
            db.execute("SELECT count(*) FROM publications").fetchone()[0]
            == MAX_PUBLICATIONS - 1
        )


@pytest.mark.parametrize("fault", ["replacement", "alias", "symlink", "journal"])
def test_finish_rejects_changed_owned_files_without_unlink(cache, monkeypatch, fault):
    import os
    from arblab.hyperliquid_copy.cache_retirement import (
        begin_retirement,
        finish_retirement,
        INTENT,
    )

    resources, target, other = cache
    inputs = prepare(resources)
    begin_retirement(resources, inputs)
    path = resources.root / target.artifacts[0].path
    if fault == "replacement":
        displaced = resources.root.parent / (path.name + "-original")
        path.rename(displaced)
        path.write_bytes(displaced.read_bytes())
    elif fault == "alias":
        os.link(path, resources.root.parent / (path.name + "-alias"))
    elif fault == "symlink":
        displaced = resources.root.parent / (path.name + "-original")
        path.rename(displaced)
        path.symlink_to(displaced)
    else:
        receipt = PublishedArtifacts(resources).lookup(INTENT, inputs)
        with (resources.root / receipt.artifacts[0].path).open("ab") as handle:
            handle.write(b"corrupt")
    with pytest.raises((ValueError, OSError)):
        finish_retirement(resources, inputs)
    assert path.exists() or path.is_symlink()


def test_prepare_keeps_original_output_inode_pinned(cache, monkeypatch):
    from arblab.hyperliquid_copy import cache_retirement_journal as module
    import os

    resources, target, other = cache
    original = module.os.fsync
    changed = False

    def replace_after_write(fd):
        nonlocal changed
        original(fd)
        name = os.readlink(f"/proc/self/fd/{fd}")
        from pathlib import Path

        path = Path(name)
        if (
            path.parent == resources.root / "artifacts"
            and path.stat().st_size > 100
            and not changed
        ):
            changed = True
            displaced = resources.root.parent / (path.name + "-displaced")
            path.rename(displaced)
            path.write_bytes(displaced.read_bytes())

    monkeypatch.setattr(module.os, "fsync", replace_after_write)
    with pytest.raises(ValueError):
        prepare(resources)
    assert changed
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 2


def test_reason_byte_work_is_bounded_before_allocation(cache):
    from arblab.hyperliquid_copy.cache_retirement import prepare_retirement

    resources, target, other = cache
    before = resources.audit()
    with pytest.raises(ValueError):
        prepare_retirement(
            resources,
            "candidate_metrics",
            {"source": "fixture"},
            reason=" " * 1024 + "x",
        )
    assert resources.audit() == before


def test_journal_schema_requires_integer_not_boolean(cache):
    from arblab.hyperliquid_copy.cache_retirement import begin_retirement

    resources, target, other = cache
    inputs = prepare(resources)
    changed = rewrite_intent(resources, inputs, lambda body: body.update(schema=True))
    with pytest.raises(ValueError):
        begin_retirement(resources, changed)


@pytest.mark.parametrize("zero", [False, True])
def test_zero_exclusive_bytes_and_shared_only_targets(cache, zero):
    import uuid
    from arblab.hyperliquid_copy.cache_retirement import (
        prepare_retirement,
        begin_retirement,
        finish_retirement,
    )

    resources, original, other = cache
    if zero:
        path = "artifacts/" + uuid.uuid4().hex
        token = resources.reserve(path, 100, "payload")
        (resources.root / path).write_bytes(b"")
        resources.settle(token)
    else:
        token = other.artifacts[0].token
    target = PublishedArtifacts(resources).publish("edge_target", {}, [token])
    inputs = prepare_retirement(resources, "edge_target", {}, reason="fixture")
    before = resources.audit()
    begin_retirement(resources, inputs)
    assert resources.audit()["reserved_bytes"] == (1 if zero else 0)
    result = finish_retirement(resources, inputs)
    assert result["retired_bytes"] == 0
    assert result["retired_files"] == (1 if zero else 0)
    assert resources.audit() == before
    assert (
        PublishedArtifacts(resources).lookup("saved_feature_ranking", {"fixture": True})
        == other
    )


def test_database_replacement_during_validation_prevents_unlink(cache, monkeypatch):
    import shutil
    from pathlib import Path
    from arblab.hyperliquid_copy import cache_retirement as module

    resources, target, _ = cache
    inputs = prepare(resources)
    module.begin_retirement(resources, inputs)
    path = resources.root / target.artifacts[0].path
    original = module.file_hash
    ready = False
    changed = False

    def arm(candidate):
        nonlocal ready
        result = original(candidate)
        if Path(candidate) == path:
            ready = True
        return result

    def replace_database():
        nonlocal changed
        if ready and not changed:
            changed = True
            displaced = resources.root.parent / "original-retirement-catalogue.sqlite"
            resources.path.rename(displaced)
            shutil.copyfile(displaced, resources.path)

    monkeypatch.setattr(module, "file_hash", arm)
    with pytest.raises(ValueError, match="database|identity|catalog"):
        module.finish_retirement(resources, inputs, validation=replace_database)
    assert changed
    assert path.exists(), (
        "validation must not replace the checked catalogue before unlink"
    )
