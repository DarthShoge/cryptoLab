import os

import pytest

from .test_candidate_day import resources


def test_owner_rechecks_manifest_after_final_engine_callback(resources, monkeypatch):
    from arblab.hyperliquid_copy import ranking_staging_owner as module

    owner = module.RankingStagingOwner.create(resources, {"query": "fixture"})
    original, calls = module._engine, 0

    def count():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(module, "_engine", count)
    owner.verify()
    final_call, calls = calls, 0

    def mutate():
        result = count()
        if calls == final_call:
            with owner.path("manifest").open("ab") as stream:
                stream.write(b"late mutation")
        return result

    monkeypatch.setattr(module, "_engine", mutate)
    try:
        with pytest.raises(ValueError):
            owner.verify()
    finally:
        owner.close()


def test_owner_rejects_pending_record_with_settled_metadata(resources):
    from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner

    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    try:
        with resources._connect() as db, db:
            db.execute(
                "UPDATE allocations SET bytes=1,sha256=? WHERE path=?",
                ("a" * 64, str(owner.path("metrics").relative_to(resources.root))),
            )
        with pytest.raises(ValueError, match="allocation"):
            owner.verify()
    finally:
        owner.close()


def test_oversized_complete_manifest_rejects_before_reservations(resources):
    from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner

    before = resources.audit()
    with pytest.raises(ValueError):
        RankingStagingOwner.create(resources, {"query": "x" * (65536 - 256)})
    assert resources.audit() == before


def test_owner_reserves_exact_envelope_and_keeps_temporaries_pending(resources):
    from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner

    before = resources.audit()
    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    try:
        owner.verify()
        assert (
            resources.audit()["reserved_bytes"] - before["reserved_bytes"]
            == int(4.5 * 1024**3) + 65536
        )
        with resources._connect() as db:
            assert db.execute("SELECT DISTINCT state FROM allocations").fetchall() == [
                ("pending",)
            ]
            assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0
    finally:
        owner.close()
    assert resources.audit()["reserved_bytes"] > before["reserved_bytes"]


@pytest.mark.parametrize("after", [1, 2, 3, 4, 5])
def test_partial_admission_stays_charged_and_blocks_another_attempt(
    resources, monkeypatch, after
):
    from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner

    original = type(resources).reserve
    count = 0

    def interrupted(self, *args, **kwargs):
        nonlocal count
        result = original(self, *args, **kwargs)
        count += 1
        if count == after:
            raise RuntimeError("reservation interrupted")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(type(resources), "reserve", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            RankingStagingOwner.create(resources, {"query": "fixture"})
    before = resources.audit()
    assert before["reserved_bytes"] > 0
    with pytest.raises(ValueError, match="pending"):
        RankingStagingOwner.create(resources, {"query": "fixture"})
    assert resources.audit() == before


def test_owner_detects_context_and_manifest_mutation(resources):
    from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner

    context = {"query": "fixture"}
    owner = RankingStagingOwner.create(resources, context)
    try:
        context["query"] = "changed"
        with pytest.raises(ValueError, match="context"):
            owner.verify()
        context["query"] = "fixture"
        with owner.path("manifest").open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="manifest"):
            owner.verify()
    finally:
        owner.close()


def test_created_fd_cannot_adopt_replacement_path(resources):
    from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner

    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    try:
        path = owner.path("metrics")
        with path.open("xb") as stream:
            path.unlink()
            with path.open("xb") as replacement:
                replacement.write(b"replacement")
            with pytest.raises(ValueError, match="identity"):
                owner.capture_created_fd("metrics", stream.fileno())
    finally:
        owner.close()


def test_created_fd_stays_pinned_after_writer_closes(resources):
    from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner

    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    try:
        with owner.path("metrics").open("xb") as stream:
            owner.capture_created_fd("metrics", stream.fileno())
            stream.write(b"fixture")
        owner.verify_file("metrics")
        os.rename(owner.path("metrics"), owner.path("metrics").with_suffix(".moved"))
        with owner.path("metrics").open("xb") as replacement:
            replacement.write(b"replacement")
        with pytest.raises(ValueError, match="identity"):
            owner.verify_file("metrics")
    finally:
        owner.close()
