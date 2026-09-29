import json
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.archive_job import ArchiveJob
from arblab.hyperliquid_copy.proxy_archive_download import download_archive
from .test_archive_job import inputs


def make_job(tmp_path, days=1):
    inventory, source, total, daily = inputs(tmp_path, days=days)
    return ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        max_download_bytes=total,
        max_batch_bytes=daily,
    ), source


def stop_at(name):
    def stop(event):
        if event.get("cleanup") == name:
            raise RuntimeError("cleanup interruption")

    return stop


def test_two_batches_dispose_only_owned_payloads_preserving_reusable_history(tmp_path):
    inventory, source, total, daily = inputs(tmp_path)
    cache = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "cache")
    cached = json.loads(cache.read_text())
    cache_bytes = cached["expected_bytes"]
    source.gets.clear()
    job = ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        cache_manifests=[cache],
        max_download_bytes=total - cache_bytes,
        max_batch_bytes=daily,
    )
    first = job.run_next()
    assert first["cleanup"]["deleted_files"] == 48
    second = ArchiveJob(job.store.root).run_next(source=source)
    assert second["cleanup"]["deleted_files"] == 96
    assert second["cleanup"]["deleted_bytes"] > total
    assert second["cleanup"]["raw_payloads_retained"] is False
    assert second["cleanup"]["policy"] == "qualified_job_owned_payloads_only"
    assert second["reserved_bytes"] == total - cache_bytes
    assert len(source.gets) == 24
    assert not list(job.store.root.glob("batches/*/raw/*/*.lz4"))
    assert not list(job.store.root.glob("batches/*/normalized/*/*.parquet"))
    assert len(list(job.store.root.glob("batches/*/compact/*/*.parquet"))) == 2
    assert len(list(job.store.root.glob("batches/*/*/*/manifest.json"))) == 8
    assert all((cache.parent / o["file"]).is_file() for o in cached["objects"])
    assert (
        ArchiveJob(job.store.root).run_next(source=source)["cleanup"]
        == second["cleanup"]
    )
    assert len(source.gets) == 24


@pytest.mark.parametrize("point", ["intent_committed", "payload_unlinked"])
def test_cleanup_crash_recovers_without_get_or_budget_refund(tmp_path, point):
    job, source = make_job(tmp_path)
    with pytest.raises(RuntimeError, match="cleanup interruption"):
        job.run_next(source=source, progress=stop_at(point))
    reserved = job.budget.reserved_bytes
    result = ArchiveJob(job.store.root).run_next(source=source)
    assert result["cleanup"]["deleted_files"] == 48
    assert result["reserved_bytes"] == reserved
    assert len(source.gets) == 24


@pytest.mark.parametrize("damage", ["missing", "changed", "symlink"])
def test_invalid_target_before_journal_preserves_all_other_payloads(tmp_path, damage):
    job, source = make_job(tmp_path)

    def damage_target(event):
        if event.get("published") == "qualified":
            manifest = job.store.records()[0, "raw"]["path"]
            data = json.loads(manifest.read_text())
            target = manifest.parent / data["objects"][0]["file"]
            if damage == "changed":
                target.write_bytes(b"changed")
            else:
                target.unlink()
                if damage == "symlink":
                    target.symlink_to(tmp_path / "inventory.json")

    with pytest.raises(ValueError, match="missing|identity|symlink"):
        job.run_next(source=source, progress=damage_target)
    assert len(list(job.store.root.glob("batches/*/normalized/*/*.parquet"))) == 24
    assert len(list(job.store.root.glob("batches/*/raw/*/*.lz4"))) >= 23


def test_corrupt_compact_after_cleanup_intent_stops_before_more_deletion(tmp_path):
    job, source = make_job(tmp_path)
    with pytest.raises(RuntimeError):
        job.run_next(source=source, progress=stop_at("intent_committed"))
    compact = next(job.store.root.glob("batches/*/compact/*/*.parquet"))
    compact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity|mismatch"):
        ArchiveJob(job.store.root).run_next(source=source)
    assert len(list(job.store.root.glob("batches/*/raw/*/*.lz4"))) == 24


def test_recreated_deleted_target_is_not_removed(tmp_path):
    job, source = make_job(tmp_path)
    job.run_next(source=source)
    manifest = job.store.records()[0, "raw"]["path"]
    target = manifest.parent / json.loads(manifest.read_text())["objects"][0]["file"]
    target.write_bytes(b"new unrelated content")
    with pytest.raises(ValueError, match="recreated"):
        ArchiveJob(job.store.root).run_next(source=source)
    assert target.read_bytes() == b"new unrelated content"


@pytest.mark.parametrize(
    "mutation", ["row_missing", "pin_changed", "path_escape", "status_changed"]
)
def test_journal_damage_blocks_deletion(tmp_path, mutation):
    job, source = make_job(tmp_path)
    with pytest.raises(RuntimeError):
        job.run_next(source=source, progress=stop_at("intent_committed"))
    with job.store._connect() as db, db:
        path = db.execute("SELECT path FROM cleanup LIMIT 1").fetchone()[0]
        if mutation == "row_missing":
            db.execute("DELETE FROM cleanup WHERE path=?", (path,))
        elif mutation == "pin_changed":
            db.execute(
                "UPDATE cleanup SET qualification_sha256='changed' WHERE path=?",
                (path,),
            )
        elif mutation == "path_escape":
            db.execute("UPDATE cleanup SET path='../outside.lz4' WHERE path=?", (path,))
        else:
            db.execute("UPDATE cleanup SET status='invalid' WHERE path=?", (path,))
    with pytest.raises(ValueError, match="journal"):
        ArchiveJob(job.store.root).run_next(source=source)
    assert len(list(job.store.root.glob("batches/*/raw/*/*.lz4"))) == 24
    assert len(list(job.store.root.glob("batches/*/normalized/*/*.parquet"))) == 24
    assert len(source.gets) == 24
