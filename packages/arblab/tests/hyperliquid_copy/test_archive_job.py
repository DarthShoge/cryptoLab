import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from arblab.hyperliquid_copy.proxy_archive_download import download_archive
from .test_proxy_archive_import import archive


class Source:
    meta = SimpleNamespace(config=SimpleNamespace(retries={"total_max_attempts": 1}))

    def __init__(self, bodies):
        self.bodies, self.gets = bodies, []
        self.heads = []

    def head_object(self, **args):
        self.heads.append(args["Key"])
        return dict(ContentLength=len(self.bodies[args["Key"]]), ETag='"fixed"')

    def get_object(self, **args):
        assert args["IfMatch"] == '"fixed"'
        self.gets.append(args["Key"])
        return self.head_object(**args) | {"Body": io.BytesIO(self.bodies[args["Key"]])}


def inputs(tmp_path, days=2):
    raw = archive(tmp_path, days=days)
    data = json.loads(raw.read_text())
    bodies = {o["key"]: (raw.parent / o["file"]).read_bytes() for o in data["objects"]}
    objects = [dict(key=k, bytes=len(v), etag='"fixed"') for k, v in bodies.items()]
    inventory = tmp_path / "inventory.json"
    total = sum(o["bytes"] for o in objects)
    inventory.write_text(
        json.dumps(
            dict(
                schema="hyperliquid_annual_metadata_audit_v1",
                start=data["start"],
                end=data["end"],
                objects=objects,
                total_bytes=total,
                expected_objects=len(objects),
                listed_expected_objects=len(objects),
                missing=[],
            )
        )
    )
    daily = max(
        sum(len(v) for k, v in bodies.items() if k.split("/")[-2] == date)
        for date in {k.split("/")[-2] for k in bodies}
    )
    return inventory, Source(bodies), total, daily


def test_job_resumes_two_batches_and_only_downloads_uncached_objects(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, daily = inputs(tmp_path)
    cache = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "cache")
    cached_bytes = json.loads(cache.read_text())["expected_bytes"]
    source.gets.clear()
    root = tmp_path / "job"
    job = ArchiveJob.create(
        inventory,
        root,
        ["BTC"],
        cache_manifests=[cache],
        max_download_bytes=total - cached_bytes,
        max_batch_bytes=daily,
    )
    first = job.run_next()
    assert first["phase"] == "prefix_qualified"
    assert first["completed_batches"] == 1
    assert first["reserved_bytes"] == 0
    assert not source.gets
    second = ArchiveJob(root).run_next(source=source)
    assert second["completed_batches"] == 2
    assert second["reserved_bytes"] == total - cached_bytes
    assert len(source.gets) == 24
    assert second["qualification"] == "canonical_prefix_validated"
    report = json.loads(Path(second["qualification_report"]["path"]).read_text())
    assert report["previous"] == first["qualification_report"]
    assert len(report["manifests"]) == 2
    assert second["research_eligible"] is False
    assert ArchiveJob(root).run_next(source=source)["phase"] == "all_batches_qualified"
    assert len(source.gets) == 24
    assert cache.is_file()


@pytest.mark.parametrize("stage", ["raw", "normalized", "compact", "qualified"])
def test_job_recovers_publication_before_sqlite_commit(tmp_path, stage):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, _ = inputs(tmp_path, days=1)
    root = tmp_path / "job"
    job = ArchiveJob.create(inventory, root, ["BTC"], max_download_bytes=total)

    def stop(event):
        if event.get("published") == stage:
            raise RuntimeError("crash before SQLite commit")

    with pytest.raises(RuntimeError, match="crash"):
        job.run_next(source=source, progress=stop)
    assert len(source.gets) == 24
    result = ArchiveJob(root).run_next(source=source)
    assert result["completed_batches"] == 1
    assert result["reserved_bytes"] == total
    assert len(source.gets) == 24
    for name in ("raw", "normalized", "compact", "qualified"):
        assert len(list((root / "batches/0000" / name).glob("*/manifest.json"))) == 1


def test_corrupt_completed_compact_blocks_later_network(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, daily = inputs(tmp_path)
    job = ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        max_download_bytes=total,
        max_batch_bytes=daily,
    )
    result = job.run_next(source=source)
    manifest = Path(result["manifest"])
    data = json.loads(manifest.read_text())
    (manifest.parent / data["files"][0]["name"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity|mismatch"):
        ArchiveJob(tmp_path / "job").run_next(source=source)
    assert len(source.gets) == 24


def test_missing_budget_or_changed_engine_does_not_reset_job(tmp_path, monkeypatch):
    import arblab.hyperliquid_copy.archive_job as module

    inventory, _, total, _ = inputs(tmp_path, days=1)
    root = tmp_path / "job"
    module.ArchiveJob.create(inventory, root, ["BTC"], max_download_bytes=total)
    original = module._engine
    monkeypatch.setattr(module, "_engine", lambda: {"changed": True})
    with pytest.raises(ValueError, match="engine"):
        module.ArchiveJob(root)
    monkeypatch.setattr(module, "_engine", original)
    (root / "budget.sqlite3").unlink()
    with pytest.raises(ValueError, match="budget"):
        module.ArchiveJob(root)
    assert not (root / "budget.sqlite3").exists()


def test_job_lock_and_offline_cache_miss_fail_before_get(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, _ = inputs(tmp_path, days=1)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    with job.store.locked():
        with pytest.raises(ValueError, match="running"):
            job.run_next(source=source)
    with pytest.raises(ValueError, match="not cached"):
        job.run_next()
    assert not source.gets


def test_failed_get_is_not_retried_after_restart(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, _ = inputs(tmp_path, days=1)

    class Broken(Source):
        def get_object(self, **args):
            self.gets.append(args["Key"])
            raise RuntimeError("network failed")

    broken = Broken(source.bodies)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    with pytest.raises(RuntimeError, match="network"):
        job.run_next(source=broken)
    with pytest.raises(ValueError, match="reservation"):
        ArchiveJob(tmp_path / "job").run_next(source=source)
    assert len(broken.gets) == 1
    assert not source.gets


def test_cli_retry_requires_explicit_network_approval_and_credentials(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, _ = inputs(tmp_path, days=1)
    first_key = next(iter(source.bodies))
    retry_bytes = len(source.bodies[first_key])

    class Broken(Source):
        def get_object(self, **args):
            self.gets.append(args["Key"])
            raise RuntimeError("network failed")

    root = tmp_path / "job"
    job = ArchiveJob.create(
        inventory, root, ["BTC"], max_download_bytes=total + retry_bytes
    )
    with pytest.raises(RuntimeError, match="network"):
        job.run_next(source=Broken(source.bodies))
    tool = Path(__file__).resolve().parents[4] / "tools/run_hyperliquid_archive_job.py"
    help_result = subprocess.run(
        [sys.executable, str(tool), "recover-request", "--help"],
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--accept-approved-download" in help_result.stdout
    assert "--credentials-file" in help_result.stdout
    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            "recover-request",
            "--root",
            str(root),
            "--batch",
            "0",
            "--object-index",
            "0",
            "--accept-additional-bytes",
            str(retry_bytes),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "credentials" in result.stderr.lower()


def test_job_refuses_insufficient_budget_and_existing_directory(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, _, total, _ = inputs(tmp_path, days=1)
    root = tmp_path / "job"
    with pytest.raises(ValueError, match="budget"):
        ArchiveJob.create(inventory, root, ["BTC"], max_download_bytes=total - 1)
    assert not root.exists()
    ArchiveJob.create(inventory, root, ["BTC"], max_download_bytes=total)
    with pytest.raises(ValueError, match="exists"):
        ArchiveJob.create(inventory, root, ["ETH"], max_download_bytes=total + 1)


def test_cache_only_zero_budget_job_rebuilds_missing_catalog(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, _, _ = inputs(tmp_path, days=1)
    cache = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "cache")
    job = ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        cache_manifests=[cache],
        max_download_bytes=0,
    )
    assert job.run_next()["reserved_bytes"] == 0
    assert not (job.store.root / "budget.sqlite3").exists()
    catalog = job.store.root / "catalog.sqlite3"
    catalog.unlink()
    assert ArchiveJob(job.store.root).run_next()["phase"] == "all_batches_qualified"
    assert catalog.is_file()


def test_empty_existing_budget_is_not_reinitialized(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, _, total, _ = inputs(tmp_path, days=1)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    path = job.store.root / "budget.sqlite3"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="budget"):
        ArchiveJob(job.store.root)
    assert path.stat().st_size == 0


def test_mid_batch_stop_resumes_only_unrequested_sources(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, _ = inputs(tmp_path, days=1)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )

    def stop(event):
        if event.get("downloaded") == 1:
            raise RuntimeError("pause")

    with pytest.raises(RuntimeError, match="pause"):
        job.run_next(source=source, progress=stop)
    assert len(source.gets) == 1
    assert ArchiveJob(job.store.root).run_next(source=source)["reserved_bytes"] == total
    assert len(source.gets) == len(set(source.gets)) == 24


def test_unpublished_normalization_is_not_duplicated(tmp_path, monkeypatch):
    import arblab.hyperliquid_copy.archive_job as module

    inventory, source, total, _ = inputs(tmp_path, days=1)
    job = module.ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    original = module.import_archive

    def incomplete(raw, coins, root, **kwargs):
        (root / "incomplete").mkdir(parents=True)
        raise RuntimeError("interrupted import")

    monkeypatch.setattr(module, "import_archive", incomplete)
    with pytest.raises(RuntimeError, match="import"):
        job.run_next(source=source)
    monkeypatch.setattr(module, "import_archive", original)
    with pytest.raises(ValueError, match="Incomplete local"):
        module.ArchiveJob(job.store.root).run_next(source=source)
    assert len(source.gets) == 24
    assert len(list((job.store.root / "batches/0000/normalized").iterdir())) == 1


def test_no_get_when_disk_reserve_or_stage_path_is_unsafe(tmp_path, monkeypatch):
    import arblab.hyperliquid_copy.archive_job as module

    inventory, source, total, _ = inputs(tmp_path, days=1)
    job = module.ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    original = module.shutil.disk_usage
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="free disk"):
        job.run_next(source=source)
    monkeypatch.setattr(module.shutil, "disk_usage", original)
    (job.store.root / "batches").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        job.run_next(source=source)
    assert not source.gets


def test_job_cli_creates_and_steps_offline_without_credentials(tmp_path):
    import subprocess
    import sys

    inventory, source, _, _ = inputs(tmp_path, days=1)
    cache = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "cache")
    tool = Path(__file__).resolve().parents[4] / "tools/run_hyperliquid_archive_job.py"
    root = tmp_path / "job"
    created = subprocess.run(
        [
            sys.executable,
            str(tool),
            "create",
            "--root",
            str(root),
            "--inventory",
            str(inventory),
            "--coins",
            "BTC",
            "--cache-manifest",
            str(cache),
            "--max-download-bytes",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(created.stdout)["phase"] == "created"
    result = subprocess.run(
        [sys.executable, str(tool), "step", "--root", str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout.splitlines()[-1])["completed_batches"] == 1
    denied = subprocess.run(
        [
            sys.executable,
            str(tool),
            "step",
            "--root",
            str(root),
            "--credentials-file",
            str(tmp_path / "not-read.csv"),
        ],
        capture_output=True,
        text=True,
    )
    assert denied.returncode != 0
    assert "requires --accept-approved-download" in denied.stderr


def test_job_budget_removed_after_open_is_not_recreated(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, _ = inputs(tmp_path, days=1)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    budget = job.store.root / "budget.sqlite3"
    budget.unlink()
    with pytest.raises(ValueError, match="budget"):
        job.run_next(source=source)
    assert not source.gets
    assert not source.heads
    assert not budget.exists()


def test_record_syncs_new_stage_ancestors_before_durable_reference(
    tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob
    import arblab.hyperliquid_copy.archive_job_store as module

    inventory, source, total, _ = inputs(tmp_path, days=1)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    synced = []
    original = module._sync

    def sync(path):
        synced.append(path)
        original(path)

    monkeypatch.setattr(module, "_sync", sync)
    job.run_next(source=source)
    required = {
        job.store.root,
        job.store.root / "batches",
        job.store.root / "batches/0000",
    }
    required.update(
        job.store.root / "batches/0000" / stage
        for stage in ("raw", "normalized", "compact")
    )
    assert required.issubset(synced)


def test_canonical_capacity_reserved_before_acquisition(tmp_path, monkeypatch):
    import arblab.hyperliquid_copy.archive_job as module

    inventory, source, total, _ = inputs(tmp_path, days=1)
    job = module.ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    monkeypatch.setattr(module, "MAX_CANONICAL_BYTES", 1)
    with pytest.raises(ValueError, match="canonical"):
        job.run_next(source=source)
    assert not source.heads
    assert not list(job.store.root.glob("batches/*/compact/*"))


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_bad_qualification_prevents_next_batch_network(tmp_path, damage):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, daily = inputs(tmp_path)
    job = ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        max_download_bytes=total,
        max_batch_bytes=daily,
    )
    result = job.run_next(source=source)
    path = Path(result["qualification_report"]["path"])
    if damage == "missing":
        path.unlink()
    else:
        path.write_text(path.read_text() + " ")
    source.heads.clear()
    with pytest.raises(ValueError, match="Incomplete|identity|missing"):
        ArchiveJob(job.store.root).run_next(source=source)
    assert len(source.gets) == 24
    assert not source.heads


def test_qualification_failure_retries_locally_before_next_batch(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob
    from arblab.hyperliquid_copy import archive_job_qualification as qualification

    inventory, source, total, daily = inputs(tmp_path)
    job = ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        max_download_bytes=total,
        max_batch_bytes=daily,
    )
    original = qualification.qualify_prefix

    def fail(*args, **kwargs):
        raise ValueError("qualification failed")

    monkeypatch.setattr(qualification, "qualify_prefix", fail)
    with pytest.raises(ValueError, match="qualification failed"):
        job.run_next(source=source)
    assert (0, "compact") in job.store.records()
    reserved = job.budget.reserved_bytes
    monkeypatch.setattr(qualification, "qualify_prefix", original)
    result = ArchiveJob(job.store.root).run_next(source=source)
    assert result["completed_batches"] == 1
    assert result["qualification"] == "canonical_prefix_validated"
    assert result["reserved_bytes"] == reserved
    assert len(source.gets) == 24


def test_uncommitted_report_requires_revalidation_before_adoption(tmp_path):
    from arblab.hyperliquid_copy.archive_job import ArchiveJob

    inventory, source, total, _ = inputs(tmp_path, days=1)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )

    def stop(event):
        if event.get("published") == "qualified":
            raise RuntimeError("stop before pin")

    with pytest.raises(RuntimeError, match="stop before pin"):
        job.run_next(source=source, progress=stop)
    path = next((job.store.root / "batches/0000/qualified").glob("*/manifest.json"))
    data = json.loads(path.read_text())
    data["files"][0]["trade_bounds"]["BTC"] = [999999, 999999]
    path.write_text(json.dumps(data, sort_keys=True, indent=2))
    with pytest.raises(ValueError, match="uncommitted|revalidation"):
        ArchiveJob(job.store.root).run_next(source=source)
    assert (0, "qualified") not in job.store.records()
    assert len(source.gets) == 24
