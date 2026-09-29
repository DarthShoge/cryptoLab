import json
import sqlite3

import pytest

from arblab.hyperliquid_copy.archive_job import ArchiveJob
from arblab.hyperliquid_copy.archive_retry_recovery import (
    recover_job_archive_request,
)
from .test_archive_job import Source, inputs


class Broken(Source):
    def get_object(self, **args):
        self.gets.append(args["Key"])
        raise RuntimeError("network failed")


def incident(tmp_path, *, extra=True):
    inventory, source, total, _ = inputs(tmp_path, days=1)
    key = next(iter(source.bodies))
    size = len(source.bodies[key])
    root = tmp_path / "job"
    job = ArchiveJob.create(
        inventory,
        root,
        ["BTC"],
        max_download_bytes=total + (size if extra else 0),
    )
    with pytest.raises(RuntimeError, match="network"):
        job.run_next(source=Broken(source.bodies))
    return root, source, key, size, total


def test_sidecar_retry_is_charged_completed_and_original_job_resumes(tmp_path):
    root, source, key, size, total = incident(tmp_path)
    receipt = recover_job_archive_request(
        root, batch=0, object_index=0, approved_bytes=size, source=source
    )
    assert receipt["key"] == key
    assert receipt["bytes"] == size
    assert receipt["attempt"] == 1
    assert receipt["state"] == "completed"
    with sqlite3.connect(root / "retry_budget.sqlite3") as db:
        assert db.execute("SELECT bytes,status,sha256 FROM attempts").fetchall() == [
            (size, "completed", receipt["sha256"])
        ]
    result = ArchiveJob(root).run_next(source=source)
    assert result["completed_batches"] == 1
    assert result["reserved_bytes"] == total
    assert source.gets.count(key) == 1


def test_sidecar_retry_rejects_wrong_approval_and_worst_case_cap(tmp_path):
    root, source, _, size, _ = incident(tmp_path, extra=False)
    with pytest.raises(ValueError, match="approval"):
        recover_job_archive_request(
            root, batch=0, object_index=0, approved_bytes=size - 1, source=source
        )
    assert not (root / "retry_budget.sqlite3").exists()
    with pytest.raises(ValueError, match="budget"):
        recover_job_archive_request(
            root, batch=0, object_index=0, approved_bytes=size, source=source
        )
    assert not source.gets


def test_failed_sidecar_retry_retains_charge_and_requires_new_attempt(tmp_path):
    root, source, key, size, _ = incident(tmp_path)
    broken = Broken(source.bodies)
    with pytest.raises(RuntimeError, match="network"):
        recover_job_archive_request(
            root, batch=0, object_index=0, approved_bytes=size, source=broken
        )
    with sqlite3.connect(root / "retry_budget.sqlite3") as db:
        assert db.execute("SELECT attempt,bytes,status FROM attempts").fetchall() == [
            (1, size, "requested")
        ]
    manifest = next((root / "batches/0000/raw").glob("*/manifest.json"))
    assert json.loads(manifest.read_text())["objects"][0]["status"] == "requested"
    assert broken.gets == [key]
