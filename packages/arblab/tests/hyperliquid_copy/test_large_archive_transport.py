import hashlib
import json
from types import SimpleNamespace

import pytest

from arblab.hyperliquid_copy.archive import archive_keys
from arblab.hyperliquid_copy.archive_budget import ArchiveBudget, BudgetedArchiveSource
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_archive_download import (
    download_archive,
    resume_archive,
)

LIMIT = 384 * 1024**2


class Body:
    def __init__(self, size):
        self.remaining = size
        self.largest_read = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def read(self, size):
        assert 0 < size <= 1024**2
        self.largest_read = max(self.largest_read, size)
        value = b"x" * min(size, self.remaining)
        self.remaining -= len(value)
        self.digest.update(value)
        return value

    def close(self):
        self.closed = True


class Source:
    meta = SimpleNamespace(config=SimpleNamespace(retries={"total_max_attempts": 1}))

    def __init__(self, size):
        self.size, self.gets, self.bodies = size, [], []
        self.keys = archive_keys("2026-08-01")

    @property
    def objects(self):
        return [
            dict(key=k, bytes=self.size if i == 0 else 3, etag='"fixed"')
            for i, k in enumerate(self.keys)
        ]

    def head_object(self, **args):
        return dict(
            ContentLength=self.size if args["Key"] == self.keys[0] else 3,
            ETag='"fixed"',
        )

    def get_object(self, **args):
        assert args["IfMatch"] == '"fixed"'
        self.gets.append(args["Key"])
        meta = self.head_object(**args)
        body = Body(meta["ContentLength"])
        self.bodies.append(body)
        return meta | {"Body": body}


@pytest.mark.parametrize("size", [359334134, LIMIT])
def test_large_transport_stays_streamed_and_reserves_once(tmp_path, size):
    source = Source(size)
    total = size + 23 * 3
    budget = ArchiveBudget(tmp_path / "budget.sqlite3", source.objects, total)
    adapter = BudgetedArchiveSource(source, budget)
    manifest = download_archive(
        adapter, "2026-08-01", "2026-08-02", tmp_path / "raw", max_bytes=total
    )
    data = json.loads(manifest.read_text())
    assert data["max_object_bytes"] == LIMIT
    first = manifest.parent / data["objects"][0]["file"]
    assert first.stat().st_size == size
    assert (
        data["objects"][0]["sha256"]
        == file_hash(first)
        == source.bodies[0].digest.hexdigest()
    )
    assert max(b.largest_read for b in source.bodies) <= 1024**2
    assert all(b.closed for b in source.bodies)
    assert budget.reserved_bytes == total
    resume_archive(adapter, manifest)
    assert len(source.gets) == 24
    assert budget.reserved_bytes == total


@pytest.mark.parametrize("size,budget", [(LIMIT + 1, 6 * 1024**3), (LIMIT, LIMIT)])
def test_unsupported_size_or_batch_budget_rejects_before_get(tmp_path, size, budget):
    source = Source(size)
    with pytest.raises(ValueError):
        download_archive(
            source, "2026-08-01", "2026-08-02", tmp_path / "raw", max_bytes=budget
        )
    assert not source.gets


@pytest.mark.parametrize("limit", [None, True, 0, LIMIT + 1])
def test_resume_does_not_widen_legacy_or_invalid_object_limits(tmp_path, limit):
    from arblab.hyperliquid_copy.download import BUCKET

    source = Source(129 * 1024**2)
    objects = [
        obj | {"file": f"fills_{i:04d}.lz4"} for i, obj in enumerate(source.objects)
    ]
    total = sum(o["bytes"] for o in objects)
    data = dict(
        schema="hyperliquid_proxy_archive_v1",
        bucket=BUCKET,
        start="2026-08-01",
        end="2026-08-02",
        objects=objects,
        expected_bytes=total,
        max_bytes=total,
        reserved_bytes=0,
        complete=False,
    )
    if limit is not None:
        data["max_object_bytes"] = limit
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        resume_archive(source, manifest)
    assert not source.gets
