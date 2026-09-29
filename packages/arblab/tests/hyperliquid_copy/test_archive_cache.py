import json

import pytest

from arblab.hyperliquid_copy.archive_budget import ArchiveBudget, BudgetedArchiveSource
from arblab.hyperliquid_copy.download import BUCKET
from arblab.hyperliquid_copy.proxy_archive_download import (
    download_archive,
    resume_archive,
)
from .test_archive_budget import Source, objects


def cached(tmp_path):
    return download_archive(
        Source(), "2026-08-01", "2026-08-02", tmp_path / "cache", max_bytes=72
    )


def test_verified_cache_avoids_network_reservations_and_resumes(tmp_path):
    from arblab.hyperliquid_copy.archive_cache import (
        VerifiedArchiveCache,
        CachedArchiveSource,
    )

    cache = VerifiedArchiveCache([cached(tmp_path)], objects())
    assert cache.total_bytes == 72
    assert len(cache.evidence) == 1
    assert len(cache.entries) == 24
    remote = Source()
    budget = ArchiveBudget(tmp_path / "budget.sqlite3", objects(), 72)
    adapter = CachedArchiveSource(cache, BudgetedArchiveSource(remote, budget))
    out = download_archive(
        adapter, "2026-08-01", "2026-08-03", tmp_path / "job", max_bytes=144
    )
    assert len(remote.gets) == 24
    assert budget.reserved_bytes == 72
    assert all("20260802" in key for key in remote.gets)
    assert json.loads(out.read_text())["complete"]
    resume_archive(adapter, out)
    assert len(remote.gets) == 24


def test_corrupt_cache_rejected_without_network_fallback(tmp_path):
    from arblab.hyperliquid_copy.archive_cache import (
        VerifiedArchiveCache,
        CachedArchiveSource,
    )

    manifest = cached(tmp_path)
    cache = VerifiedArchiveCache([manifest], objects())
    (manifest.parent / "fills_0000.lz4").write_bytes(b"bad")
    with pytest.raises(ValueError, match="identity"):
        VerifiedArchiveCache([manifest], objects())
    with pytest.raises(ValueError, match="identity"):
        CachedArchiveSource(cache).get_object(
            Bucket=BUCKET,
            Key=objects()[0]["key"],
            RequestPayer="requester",
            IfMatch='"fixed"',
        )


def test_cache_body_checks_hash_after_open_before_successful_eof(tmp_path):
    from arblab.hyperliquid_copy.archive_cache import (
        VerifiedArchiveCache,
        CachedArchiveSource,
    )

    manifest = cached(tmp_path)
    cache = VerifiedArchiveCache([manifest], objects())
    body = CachedArchiveSource(cache).get_object(
        Bucket=BUCKET,
        Key=objects()[0]["key"],
        RequestPayer="requester",
        IfMatch='"fixed"',
    )["Body"]
    try:
        (manifest.parent / "fills_0000.lz4").write_bytes(b"bad")
        body.read(3)
        with pytest.raises(ValueError, match="identity"):
            body.read(1)
    finally:
        body.close()


def test_missing_cache_only_object_and_unscoped_requests_fail_closed(tmp_path):
    from arblab.hyperliquid_copy.archive_cache import (
        VerifiedArchiveCache,
        CachedArchiveSource,
    )

    adapter = CachedArchiveSource(VerifiedArchiveCache([cached(tmp_path)], objects()))
    with pytest.raises(ValueError, match="not cached"):
        adapter.head_object(
            Bucket=BUCKET, Key=objects()[-1]["key"], RequestPayer="requester"
        )
    with pytest.raises(ValueError, match="request"):
        adapter.get_object(
            Bucket=BUCKET,
            Key=objects()[0]["key"],
            RequestPayer="requester",
            IfMatch='"changed"',
        )


def test_duplicate_and_changed_frozen_cache_identity_rejected(tmp_path):
    from arblab.hyperliquid_copy.archive_cache import VerifiedArchiveCache

    manifest = cached(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        VerifiedArchiveCache([manifest, manifest], objects())
    scope = objects()
    scope[0]["etag"] = '"changed"'
    with pytest.raises(ValueError, match="identity"):
        VerifiedArchiveCache([manifest], scope)


def test_mutating_cached_body_cannot_publish_a_successful_download(tmp_path):
    from arblab.hyperliquid_copy.archive_cache import (
        VerifiedArchiveCache,
        CachedArchiveSource,
    )

    manifest = cached(tmp_path)

    class Changed(CachedArchiveSource):
        def get_object(self, **args):
            response = super().get_object(**args)
            (manifest.parent / "fills_0000.lz4").write_bytes(b"bad")
            return response

    source = Changed(VerifiedArchiveCache([manifest], objects()))
    with pytest.raises(ValueError, match="identity"):
        download_archive(
            source, "2026-08-01", "2026-08-02", tmp_path / "job", max_bytes=72
        )
    result = json.loads(next((tmp_path / "job").glob("*/manifest.json")).read_text())
    assert not result["complete"]
    assert result["objects"][0]["status"] == "requested"


def test_symlink_cache_and_unbudgeted_remote_rejected(tmp_path):
    from arblab.hyperliquid_copy.archive_cache import (
        VerifiedArchiveCache,
        CachedArchiveSource,
    )

    manifest = cached(tmp_path)
    cache = VerifiedArchiveCache([manifest], objects())
    with pytest.raises(ValueError, match="budget"):
        CachedArchiveSource(cache, Source())
    link = tmp_path / "linked"
    link.symlink_to(manifest.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        VerifiedArchiveCache([link / "manifest.json"], objects())
