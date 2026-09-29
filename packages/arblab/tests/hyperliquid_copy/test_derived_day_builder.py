import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from .test_qualified_day import qualified


def test_whole_day_publication_reuses_without_queries_or_charges(
    qualified, tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import derived_day_builder as module

    root = tmp_path / "derived"
    root.mkdir()
    with CacheLease(root) as lease:
        resources = CacheResources.create(lease, "builder-test")
        first = module.build_projection_day(
            resources, qualified, "2026-08-01", max_partition_rows=4
        )
        assert (
            sum(pq.read_metadata(root / a.path).num_rows for a in first.artifacts) == 48
        )
        before = resources.audit()
        assert before["reserved_bytes"] == 0
        assert list((root / "scratch").iterdir()) == []

        def unexpected(*args, **kwargs):
            raise AssertionError("cache hit reran projection")

        monkeypatch.setattr(module.ProjectionQuery, "__enter__", unexpected)
        second = module.build_projection_day(
            resources, qualified, "2026-08-01", max_partition_rows=4
        )
        assert second == first
        assert resources.audit() == before


def test_changed_source_rejects_even_a_cache_hit(qualified, tmp_path):
    from arblab.hyperliquid_copy.derived_day_builder import build_projection_day

    root = tmp_path / "derived"
    root.mkdir()
    with CacheLease(root) as lease:
        resources = CacheResources.create(lease, "builder-test")
        build_projection_day(resources, qualified, "2026-08-01")
        source = Path(
            json.loads(Path(qualified["path"]).read_text())["files"][0]["path"]
        )
        with source.open("ab") as handle:
            handle.write(b"changed")
        with pytest.raises(ValueError):
            build_projection_day(resources, qualified, "2026-08-01")


def test_failed_write_stays_charged_without_visible_day(qualified, tmp_path):
    from arblab.hyperliquid_copy.derived_day_builder import build_projection_day

    root = tmp_path / "derived"
    root.mkdir()
    with CacheLease(root) as lease:
        resources = CacheResources.create(lease, "builder-test")
        with pytest.raises(ValueError, match="byte limit"):
            build_projection_day(
                resources, qualified, "2026-08-01", max_output_bytes=10
            )
        assert resources.audit()["reserved_bytes"] == 10
        with resources._connect() as db:
            assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0


def test_source_changed_during_projection_is_not_published(
    qualified, tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import derived_day_builder as module

    root = tmp_path / "derived"
    root.mkdir()
    original = module.ProjectionQuery.write

    def changed(query, *args, **kwargs):
        result = original(query, *args, **kwargs)
        with query.day.entries[0].path.open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(module.ProjectionQuery, "write", changed)
    with CacheLease(root) as lease:
        resources = CacheResources.create(lease, "builder-test")
        with pytest.raises(ValueError, match="identity"):
            module.build_projection_day(resources, qualified, "2026-08-01")
        with resources._connect() as db:
            assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0
        assert resources.audit()["retained_bytes"] > 0
