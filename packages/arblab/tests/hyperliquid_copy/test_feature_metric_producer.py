from dataclasses import replace
from datetime import timedelta

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from .test_candidate_day import resources
from .test_feature_window import days
from .test_feature_day_builder import DAY, SEMANTICS
from .test_qualified_day import qualified
from .test_lab_ranking import settings


def args(resources, qualified, days, offset=2):
    return (
        resources,
        qualified,
        "2026-08-01",
        days[offset - 1 : offset],
        DAY + timedelta(days=offset),
        settings(lookback_days=1, min_volume=0, metric_weights={"gross_volume": 1}),
        "BTC",
        SEMANTICS,
    )


def assert_metric_rows_close(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        left, right = dict(left), dict(right)
        left_metrics, right_metrics = left.pop("metrics"), right.pop("metrics")
        assert left == right
        assert left_metrics == pytest.approx(right_metrics, nan_ok=True)


def test_feature_ranking_matches_raw_and_reuses_monday_inputs(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as module
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_and_score_candidates,
    )

    values = args(resources, qualified, days)
    actual = module.build_and_score_features(*values)
    expected = build_and_score_candidates(*values[:3], *values[4:])
    assert_metric_rows_close(
        [r for batch in actual.iter_batches() for r in batch],
        [r for batch in expected.iter_batches() for r in batch],
    )
    assert actual.candidate_count == expected.candidate_count == 2
    before = resources.audit()

    def forbidden(*a, **kw):
        pytest.fail("feature ranking reuse repeated sorting")

    monkeypatch.setattr(module, "merge_feature_metric_rows_from_days", forbidden)
    weekly = (*values[:5], replace(values[5], reselection="weekly"), *values[6:])
    assert module.build_and_score_features(*weekly) == actual
    assert resources.audit() == before


def test_dormant_feature_day_retains_all_candidates(qualified, resources, days):
    from arblab.hyperliquid_copy.feature_metric_producer import (
        build_feature_candidate_metrics,
    )

    inputs = build_feature_candidate_metrics(
        *args(resources, qualified, days, offset=3)
    )
    publication = PublishedArtifacts(resources).lookup("candidate_metrics", inputs)
    rows = pq.read_table(resources.root / publication.artifacts[0].path).to_pylist()
    assert len(rows) == 2
    assert all(
        r["metrics"]["gross_volume"] == 0
        and r["exclusions"] == ["no_activity_in_lookback"]
        for r in rows
    )
    assert resources.audit()["reserved_bytes"] == 0


def test_output_and_scratch_reserved_before_order_query(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as module

    original, seen = module.stage_metric_summaries, []

    def stage(*values, **kw):
        seen.append(resources.audit()["reserved_bytes"])
        assert seen[-1] == 3 * 1024**3 + 512 * 1024**2
        return original(*values, **kw)

    monkeypatch.setattr(module, "stage_metric_summaries", stage)
    module.build_feature_candidate_metrics(*args(resources, qualified, days))
    assert seen


def test_new_feature_ranking_merges_days_without_external_sort(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as module
    from arblab.hyperliquid_copy import ordered_feature_partitions

    class ForbiddenSorter:
        def __init__(self, *args, **kwargs):
            pytest.fail("feature-backed ranking materialised an external sort")

    monkeypatch.setattr(
        ordered_feature_partitions, "OrderedFeaturePartitions", ForbiddenSorter
    )
    monkeypatch.setattr(
        module,
        "merge_feature_metric_rows_from_days",
        lambda *a, **k: pytest.fail("feature-backed ranking decoded rows in Python"),
    )
    result = module.build_and_score_features(*args(resources, qualified, days))
    assert result.candidate_count == 2
    assert not list((resources.root / "scratch").iterdir())


def test_cleanup_failure_cannot_publish_settled_metrics(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as module

    def fail(*a, **kw):
        raise ValueError("injected cleanup failure")

    monkeypatch.setattr(module, "_release_empty_scratch", fail)
    with pytest.raises(ValueError, match="cleanup failure"):
        module.build_feature_candidate_metrics(*args(resources, qualified, days))
    with resources._connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM publications WHERE json_extract(descriptor,'$.kind')='candidate_metrics'"
            ).fetchone()[0]
            == 0
        )
    assert resources.audit()["reserved_bytes"] == 3 * 1024**3


@pytest.mark.parametrize("target", ["feature", "config", "engine"])
def test_changes_after_metric_settlement_block_publication(
    qualified, resources, days, monkeypatch, target
):
    from arblab.hyperliquid_copy import feature_metric_producer as module

    values = args(resources, qualified, days)
    original = module.write_metric_rows

    def write(*a, **kw):
        result = original(*a, **kw)
        if target == "feature":
            with (resources.root / days[1].publication.artifacts[0].path).open(
                "ab"
            ) as handle:
                handle.write(b"changed feature")
        elif target == "config":
            values[5].metric_weights["gross_volume"] = 0.5
        else:
            monkeypatch.setattr(module, "_engine", lambda: {"changed": True})
        return result

    monkeypatch.setattr(module, "write_metric_rows", write)
    with pytest.raises(ValueError):
        module.build_feature_candidate_metrics(*values)
    with resources._connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM publications WHERE json_extract(descriptor,'$.kind')='candidate_metrics'"
            ).fetchone()[0]
            == 0
        )


def test_candidate_mutation_after_stream_exhaustion_blocks_publication(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as module

    original = module._candidate_rows

    def candidates(r, publication):
        yield from original(r, publication)
        with (r.root / publication.artifacts[0].path).open("ab") as handle:
            handle.write(b"changed candidate")

    monkeypatch.setattr(module, "_candidate_rows", candidates)
    with pytest.raises(ValueError):
        module.build_feature_candidate_metrics(*args(resources, qualified, days))


def test_output_overflow_keeps_unpublished_obligation(qualified, resources, days):
    from arblab.hyperliquid_copy.feature_metric_producer import (
        build_feature_candidate_metrics,
    )

    with pytest.raises(ValueError, match="byte limit"):
        build_feature_candidate_metrics(*args(resources, qualified, days), max_bytes=10)
    with resources._connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM publications WHERE json_extract(descriptor,'$.kind')='candidate_metrics'"
            ).fetchone()[0]
            == 0
        )
    assert resources.audit()["reserved_bytes"] >= 10


@pytest.mark.parametrize(
    "target",
    ["feature_engine", "qualification_engine", "feature_file", "candidate_file"],
)
def test_mutation_in_final_candidate_hash_tail_is_rejected(
    qualified, resources, days, monkeypatch, target
):
    from arblab.hyperliquid_copy import feature_metric_producer as module
    from arblab.hyperliquid_copy import feature_publication, qualified_window

    window, candidate, _, verify = module._prepare(
        *args(resources, qualified, days),
        512 * 1024**2,
        250000,
    )
    candidate_path = resources.root / candidate.artifacts[0].path
    original = module.file_hash

    def digest(path):
        result = original(path)
        if path == candidate_path:
            if target == "feature_engine":
                monkeypatch.setattr(
                    feature_publication, "feature_engine", lambda: {"changed": True}
                )
            elif target == "qualification_engine":
                monkeypatch.setattr(
                    qualified_window, "_engine", lambda: {"changed": True}
                )
            else:
                changed = (
                    candidate_path
                    if target == "candidate_file"
                    else resources.root / window.observation_pins[0].path
                )
                with changed.open("ab") as handle:
                    handle.write(b"changed during final hash")
        return result

    monkeypatch.setattr(module, "file_hash", digest)
    with pytest.raises(ValueError):
        verify()


def test_old_candidate_source_outside_lookback_cannot_change_in_hash_tail(
    qualified, resources, days, monkeypatch
):
    import json
    from pathlib import Path
    from arblab.hyperliquid_copy import feature_metric_producer as module

    window, candidate, _, verify = module._prepare(
        *args(resources, qualified, days, offset=3), 512 * 1024**2, 250000
    )
    excluded = {window.source.witness.path, window.source._anchor.witness.path}
    excluded.update(e.path for e in window.source.entries)
    report = json.loads(Path(qualified["path"]).read_text())
    older = next(
        Path(e["path"]) for e in report["files"] if Path(e["path"]) not in excluded
    )
    candidate_path = resources.root / candidate.artifacts[0].path
    original = module.file_hash

    def digest(path):
        value = original(path)
        if path == candidate_path:
            with older.open("ab") as handle:
                handle.write(b"changed dormant candidate source")
        return value

    monkeypatch.setattr(module, "file_hash", digest)
    with pytest.raises(ValueError):
        verify()


def test_multiple_partitions_finish_wallet_spool_before_next_sort(
    qualified, resources, days
):
    from arblab.hyperliquid_copy import feature_metric_producer as module
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_candidate_metrics,
    )

    inputs = module.build_feature_candidate_metrics(
        *args(resources, qualified, days), max_partition_rows=4
    )
    publication = PublishedArtifacts(resources).lookup("candidate_metrics", inputs)
    rows = pq.read_table(resources.root / publication.artifacts[0].path).to_pylist()
    assert len(rows) == 2
    values = args(resources, qualified, days)
    reference_inputs = build_candidate_metrics(*values[:3], *values[4:])
    reference = PublishedArtifacts(resources).lookup(
        "candidate_metrics", reference_inputs
    )
    assert_metric_rows_close(
        rows,
        pq.read_table(resources.root / reference.artifacts[0].path).to_pylist(),
    )
    assert resources.audit()["reserved_bytes"] == 0
    assert not list((resources.root / "scratch").iterdir())


def test_feature_ranking_reopens_under_fresh_lease_without_sort(
    qualified, tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as module
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.feature_publication import FeatureDay
    from .test_feature_day_builder import build

    root = tmp_path / "fresh_ranking_cache"
    root.mkdir()
    with CacheLease(root) as lease:
        original = CacheResources.create(lease, "feature-ranking-resume")
        first = build(original, qualified)
        second = build(original, qualified, DAY + timedelta(days=1), first)
        previous = module.build_and_score_features(
            *args(original, qualified, [first, second])
        )
        saved = [first.inputs, second.inputs]
        before = original.audit()
    with CacheLease(root) as lease:
        reopened = CacheResources(lease, "feature-ranking-resume")
        restored = [FeatureDay(reopened, data) for data in saved]

        def forbidden(*a, **kw):
            pytest.fail("fresh-lease feature ranking reuse sorted input again")

        monkeypatch.setattr(module, "merge_feature_metric_rows_from_days", forbidden)
        actual = module.build_and_score_features(*args(reopened, qualified, restored))
        assert actual == previous
        assert reopened.audit() == before
