import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner
from .test_candidate_day import resources
from .test_feature_window import days
from .test_qualified_day import qualified
from .test_feature_metric_producer import args


def assert_metric_equivalent(actual, expected):
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        observed, reference = dict(observed), dict(reference)
        observed_metrics = observed.pop("metrics")
        reference_metrics = reference.pop("metrics")
        assert observed == reference
        assert observed_metrics.keys() == reference_metrics.keys()
        for key in observed_metrics:
            if observed_metrics[key] is None or reference_metrics[key] is None:
                assert observed_metrics[key] is reference_metrics[key]
            else:
                assert observed_metrics[key] == pytest.approx(
                    reference_metrics[key], rel=1e-14, abs=1e-14
                )


@pytest.mark.parametrize("route", ["raw", "features"])
@pytest.mark.parametrize("scope,offset", [("BTC", 2), (None, 2), ("BTC", 3)])
def test_staged_metrics_match_complete_published_reference(
    qualified, resources, days, route, scope, offset
):
    from arblab.hyperliquid_copy.ranking_staging_sources import prepare_staging_source
    from arblab.hyperliquid_copy.ranking_staging_producer import produce_staged_metrics
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_candidate_metrics,
    )

    values = list(args(resources, qualified, days, offset=offset))
    values[6] = scope
    # Published raw computation is independent of staged raw/feature orchestration.
    inputs = build_candidate_metrics(*values[:3], *values[4:])
    reference = PublishedArtifacts(resources).lookup("candidate_metrics", inputs)
    expected = pq.read_table(resources.root / reference.artifacts[0].path).to_pylist()
    prepared = prepare_staging_source(
        *values[:3],
        *values[4:],
        days=values[3] if route == "features" else None,
        max_partition_rows=4,
    )
    before = resources.audit()
    owner = RankingStagingOwner.create(resources, {"producer": prepared.key})
    try:
        artifact = produce_staged_metrics(owner, prepared)
        actual = pq.read_table(artifact.path).to_pylist()
        if route == "features":
            assert_metric_equivalent(actual, expected)
        else:
            assert actual == expected
        assert len(expected) == 2
        assert list(owner.path("scratch").iterdir()) == []
        assert resources.audit()["retained_bytes"] == before["retained_bytes"]
        assert resources.audit()["reserved_bytes"] == int(4.5 * 1024**3) + 65536
        owner.verify()
    finally:
        owner.close()


def test_feature_staging_has_no_legacy_sort_or_wallet_spool_route():
    from arblab.hyperliquid_copy import ranking_staging_producer as module

    assert not hasattr(module, "OrderedFeaturePartitions")
    assert not hasattr(module, "partition_metric_rows")


@pytest.mark.parametrize("route", ["raw", "features"])
def test_prepared_source_rejects_changed_config_before_output(
    qualified, resources, days, route
):
    from arblab.hyperliquid_copy.ranking_staging_sources import prepare_staging_source
    from arblab.hyperliquid_copy.ranking_staging_producer import produce_staged_metrics

    values = args(resources, qualified, days)
    prepared = prepare_staging_source(
        *values[:3], *values[4:], days=values[3] if route == "features" else None
    )
    owner = RankingStagingOwner.create(resources, {"producer": prepared.key})
    try:
        values[5].metric_weights["gross_volume"] = 0.5
        with pytest.raises(ValueError):
            produce_staged_metrics(owner, prepared)
        assert not owner.path("metrics").exists()
    finally:
        owner.close()


@pytest.mark.parametrize("route", ["raw", "features"])
def test_writer_must_consume_complete_metric_stream(
    qualified, resources, days, monkeypatch, route
):
    from arblab.hyperliquid_copy.ranking_staging_sources import prepare_staging_source
    from arblab.hyperliquid_copy import ranking_staging_producer as module

    values = args(resources, qualified, days)
    prepared = prepare_staging_source(
        *values[:3], *values[4:], days=values[3] if route == "features" else None
    )
    owner = RankingStagingOwner.create(resources, {"producer": prepared.key})
    original = module.write_pending_metrics

    def partial(owner, rows, required):
        return original(owner, [next(rows)], required)

    monkeypatch.setattr(module, "write_pending_metrics", partial)
    try:
        with pytest.raises(ValueError, match="complete"):
            module.produce_staged_metrics(owner, prepared)
        assert resources.audit()["reserved_bytes"] == int(4.5 * 1024**3) + 65536
    finally:
        owner.close()


@pytest.mark.parametrize("kind", ["candidate_day", "candidate_history"])
def test_verifier_never_rebuilds_missing_publication_in_callback(
    qualified, resources, days, monkeypatch, kind
):
    from arblab.hyperliquid_copy.ranking_staging_sources import prepare_staging_source

    values = args(resources, qualified, days)
    prepared = prepare_staging_source(*values[:3], *values[4:])
    with resources._connect() as db, db:
        db.execute(
            "DELETE FROM publications WHERE json_extract(descriptor,'$.kind')=?",
            (kind,),
        )

    def forbidden(*args, **kwargs):
        pytest.fail("read-only staging verification attempted reservation")

    monkeypatch.setattr(type(resources), "reserve", forbidden)
    with pytest.raises(ValueError, match="publication"):
        prepared.verify()


def test_repeated_prepared_guards_do_not_rehash_feature_window(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy.ranking_staging_sources import prepare_staging_source

    values = args(resources, qualified, days)
    prepared = prepare_staging_source(*values[:3], *values[4:], days=values[3])

    def forbidden(*args, **kwargs):
        pytest.fail("repeated prepared guard rehashed the feature window")

    monkeypatch.setattr(type(prepared.window), "verify", forbidden)
    prepared.verify()
    prepared.verify()


@pytest.mark.parametrize("route", ["raw", "features"])
def test_final_source_callback_cannot_replace_spill_parent(
    qualified, resources, days, monkeypatch, route
):
    from arblab.hyperliquid_copy import ranking_staging_sources as sources
    from arblab.hyperliquid_copy.ranking_staging_producer import produce_staged_metrics

    values = args(resources, qualified, days)
    prepared = sources.prepare_staging_source(
        *values[:3], *values[4:], days=values[3] if route == "features" else None
    )
    owner = RankingStagingOwner.create(resources, {"producer": prepared.key})
    original, mutated = sources.PreparedStagingSource.verify, False

    def mutate(current):
        nonlocal mutated
        result = original(current)
        if owner.path("metrics").exists() and not mutated:
            scratch = owner.path("scratch")
            displaced = scratch.with_name(scratch.name + "-displaced")
            scratch.rename(displaced)
            scratch.mkdir()
            (displaced / "spill").rename(scratch / "spill")
            mutated = True
        return result

    monkeypatch.setattr(sources.PreparedStagingSource, "verify", mutate)
    try:
        with pytest.raises(ValueError):
            produce_staged_metrics(owner, prepared)
        assert mutated
        assert (owner.path("scratch") / "spill").exists()
    finally:
        owner.close()


def test_changed_prepared_fields_cannot_reuse_original_verifier(
    qualified, resources, days
):
    from arblab.hyperliquid_copy.ranking_staging_sources import prepare_staging_source

    values = args(resources, qualified, days)
    prepared = prepare_staging_source(*values[:3], *values[4:])
    forged = replace(prepared, scope=None)
    with pytest.raises(ValueError):
        forged.verify()


def test_last_source_callback_cannot_change_completed_metrics(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import ranking_staging_sources as sources
    from arblab.hyperliquid_copy.ranking_staging_producer import produce_staged_metrics

    values = args(resources, qualified, days)
    prepared = sources.prepare_staging_source(*values[:3], *values[4:])
    owner = RankingStagingOwner.create(resources, {"producer": prepared.key})
    original, count = sources.PreparedStagingSource.verify, 0

    def mutate(current):
        nonlocal count
        result = original(current)
        count += 1
        if count == 3:
            with owner.path("metrics").open("ab") as stream:
                stream.write(b"late source callback")
        return result

    monkeypatch.setattr(sources.PreparedStagingSource, "verify", mutate)
    try:
        with pytest.raises(ValueError):
            produce_staged_metrics(owner, prepared)
    finally:
        owner.close()


def test_metric_kernel_must_exhaust_candidates_and_fills(
    qualified, resources, days, monkeypatch
):
    from contextlib import closing
    from arblab.hyperliquid_copy.ranking_staging_sources import prepare_staging_source
    from arblab.hyperliquid_copy import ranking_staging_producer as module

    values = args(resources, qualified, days)
    prepared = prepare_staging_source(*values[:3], *values[4:])
    owner = RankingStagingOwner.create(resources, {"producer": prepared.key})
    original = module.merge_metric_rows

    def truncated(*args, **kwargs):
        with closing(original(*args, **kwargs)) as rows:
            yield next(rows)

    monkeypatch.setattr(module, "merge_metric_rows", truncated)
    try:
        with pytest.raises(ValueError, match="complete"):
            module.produce_staged_metrics(owner, prepared)
    finally:
        owner.close()


@pytest.mark.parametrize("route", ["raw", "features"])
@pytest.mark.parametrize("minimum", [0, 1e12])
def test_full_staged_rankings_match_reference_including_empty_eligible_cohort(
    qualified, resources, days, route, minimum
):
    from arblab.hyperliquid_copy.ranking_staging_sources import prepare_staging_source
    from arblab.hyperliquid_copy.ranking_staging_producer import produce_staged_metrics
    from arblab.hyperliquid_copy.ranking_staging_score import score_pending_metrics
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_and_score_candidates,
    )

    values = list(args(resources, qualified, days))
    values[5] = replace(values[5], min_volume=minimum)
    reference = build_and_score_candidates(*values[:3], *values[4:])
    expected = pq.read_table(reference.artifact.path).to_pylist()
    prepared = prepare_staging_source(
        *values[:3], *values[4:], days=values[3] if route == "features" else None
    )
    owner = RankingStagingOwner.create(resources, {"producer": prepared.key})
    try:
        metrics = produce_staged_metrics(owner, prepared)
        actual = score_pending_metrics(
            owner,
            metrics,
            values[5],
            values[4],
            values[6],
            values[7],
            verify_source=prepared.verify,
        )
        rows = pq.read_table(actual.ranking.path).to_pylist()
        if route == "features":
            assert_metric_equivalent(rows, expected)
        else:
            assert rows == expected
        assert actual.summary["candidate_count"] == reference.candidate_count == 2
        if minimum:
            assert not any(row["selected"] for row in rows)
    finally:
        owner.close()


def test_final_engine_callback_cannot_remove_candidate_receipt(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import ranking_staging_sources as sources

    values = args(resources, qualified, days)
    prepared = sources.prepare_staging_source(*values[:3], *values[4:])
    original = sources._engine

    def changed():
        result = original()
        with resources._connect() as db, db:
            db.execute(
                "DELETE FROM publications WHERE key=?", (prepared.candidate.key,)
            )
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("source verification attempted to rebuild")

    monkeypatch.setattr(sources, "_engine", changed)
    monkeypatch.setattr(type(resources), "reserve", forbidden)
    with pytest.raises(ValueError):
        prepared.verify()


from dataclasses import replace
