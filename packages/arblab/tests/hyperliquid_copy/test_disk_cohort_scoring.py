from dataclasses import replace

import pytest

from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows
from arblab.hyperliquid_copy.lab_config import day, METRICS
from .test_disk_metric_rows import row
from .test_disk_score_query import resources
from .test_lab_ranking import settings


def fixture(resources, config=None, records=None):
    from arblab.hyperliquid_copy.disk_cohort_scoring import metric_context

    config = config or settings()
    decision = day(config.start)
    provenance = {"fixture_sha256": "a" * 64}
    inputs = dict(
        context=metric_context(config, decision, "BTC", "gross_excludes_fee"),
        provenance=provenance,
    )
    token = write_metric_rows(
        resources,
        iter(records if records is not None else [row(i, i) for i in range(7)]),
        METRICS,
    )
    PublishedArtifacts(resources).publish("candidate_metrics", inputs, [token])
    return config, decision, inputs


def run(resources, config, decision, inputs, **kwargs):
    from arblab.hyperliquid_copy.disk_cohort_scoring import score_metric_artifact

    return score_metric_artifact(
        resources,
        inputs,
        config,
        decision,
        "BTC",
        "gross_excludes_fee",
        verify_source=kwargs.pop("verify_source", lambda: inputs["provenance"]),
        **kwargs,
    )


def test_publication_reuse_does_not_query_or_add_charge(resources, monkeypatch):
    from arblab.hyperliquid_copy import disk_cohort_scoring as module

    config, decision, inputs = fixture(resources)
    result = run(resources, config, decision, inputs)
    assert result.candidate_count == result.eligible_count == 7
    assert [r["user"] for r in result.selected] == [row(i)["user"] for i in (6, 5)]
    assert result.cutoff_address == row(5)["user"]
    assert sum(len(batch) for batch in result.iter_batches()) == 7
    accounting = resources.audit()
    assert accounting["reserved_bytes"] == 0

    def unexpected(*args, **kwargs):
        pytest.fail("cache hit reran scoring or cohort query")

    monkeypatch.setattr(module.ScoreQuery, "__enter__", unexpected)
    monkeypatch.setattr(module.CohortQuery, "__enter__", unexpected)
    reused = run(resources, replace(config, reselection="weekly"), decision, inputs)
    assert reused == result
    assert resources.audit() == accounting
    selected = result.selected
    selected[0]["metrics"]["gross_volume"] = -100
    assert result.selected[0]["metrics"]["gross_volume"] == 6


def test_changed_cohort_reuses_scores_but_has_new_final_identity(
    resources, monkeypatch
):
    from arblab.hyperliquid_copy import disk_cohort_scoring as module

    config, decision, inputs = fixture(resources)
    first = run(resources, config, decision, inputs)

    def unexpected(*args, **kwargs):
        pytest.fail("cohort-only change repeated percentile pass")

    monkeypatch.setattr(module.ScoreQuery, "__enter__", unexpected)
    second = run(resources, replace(config, top_n=3), decision, inputs)
    assert first.publication.key != second.publication.key
    assert len(second.selected) == 3


def test_input_context_mismatch_rejects_before_reserving(resources):
    config, decision, inputs = fixture(resources)
    before = resources.audit()
    with pytest.raises(ValueError, match="context"):
        run(resources, replace(config, lookback_days=89), decision, inputs)
    assert resources.audit() == before


def test_changed_upstream_rejects_cache_hit(resources):
    config, decision, inputs = fixture(resources)
    run(resources, config, decision, inputs)
    with pytest.raises(ValueError, match="provenance"):
        run(
            resources,
            config,
            decision,
            inputs,
            verify_source=lambda: {"fixture_sha256": "b" * 64},
        )


def test_changed_input_payload_rejects_cache_hit(resources):
    config, decision, inputs = fixture(resources)
    run(resources, config, decision, inputs)
    pin = PublishedArtifacts(resources).lookup("candidate_metrics", inputs).artifacts[0]
    with (resources.root / pin.path).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        run(resources, config, decision, inputs)


def test_failed_output_stays_charged_and_unpublished(resources):
    config, decision, inputs = fixture(resources)
    with pytest.raises(ValueError, match="byte limit"):
        run(resources, config, decision, inputs, max_bytes=10)
    assert resources.audit()["reserved_bytes"] == 10
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 1


def test_upstream_change_during_write_prevents_publication(resources, monkeypatch):
    from arblab.hyperliquid_copy import disk_cohort_scoring as module

    config, decision, inputs = fixture(resources)
    current = dict(inputs["provenance"])
    original = module.ScoreQuery.write_scores

    def change(query, *args, **kwargs):
        result = original(query, *args, **kwargs)
        current["fixture_sha256"] = "b" * 64
        return result

    monkeypatch.setattr(module.ScoreQuery, "write_scores", change)
    with pytest.raises(ValueError, match="provenance"):
        run(resources, config, decision, inputs, verify_source=lambda: current)
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 1
    assert resources.audit()["reserved_bytes"] == 0


def test_duplicate_candidates_never_publish_scores(resources):
    config, decision, inputs = fixture(resources, records=[row(), row()])
    with pytest.raises(ValueError, match="Duplicate"):
        run(resources, config, decision, inputs)
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 1


def test_reopens_same_publication_with_new_lease(resources):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

    config, decision, inputs = fixture(resources)
    first = run(resources, config, decision, inputs)
    before = resources.audit()
    resources.lease.__exit__(None, None, None)
    with CacheLease(resources.root) as lease:
        reopened = CacheResources(lease, "score-test")
        assert run(reopened, config, decision, inputs) == first
        assert reopened.audit() == before


def test_empty_candidate_publication_stays_typed(resources):
    config, decision, inputs = fixture(resources, records=[])
    result = run(resources, config, decision, inputs)
    assert result.candidate_count == result.eligible_count == 0
    assert result.selected == () and result.cutoff_address is None
    assert list(result.iter_batches()) == []


def test_reordered_metric_terms_cannot_reuse_old_context(resources):
    config, decision, inputs = fixture(
        resources, config=settings(metric_weights=dict.fromkeys(METRICS, 1))
    )
    run(resources, config, decision, inputs)
    terms = tuple(config.metric_weights.items())
    config.metric_weights.clear()
    config.metric_weights.update(reversed(terms))
    with pytest.raises(ValueError, match="context"):
        run(resources, config, decision, inputs)


def test_failed_final_output_keeps_score_publication_and_pending_bytes(
    resources, monkeypatch
):
    from arblab.hyperliquid_copy import disk_cohort_scoring as module

    config, decision, inputs = fixture(resources)
    original = module.CohortQuery.write_rankings

    def fail(query, *args, **kwargs):
        original(query, *args, **kwargs)
        raise OSError("injected interruption")

    monkeypatch.setattr(module.CohortQuery, "write_rankings", fail)
    with pytest.raises(OSError, match="interruption"):
        run(resources, config, decision, inputs)
    assert resources.audit()["reserved_bytes"] == 512 * 1024**2
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 2


def test_published_ranking_corruption_is_rejected_by_stream(resources):
    config, decision, inputs = fixture(resources)
    result = run(resources, config, decision, inputs)
    with result.artifact.path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        list(result.iter_batches())


def test_effective_cross_class_config_is_supported(resources):
    from arblab.hyperliquid_copy.disk_cohort_scoring import (
        metric_context,
        score_metric_artifact,
    )
    from arblab.hyperliquid_copy.lab_config_v2 import LabConfigV2

    config = LabConfigV2().effective(["xyz:GOLD"], {"xyz:GOLD": 1})
    decision = day("2026-08-01")
    provenance = {"fixture_sha256": "a" * 64}
    inputs = dict(
        context=metric_context(config, decision, "xyz:GOLD", "gross_excludes_fee"),
        provenance=provenance,
    )
    token = write_metric_rows(resources, (row(i, i) for i in range(7)), METRICS)
    PublishedArtifacts(resources).publish("candidate_metrics", inputs, [token])
    result = score_metric_artifact(
        resources,
        inputs,
        config,
        decision,
        "xyz:GOLD",
        "gross_excludes_fee",
        verify_source=lambda: provenance,
    )
    assert result.candidate_count == 7
    assert all(r["coin"] == "xyz:GOLD" for r in result.selected)


def test_shared_budget_rejects_before_starting_queries(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import disk_cohort_scoring as module
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

    root = tmp_path / "small-cache"
    root.mkdir()
    config = settings()
    decision = day(config.start)
    inputs = dict(
        context=module.metric_context(config, decision, "BTC", "gross_excludes_fee"),
        provenance={"fixture_sha256": "a" * 64},
    )
    with CacheLease(root) as lease:
        resources = CacheResources.create(
            lease, "small-score-test", limit_bytes=64 * 1024**2
        )
        token = write_metric_rows(resources, iter([row()]), METRICS, max_bytes=1024**2)
        PublishedArtifacts(resources).publish("candidate_metrics", inputs, [token])
        before = resources.audit()

        def unexpected(*args, **kwargs):
            pytest.fail("query started before budget reservation")

        monkeypatch.setattr(module.ScoreQuery, "__enter__", unexpected)
        with pytest.raises(ValueError, match="budget"):
            run(resources, config, decision, inputs)
        assert resources.audit() == before
