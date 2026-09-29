import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.lab_config import METRICS, day
from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner
from arblab.hyperliquid_copy.ranking_staging_rows import write_pending_metrics
from .test_candidate_day import resources
from .test_disk_metric_rows import row
from .test_lab_ranking import settings


def test_pending_scoring_rejects_empty_provenance_before_outputs(resources):
    from arblab.hyperliquid_copy.ranking_staging_score import score_pending_metrics

    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    metrics = write_pending_metrics(owner, [row()], METRICS)
    try:
        with pytest.raises(ValueError, match="provenance"):
            score_pending_metrics(
                owner,
                metrics,
                settings(),
                day("2026-08-03"),
                "BTC",
                "gross_excludes_fee",
                verify_source=lambda: {},
            )
        assert not owner.path("scores").exists() and not owner.path("ranking").exists()
    finally:
        owner.close()


def test_pending_scoring_rechecks_config_after_final_artifact_hash(
    resources, monkeypatch
):
    from arblab.hyperliquid_copy.ranking_staging_score import score_pending_metrics
    from arblab.hyperliquid_copy.ranking_staging_artifact import StagingArtifact

    config = settings()
    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    metrics = write_pending_metrics(owner, [row()], METRICS)
    original = StagingArtifact.verify

    def changed(self):
        original(self)
        if self.path == owner.path("ranking"):
            config.metric_weights["pnl_efficiency"] = 2

    monkeypatch.setattr(StagingArtifact, "verify", changed)
    try:
        with pytest.raises(ValueError):
            score_pending_metrics(
                owner,
                metrics,
                config,
                day("2026-08-03"),
                "BTC",
                "gross_excludes_fee",
                verify_source=lambda: {"verified": "fixture"},
            )
    finally:
        owner.close()


def test_pending_scoring_rejects_config_change_in_final_owner_check(
    resources, monkeypatch
):
    from arblab.hyperliquid_copy.ranking_staging_score import score_pending_metrics

    config, decision = settings(), day("2026-08-03")
    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    metrics = write_pending_metrics(owner, [row()], METRICS)
    original, calls = RankingStagingOwner.verify, 0

    def mutate(self):
        nonlocal calls
        original(self)
        calls += 1
        if calls == 6:
            config.metric_weights["pnl_efficiency"] = 2

    monkeypatch.setattr(RankingStagingOwner, "verify", mutate)
    try:
        with pytest.raises(ValueError):
            score_pending_metrics(
                owner,
                metrics,
                config,
                decision,
                "BTC",
                "gross_excludes_fee",
                verify_source=lambda: {"verified": "fixture"},
            )
        assert calls == 6
    finally:
        owner.close()


@pytest.mark.parametrize("fault", ["config", "metrics"])
def test_pending_scoring_checks_inputs_after_score_output(
    resources, monkeypatch, fault
):
    from arblab.hyperliquid_copy.ranking_staging_score import score_pending_metrics
    from arblab.hyperliquid_copy.disk_score_query import ScoreQuery

    config, decision = settings(), day("2026-08-03")
    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    metrics = write_pending_metrics(owner, [row()], METRICS)
    original = ScoreQuery.write_scores

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if fault == "config":
            config.metric_weights["pnl_efficiency"] = 2
        else:
            with metrics.path.open("ab") as stream:
                stream.write(b"changed")
        return result

    monkeypatch.setattr(ScoreQuery, "write_scores", changed)
    try:
        with pytest.raises(ValueError):
            score_pending_metrics(
                owner,
                metrics,
                config,
                decision,
                "BTC",
                "gross_excludes_fee",
                verify_source=lambda: {"verified": "fixture"},
            )
        assert not owner.path("ranking").exists()
    finally:
        owner.close()


@pytest.mark.parametrize("count,scope", [(0, "BTC"), (5, "BTC"), (5, None)])
def test_pending_score_passes_match_existing_kernels(resources, tmp_path, count, scope):
    from arblab.hyperliquid_copy.ranking_staging_score import score_pending_metrics
    from arblab.hyperliquid_copy.disk_score_query import ScoreQuery
    from arblab.hyperliquid_copy.disk_cohort_query import CohortQuery

    config, decision = settings(), day("2026-08-03")
    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    try:
        metrics = write_pending_metrics(
            owner, (row(i, float(i)) for i in range(count)), METRICS
        )
        ref_scores, ref_ranks = (
            tmp_path / "scores.parquet",
            tmp_path / "rankings.parquet",
        )
        with ScoreQuery(metrics.path, tmp_path, config) as query:
            query.write_scores(ref_scores)
        with CohortQuery(ref_scores, tmp_path, config, decision, scope) as query:
            expected = query.write_rankings(ref_ranks)
        before = resources.audit()
        result = score_pending_metrics(
            owner,
            metrics,
            config,
            decision,
            scope,
            "gross_excludes_fee",
            verify_source=lambda: {"verified": "fixture"},
        )
        assert (
            pq.read_table(result.ranking.path).to_pylist()
            == pq.read_table(ref_ranks).to_pylist()
        )
        assert (
            pq.read_table(result.scores.path).to_pylist()
            == pq.read_table(ref_scores).to_pylist()
        )
        assert result.summary == expected
        result.ranking.verify()
        result.scores.verify()
        assert resources.audit() == before
    finally:
        owner.close()


@pytest.mark.parametrize("fault", ["provenance", "config", "metrics"])
def test_pending_scoring_rejects_changes(resources, fault):
    from arblab.hyperliquid_copy.ranking_staging_score import score_pending_metrics

    config, decision = settings(), day("2026-08-03")
    owner = RankingStagingOwner.create(resources, {"query": "fixture"})
    metrics = write_pending_metrics(owner, [row()], METRICS)
    calls = 0

    def verify():
        nonlocal calls
        calls += 1
        if calls > 1:
            if fault == "provenance":
                return {"changed": True}
            if fault == "config":
                config.metric_weights["pnl_efficiency"] = 2
            if fault == "metrics":
                with metrics.path.open("ab") as stream:
                    stream.write(b"corruption")
        return {"verified": "fixture"}

    try:
        with pytest.raises(ValueError):
            score_pending_metrics(
                owner,
                metrics,
                config,
                decision,
                "BTC",
                "gross_excludes_fee",
                verify_source=verify,
            )
    finally:
        owner.close()
