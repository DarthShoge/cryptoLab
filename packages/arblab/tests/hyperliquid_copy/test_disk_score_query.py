from datetime import timedelta
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows
from arblab.hyperliquid_copy.lab_config import METRICS
from arblab.hyperliquid_copy.lab_ranking import rank_universe
from .test_disk_metric_rows import output, row
from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from .test_lab_ranking import settings
from .test_ranking import history


@pytest.fixture
def resources(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    with CacheLease(root) as lease:
        yield CacheResources.create(lease, "score-test")


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {
            "metric_weights": {"gross_volume": 1},
            "metric_directions": {"gross_volume": "asc"},
        },
        {"metric_weights": dict.fromkeys(METRICS, 1)},
        {"lookback_days": 1},
    ],
)
def test_scores_match_reference_exactly(resources, tmp_path, changes):
    from arblab.hyperliquid_copy.disk_score_query import ScoreQuery

    fills = history()
    decision = max(f.exchange_time for f in fills) + timedelta(days=2)
    config = settings(**changes)
    expected = rank_universe(fills, decision, config, "BTC", "gross_excludes_fee")
    records = (
        dict(
            user=r["user"],
            metrics=dict.fromkeys(METRICS) | r["metrics"],
            exclusions=r["exclusions"],
        )
        for r in expected
    )
    token = write_metric_rows(resources, records, tuple(config.metric_weights))
    scratch = tmp_path / "query-temp"
    scratch.mkdir()
    target = tmp_path / "scores.parquet"
    with ScoreQuery(output(resources, token), scratch, config) as query:
        assert query.counts == (len(expected), sum(r["eligible"] for r in expected))
        assert query.write_scores(target) == len(expected)
    actual = {r["user"]: r for r in pq.read_table(target).to_pylist()}
    for ref in expected:
        result = actual[ref["user"]]
        assert result["score"] == ref["score"]
        assert {k: v for k, v in result["percentiles"].items() if v is not None} == ref[
            "percentiles"
        ]
        assert result["exclusions"] == ref["exclusions"]


@pytest.mark.parametrize(
    "values,expected",
    [([], []), ([1], [0.5]), ([1, 1, 2, 4, 4], [0.125, 0.125, 0.5, 0.875, 0.875])],
)
def test_exact_tied_singleton_and_empty_percentiles(
    resources, tmp_path, values, expected
):
    from arblab.hyperliquid_copy.disk_score_query import ScoreQuery

    token = write_metric_rows(
        resources, (row(i, v) for i, v in enumerate(values)), METRICS
    )
    scratch = tmp_path / "query-temp"
    scratch.mkdir()
    target = tmp_path / "scores.parquet"
    with ScoreQuery(output(resources, token), scratch, settings()) as query:
        query.write_scores(target)
    actual = sorted(pq.read_table(target).to_pylist(), key=lambda r: r["user"])
    assert [r["score"] for r in actual] == expected


def test_duplicate_candidates_reject_before_output(resources, tmp_path):
    from arblab.hyperliquid_copy.disk_score_query import ScoreQuery

    token = write_metric_rows(resources, iter([row(), row()]), METRICS)
    with pytest.raises(ValueError, match="Duplicate"):
        with ScoreQuery(output(resources, token), tmp_path, settings()):
            pytest.fail("duplicate candidates accepted")


def test_bounds_and_query_lifecycle(resources, tmp_path):
    from arblab.hyperliquid_copy.disk_score_query import ScoreQuery

    token = write_metric_rows(resources, iter([row()]), METRICS)
    with ScoreQuery(output(resources, token), tmp_path, settings()) as query:
        assert query.db.execute(
            "SELECT current_setting('memory_limit'), current_setting('threads')"
        ).fetchone() == ("244.1 MiB", 1)
        with pytest.raises(ValueError, match="byte limit"):
            query.write_scores(tmp_path / "failed.parquet", max_bytes=10)
    assert query.db is None
    with pytest.raises(ValueError, match="open"):
        query.write_scores(tmp_path / "closed.parquet")


def test_scoring_identity_retains_term_order():
    from arblab.hyperliquid_copy.disk_score_query import scoring_terms

    first = SimpleNamespace(
        metric_weights={"gross_volume": 0.4, "pnl_efficiency": 0.6},
        metric_directions={"gross_volume": "desc", "pnl_efficiency": "asc"},
    )
    second = SimpleNamespace(
        metric_weights=dict(reversed(tuple(first.metric_weights.items()))),
        metric_directions=first.metric_directions,
    )
    assert scoring_terms(first) == (
        ("gross_volume", 0.4, "desc"),
        ("pnl_efficiency", 0.6, "asc"),
    )
    assert scoring_terms(second) != scoring_terms(first)
    with pytest.raises(ValueError):
        scoring_terms(
            SimpleNamespace(
                metric_weights={"bad": 1}, metric_directions={"bad": "desc"}
            )
        )


@pytest.mark.parametrize(
    "weight", [10**400, float("nan"), float("inf"), True, -1, 0, 2]
)
def test_invalid_weight_is_explicit_validation(weight):
    from arblab.hyperliquid_copy.disk_score_query import scoring_terms

    with pytest.raises(ValueError):
        scoring_terms(
            SimpleNamespace(
                metric_weights={"gross_volume": weight},
                metric_directions={"gross_volume": "desc"},
            )
        )
