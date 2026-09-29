from datetime import timedelta

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows
from arblab.hyperliquid_copy.disk_score_query import ScoreQuery
from arblab.hyperliquid_copy.lab_config import METRICS
from arblab.hyperliquid_copy.lab_ranking import rank_universe
from .test_disk_metric_rows import output
from .test_disk_score_query import resources
from .test_lab_ranking import settings
from .test_ranking import history


@pytest.mark.parametrize(
    "changes,scope",
    [
        ({}, "BTC"),
        ({"selection": "fraction", "top_n": None, "top_fraction": 0.5}, "BTC"),
        ({"min_cohort": 10, "max_cohort": 10}, "BTC"),
        ({"aggregation": "direction_score_weighted"}, None),
        (
            {
                "metric_weights": {"profit_factor": 1},
                "aggregation": "direction_score_weighted",
            },
            "BTC",
        ),
        ({"lookback_days": 1}, "BTC"),
        ({"top_n": 250}, "BTC"),
    ],
)
def test_entire_evidence_and_selected_match_reference(
    resources, tmp_path, changes, scope
):
    from arblab.hyperliquid_copy.disk_cohort_query import CohortQuery

    fills = history()
    decision = max(f.exchange_time for f in fills) + timedelta(days=2)
    config = settings(**changes)
    expected = rank_universe(fills, decision, config, scope, "gross_excludes_fee")
    records = (
        dict(
            user=r["user"],
            metrics=dict.fromkeys(METRICS) | r["metrics"],
            exclusions=r["exclusions"],
        )
        for r in expected
    )
    token = write_metric_rows(resources, records, tuple(config.metric_weights))
    scores, target = tmp_path / "scores.parquet", tmp_path / "ranks.parquet"
    with ScoreQuery(output(resources, token), tmp_path, config) as query:
        query.write_scores(scores)
    with CohortQuery(scores, tmp_path, config, decision, scope) as query:
        summary = query.write_rankings(target)
    actual = pq.read_table(target).to_pylist()
    for value, ref in zip(actual, expected, strict=True):
        value["percentiles"] = {
            k: v for k, v in value["percentiles"].items() if v is not None
        }
        value["metrics"] = {k: value["metrics"][k] for k in ref["metrics"]}
        assert {k: value[k] for k in ref} == ref
    assert list(summary["selected"]) == [r for r in expected if r["selected"]]
    assert summary["candidate_count"] == len(expected)
    assert summary["eligible_count"] == sum(r["eligible"] for r in expected)
    assert summary["cutoff_address"] == next(
        (r["user"] for r in reversed(expected) if r["selected"]), None
    )


def test_empty_output_and_capped_failure(resources, tmp_path):
    from arblab.hyperliquid_copy.disk_cohort_query import CohortQuery
    from arblab.hyperliquid_copy.ranking_artifact import RANKING_SCHEMA

    config = settings()
    token = write_metric_rows(resources, iter(()), tuple(config.metric_weights))
    scores = tmp_path / "scores.parquet"
    from arblab.hyperliquid_copy.lab_config import day

    with ScoreQuery(output(resources, token), tmp_path, config) as query:
        query.write_scores(scores)
    with CohortQuery(scores, tmp_path, config, day(config.start), "BTC") as query:
        result = query.write_rankings(tmp_path / "ranks.parquet")
        assert result["selected"] == ()
        assert result["candidate_count"] == result["eligible_count"] == 0
        with pytest.raises(ValueError, match="byte limit"):
            query.write_rankings(tmp_path / "failed.parquet", max_bytes=10)
    assert pq.read_schema(tmp_path / "ranks.parquet") == RANKING_SCHEMA


def test_over_100000_candidates_keep_all_evidence_with_bounded_selection(
    resources, tmp_path
):
    from arblab.hyperliquid_copy.disk_cohort_query import CohortQuery
    from arblab.hyperliquid_copy.lab_config import day
    from .test_disk_metric_rows import row

    config = settings()
    count = 100001
    records = (
        row(i, i % 17, ["fixture_exclusion"] if i % 997 == 0 else ())
        for i in range(count)
    )
    token = write_metric_rows(resources, records, METRICS)
    scores, ranks = tmp_path / "large_scores.parquet", tmp_path / "large_ranks.parquet"
    with ScoreQuery(output(resources, token), tmp_path, config) as query:
        assert query.write_scores(scores) == count
    with CohortQuery(scores, tmp_path, config, day(config.start), "BTC") as query:
        summary = query.write_rankings(ranks)
    assert summary["candidate_count"] == count
    assert summary["eligible_count"] == count - len(range(0, count, 997))
    assert [r["user"] for r in summary["selected"]] == [f"0x{i:040x}" for i in (16, 33)]
    assert summary["selected_count"] == 2
    meta = pq.read_metadata(ranks)
    assert meta.num_rows == count
    assert all(meta.row_group(i).num_rows <= 4096 for i in range(meta.num_row_groups))
