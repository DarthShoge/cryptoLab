from datetime import timedelta
from contextlib import closing

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.feature_window import FeatureWindow
from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS
from .test_feature_window import days
from .test_qualified_day import qualified


@pytest.fixture
def window(qualified, resources, days):
    return FeatureWindow(
        resources,
        qualified,
        days,
        DAY + timedelta(hours=1),
        DAY + timedelta(days=2, hours=1),
        ["BTC"],
        SEMANTICS,
    )


def test_stages_bounded_daily_and_episode_summaries(window, tmp_path):
    from arblab.hyperliquid_copy.feature_metric_summary import stage_metric_summaries
    from arblab.hyperliquid_copy.merged_feature_stream import merge_feature_days
    from arblab.hyperliquid_copy.wallet_day_features import EpisodeObservation

    (tmp_path / "spill").mkdir()
    expected = list(merge_feature_days(window))
    result = stage_metric_summaries(window, tmp_path)
    daily = pq.read_table(result.daily_path).to_pylist()
    episodes = pq.read_table(result.episode_path).to_pylist()

    assert result.daily_rows == len(daily)
    assert (
        result.episode_rows
        == len(episodes)
        == sum(isinstance(row, EpisodeObservation) for row in expected)
    )
    assert sum(row["fill_count"] for row in daily) == sum(
        not isinstance(row, EpisodeObservation) for row in expected
    )
    assert sum(row["volume"] for row in daily) == pytest.approx(
        sum(
            row.gross_volume
            for row in expected
            if not isinstance(row, EpisodeObservation)
        )
    )
    assert (
        result.bytes
        == result.daily_path.stat().st_size + result.episode_path.stat().st_size
    )


def test_summary_cap_retains_partial_evidence(window, tmp_path):
    from arblab.hyperliquid_copy.feature_metric_summary import stage_metric_summaries

    (tmp_path / "spill").mkdir()
    with pytest.raises(ValueError, match="byte limit"):
        stage_metric_summaries(window, tmp_path, max_daily_bytes=1, max_episode_bytes=1)
    assert any(tmp_path.glob("metric_*.parquet"))


def test_summary_early_failure_rechecks_window(window, tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import feature_metric_summary as module

    (tmp_path / "spill").mkdir()
    original = module.FeatureWindow.verify
    calls = 0

    def verify(value):
        nonlocal calls
        calls += 1
        result = original(value)
        if calls == 2:
            raise ValueError("changed feature window")
        return result

    monkeypatch.setattr(module.FeatureWindow, "verify", verify)
    with pytest.raises(ValueError, match="changed feature window"):
        module.stage_metric_summaries(window, tmp_path)


def test_summary_metrics_match_verified_observation_replay(window, tmp_path):
    from arblab.hyperliquid_copy.feature_candidate_merge import (
        merge_feature_metric_rows,
    )
    from arblab.hyperliquid_copy.feature_metric_summary import (
        stage_metric_summaries,
        summary_metric_rows,
    )
    from arblab.hyperliquid_copy.merged_feature_stream import merge_feature_days
    from .test_lab_ranking import settings

    (tmp_path / "spill").mkdir()
    candidates = sorted({row.user for row in merge_feature_days(window)})
    candidates.append("0x" + "f" * 40)
    candidates.sort()
    config = settings(
        lookback_days=2,
        min_active_days=1,
        min_episodes=1,
        min_notional=0,
        min_minutes=0,
        min_volume=0,
        metric_weights={"gross_volume": 1},
    )
    with closing(merge_feature_days(window)) as observations:
        expected = list(
            merge_feature_metric_rows(
                iter(candidates),
                observations,
                window.end,
                config,
                SEMANTICS,
                temp_root=tmp_path,
            )
        )
    summary = stage_metric_summaries(window, tmp_path)
    with closing(
        summary_metric_rows(summary, iter(candidates), config, temp_root=tmp_path)
    ) as rows:
        actual = list(rows)
    assert [row["user"] for row in actual] == candidates
    assert [row["exclusions"] for row in actual] == [
        row["exclusions"] for row in expected
    ]
    for left, right in zip(actual, expected):
        assert left["metrics"] == pytest.approx(right["metrics"], nan_ok=True)
