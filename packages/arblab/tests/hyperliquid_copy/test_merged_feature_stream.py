from contextlib import closing
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.feature_window import FeatureWindow
from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS
from .test_feature_window import days
from .test_lab_ranking import settings
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


def _reference(window):
    rows = [
        row
        for day in window.days
        for row in day.observations()
        if window.start <= row.order_key[0] < window.end and row.coin in window.coins
    ]
    rows.sort(
        key=lambda row: (
            row.user,
            *row.order_key,
            0 if hasattr(row, "fragments") else 1,
        )
    )
    return rows


def test_merges_ordered_days_without_materialising_sorted_partitions(window):
    from arblab.hyperliquid_copy.merged_feature_stream import merge_feature_days

    with closing(merge_feature_days(window)) as rows:
        assert list(rows) == _reference(window)


def test_early_close_rechecks_every_open_day(window, monkeypatch):
    from arblab.hyperliquid_copy import merged_feature_stream as module

    original = module.FeatureDay.verify
    calls = {day.publication.key: 0 for day in window.days}

    def verify(day):
        calls[day.publication.key] += 1
        return original(day)

    monkeypatch.setattr(module.FeatureDay, "verify", verify)
    rows = module.merge_feature_days(window)
    next(rows)
    rows.close()

    # Each stream verifies on entry and in its finally block even if the merged
    # consumer stops before exhaustion.
    assert all(count >= 2 for count in calls.values())


def test_window_change_during_final_verification_is_rejected(window, monkeypatch):
    from arblab.hyperliquid_copy import merged_feature_stream as module

    original = module.FeatureWindow.verify
    calls = 0

    def verify(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("changed window")
        return original(value)

    monkeypatch.setattr(module.FeatureWindow, "verify", verify)
    rows = module.merge_feature_days(window)
    next(rows)
    with pytest.raises(ValueError, match="changed window"):
        rows.close()


def test_coin_and_intraday_filters_preserve_complete_source_validation(window):
    from arblab.hyperliquid_copy.merged_feature_stream import merge_feature_days

    with closing(merge_feature_days(window)) as rows:
        actual = list(rows)
    assert actual
    assert all(
        row.coin in window.coins and window.start <= row.order_key[0] < window.end
        for row in actual
    )


def test_streamed_metrics_match_materialised_sort(window, tmp_path):
    from arblab.hyperliquid_copy.feature_candidate_merge import (
        merge_feature_metric_rows,
    )
    from arblab.hyperliquid_copy.merged_feature_stream import (
        merge_feature_metric_rows_from_days,
    )

    candidates = sorted({row.user for row in _reference(window)})
    config = settings(lookback_days=2, min_volume=0, metric_weights={"gross_volume": 1})
    expected = list(
        merge_feature_metric_rows(
            iter(candidates),
            iter(_reference(window)),
            window.end,
            config,
            SEMANTICS,
            temp_root=tmp_path,
        )
    )
    with closing(
        merge_feature_metric_rows_from_days(
            window,
            iter(candidates),
            config,
            SEMANTICS,
            temp_root=tmp_path,
        )
    ) as rows:
        actual = list(rows)
    assert actual == expected


def test_streamed_metrics_include_dormant_candidates(window, tmp_path):
    from arblab.hyperliquid_copy.merged_feature_stream import (
        merge_feature_metric_rows_from_days,
    )

    active = sorted({row.user for row in _reference(window)})
    dormant = "0x" + "f" * 40
    candidates = sorted([*active, dormant])
    with closing(
        merge_feature_metric_rows_from_days(
            window,
            iter(candidates),
            settings(
                lookback_days=2,
                min_volume=0,
                metric_weights={"gross_volume": 1},
            ),
            SEMANTICS,
            temp_root=tmp_path,
        )
    ) as rows:
        actual = list(rows)
    dormant_row = next(row for row in actual if row["user"] == dormant)
    assert dormant_row["metrics"]["gross_volume"] == 0
    assert dormant_row["exclusions"] == ["no_activity_in_lookback"]
