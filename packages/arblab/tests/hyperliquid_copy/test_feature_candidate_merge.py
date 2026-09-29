from datetime import timedelta
from itertools import groupby

import pytest

from arblab.hyperliquid_copy.candidate_metric_producer import merge_metric_rows
from arblab.hyperliquid_copy.episode_state import encode_episode_state
from arblab.hyperliquid_copy.wallet_day_features import derive_wallet_day
from .test_lab_ranking import settings
from .test_ranking import history


def feature_rows(fills, semantics):
    result = []
    for user, group in groupby(fills, key=lambda r: r.user):
        rows = list(group)
        day = rows[0].exchange_time.replace(hour=0, minute=0, second=0, microsecond=0)
        checkpoint = encode_episode_state(
            {}, user=user, cutoff=day, semantics=semantics
        )
        while day <= rows[-1].exchange_time:
            next_day = day + timedelta(days=1)
            derived = derive_wallet_day(
                (r for r in rows if day <= r.exchange_time < next_day),
                user=user,
                day=day,
                semantics=semantics,
                checkpoint=checkpoint,
                on_fill=result.append,
                on_episode=result.append,
            )
            checkpoint, day = derived.checkpoint, next_day
    return result


@pytest.mark.parametrize("semantics", ["gross_excludes_fee", "net_includes_fee"])
def test_complete_merge_matches_raw_metrics_with_dormant_candidate(tmp_path, semantics):
    from arblab.hyperliquid_copy.feature_candidate_merge import (
        merge_feature_metric_rows,
    )

    fills = sorted(history(2), key=lambda r: (r.user, *r.order_key))
    candidates = sorted({r.user for r in fills} | {"0x" + "f" * 40})
    decision = max(r.exchange_time for r in fills) + timedelta(seconds=1)
    config = settings(lookback_days=1, min_volume=0)
    expected = list(
        merge_metric_rows(
            iter(candidates),
            iter(fills),
            decision,
            config,
            semantics,
            temp_root=tmp_path,
        )
    )
    actual = list(
        merge_feature_metric_rows(
            iter(candidates),
            iter(feature_rows(fills, semantics)),
            decision,
            config,
            semantics,
            temp_root=tmp_path,
        )
    )
    assert actual == expected
    assert len(actual) == 3
    assert "no_activity_in_lookback" in actual[-1]["exclusions"]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "reversed", "invalid"])
def test_invalid_candidate_domain_rejected(tmp_path, mode):
    from arblab.hyperliquid_copy.feature_candidate_merge import (
        merge_feature_metric_rows,
    )

    fills = sorted(history(2), key=lambda r: (r.user, *r.order_key))
    candidates = sorted({r.user for r in fills})
    if mode == "missing":
        candidates.pop()
    elif mode == "duplicate":
        candidates.insert(1, candidates[0])
    elif mode == "reversed":
        candidates.reverse()
    else:
        candidates[0] = "not-an-address"
    with pytest.raises(ValueError):
        list(
            merge_feature_metric_rows(
                iter(candidates),
                iter(feature_rows(fills, "gross_excludes_fee")),
                max(r.exchange_time for r in fills) + timedelta(seconds=1),
                settings(lookback_days=1),
                "gross_excludes_fee",
                temp_root=tmp_path,
            )
        )


def test_more_than_100000_dormant_candidates_are_streamed(tmp_path):
    from arblab.hyperliquid_copy.feature_candidate_merge import (
        merge_feature_metric_rows,
    )

    count = 100_001
    decision = history(1)[0].exchange_time
    config = settings(lookback_days=1, min_volume=0, metric_weights={"gross_volume": 1})
    candidates = (f"0x{i:040x}" for i in range(count))
    seen = 0
    for row in merge_feature_metric_rows(
        candidates, iter(()), decision, config, "gross_excludes_fee", temp_root=tmp_path
    ):
        assert row["user"] == f"0x{seen:040x}"
        assert row["metrics"]["gross_volume"] == 0
        assert row["exclusions"] == ["no_activity_in_lookback"]
        seen += 1
    assert seen == count
    assert not list(tmp_path.iterdir())


def test_null_partition_candidate_is_not_treated_as_end_of_stream():
    from arblab.hyperliquid_copy.feature_metric_stream import _Candidates

    with pytest.raises(ValueError):
        _Candidates(iter([None, "0x" + "0" * 40]))
