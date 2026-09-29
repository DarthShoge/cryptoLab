from dataclasses import replace
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from arblab.hyperliquid_copy.proxy_conviction import ProxyConviction
from .test_lab_pipeline_proxy import config, dataset


def test_conviction_uses_only_current_window_and_prunes_as_time_advances(tmp_path):
    data = dataset(tmp_path)
    c = config()
    c = replace(
        c,
        follower=replace(
            c.follower, aggregation="conviction_trimmed", scale_lookback_days=1
        ),
    )
    at = day(c.start)
    begin = at - timedelta(days=1)
    key = (
        data.activity.db.execute(
            'SELECT "user" FROM fills ORDER BY "user" LIMIT 1'
        ).fetchone()[0],
        "BTC",
    )
    try:
        with ProxyActivity(
            [tmp_path / "fills.parquet"],
            temp_root=tmp_path,
            query_window=(begin, at + timedelta(hours=3)),
        ) as bounded:
            conviction = ProxyConviction(bounded, c, at + timedelta(days=365))
            for hour in range(3):
                decision = at + timedelta(hours=hour)
                conviction.advance({key}, decision)
                expected = dict(
                    data.activity.hourly_exposure(
                        *key,
                        decision - timedelta(days=1),
                        decision + timedelta(hours=1),
                        max_price_age_seconds=c.proxy.max_mark_age_seconds,
                    )
                )
                assert conviction.samples[key] == expected
                assert len(conviction.samples[key]) == 25
            conviction.advance(set(), at + timedelta(hours=2))
            assert conviction.samples == {}
    finally:
        data.activity.close()


def test_conviction_rejects_reverse_or_out_of_run_decisions(tmp_path):
    data = dataset(tmp_path)
    c = config()
    at = day(c.start)
    conviction = ProxyConviction(data.activity, c, at + timedelta(days=1))
    try:
        conviction.advance(set(), at)
        with pytest.raises(ValueError, match="decision"):
            conviction.advance(set(), at - timedelta(hours=1))
        with pytest.raises(ValueError, match="decision"):
            conviction.advance(set(), at + timedelta(days=1))
    finally:
        data.activity.close()
