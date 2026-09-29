import pytest


def test_proxy_config_is_versioned_hourly_and_reuses_normalized_strategy_controls():
    from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxy

    config = LabConfigProxy(trader={"metric_weights": {"gross_volume": 2}})
    assert config.schema_version == "hyperliquid_copy_lab_proxy_v1"
    assert config.follower.update_minutes == 60
    assert config.trader.metric_weights == {"gross_volume": 1}
    assert LabConfigProxy(**config.to_dict()) == config
    assert config.effective(["BTC"], {"BTC": 1}).coins == ["BTC"]
    assert "proxy" in config.summary()


@pytest.mark.parametrize(
    "changes",
    [
        {"follower": {"update_minutes": 1}},
        {"proxy": {"slippage_bps": -1}},
        {"proxy": {"max_mark_age_seconds": 0}},
        {"schema_version": "hyperliquid_copy_lab_v2"},
    ],
)
def test_proxy_config_rejects_invalid_or_silently_changed_clocks(changes):
    from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxy

    with pytest.raises(ValueError):
        LabConfigProxy(**changes)


def test_scheduled_config_roundtrip_and_legacy_payload_unchanged():
    from arblab.hyperliquid_copy.lab_config_proxy import (
        LabConfigProxy,
        LabConfigProxyScheduled,
    )
    from arblab.hyperliquid_copy.lab_config_codec import parse_lab_config

    legacy = LabConfigProxy()
    assert "rebalance" not in legacy.to_dict()
    c = LabConfigProxyScheduled()
    assert c.schema_version == "hyperliquid_copy_lab_proxy_v2"
    assert (
        c.rebalance == c.trader.reselection == c.market_universe.reselection == "weekly"
    )
    assert c.follower.update_minutes == 60
    assert c.effective(["BTC"], {"BTC": 1}).lookback_days == 90
    assert parse_lab_config(c.to_dict()) == c
    assert "weekly" in c.summary()
    with pytest.raises(ValueError):
        LabConfigProxyScheduled(trader={"reselection": "daily"})
    with pytest.raises(ValueError):
        LabConfigProxyScheduled(rebalance="monthly")


def test_weekly_clock_is_calendar_anchored_not_run_start():
    from arblab.hyperliquid_copy.proxy_schedule import decision_times
    from arblab.hyperliquid_copy.lab_config import day

    assert list(decision_times(day("2026-08-04"), day("2026-08-18"), "weekly")) == [
        day("2026-08-10"),
        day("2026-08-17"),
    ]
    assert len(list(decision_times(day("2026-08-04"), day("2026-08-06"), "daily"))) == 2
    assert (
        len(list(decision_times(day("2026-08-04"), day("2026-08-06"), "hourly"))) == 48
    )
