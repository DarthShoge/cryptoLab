from dataclasses import asdict

import pytest


def test_config_roundtrip_and_independent_benchmark():
    from arblab.hyperliquid_copy.lab_config import LabConfig

    config = LabConfig.from_dict(
        {
            "coins": ["SOL", "ETH"],
            "metric_weights": {"gross_volume": 2, "pnl_efficiency": 2},
        }
    )
    assert config.coins == ["ETH", "SOL"]
    assert config.metric_weights == {"gross_volume": 0.5, "pnl_efficiency": 0.5}
    assert config.benchmark == "btc_perp_buy_hold" and "BTC" not in config.coins
    assert LabConfig.from_dict(asdict(config)) == config
    assert "top 5%" in config.summary()


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": 1},
        {"metric_weights": {"return": 1}},
        {"metric_weights": {"gross_volume": 0}},
        {"metric_weights": {"gross_volume": -1}},
        {"metric_weights": {"gross_volume": float("nan")}},
        {"top_n": 5},
        {"selection": "n", "top_n": 5},
        {"top_fraction": 0},
        {"coins": ["DOGE"]},
        {"fee_bps": -1},
        {"gross_cap": 2},
        {"lookback_days": True},
        {"asset_weights": {"ETH": 0.9, "SOL": 0.9}},
        {"start": "2026-01-01", "end": "2025-01-01"},
        {"split": "locked_test"},
        {"metric_directions": {"return": "desc"}},
    ],
)
def test_invalid_settings_are_not_ignored(changes):
    from arblab.hyperliquid_copy.lab_config import LabConfig

    with pytest.raises((ValueError, TypeError)):
        LabConfig.from_dict(changes)
