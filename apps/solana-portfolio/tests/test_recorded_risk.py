import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.kamino import recorded_risk


def test_recorded_risk_uses_protocol_thresholds_including_elevation_groups():
    unit = 2**60
    obligation = SimpleNamespace(
        deposited_value_sf=10000 * unit,
        borrow_factor_adjusted_debt_value_sf=7000 * unit,
        borrowed_assets_market_value_sf=6500 * unit,
        allowed_borrow_value_sf=8000 * unit,
        unhealthy_borrow_value_sf=9000 * unit,
        elevation_group=3,
    )
    result = recorded_risk(obligation)
    assert result["ltv"] == 0.7
    assert result["liquidationLtv"] == 0.9
    assert result["health"] == pytest.approx(9 / 7)
    assert result["borrowHealth"] == pytest.approx(8 / 7)
    assert result["liquidationBuffer"] == 2000
    assert result["elevationGroup"] == 3


def test_no_debt_health_is_null_for_json():
    unit = 2**60
    obligation = SimpleNamespace(
        deposited_value_sf=1000 * unit,
        borrow_factor_adjusted_debt_value_sf=0,
        borrowed_assets_market_value_sf=0,
        allowed_borrow_value_sf=650 * unit,
        unhealthy_borrow_value_sf=750 * unit,
        elevation_group=0,
    )
    assert recorded_risk(obligation)["health"] is None
