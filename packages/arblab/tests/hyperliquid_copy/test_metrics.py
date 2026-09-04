from datetime import timedelta

import pytest

from .test_market_data import T


def test_performance_golden_and_undefined():
    from arblab.hyperliquid_copy.metrics import compute_performance
    rows = [dict(time=T+timedelta(minutes=i), equity=v) for i,v in enumerate([100,110,99,100])]
    result = compute_performance(rows)
    assert result["total_return"] == 0
    assert result["max_drawdown"] == pytest.approx(.1)
    assert result["max_drawdown_minutes"] == 2
    assert result["terminal_open_drawdown"]
    flat = compute_performance([r | {"equity":100} for r in rows])
    assert flat["sharpe"] is None and "undefined_zero_variance" in flat["warnings"]
    with pytest.raises(ValueError, match="minute"):
        compute_performance(rows[::2])
    assert "nonpositive_equity" in compute_performance([rows[0], rows[1] | {"equity":0}])["warnings"]
