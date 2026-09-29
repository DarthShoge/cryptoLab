import pytest


def test_unknown_weight_and_trimming():
    from arblab.hyperliquid_copy.signals import aggregate
    values = {str(i): 1.0 if i < 6 else None for i in range(10)}
    weights = {str(i): 1 for i in range(10)}
    assert aggregate(values, weights, "direction_equal").value == pytest.approx(.6)
    assert aggregate(values, weights, "conviction_trimmed").value == pytest.approx(.5)
    values["5"] = None
    assert aggregate(values, weights, "direction_equal").reason == "insufficient_coverage"


def test_conviction_uses_notional_not_coin_quantity():
    from arblab.hyperliquid_copy.signals import conviction_value
    assert conviction_value(position_qty=2, mid=100, scale=400) == .5
    assert conviction_value(position_qty=-10, mid=100, scale=400) == -1
    assert conviction_value(position_qty=1, mid=100, scale=0) is None


def test_trimming_fractional_boundary_and_zero_scores():
    from arblab.hyperliquid_copy.signals import aggregate
    values = {str(i): (-1.0 if i == 0 else 1.0) for i in range(5)}
    result = aggregate(values, {str(i): 0 for i in range(5)}, "conviction_trimmed")
    assert result.value == pytest.approx(.75)
