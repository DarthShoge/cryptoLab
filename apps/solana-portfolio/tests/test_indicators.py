import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.indicators import supertrend


def test_wilder_atr_has_warmup_and_known_value():
    candles = [
        {"time": i, "open": 10, "high": 12, "low": 8, "close": 10} for i in range(7)
    ]
    result = supertrend(candles, period=3, multiplier=2)
    assert result[0]["value"] is None
    assert result[1]["value"] is None
    assert result[2]["atr"] == 4
    assert result[2]["value"] == 18
    assert result[-1]["atr"] == 4


def test_supertrend_flips_only_after_close_crosses_the_band():
    candles = [
        {"time": i, "open": 10, "high": 12, "low": 8, "close": 10} for i in range(4)
    ]
    candles += [{"time": 4, "open": 10, "high": 25, "low": 10, "close": 24}]
    result = supertrend(candles, period=3, multiplier=1)
    assert result[-2]["direction"] == "bearish"
    assert result[-1]["direction"] == "bullish"
    assert result[-1]["value"] < 24


def test_indicator_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        supertrend([], period=0)
