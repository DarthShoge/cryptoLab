from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from hyperliquid_explorer_api import diagnostic_statistics as stats


UTC = timezone.utc
START = datetime(2026, 6, 1, tzinfo=UTC)  # Monday and month boundary


def history(hours, start=START):
    return [{"time": start + timedelta(hours=i), "equity": 100 + i} for i in range(hours + 1)]


def weeks(values, start=START):
    return [dict(start=start + timedelta(weeks=i), end=start + timedelta(weeks=i+1),
                 return_value=value, partial=False) for i, value in enumerate(values)]


def test_profiles_full_source_and_calendar_boundaries():
    result = stats.profiles(history(24 * 31), 3600)
    assert result["samples"] == 745
    assert result["points"][0]["growth"] == 1
    assert result["points"][-1]["growth"] == 8.44
    assert result["weeks"][0] == dict(start=START, end=START + timedelta(days=7), return_value=pytest.approx(1.68), partial=False)
    assert result["weeks"][-1]["partial"] is True
    assert result["months"][0]["end"] == datetime(2026, 7, 1, tzinfo=UTC)
    assert result["months"][0]["partial"] is False
    assert result["months"][1]["partial"] is True
    assert result["max_drawdown"] == 0


def test_daily_reduction_preserves_peaks_drawdowns_and_exposures():
    rows = history(47)
    rows[5]["equity"] = 1000
    rows[8]["equity"] = 50
    rows[12]["gross_exposure"] = 5
    rows[13]["net_exposure"] = -3
    result = stats.profiles(rows, 3600)
    assert len(result["points"]) <= 20
    by_time = {p["time"]: p for p in result["points"]}
    for i in [0, 5, 8, 12, 13, 23, 24, 47]:
        assert rows[i]["time"] in by_time
    assert result["max_drawdown"] == pytest.approx(-0.95)
    assert by_time[rows[8]["time"]]["drawdown"] == pytest.approx(-0.95)
    assert by_time[rows[0]["time"]]["gross_exposure"] is None


def test_partial_periods_and_timezone_normalization():
    start = datetime(2026, 6, 3, 2, tzinfo=timezone(timedelta(hours=2)))
    result = stats.profiles(history(24 * 7, start), 3600)
    assert all(w["partial"] for w in result["weeks"])
    assert result["points"][0]["time"] == datetime(2026, 6, 3, tzinfo=UTC)


@pytest.mark.parametrize("kind", ["empty", "gap", "duplicate", "unsorted", "nan", "inf", "zero", "negative", "naive", "interval"])
def test_invalid_history_is_rejected(kind):
    rows = history(3)
    interval = 3600
    if kind == "empty": rows = []
    elif kind == "gap": rows.pop(1)
    elif kind == "duplicate": rows[1]["time"] = rows[0]["time"]
    elif kind == "unsorted": rows.reverse()
    elif kind in {"nan", "inf", "zero", "negative"}: rows[1]["equity"] = {"nan": float("nan"), "inf": float("inf"), "zero": 0, "negative": -1}[kind]
    elif kind == "naive": rows[0]["time"] = rows[0]["time"].replace(tzinfo=None)
    elif kind == "interval": interval = 300
    with pytest.raises(ValueError): stats.profiles(rows, interval)


def test_known_weekly_series_and_sample_deviation():
    values = [-0.2, -0.1, 0, 0.1, 0.2]
    result = stats.weekly_statistics(weeks(values), [])
    expected = dict(count=5, mean=0, median=0, weekly_volatility=np.std(values, ddof=1),
                    win_rate=0.4, best_week=0.2, worst_week=-0.2, percentile_05=-0.18,
                    tail_mean_05=-0.2, paired_count=0)
    for key, value in expected.items(): assert result[key]["value"] == pytest.approx(value)
    for key in ["correlation", "beta", "tracking_error", "excess_mean", "mean_ci_low", "mean_ci_high"]:
        assert result[key]["value"] is None and result[key]["reason"]


def test_annualized_weekly_volatility_uses_crypto_calendar_and_requires_two_weeks():
    values = [-0.1, 0.2, 0.05]
    result = stats.weekly_statistics(weeks(values), [])
    assert result["annualized_weekly_volatility"] == dict(
        value=pytest.approx(np.std(values, ddof=1) * np.sqrt(365 / 7)),
        reason=None, unit="percent",
    )
    unavailable = stats.weekly_statistics(weeks([0.1]), [])["annualized_weekly_volatility"]
    assert unavailable["value"] is None
    assert unavailable["reason"]


def test_pairing_requires_exact_complete_periods_and_reports_relationship():
    source = weeks([0.01, 0.02, 0.03, 0.04])
    benchmark = weeks([0.005, 0.01, 0.015, 0.02])
    benchmark[0]["partial"] = True
    benchmark[1]["end"] += timedelta(seconds=1)
    source.append(dict(source[0], partial=True))
    result = stats.weekly_statistics(source, benchmark)
    assert result["count"]["value"] == 4
    assert result["paired_count"]["value"] == 2
    assert result["correlation"]["value"] == pytest.approx(1)
    assert result["beta"]["value"] == pytest.approx(2)
    assert result["excess_mean"]["value"] == pytest.approx(0.0175)
    assert result["tracking_error"]["value"] == pytest.approx(np.std([0.015, 0.02], ddof=1) * np.sqrt(365 / 7))


def test_zero_variance_and_missing_values_never_emit_nonfinite_scalars():
    result = stats.weekly_statistics(weeks([0.1] * 20), weeks([0.2] * 20))
    for key in ["correlation", "beta"]:
        assert result[key]["value"] is None and result[key]["reason"]
    assert result["weekly_volatility"]["value"] == pytest.approx(0)
    for value in result.values():
        assert value["value"] is None or np.isfinite(value["value"])
    empty = stats.weekly_statistics([], [])
    assert empty["count"]["value"] == 0
    assert empty["mean"]["value"] is None


def test_bootstrap_is_deterministic_uses_paired_blocks_and_twenty_week_minimum():
    values = np.linspace(-0.08, 0.15, 24)
    source, benchmark = weeks(values), weeks(values - 0.01)
    first = stats.weekly_statistics(source, benchmark)
    assert first == stats.weekly_statistics(source, benchmark)
    rng = np.random.default_rng(20260926)
    starts = rng.integers(0, 24, size=(2000, 6))
    indices = ((starts[:, :, None] + np.arange(4)) % 24).reshape(2000, -1)
    expected = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    assert first["mean_ci_low"]["value"] == pytest.approx(expected[0])
    assert first["mean_ci_high"]["value"] == pytest.approx(expected[1])
    assert first["excess_ci_low"]["value"] == pytest.approx(0.01)
    assert first["excess_ci_high"]["value"] == pytest.approx(0.01)
    assert stats.weekly_statistics(source[:19], benchmark)["mean_ci_low"]["value"] is None
