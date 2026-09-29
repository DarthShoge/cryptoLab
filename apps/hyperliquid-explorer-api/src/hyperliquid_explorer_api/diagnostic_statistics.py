"""Pure UTC equity profiles and descriptive statistics over complete weeks."""

from datetime import datetime, timedelta, timezone
import math

import numpy as np


def profiles(rows, interval_seconds):
    """Validate an uninterrupted history and calculate before reducing chart points."""
    if interval_seconds not in (60, 3600) or not rows:
        raise ValueError("History requires observations at a supported interval.")
    points = []
    peak = 0.0
    initial = None
    for row in rows:
        try:
            timestamp = row["time"]
            if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
                raise ValueError
            timestamp = timestamp.astimezone(timezone.utc)
            equity = float(row["equity"])
            if not math.isfinite(equity) or equity <= 0:
                raise ValueError
            if points and (timestamp - points[-1]["time"]).total_seconds() != interval_seconds:
                raise ValueError
            exposures = {}
            for key in ("gross_exposure", "net_exposure"):
                value = row.get(key)
                value = None if value is None else float(value)
                if value is not None and not math.isfinite(value):
                    raise ValueError
                exposures[key] = value
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("History contains invalid or noncontiguous observations.") from None
        initial = equity if initial is None else initial
        peak = max(peak, equity)
        growth = equity / initial
        if not math.isfinite(growth):
            raise ValueError("History growth exceeds the supported numeric range.")
        points.append(dict(time=timestamp, equity=equity, growth=growth,
                           drawdown=equity / peak - 1, **exposures))
    return dict(points=_daily_points(points), weeks=_periods(points, "week"),
                months=_periods(points, "month"), samples=len(points),
                max_drawdown=min(p["drawdown"] for p in points))


def _daily_points(points):
    result = []
    day = []

    def retain(group):
        selected = {0, len(group) - 1}
        for key in ("equity", "drawdown", "gross_exposure", "net_exposure"):
            available = [i for i, point in enumerate(group) if point[key] is not None]
            if available:
                selected.add(min(available, key=lambda i: group[i][key]))
                selected.add(max(available, key=lambda i: group[i][key]))
        result.extend(group[i] for i in sorted(selected))

    for point in points:
        if day and day[-1]["time"].date() != point["time"].date():
            retain(day)
            day = []
        day.append(point)
    retain(day)
    return result


def _periods(points, kind):
    first, last = points[0]["time"], points[-1]["time"]
    boundary = first.replace(hour=0, minute=0, second=0, microsecond=0)
    boundary = boundary - timedelta(days=boundary.weekday()) if kind == "week" else boundary.replace(day=1)
    result = []
    index = 0
    while boundary < last:
        if kind == "week":
            end = boundary + timedelta(days=7)
        else:
            end = boundary.replace(year=boundary.year + (boundary.month == 12), month=boundary.month % 12 + 1)
        while index < len(points) - 1 and points[index]["time"] < boundary:
            index += 1
        start_index = index
        while index < len(points) - 1 and points[index + 1]["time"] <= end:
            index += 1
        start_point, end_point = points[start_index], points[index]
        if start_point["time"] < end_point["time"]:
            value = end_point["equity"] / start_point["equity"] - 1
            if not math.isfinite(value):
                raise ValueError("Period return exceeds the supported numeric range.")
            result.append(dict(start=start_point["time"], end=end_point["time"],
                               return_value=value, partial=start_point["time"] != boundary or end_point["time"] != end))
        boundary = end
    return result


def _complete(weeks):
    return {(w["start"], w["end"]): float(w["return_value"])
            for w in weeks if not w["partial"] and math.isfinite(w["return_value"])}


def _confidence(values):
    if len(values) < 20:
        return None, None
    rng = np.random.default_rng(20260926)
    starts = rng.integers(0, len(values), size=(2000, math.ceil(len(values) / 4)))
    indices = ((starts[:, :, None] + np.arange(4)) % len(values)).reshape(2000, -1)[:, :len(values)]
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975])


def weekly_statistics(weeks, benchmark_weeks):
    """Estimate complete-week statistics, pairing benchmark periods exactly."""
    source, benchmark = _complete(weeks), _complete(benchmark_weeks)
    values = np.array([source[key] for key in sorted(source)], dtype=float)
    matching = sorted(source.keys() & benchmark.keys())
    paired = np.array([source[key] for key in matching], dtype=float)
    reference = np.array([benchmark[key] for key in matching], dtype=float)
    excess = paired - reference
    result = {}

    def put(key, value, reason="No complete weeks are available.", unit="percent"):
        valid = value is not None and math.isfinite(value)
        result[key] = dict(value=float(value) if valid else None,
                           reason=None if valid else reason, unit=unit)

    n = len(values)
    put("count", n, unit="count")
    put("paired_count", len(paired), unit="count")
    for key, function in [("mean", np.mean), ("median", np.median),
                          ("best_week", np.max), ("worst_week", np.min)]:
        put(key, function(values) if n else None)
    put("win_rate", float(np.mean(values > 0)) if n else None)
    quantile = np.quantile(values, 0.05) if n else None
    put("percentile_05", quantile)
    put("tail_mean_05", np.mean(values[values <= quantile]) if n else None)
    put("weekly_volatility", np.std(values, ddof=1) if n >= 2 else None,
        "At least two complete weeks are required.")
    put("annualized_weekly_volatility", np.std(values, ddof=1) * math.sqrt(365 / 7) if n >= 2 else None,
        "At least two complete weeks are required.")
    low, high = _confidence(values)
    for key, value in [("mean_ci_low", low), ("mean_ci_high", high)]:
        put(key, value, "At least 20 complete weeks are required.")
    put("excess_mean", np.mean(excess) if len(excess) else None,
        "No matching complete benchmark weeks are available.")
    low, high = _confidence(excess)
    for key, value in [("excess_ci_low", low), ("excess_ci_high", high)]:
        put(key, value, "At least 20 matching complete benchmark weeks are required.")
    enough = len(paired) >= 2
    reference_varies = enough and np.ptp(reference) > 0
    both_vary = reference_varies and np.ptp(paired) > 0
    put("correlation", np.corrcoef(paired, reference)[0, 1] if both_vary else None,
        "At least two matching weeks with nonzero variance in both series are required.", "ratio")
    put("beta", np.cov(paired, reference, ddof=1)[0, 1] / np.var(reference, ddof=1) if reference_varies else None,
        "At least two matching weeks with nonzero benchmark variance are required.", "ratio")
    put("tracking_error", np.std(excess, ddof=1) * math.sqrt(365 / 7) if enough else None,
        "At least two matching complete benchmark weeks are required.")
    return result
