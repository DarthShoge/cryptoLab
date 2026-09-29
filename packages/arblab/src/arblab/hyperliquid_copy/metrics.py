"""Minute-grid performance statistics; undefined ratios remain null."""
from math import isfinite, sqrt
from statistics import mean, stdev

from .contracts import finite, utc

MINUTES_PER_YEAR = 365*1440


def compute_performance(equity_rows, *, interval_seconds=60):
    if type(interval_seconds) is not int or interval_seconds not in (60, 3600):
        raise ValueError("Expected minute or hourly performance clock")
    periods_per_year = 365 * 86400 / interval_seconds
    rows = sorted(equity_rows, key=lambda r:r["time"])
    if not rows:
        raise ValueError("empty equity curve")
    values = [finite(r["equity"]) for r in rows]
    times = [utc(r["time"]) for r in rows]
    if any((b-a).total_seconds() != interval_seconds for a,b in zip(times,times[1:])):
        raise ValueError("missing/duplicate minute mark" if interval_seconds == 60 else "missing/duplicate hourly mark")
    warnings = []
    result = dict(total_return=None, annualized_return=None, annualized_volatility=None,
                  sharpe=None, sortino=None, max_drawdown=None, max_drawdown_minutes=0,
                  terminal_open_drawdown=False, warnings=warnings)
    if any(v <= 0 for v in values):
        warnings.append("nonpositive_equity")
        return result
    ratio = values[-1]/values[0]
    result["total_return"] = ratio-1
    if len(values) > 1:
        try:
            cagr = ratio**(periods_per_year/(len(values)-1))-1
            if isfinite(cagr):
                result["annualized_return"] = cagr
            else:
                warnings.append("annualization_overflow")
        except OverflowError:
            warnings.append("annualization_overflow")
    returns = [b/a-1 for a,b in zip(values,values[1:])]
    if len(returns) >= 2 and stdev(returns) > 0:
        std = stdev(returns)
        result["annualized_volatility"] = std*sqrt(periods_per_year)
        result["sharpe"] = mean(returns)/std*sqrt(periods_per_year)
    else:
        warnings.append("undefined_zero_variance")
    downside = sqrt(mean(min(r,0)**2 for r in returns)) if returns else 0
    if len(returns) >= 2 and downside > 0:
        result["sortino"] = mean(returns)/downside*sqrt(periods_per_year)
    elif "undefined_zero_variance" not in warnings:
        warnings.append("undefined_zero_variance")
    peak, drawdown, duration = values[0], 0., 0
    for value in values:
        peak = max(peak,value)
        drawdown = max(drawdown,1-value/peak)
        duration = duration+1 if value < peak else 0
        result["max_drawdown_minutes"] = max(result["max_drawdown_minutes"],duration * (interval_seconds // 60))
    result.update(max_drawdown=drawdown, terminal_open_drawdown=duration > 0)
    return result
