"""Descriptive report analytics with explicit availability, not strategy logic."""

from .models import Analytics, Metric
from .queries import Curve
from .repository import ANNUALIZED, number

CONVENTIONS = "Net simple UTC minute equity returns; zero risk-free rate; sample standard deviation (ddof=1); 365 × 1440 annualization. Sortino uses downside RMS over all minute returns."
SPECS = {
    "initial_equity": ("usd", "Return & capital", "Starting marked equity."),
    "final_equity": (
        "usd",
        "Return & capital",
        "Final collateral plus unrealized PnL.",
    ),
    "net_pnl": (
        "usd",
        "Return & capital",
        "Final minus starting equity; includes modeled costs.",
    ),
    "total_return": (
        "percent",
        "Return & capital",
        "Final / starting equity minus one.",
    ),
    "annualized_return": (
        "percent",
        "Return & capital",
        "CAGR over the report window; short samples are unreliable.",
    ),
    "sharpe": (
        "ratio",
        "Risk-adjusted performance",
        "Mean net minute return / sample standard deviation, annualized; risk-free rate is zero.",
    ),
    "sortino": (
        "ratio",
        "Risk-adjusted performance",
        "Mean net minute return / downside RMS over all returns, annualized; target return is zero.",
    ),
    "calmar": (
        "ratio",
        "Risk-adjusted performance",
        "CAGR / maximum fractional drawdown for the same window.",
    ),
    "annualized_volatility": (
        "percent",
        "Risk-adjusted performance",
        "Sample minute-return standard deviation × sqrt(365 × 1440).",
    ),
    "max_drawdown": (
        "percent",
        "Drawdown",
        "Largest decline from the running equity high.",
    ),
    "current_drawdown": (
        "percent",
        "Drawdown",
        "Final equity decline from its running high.",
    ),
    "max_drawdown_minutes": (
        "minutes",
        "Drawdown",
        "Longest uninterrupted underwater period on the full minute grid.",
    ),
    "terminal_open_drawdown": (
        "count",
        "Drawdown",
        "1 means the final drawdown is unrecovered; 0 means recovered.",
    ),
    "fees_usd": ("usd", "Costs & execution", "Executed trading fees in dollars."),
    "fee_drag": ("percent", "Costs & execution", "Fees / starting equity."),
    "funding_usd": (
        "usd",
        "Costs & execution",
        "Net funding cost; negative values are credits received.",
    ),
    "funding_drag": (
        "percent",
        "Costs & execution",
        "Net funding cost / starting equity; negative means a credit.",
    ),
    "turnover": (
        "ratio",
        "Costs & execution",
        "One-way absolute executed notional / starting equity.",
    ),
    "fill_ratio": (
        "percent",
        "Costs & execution",
        "Requested-notional-weighted fill ratio including stale requests; N/A if no orders.",
    ),
    "stale_requests": (
        "count",
        "Costs & execution",
        "Requests canceled because a timely executable book was unavailable.",
    ),
    "mean_gross_leverage": (
        "ratio",
        "Exposure & accounting",
        "Mean gross notional / positive equity over all minute observations.",
    ),
    "mean_net_leverage": (
        "ratio",
        "Exposure & accounting",
        "Mean signed net notional / positive equity over all minute observations.",
    ),
    "max_gross_leverage": (
        "ratio",
        "Exposure & accounting",
        "Maximum observed gross notional / equity; delayed execution can cause drift.",
    ),
    "residual_gross_exposure": (
        "usd",
        "Exposure & accounting",
        "Gross notional still open at the end; positions are not force-closed.",
    ),
    "residual_net_exposure": (
        "usd",
        "Exposure & accounting",
        "Signed net notional still open at the end.",
    ),
    "final_collateral": (
        "usd",
        "Exposure & accounting",
        "Remaining perpetual collateral, separate from unrealized PnL.",
    ),
    "final_unrealized_pnl": (
        "usd",
        "Exposure & accounting",
        "Mark-to-market PnL of residual positions.",
    ),
    "time_net_long": (
        "percent",
        "Exposure & accounting",
        "Fraction of observations with positive net exposure.",
    ),
    "time_net_short": (
        "percent",
        "Exposure & accounting",
        "Fraction of observations with negative net exposure.",
    ),
    "time_net_flat": (
        "percent",
        "Exposure & accounting",
        "Net-neutral observations may still have offsetting gross exposure.",
    ),
    "benchmark_excess_return": (
        "percent",
        "Return & capital",
        "Difference in total returns in percentage points; not risk-adjusted alpha.",
    ),
}


def build_metrics(scenario, stats, synthetic):
    start, end = stats.get("start"), stats.get("end")
    elapsed = (end - start).total_seconds() if start and end else 0
    initial, final = stats.get("initial"), stats.get("final")
    valid = (
        bool(stats.get("samples"))
        and not stats.get("gaps")
        and not stats.get("invalid_equity")
    )
    derived = dict(
        initial_equity=initial,
        net_pnl=final - initial if final is not None and initial is not None else None,
        current_drawdown=stats.get("current_drawdown"),
        mean_gross_leverage=stats.get("mean_gross_leverage"),
        mean_net_leverage=stats.get("mean_net_leverage"),
    )
    for key, source in (("fees_usd", "fee_drag"), ("funding_usd", "funding_drag")):
        value = scenario.metrics.get(source)
        derived[key] = (
            value * initial if value is not None and initial is not None else None
        )
    cagr, dd = (
        scenario.metrics.get("annualized_return"),
        scenario.metrics.get("max_drawdown"),
    )
    derived["calmar"] = (
        cagr / dd if cagr is not None and dd is not None and dd > 0 else None
    )
    output = {}
    for key, (unit, group, description) in SPECS.items():
        if scenario.metrics.get("sampling_interval_seconds") == 3600:
            description = description.replace("minute", "hourly").replace(
                "365 × 1440", "365 × 24"
            )
        value = number(derived[key] if key in derived else scenario.metrics.get(key))
        reason = None if value is not None else "not provided or undefined"
        if key in derived and not valid:
            value, reason = None, "incomplete or invalid equity history"
        if key in ANNUALIZED:
            if synthetic:
                value, reason = None, "synthetic demo"
            elif elapsed < 86400:
                value, reason = None, "insufficient history"
        output[key] = Metric(
            value=value,
            unit=unit,
            reason=reason,
            source="Python-derived" if key in derived else "stored summary",
            start=start,
            end=end,
            samples=stats.get("samples", 0),
            description=description,
            group=group,
        )
    return output


def analyze(repo, run_id, scenario, benchmark=None):
    detail = repo.detail(run_id)
    stats = Curve(repo, run_id, scenario).stats()
    metrics = build_metrics(scenario, stats, detail.synthetic)
    warnings = list(detail.warnings) + list(scenario.warnings)
    if (
        stats.get("start")
        and 86400 <= (stats["end"] - stats["start"]).total_seconds() < 30 * 86400
    ):
        warnings.append(
            "Short sample: fewer than 30 days; risk ratios are not evidence of statistical validity."
        )
    other = None
    if benchmark:
        other_stats = Curve(repo, run_id, benchmark).stats()
        if (stats.get("start"), stats.get("end")) == (
            other_stats.get("start"),
            other_stats.get("end"),
        ):
            other = build_metrics(benchmark, other_stats, detail.synthetic)
            a, b = metrics["total_return"].value, other["total_return"].value
            if a is not None and b is not None:
                metrics["benchmark_excess_return"] = metrics[
                    "benchmark_excess_return"
                ].model_copy(
                    update={"value": a - b, "reason": None, "source": "Python-derived"}
                )
        else:
            warnings.append("Benchmark period does not match; comparison unavailable.")
    return Analytics(
        metrics=metrics,
        benchmark=other,
        warnings=list(dict.fromkeys(warnings)),
        conventions=(
            CONVENTIONS.replace("minute", "hourly").replace("365 × 1440", "365 × 24")
            if scenario.metrics.get("sampling_interval_seconds") == 3600
            else CONVENTIONS
        ),
    )
