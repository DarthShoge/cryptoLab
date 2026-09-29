"""Bounded diagnostics from immutable artifacts. No worker or cache access."""

import duckdb
from datetime import datetime

from .diagnostic_accounting import accounting, metric, worst_context
from .diagnostic_models import Diagnostics
from .diagnostic_statistics import profiles, weekly_statistics
from .queries import connection, dictionaries
from .repository import ANNUALIZED, ReportError, public_json, number

MAX_ROWS = 600_000
METRIC_UNITS = {
    "total_return": "percent", "annualized_return": "percent", "final_equity": "usd",
    "sharpe": "ratio", "sortino": "ratio", "annualized_volatility": "percent",
    "max_drawdown": "percent", "max_drawdown_minutes": "minutes",
    "fee_drag": "percent", "funding_drag": "percent",
}
METHODS = [
    "Stored risk metrics use the saved sampling interval, zero risk-free rate and sample standard deviation. Hourly annualization is sqrt(365 × 24); minute annualization is sqrt(365 × 1440). Sortino uses downside RMS over all returns.",
    "Profiles and drawdowns are computed on validated full-resolution equity before extrema-preserving daily display sampling. Equity is already net of modeled costs. No additional fees are subtracted.",
    "Weeks run Monday 00:00 UTC to Monday 00:00 UTC; months use UTC calendar boundaries. Clipped periods are partial and excluded from weekly statistics. Benchmark statistics pair exact complete start/end boundaries, never forward-fill.",
    "Weekly volatility is sample standard deviation; tracking error uses sqrt(365/7) annualization. Beta = covariance(strategy, BTC) / variance(BTC). Correlation and beta are unavailable for zero benchmark variance.",
    "Quantiles use linear interpolation. Tail mean averages observed weekly returns at or below the 5th percentile; one year gives very few tail observations.",
    "95% mean-return intervals: 2,000 circular moving-block bootstrap samples, four-week blocks, seed 20260926, percentile endpoints; at least 20 full weeks required. Excess-return blocks are paired. Assumes approximately stationary dependence; one development window cannot establish trader skill. Intervals exclude model and missing-data risk and do not correct for strategy selection/multiple testing.",
    "Cohort turnover = (entries + exits) / (previous + current membership counts). Unknown previous cohort is N/A; known empty-to-nonempty is 100%; both known empty is 0%. Missing cohort dates are not zero membership.",
    "Accounting checks use USD 0.01 tolerance. They test internal consistency only, not independent native marks, cash flows or liquidation realism; original reconciliation and research status are unchanged.",
]


def read_rows(repo, run, filename, scenario=None, limit=MAX_ROWS):
    path = repo.artifact(run, filename, optional=True)
    if path is None:
        return None, f"{filename}: unavailable"
    where, params = "", [str(path)]
    if scenario:
        column = "signal_name" if scenario["scenario_type"] == "strategy" else "control_name"
        name = scenario.get("signal_name") or scenario.get("control_name")
        where = f" WHERE {column} = ? AND latency_seconds IS NOT DISTINCT FROM ?"
        params += [name, scenario.get("latency_seconds")]
    try:
        with connection() as db:
            result = dictionaries(db.execute(
                "SELECT * FROM read_parquet(?, hive_partitioning=false)" + where + " LIMIT ?",
                [*params, limit + 1],
            ))
        if len(result) > limit:
            return None, f"{filename}: exceeds diagnostic row limit ({limit})"
        return result, None
    except (duckdb.Error, OSError, ValueError):
        return None, f"{filename}: malformed or unsupported artifact"


def series_data(repo, run, scenario):
    if scenario is None:
        return dict(reason="Configured scenario unavailable"), []
    filename = "equity_curve.parquet" if scenario["scenario_type"] == "strategy" else "control_equity_curve.parquet"
    rows, reason = read_rows(repo, run, filename, scenario)
    if reason:
        return dict(reason=reason), []
    try:
        rows.sort(key=lambda r: r["time"])
        result = profiles(rows, scenario.get("sampling_interval_seconds", 60))
        return result, rows
    except (ValueError, TypeError, KeyError):
        return dict(reason="Invalid, empty or incomplete equity history; no return inference available"), []


def valid_time(value):
    return isinstance(value, datetime) and value.utcoffset() is not None


def cohort_rows(rows, reason):
    if rows is None:
        return [], reason
    if any(not valid_time(r.get("decision_time")) or not isinstance(r.get("coin"), (str, type(None))) for r in rows):
        return [], "Historical cohorts unavailable: invalid decision timestamps or asset identifiers"
    cleaned = []
    for row in rows:
        entry = {"decision_time": row["decision_time"], "coin": row.get("coin")}
        for key in ("candidate_count", "eligible_count", "selected_count", "membership_turnover"):
            value = number(row.get(key))
            if value is not None and (value < 0 or (key.endswith("count") and not value.is_integer()) or (key == "membership_turnover" and value > 1)):
                value = None
            entry[key] = value
            if value is None:
                reason = "Some cohort counts/turnover are unavailable or invalid; shown as N/A"
        cleaned.append(entry)
    return sorted(cleaned, key=lambda r: (r["decision_time"], r.get("coin") or "")), reason


def get_diagnostics(jobs, repo, identifier):
    try:
        item = jobs.store.get(identifier)
    except ValueError:
        raise ReportError("Unknown experiment", 404) from None
    if item["status"] != "completed" or item["kind"] != "backtest":
        raise ReportError("Diagnostics are available only for completed backtests", 409)
    return build_diagnostics(repo, item)


def build_diagnostics(repo, item):
    run, config = item["run_id"], item["config"]
    summary = repo.json(run, "summary.json")
    reconciliation = repo.json(run, "reconciliation.json", optional=True)
    follower = config.get("follower", config)
    name, delay = follower.get("aggregation"), follower.get("latency_seconds")

    def scenario(kind, key, value, latency):
        matches = [s for s in summary.get("scenarios", []) if s.get("scenario_type") == kind
                   and s.get(key) == value and s.get("latency_seconds") == latency]
        return matches[0] if len(matches) == 1 else None

    selected = scenario("strategy", "signal_name", name, delay)
    btc = scenario("control", "control_name", "btc_buy_hold", delay)
    cash = scenario("control", "control_name", "cash", None)
    series, raw = {}, {}
    warnings = list(summary.get("warnings", [])) + list(reconciliation.get("issues", []))
    for key, value in (("strategy", selected), ("btc", btc), ("cash", cash)):
        series[key], raw[key] = series_data(repo, run, value)
        if series[key].get("reason"):
            warnings.append(f"{key}: {series[key]['reason']}")
    synthetic = item.get("provenance", {}).get("manifest", {}).get("synthetic") is True
    if synthetic:
        warnings.append("Synthetic demonstration: not evidence of profitability or trader skill")
    statistics = weekly_statistics(series["strategy"].get("weeks", []), series["btc"].get("weeks", []))
    # Comparing total returns requires identical endpoints; weekly paired stats above
    # explicitly report their possibly smaller common sample.
    a, b = raw["strategy"], raw["btc"]
    aligned = bool(a and b and a[0]["time"] == b[0]["time"] and a[-1]["time"] == b[-1]["time"])
    statistics["excess_total_return_pp"] = metric(
        a[-1]["equity"] / a[0]["equity"] - b[-1]["equity"] / b[0]["equity"] if aligned else None,
        "percent", "Different or unavailable full equity windows",
    )
    if a and b and not aligned:
        warnings.append("BTC full window differs; total-return outperformance unavailable. Weekly statistics use only paired full weeks.")
    for key, rows in raw.items():
        if rows and (rows[0]["time"].date().isoformat() != config.get("start") or rows[-1]["time"].date().isoformat() != config.get("end")):
            warnings.append(f"{key}: observed equity window differs from configured dates")

    ledgers = []
    for filename in ("simulated_fills.parquet", "funding_ledger.parquet"):
        rows, reason = read_rows(repo, run, filename, selected) if selected else (None, "Configured strategy unavailable")
        field = "fee" if filename == "simulated_fills.parquet" else "cash_delta"
        if rows is not None and any(not valid_time(r.get("time")) or number(r.get(field)) is None for r in rows):
            rows, reason = None, f"{filename}: invalid ledger timestamps or amounts"
        ledgers.append(rows)
        if reason:
            warnings.append(reason)
    costs, checks = accounting(raw["strategy"], selected or {}, *ledgers)
    cohorts, cohort_reason = read_rows(repo, run, "cohort_history.parquet", limit=5000)
    cohorts, cohort_reason = cohort_rows(cohorts, cohort_reason)
    if not cohorts and not cohort_reason:
        cohort_reason = "No historical cohort records"

    def stored(row, valid):
        if row is None:
            return {}
        return {k: metric(None if (not valid or synthetic and k in ANNUALIZED) else row.get(k), unit,
                          "Invalid equity history" if not valid else "Synthetic demonstration" if synthetic and k in ANNUALIZED else "Not provided or undefined")
                for k, unit in METRIC_UNITS.items()}

    return Diagnostics(
        experiment_id=item["id"], run_id=run, name=item["name"], dataset_id=item["dataset_id"],
        config_hash=item["config_hash"], config=public_json(config),
        provenance=public_json({k: item.get("provenance", {}).get(k) for k in ("dataset_hash", "manifest")}),
        research_eligible=summary.get("research_eligible"), synthetic=synthetic,
        reconciliation=public_json(reconciliation), warnings=list(dict.fromkeys(warnings)), series=series,
        stored_metrics=stored(selected, bool(a)), benchmark_metrics=stored(btc, bool(b)), statistics=statistics,
        accounting=costs, checks=checks, positions=public_json((selected or {}).get("residual_positions", {})),
        cohorts=cohorts or [], cohort_reason=cohort_reason,
        worst_weeks=worst_context(series["strategy"].get("weeks", []), *ledgers, raw["strategy"]),
        methods=METHODS,
    )
