"""Bounded immutable experiment evidence and explicit cross-run comparisons."""

from .analytics import analyze
from .models import EquityRow, Page
from .lab_models import Comparison, ComparisonSeries, UniverseRow
from .queries import Curve, connection, dictionaries
from .repository import ReportError


def universe(
    jobs,
    identifier,
    table,
    *,
    page=1,
    page_size=50,
    decision_date=None,
    scope=None,
    wallet=None,
    at=None,
    selection=None,
):
    item = jobs.store.get(identifier)
    if item["status"] != "completed":
        raise ValueError("Evidence is available only after job completion")
    name = {
        "rankings": "rankings.parquet",
        "cohorts": "cohorts.parquet",
        "contributions": "contributions.parquet",
    }[table]
    directory = jobs.root / "results" / identifier
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        return Page[UniverseRow](
            rows=[], total=0, available=False, reason="Historical evidence unavailable"
        )
    clauses, params = [], [str(path)]
    for condition, value in (
        ("CAST(decision_time AS DATE) = ?", decision_date),
        ('contains(lower("user"),lower(?))', wallet),
        ("time = ?", at),
    ):
        if value:
            if table == "cohorts" and wallet or table != "contributions" and at:
                raise ValueError("Filter is not applicable to this table")
            clauses.append(condition)
            params.append(value)
    if scope:
        clauses.append("coin IS NULL" if scope == "pooled" else "coin = ?")
        if scope != "pooled":
            params.append(scope)
    if selection:
        if table != "rankings":
            raise ValueError("Selection filter applies only to rankings")
        clauses.append(
            {
                "selected": "selected",
                "eligible": "eligible AND NOT selected",
                "excluded": "NOT eligible",
            }[selection]
        )
    source = " FROM read_parquet(?,hive_partitioning=false)"
    with connection() as db:
        if db.execute("SELECT count(*)" + source, [str(path)]).fetchone()[0] == 0:
            return Page[UniverseRow](rows=[], total=0)
        if clauses:
            source += " WHERE " + " AND ".join(clauses)
        total = db.execute("SELECT count(*)" + source, params).fetchone()[0]
        order = {
            "rankings": 'decision_time DESC,coin,rank NULLS LAST,"user"',
            "cohorts": "decision_time DESC,coin",
            "contributions": 'time DESC,coin,"user"',
        }[table]
        rows = dictionaries(
            db.execute(
                "SELECT *" + source + f" ORDER BY {order} LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            )
        )
    return Page[UniverseRow](rows=rows, total=total)


def compare(jobs, repo, identifiers, units):
    if not 2 <= len(identifiers) <= 6 or len(set(identifiers)) != len(identifiers):
        raise ValueError("Choose two to six distinct completed backtests")
    series, metadata = [], []
    for identifier in identifiers:
        item = jobs.store.get(identifier)
        if item["status"] != "completed" or item["kind"] != "backtest":
            raise ValueError("Only completed backtests can be compared")
        config = item["config"]
        follower = config.get("follower", config)
        scenario = repo.scenario(
            item["run_id"],
            "strategy",
            follower["aggregation"],
            follower["latency_seconds"],
        )
        benchmark = repo.scenario(
            item["run_id"], "control", "btc_buy_hold", follower["latency_seconds"]
        )
        analytics = analyze(repo, item["run_id"], scenario, benchmark)
        curve = Curve(repo, item["run_id"], scenario).page()
        initial = analytics.metrics["initial_equity"].value
        if units == "growth":
            if initial is None or initial <= 0:
                curve = curve.model_copy(
                    update={
                        "rows": [],
                        "available": False,
                        "reason": "Initial equity unavailable for rebasing",
                    }
                )
            else:
                curve = curve.model_copy(
                    update={
                        "rows": [
                            row.model_copy(
                                update={
                                    key: getattr(row, key) / initial
                                    for key in (
                                        "equity",
                                        "cash",
                                        "unrealized_pnl",
                                        "gross_exposure",
                                        "net_exposure",
                                    )
                                }
                            )
                            for row in curve.rows
                        ]
                    }
                )
        cohort_path = jobs.root / "results" / identifier / "cohorts.parquet"
        market_path = jobs.root / "results" / identifier / "market_cohorts.parquet"
        market_turnover, mean_assets = None, None
        with connection() as db:
            turnover = db.execute(
                "SELECT avg(membership_turnover) FROM read_parquet(?)",
                [str(cohort_path)],
            ).fetchone()[0]
            if market_path.is_file():
                market_turnover, mean_assets = db.execute(
                    "SELECT avg(membership_turnover), avg(selected_count) FROM read_parquet(?)",
                    [str(market_path)],
                ).fetchone()
        series.append(
            ComparisonSeries(
                id=identifier,
                name=item["name"],
                config=config,
                analytics=analytics,
                curve=curve,
                membership_turnover=turnover,
                market_membership_turnover=market_turnover,
                mean_selected_assets=mean_assets,
                synthetic=item["provenance"]["manifest"]["synthetic"],
            )
        )
        metadata.append(item)
    differences = sorted(
        k
        for k in set().union(*(s.config for s in series))
        if any(s.config.get(k) != series[0].config.get(k) for s in series[1:])
    )
    warnings = []
    for key, message in (
        (
            "start",
            "Different start dates: original windows and metrics retained; not a ranked comparison",
        ),
        (
            "end",
            "Different end dates: original windows and metrics retained; not a ranked comparison",
        ),
        (
            "initial_equity",
            "Different starting capital; growth curves are explicitly rebased",
        ),
        ("fee_bps", "Different fee assumptions"),
        ("latency_seconds", "Different execution delays"),
        ("benchmark", "Different benchmarks"),
        ("split", "Different research splits"),
    ):
        if key in differences:
            warnings.append(message)
    if len({m["provenance"]["dataset_hash"] for m in metadata}) > 1:
        warnings.append(
            "Different dataset versions/coverage; inspect assumptions before interpreting differences"
        )
    if any(s.synthetic for s in series):
        warnings.append(
            "Synthetic demo results are not evidence of trader skill or profitability"
        )
    warnings.append(
        "Development comparisons only; this does not select or unlock a validation/test winner"
    )
    return Comparison(
        series=series, differences=differences, warnings=warnings, units=units
    )
