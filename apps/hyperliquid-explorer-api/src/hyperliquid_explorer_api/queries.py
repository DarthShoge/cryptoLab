"""Bounded output SQL queries; computations precede chart downsampling."""

from contextlib import contextmanager

import duckdb

from .models import EquityRow, Page, Record
from .repository import ReportError


@contextmanager
def connection():
    with duckdb.connect(config={"threads": 2, "memory_limit": "256MB"}) as db:
        db.execute("SET TimeZone = 'UTC'")
        yield db


def dictionaries(cursor):
    names = [c[0] for c in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def scenario_where(scenario):
    column = "signal_name" if scenario.scenario_type == "strategy" else "control_name"
    return f"{column} = ? AND latency_seconds IS NOT DISTINCT FROM ?", [
        scenario.name,
        scenario.latency_seconds,
    ]


class Curve:
    def __init__(self, repo, run_id, scenario):
        file = (
            "equity_curve.parquet"
            if scenario.scenario_type == "strategy"
            else "control_equity_curve.parquet"
        )
        self.path = repo.artifact(run_id, file, optional=True)
        self.where, self.params = scenario_where(scenario)
        self.interval_seconds = scenario.metrics.get("sampling_interval_seconds", 60)
        if self.interval_seconds not in (60, 3600):
            raise ReportError("Unsupported equity sampling interval")

    @property
    def cte(self):
        return f"""WITH filtered AS (SELECT * FROM read_parquet(?, hive_partitioning=false) WHERE {self.where}),
            curve AS (SELECT *, row_number() OVER (ORDER BY time) AS seq,
                1-equity/max(equity) OVER (ORDER BY time ROWS UNBOUNDED PRECEDING) AS drawdown,
                lag(time) OVER (ORDER BY time) AS prev_time FROM filtered)"""

    def stats(self):
        if self.path is None:
            return {}
        with connection() as db:
            return dictionaries(
                db.execute(
                    self.cte
                    + """ SELECT count(*) AS samples, min(time) AS start, max(time) AS end,
                first(equity ORDER BY time) AS initial, last(equity ORDER BY time) AS final,
                last(drawdown ORDER BY time) AS current_drawdown, max(drawdown) AS maximum_drawdown,
                avg(gross_exposure/nullif(equity,0)) AS mean_gross_leverage,
                avg(net_exposure/nullif(equity,0)) AS mean_net_leverage,
                max(gross_exposure/nullif(equity,0)) AS maximum_gross_leverage,
                count(*) FILTER (WHERE time IS NULL OR equity IS NULL OR equity <= 0 OR NOT isfinite(equity)
                    OR cash IS NULL OR NOT isfinite(cash)
                    OR unrealized_pnl IS NULL OR NOT isfinite(unrealized_pnl)
                    OR gross_exposure IS NULL OR NOT isfinite(gross_exposure)
                    OR net_exposure IS NULL OR NOT isfinite(net_exposure)) AS invalid_equity,
                count(*) FILTER (WHERE prev_time IS NOT NULL AND epoch(time-prev_time) != ?) AS gaps
                FROM curve""",
                    [str(self.path), *self.params, self.interval_seconds],
                )
            )[0]

    def page(self, max_points=2000):
        if self.path is None:
            return Page[EquityRow](
                rows=[], total=0, available=False, reason="Equity artifact unavailable"
            )
        stats = self.stats()
        count = stats["samples"]
        if not count:
            return Page[EquityRow](rows=[], total=0)
        if stats["invalid_equity"] or stats["gaps"]:
            raise ReportError(
                "Equity contains invalid values or a missing/duplicate sample"
            )
        sampling = ""
        parameters = [str(self.path), *self.params]
        if count > max_points:
            buckets = max(1, (max_points - 2) // 8)
            terms = [
                f"SELECT {fn}(seq,{column}) AS seq FROM bucketed GROUP BY bucket"
                for column in ("equity", "drawdown", "gross_exposure", "net_exposure")
                for fn in ("arg_min", "arg_max")
            ]
            sampling = (
                f", bucketed AS (SELECT *, floor((seq-1)*{buckets}.0/{count}) AS bucket FROM curve), chosen AS ("
                + " UNION ".join(terms)
                + f" UNION SELECT 1 UNION SELECT {count})"
            )
            tail = " WHERE seq IN (SELECT seq FROM chosen)"
        else:
            tail = ""
        with connection() as db:
            rows = dictionaries(
                db.execute(
                    self.cte
                    + sampling
                    + " SELECT time,equity,cash,unrealized_pnl,gross_exposure,net_exposure,drawdown FROM curve"
                    + tail
                    + " ORDER BY time",
                    parameters,
                )
            )
        return Page[EquityRow](rows=rows, total=count, downsampled=count > max_points)


def records(
    repo,
    run_id,
    table,
    *,
    scenario=None,
    page=1,
    page_size=50,
    coin=None,
    reason=None,
    wallet=None,
    decision_date=None,
    scope=None,
):
    file = {
        "traders": "trader_scores.parquet",
        "cohorts": "cohort_history.parquet",
        "fills": "simulated_fills.parquet",
        "funding": "funding_ledger.parquet",
    }[table]
    if scenario and scenario.scenario_type == "control":
        file = (
            "control_funding_ledger.parquet"
            if table == "funding"
            else "control_fills.parquet"
        )
    path = repo.artifact(run_id, file, optional=True)
    if path is None:
        return Page[Record](
            rows=[], total=0, available=False, reason="Artifact unavailable"
        )
    with connection() as db:
        # Empty prototype files have intentionally minimal schemas.
        total = db.execute(
            "SELECT count(*) FROM read_parquet(?, hive_partitioning=false)", [str(path)]
        ).fetchone()[0]
        if total == 0:
            return Page[Record](rows=[], total=0)
        clauses, params = [], [str(path)]
        if scenario:
            clause, values = scenario_where(scenario)
            clauses.append(clause)
            params.extend(values)
        for column, value in (("coin", coin), ("reason", reason)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if wallet:
            clauses.append('contains(lower("user"), lower(?))')
            params.append(wallet)
        if decision_date:
            clauses.append("CAST(decision_time AS DATE) = ?")
            params.append(decision_date)
        if scope:
            clauses.append("coin IS NULL" if scope == "global" else "coin = ?")
            if scope != "global":
                params.append(scope)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        source = " FROM read_parquet(?, hive_partitioning=false)" + where
        total = db.execute("SELECT count(*)" + source, params).fetchone()[0]
        order = {
            "traders": 'decision_time DESC, score DESC NULLS LAST, "user", coin',
            "cohorts": "decision_time DESC, coin",
            "fills": "signal_time, coin, book_time, requested_qty, vwap",
            "funding": "time, coin",
        }[table]
        rows = dictionaries(
            db.execute(
                "SELECT *" + source + f" ORDER BY {order} LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            )
        )
    return Page[Record](rows=rows, total=total)
