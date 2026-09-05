"""Bounded, parameterized queries over immutable market selection evidence."""

import json

from .models import Page
from .lab_models import MarketRow
from .queries import connection, dictionaries


def preview_info(jobs, identifier):
    item = jobs.store.get(identifier)
    if item["status"] != "completed" or item["kind"] != "cohort_preview":
        raise ValueError("Completed preview required")
    directory = jobs.root / "results" / identifier
    path = (directory / "preview.json").resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Invalid preview evidence")
    if not path.is_file():
        return {"hypothetical": None}
    if path.stat().st_size > 1024:
        raise ValueError("Invalid preview evidence")
    return json.loads(path.read_text())


def market_universe(
    jobs,
    identifier,
    table,
    *,
    page=1,
    page_size=50,
    decision_date=None,
    asset_class=None,
    instrument_id=None,
):
    item = jobs.store.get(identifier)
    if item["status"] != "completed":
        raise ValueError("Evidence requires a completed experiment")
    directory = jobs.root / "results" / identifier
    path = (
        directory
        / {"rankings": "market_rankings.parquet", "cohorts": "market_cohorts.parquet"}[
            table
        ]
    ).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        return Page[MarketRow](
            rows=[],
            total=0,
            available=False,
            reason="Market history unavailable for this experiment",
        )
    if table == "cohorts" and (asset_class or instrument_id):
        raise ValueError("Instrument filters require rankings")
    clauses, params = [], [str(path)]
    for condition, value in (
        ("CAST(decision_time AS DATE) = ?", decision_date),
        ("asset_class = ?", asset_class),
        ("instrument_id = ?", instrument_id),
    ):
        if value is not None:
            clauses.append(condition)
            params.append(value)
    source = " FROM read_parquet(?, hive_partitioning=false)"
    if clauses:
        source += " WHERE " + " AND ".join(clauses)
    order = "decision_time DESC" + (", instrument_id" if table == "rankings" else "")
    with connection() as db:
        total = db.execute("SELECT count(*)" + source, params).fetchone()[0]
        rows = dictionaries(
            db.execute(
                "SELECT *" + source + f" ORDER BY {order} LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            )
        )
    return Page[MarketRow](rows=rows, total=total)
