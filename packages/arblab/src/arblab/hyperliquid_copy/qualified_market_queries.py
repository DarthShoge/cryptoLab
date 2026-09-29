"""Causal market observations and unique trade volume, not listing evidence."""

from datetime import datetime
from pathlib import Path
import uuid

import duckdb
import pyarrow

from .contracts import finite, symbol, utc
from .derived_day_builder import _release_empty_scratch
from .download import file_hash
from .prefix_qualification import _previous, _engine as qualification_engine
from .proxy_activity import IDENTITY, ORDER
from .qualified_day import QualifiedDay, _day
from .qualified_window import QualifiedWindow
from .query_directory_pin import pin_directory

SPILL_BYTES = 2 * 1024**3


def _engine():
    root = Path(__file__).parent
    names = (
        "qualified_market_queries.py",
        "qualified_window.py",
        "qualified_day.py",
        "contracts.py",
        "derived_publication.py",
        "derived_day_builder.py",
        "derived_cache_resources.py",
        "derived_cache_lease.py",
        "download.py",
        "proxy_activity.py",
        "query_directory_pin.py",
    )
    return dict(
        duckdb=duckdb.__version__,
        pyarrow=pyarrow.__version__,
        code={name: file_hash(root / name) for name in names},
    )


def _time(value):
    if not isinstance(value, datetime):
        raise ValueError("Timezone-aware market query timestamp required")
    return utc(value)


class _Query:
    """One immutable interval and one shared, explicitly owned scratch query."""

    def __init__(self, resources, pin, start, end):
        resources.lease.check()
        if type(pin) is not dict or set(pin) != {"path", "sha256"}:
            raise ValueError("Pinned market qualification required")
        self.resources, self.original_pin, self.pin = resources, pin, dict(pin)
        self.engine = _engine()
        self.qualification = qualification_engine()
        report = _previous(self.pin, self.qualification)
        origin, finish = _day(report["source_start"]), _day(report["source_end"])
        if not 0 < (finish - origin).days <= 732:
            raise ValueError("Qualified market history day bound exceeded")
        self.start = origin if start is None else _time(start)
        self.end = _time(end)
        if not origin <= self.start <= self.end <= finish:
            raise ValueError("Market query outside qualified coverage")
        if self.start == self.end:
            self.source = QualifiedDay(self.pin, report["source_start"])
        else:
            self.source = QualifiedWindow(self.pin, self.start, self.end)

    def verify(self):
        self.resources.lease.check()
        self.source.verify()
        if self.original_pin != self.pin:
            raise ValueError("Market query report pin changed")
        if _engine() != self.engine or qualification_engine() != self.qualification:
            raise ValueError("Market query engine changed")

    def rows(self, sql, parameters, maximum):
        self.verify()
        relative = f"scratch/{uuid.uuid4().hex}"
        token = self.resources.reserve(relative, SPILL_BYTES, "scratch")
        scratch = self.resources.root / relative
        scratch.mkdir()
        info = scratch.stat()
        identity = info.st_dev, info.st_ino
        db = None
        with pin_directory(scratch, identity):
            try:
                db = duckdb.connect(
                    config={
                        "memory_limit": "256MB",
                        "threads": 1,
                        "TimeZone": "UTC",
                        "max_temp_directory_size": f"{SPILL_BYTES}B",
                        "temp_directory": str(scratch),
                        "preserve_insertion_order": False,
                    }
                )
                entries = self.source.entries or (self.source.witness,)
                db.read_parquet(
                    [str(entry.path) for entry in entries], hive_partitioning=False
                ).create_view("market_source")
                result = _execute(db, sql, parameters, maximum)
                if len(result) > maximum:
                    raise ValueError("Market query result cardinality exceeded")
                self.verify()
            finally:
                if db is not None:
                    db.close()
                _release_empty_scratch(self.resources, token, scratch, identity)
        self.verify()
        return result


def _execute(db, sql, parameters, maximum):
    return db.execute(sql, parameters).fetchmany(maximum + 1)


def observed_markets(resources, report_pin, decision):
    """Sorted markets observed in [report.source_start, decision), at most50.

    Physical out-of-coverage spill is not extra history. A registered dataset
    with a later coverage origin requires explicit downstream reconciliation.
    """
    query = _Query(resources, report_pin, None, decision)
    if query.start == query.end:
        query.verify()
        return []
    rows = query.rows(
        "SELECT DISTINCT coin FROM market_source "
        "WHERE exchange_time >= ? AND exchange_time < ? ORDER BY coin",
        [query.start, query.end],
        50,
    )
    result = [symbol(row[0]) for row in rows]
    if result != sorted(set(result)) or not set(result) <= set(query.source.coins):
        raise ValueError("Observed markets outside qualified scope")
    return result


def market_volume(resources, report_pin, coin, start, decision):
    """Unique native trade notional in [start, decision); not wallet-side volume.

    Preserve native dedup before the global aggregate: although MAX(px*sz) is
    algebraically unchanged without it, changed floating SUM order can alter
    threshold eligibility or tied market ranks. Do not sum rounded day totals.
    """
    start, decision, coin = _time(start), _time(decision), symbol(coin)
    if start >= decision:
        raise ValueError("Positive market volume interval required")
    query = _Query(resources, report_pin, start, decision)
    if coin not in query.source.coins:
        raise ValueError("Volume coin outside qualified scope")
    rows = query.rows(
        "WITH unique_fills AS MATERIALIZED (SELECT * FROM market_source "
        f"QUALIFY row_number() OVER (PARTITION BY {IDENTITY} ORDER BY {ORDER}) = 1) "
        "SELECT coalesce(sum(notional), 0) FROM ("
        "SELECT exchange_time, tid, max(px * sz) AS notional FROM unique_fills "
        "WHERE coin = ? AND exchange_time >= ? AND exchange_time < ? "
        "GROUP BY exchange_time, tid)",
        [coin, start, decision],
        1,
    )
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError("Expected one market volume result")
    result = finite(rows[0][0])
    if result < 0:
        raise ValueError("Negative native market volume")
    return result
