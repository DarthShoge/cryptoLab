"""Bounded selected-cohort positions from strictly past qualified native fills."""

from datetime import datetime, timedelta
from pathlib import Path
import uuid

import duckdb
import pyarrow

from .archive_cache import _safe
from .contracts import address, finite, symbol, utc
from .derived_day_builder import _release_empty_scratch
from .download import file_hash
from .prefix_qualification import _engine as qualification_engine, _previous
from .proxy_activity import IDENTITY, ORDER, REVERSE_ORDER
from .qualified_day import QualifiedDay, _day
from .query_directory_pin import pin_directory

SPILL_BYTES = 2 * 1024**3
MAX_USERS = 250
MAX_DAYS = 732


def _engine():
    root = Path(__file__).parent
    names = (
        "qualified_native_positions.py",
        "qualified_day.py",
        "contracts.py",
        "proxy_activity.py",
        "derived_day_builder.py",
        "derived_cache_resources.py",
        "derived_cache_lease.py",
        "archive_cache.py",
        "download.py",
        "query_directory_pin.py",
    )
    return dict(
        duckdb=duckdb.__version__,
        pyarrow=pyarrow.__version__,
        code={name: file_hash(root / name) for name in names},
    )


def _request(decision, coin, users):
    if not isinstance(decision, datetime):
        raise ValueError("Timezone-aware decision required")
    if not isinstance(users, (list, tuple)) or len(users) > MAX_USERS:
        raise ValueError("Expected at most 250 selected users")
    selected = tuple(address(user) for user in users)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate selected user")
    return utc(decision), symbol(coin), selected


class _Source:
    """One report plus deduplicated used-file pins, never a list of day objects."""

    def __init__(self, pin, decision, coin):
        if type(pin) is not dict or set(pin) != {"path", "sha256"}:
            raise ValueError("Pinned qualification report required")
        self.pin = dict(pin)
        self.engine = _engine()
        self.qualification = qualification_engine()
        report = _previous(self.pin, self.qualification)
        self.start, self.end = _day(report["source_start"]), _day(report["source_end"])
        if not 0 < (self.end - self.start).days <= MAX_DAYS:
            raise ValueError("Qualified position history day bound exceeded")
        if not self.start <= decision <= self.end:
            raise ValueError("Decision outside qualified position history")
        self.files = {}
        anchor = QualifiedDay(self.pin, report["source_start"])
        if coin not in anchor.coins:
            raise ValueError("Coin outside qualified market scope")
        self.remember(anchor)

    def remember(self, day):
        for entry in day.entries or (day.witness,):
            previous = self.files.setdefault(entry.path, entry)
            if previous != entry or len(self.files) > 5000:
                raise ValueError("Qualified file registry identity/count changed")

    def verify(self):
        path = Path(self.pin["path"]).absolute()
        _safe(path)
        if file_hash(path) != self.pin["sha256"]:
            raise ValueError("Qualification report identity changed")
        for entry in self.files.values():
            entry.verify()
        if (
            file_hash(path) != self.pin["sha256"]
            or _engine() != self.engine
            or qualification_engine() != self.qualification
        ):
            raise ValueError("Position source/report/engine changed")


def _query_day(db, day, decision, coin, users):
    db.read_parquet(
        [str(entry.path) for entry in day.entries], hive_partitioning=False
    ).create_view("native_day", replace=True)
    rows = db.execute(
        f"WITH unique_fills AS (SELECT * FROM native_day "
        "WHERE coin = ? AND user IN (SELECT unnest(?)) "
        "AND exchange_time >= ? AND exchange_time < ? "
        f"QUALIFY row_number() OVER (PARTITION BY {IDENTITY} ORDER BY {ORDER}) = 1) "
        "SELECT user, post_position FROM unique_fills "
        f"QUALIFY row_number() OVER (PARTITION BY user ORDER BY {REVERSE_ORDER}) = 1 "
        "ORDER BY user",
        [coin, list(users), day.start, min(day.end, decision)],
    ).fetchmany(MAX_USERS + 1)
    if len(rows) > len(users):
        raise ValueError("Native position result cardinality exceeded")
    result = {}
    for user, quantity in rows:
        if user not in users or user in result:
            raise ValueError("Unexpected/duplicate native position user")
        result[user] = finite(quantity)
    return result


def native_positions(resources, report_pin, decision, coin, users):
    """Return known quantities or None; no future fill can backfill knownness.

    Authorized event history is [report.source_start, decision). Physical spill
    before that origin does not extend coverage. Reference comparisons must use
    this same interior interval; a later registered-dataset origin must be
    reconciled explicitly by the downstream adapter, not silently substituted.

    The caller holds the shared cache lease and must not run another scratch
    query concurrently. All native fills for each selected user are considered;
    the selected-cohort bound is not a per-wallet fill limit.
    """
    resources.lease.check()
    request = _request(decision, coin, users)
    cutoff, market, selected = request
    source = _Source(report_pin, cutoff, market)
    result = dict.fromkeys(selected)

    def verify():
        resources.lease.check()
        source.verify()
        if report_pin != source.pin or _request(decision, coin, users) != request:
            raise ValueError("Native position request changed")

    if not selected or cutoff == source.start:
        verify()
        return result

    relative = f"scratch/{uuid.uuid4().hex}"
    token = resources.reserve(relative, SPILL_BYTES, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    info = scratch.stat()
    identity = info.st_dev, info.st_ino
    db = None
    with pin_directory(scratch, identity):
        try:
            db = duckdb.connect(
                config={
                    "memory_limit": "256MB",
                    "max_temp_directory_size": f"{SPILL_BYTES}B",
                    "temp_directory": str(scratch),
                    "threads": 1,
                    "TimeZone": "UTC",
                    "preserve_insertion_order": False,
                }
            )
            pending = set(selected)
            cursor = (cutoff - timedelta(microseconds=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            while pending and cursor >= source.start:
                day = QualifiedDay(source.pin, cursor.date().isoformat())
                source.remember(day)
                if day.entries:
                    values = _query_day(db, day, cutoff, market, tuple(sorted(pending)))
                    result.update(values)
                    pending.difference_update(values)
                day.verify()
                del day
                cursor -= timedelta(days=1)
            verify()
        finally:
            if db is not None:
                db.close()
            _release_empty_scratch(resources, token, scratch, identity)
    verify()
    return result
