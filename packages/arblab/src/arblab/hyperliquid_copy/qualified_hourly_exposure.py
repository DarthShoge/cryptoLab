"""Bounded native quantity-times-price samples for historical conviction."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
import uuid

import duckdb
import pyarrow

from .contracts import address, finite, symbol, utc
from .derived_day_builder import _release_empty_scratch
from .download import file_hash
from .proxy_activity import IDENTITY, ORDER, REVERSE_ORDER
from .qualified_day import QualifiedDay
from .qualified_native_positions import _Source
from .qualified_window import QualifiedWindow
from .query_directory_pin import pin_directory

HOUR = timedelta(hours=1)
SPILL_BYTES = 2 * 1024**3
SAMPLE_BATCH = 24


def _engine():
    root = Path(__file__).parent
    names = (
        "qualified_hourly_exposure.py",
        "qualified_native_positions.py",
        "qualified_day.py",
        "qualified_window.py",
        "proxy_activity.py",
        "contracts.py",
        "derived_publication.py",
        "derived_day_builder.py",
        "derived_cache_resources.py",
        "derived_cache_lease.py",
        "download.py",
        "query_directory_pin.py",
    )
    return dict(
        duckdb=duckdb.__version__,
        pyarrow=pyarrow.__version__,
        code={name: file_hash(root / name) for name in names},
    )


def _time(value):
    if not isinstance(value, datetime):
        raise ValueError("Timezone-aware native sample timestamp required")
    return utc(value)


@dataclass(frozen=True)
class _State:
    quantity: float | None = None
    quantity_time: datetime | None = None
    price: float | None = None
    price_time: datetime | None = None

    def check(self, origin, sample):
        for value, moment in (
            (self.quantity, self.quantity_time),
            (self.price, self.price_time),
        ):
            if (value is None) != (moment is None):
                raise ValueError("Inconsistent native sample knownness")
            if value is not None:
                finite(value)
                if not origin <= _time(moment) < sample:
                    raise ValueError("Native state outside strictly past history")
        if self.price is not None and self.price <= 0:
            raise ValueError("Invalid native sample price")

    def exposure(self, sample, age):
        if (
            self.quantity is None
            or self.price is None
            or (sample - self.price_time).total_seconds() > age
        ):
            return None
        return finite(self.quantity * self.price)


def _view(db, source):
    entries = source.entries or (source.witness,)
    db.read_parquet(
        [str(entry.path) for entry in entries], hive_partitioning=False
    ).create_view("native_source", replace=True)


def _seed_rows(db, day, start, user, coin, need_price):
    _view(db, day)
    return db.execute(
        "WITH unique_fills AS MATERIALIZED (SELECT * FROM native_source "
        "WHERE coin=? AND exchange_time>=? AND exchange_time<? AND (? OR user=?) "
        f"QUALIFY row_number() OVER (PARTITION BY {IDENTITY} ORDER BY {ORDER})=1) "
        "SELECT 'quantity' AS kind, exchange_time, post_position AS value FROM ("
        f"SELECT * FROM unique_fills WHERE user=? ORDER BY {REVERSE_ORDER} LIMIT 1) "
        "UNION ALL SELECT 'price', exchange_time, px FROM ("
        f"SELECT * FROM unique_fills ORDER BY {REVERSE_ORDER} LIMIT 1)",
        [coin, day.start, min(day.end, start), need_price, user, user],
    ).fetchmany(3)


def _seed(db, source, start, user, coin):
    state = _State()
    cursor = (start - timedelta(microseconds=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    while cursor >= source.start and (state.quantity is None or state.price is None):
        day = QualifiedDay(source.pin, cursor.date().isoformat())
        source.remember(day)
        if day.entries:
            rows = _seed_rows(db, day, start, user, coin, state.price is None)
            if len(rows) > 2 or len({row[0] for row in rows}) != len(rows):
                raise ValueError("Invalid native seed cardinality")
            for kind, moment, value in rows:
                if kind not in ("quantity", "price") or not day.start <= _time(
                    moment
                ) < min(day.end, start):
                    raise ValueError("Native seed outside queried interval")
                if getattr(state, kind) is None:
                    state = replace(
                        state, **{kind: finite(value), kind + "_time": moment}
                    )
        day.verify()
        del day
        state.check(source.start, start)
        cursor -= timedelta(days=1)
    return state


def _sample_rows(db, window, state, first, last, user, coin):
    _view(db, window)
    return db.execute(
        "WITH unique_fills AS MATERIALIZED (SELECT * FROM native_source "
        "WHERE coin=? AND exchange_time>=? AND exchange_time<? "
        f"QUALIFY row_number() OVER (PARTITION BY {IDENTITY} ORDER BY {ORDER})=1), "
        "positions AS (SELECT exchange_time,post_position FROM unique_fills WHERE user=? "
        f"QUALIFY row_number() OVER (PARTITION BY exchange_time ORDER BY {REVERSE_ORDER})=1 "
        "UNION ALL SELECT ?::TIMESTAMPTZ,?::DOUBLE WHERE ?::TIMESTAMPTZ IS NOT NULL), "
        "prices AS (SELECT exchange_time,px FROM unique_fills "
        f"QUALIFY row_number() OVER (PARTITION BY exchange_time ORDER BY {REVERSE_ORDER})=1 "
        "UNION ALL SELECT ?::TIMESTAMPTZ,?::DOUBLE WHERE ?::TIMESTAMPTZ IS NOT NULL) "
        "SELECT s.generate_series,p.post_position,p.exchange_time,m.px,m.exchange_time "
        "FROM generate_series(?, ?, INTERVAL '1 hour') s "
        "ASOF LEFT JOIN positions p ON s.generate_series>p.exchange_time "
        "ASOF LEFT JOIN prices m ON s.generate_series>m.exchange_time ORDER BY s.generate_series",
        [
            coin,
            window.start,
            window.end,
            user,
            state.quantity_time,
            state.quantity,
            state.quantity_time,
            state.price_time,
            state.price,
            state.price_time,
            first,
            last,
        ],
    ).fetchmany(SAMPLE_BATCH + 1)


def _samples(db, source, state, user, coin, start, count, age):
    result = []
    lower = start
    for offset in range(0, count, SAMPLE_BATCH):
        size = min(SAMPLE_BATCH, count - offset)
        first = start + offset * HOUR
        last = first + (size - 1) * HOUR
        if lower == last:
            # Only the first, single-sample chunk can have an empty event span.
            state.check(source.start, first)
            result.append((first, state.exposure(first, age)))
        else:
            window = QualifiedWindow(source.pin, lower, last)
            source.remember(window)
            rows = _sample_rows(db, window, state, first, last, user, coin)
            if len(rows) != size:
                raise ValueError("Native sample chunk cardinality changed")
            for index, (
                sample,
                quantity,
                quantity_time,
                price,
                price_time,
            ) in enumerate(rows):
                if sample != first + index * HOUR:
                    raise ValueError("Native hourly sample timestamps changed")
                state = _State(quantity, quantity_time, price, price_time)
                state.check(source.start, sample)
                result.append((sample, state.exposure(sample, age)))
            window.verify()
            del window
        lower = last
    return result


def hourly_exposure(
    resources, report_pin, user, coin, start, end, *, max_price_age_seconds
):
    """Hourly samples in [start,end), using native evidence strictly before each.

    History begins at the report origin; physical pre-origin spill does not
    extend authorization. Later registered coverage must be reconciled by the
    adapter. This is native trade-price notional, not a mark or account equity.
    """
    resources.lease.check()
    user, coin = address(user), symbol(coin)
    start, end = _time(start), _time(end)
    count, remainder = divmod(end - start, HOUR)
    age = finite(max_price_age_seconds)
    if not 0 < count <= 100_000 or remainder or age <= 0:
        raise ValueError("Invalid or oversized hourly exposure request")
    engine = _engine()
    source = _Source(report_pin, end, coin)
    if start < source.start:
        raise ValueError("Native samples outside qualified coverage")

    def verify():
        resources.lease.check()
        source.verify()
        if report_pin != source.pin or _engine() != engine:
            raise ValueError("Native sample request/engine changed")

    if start == source.start and count == 1:
        verify()
        return [(start, None)]
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
                    "threads": 1,
                    "TimeZone": "UTC",
                    "max_temp_directory_size": f"{SPILL_BYTES}B",
                    "temp_directory": str(scratch),
                    "preserve_insertion_order": False,
                }
            )
            state = _seed(db, source, start, user, coin)
            result = _samples(db, source, state, user, coin, start, count, age)
            verify()
        finally:
            if db is not None:
                db.close()
            _release_empty_scratch(resources, token, scratch, identity)
    verify()
    return result
