"""Canonical market rows, deterministic lookups, and upstream Parquet access."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import duckdb

from .contracts import finite, symbol, utc


@dataclass(frozen=True)
class Book:
    time: object
    coin: str
    bids: tuple
    asks: tuple

    @property
    def mid(self):
        return (self.bids[0][0] + self.asks[0][0]) / 2


def parse_book(row):
    sides = []
    for name in ("bids", "asks"):
        levels = tuple((finite(v["px"]), finite(v["sz"])) for v in row[name])
        if not levels or any(px <= 0 or sz <= 0 for px, sz in levels):
            raise ValueError("invalid book levels")
        if list(levels) != sorted(levels, key=lambda x: x[0], reverse=name == "bids"):
            raise ValueError("unordered book")
        sides.append(levels)
    if sides[0][0][0] >= sides[1][0][0]:
        raise ValueError("crossed book")
    return Book(utc(row["exch_time"]), symbol(row["coin"]), *sides)


def read_parquet(paths, time_column, start, end):
    if not paths:
        return []
    if time_column not in ("exch_time", "timestamp", "exchange_time"):
        raise ValueError("invalid timestamp column")
    with duckdb.connect() as db:
        rows = db.execute(f'SELECT * FROM read_parquet(?, hive_partitioning=false, union_by_name=true) WHERE "{time_column}" >= ? AND "{time_column}" <= ?',
                          [[str(p) for p in paths], start, end])
        columns = [c[0] for c in rows.description]
        return [dict(zip(columns, row)) for row in rows.fetchall()]


class MarketData:
    def __init__(self, books, funding):
        by_key = {}
        for row in books:
            book = parse_book(row)
            key = book.coin, book.time
            if key in by_key and by_key[key] != book:
                raise ValueError("conflicting book rows")
            by_key[key] = book
        self.books, self.times = {}, {}
        for (coin, at), book in sorted(by_key.items()):
            self.books.setdefault(coin, []).append(book)
            self.times.setdefault(coin, []).append(at)
        self.funding = {}
        for row in funding:
            coin = symbol(row.get("coin") or row["instrument"].removesuffix("-PERP").removesuffix("_perp"))
            key = coin, utc(row["timestamp"])
            rate = finite(row["rate"])
            if key in self.funding and self.funding[key] != rate:
                raise ValueError("conflicting funding rows")
            self.funding[key] = rate

    @classmethod
    def from_parquet(cls, books, funding, start, end):
        return cls(read_parquet(books, "exch_time", start, end),
                   read_parquet(funding, "timestamp", start, end))

    def mark(self, coin, at, *, max_age=60):
        index = bisect_right(self.times.get(coin, []), utc(at)) - 1
        if index < 0 or (at - self.times[coin][index]).total_seconds() > max_age:
            raise ValueError(f"stale mark: {coin} at {at.isoformat()}")
        return self.books[coin][index].mid

    def execution_book(self, coin, at, *, max_lag=2):
        index = bisect_left(self.times.get(coin, []), utc(at))
        if index >= len(self.times.get(coin, [])):
            return None
        book = self.books[coin][index]
        return book if (book.time - at).total_seconds() <= max_lag else None
