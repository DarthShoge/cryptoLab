from collections import Counter
from hashlib import sha256

import duckdb
import pytest


def reference(users, limit):
    counts = Counter(users)
    pending, result = [""], []
    while pending:
        prefix = pending.pop()
        low = int(prefix.ljust(40, "0"), 16)
        high = low + 16 ** (40 - len(prefix))
        lower = f"0x{low:040x}"
        upper = f"0x{high:040x}" if high < 2**160 else "0y"
        matching = sorted(u for u in counts if lower <= u < upper)
        count = sum(counts[u] for u in matching)
        if count <= limit or len(matching) <= 1:
            result.append(
                (lower, upper, count, matching[0] if len(matching) == 1 else None)
            )
        else:
            pending.extend(prefix + d for d in reversed("0123456789abcdef"))
    return tuple(result)


@pytest.fixture
def db(tmp_path):
    with duckdb.connect(
        config={
            "memory_limit": "256MB",
            "threads": 1,
            "max_temp_directory_size": "2GB",
            "temp_directory": str(tmp_path / "spill"),
        }
    ) as connection:
        connection.execute("CREATE TABLE source(user VARCHAR)")
        yield connection


@pytest.mark.parametrize(
    "users,limit",
    [
        ([], 1),
        (["0x" + "0" * 40, "0x" + "f" * 40], 1),
        (["0x" + "a" * 40] * 20, 1),
        (["0x" + "a" * 40, "0x" + "b" * 40] * 2, 4),
        (["0x" + sha256(str(i).encode()).hexdigest()[:40] for i in range(150)] * 3, 9),
    ],
)
def test_matches_complete_reference_physical_plan(db, users, limit):
    from arblab.hyperliquid_copy.wallet_partition_plan import counted_plan

    if users:
        db.executemany("INSERT INTO source VALUES (?)", [(u,) for u in users])
    actual = counted_plan(db, max_rows=limit)
    assert actual == reference(users, limit)
    assert sum(p[2] for p in actual) == len(users)
    assert all(a[1] == b[0] for a, b in zip(actual, actual[1:]))
    assert not db.execute(
        "SELECT table_name FROM duckdb_tables() WHERE table_name='wallet_partition_counts'"
    ).fetchall()


class Traced:
    def __init__(self, db, interrupt=False):
        self.db, self.interrupt, self.queries = db, interrupt, []

    def execute(self, sql, *args):
        self.queries.append(sql)
        if self.interrupt and "coalesce" in sql.lower():
            raise KeyboardInterrupt("prefix interrupted")
        return self.db.execute(sql, *args)


def test_scans_source_once_and_queries_only_aggregated_counts(db):
    from arblab.hyperliquid_copy.wallet_partition_plan import counted_plan

    db.execute(
        "INSERT INTO source VALUES ('0x' || repeat('0',40)), ('0x' || repeat('f',40))"
    )
    traced = Traced(db)
    assert counted_plan(traced, max_rows=1) == reference(
        ["0x" + "0" * 40, "0x" + "f" * 40], 1
    )
    scans = [q for q in traced.queries if "from source" in q.lower()]
    assert len(scans) == 1 and "group by user" in scans[0].lower()
    assert len([q for q in traced.queries if "coalesce" in q.lower()]) == 17


def test_cleans_own_counts_on_interruption(db):
    from arblab.hyperliquid_copy.wallet_partition_plan import counted_plan

    with pytest.raises(KeyboardInterrupt, match="prefix interrupted"):
        counted_plan(Traced(db, interrupt=True))
    assert not db.execute(
        "SELECT table_name FROM duckdb_tables() WHERE table_name='wallet_partition_counts'"
    ).fetchall()
    assert counted_plan(db) == reference([], 250000)


def test_does_not_drop_preexisting_table(db):
    from arblab.hyperliquid_copy.wallet_partition_plan import counted_plan

    db.execute("CREATE TEMP TABLE wallet_partition_counts AS SELECT 42 AS sentinel")
    with pytest.raises(duckdb.CatalogException):
        counted_plan(db)
    assert db.execute("SELECT sentinel FROM wallet_partition_counts").fetchone() == (
        42,
    )


@pytest.mark.parametrize("limit", [True, False, 0, -1, 250001, 1.5, None])
def test_invalid_bounds_rejected_before_query(db, limit):
    from arblab.hyperliquid_copy.wallet_partition_plan import counted_plan

    traced = Traced(db)
    with pytest.raises(ValueError):
        counted_plan(traced, max_rows=limit)
    assert not traced.queries


def test_leaf_limit_fails_without_partial_result_and_releases_counts(db):
    from arblab.hyperliquid_copy.wallet_partition_plan import counted_plan

    db.execute(
        "INSERT INTO source SELECT '0x' || md5(cast(i AS VARCHAR)) || '00000000' FROM range(4096) r(i)"
    )
    with pytest.raises(ValueError, match="partition/depth limit"):
        counted_plan(db, max_rows=1)
    assert not db.execute(
        "SELECT table_name FROM duckdb_tables() WHERE table_name='wallet_partition_counts'"
    ).fetchall()
