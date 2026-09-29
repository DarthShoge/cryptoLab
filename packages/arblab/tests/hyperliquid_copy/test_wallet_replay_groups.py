from hashlib import sha256

import duckdb
import pytest


def address(n):
    return f"0x{n:040x}"


def leaves(counts):
    return tuple(
        (
            address(i),
            address(i + 1) if i + 1 < len(counts) else "0y",
            count,
            address(i) if count else None,
        )
        for i, count in enumerate(counts)
    )


def group(parts, limit=5):
    from arblab.hyperliquid_copy.wallet_replay_groups import coalesce_plan

    return coalesce_plan(parts, max_rows=limit)


def test_empty_source_retains_full_coverage():
    assert group(leaves([0, 0, 0])) == ((address(0), "0y", 0, None),)


def test_merge_exact_limit_across_empty_leaves_and_preserve_whale():
    parts = leaves([0, 2, 0, 3, 0, 7, 0, 1, 0])
    assert group(parts) == (
        (address(0), address(5), 5, None),
        parts[5],
        (address(6), "0y", 1, address(7)),
    )


def test_zero_neighbors_preserve_single_wallet_witness():
    assert group(leaves([0, 2, 0])) == ((address(0), "0y", 2, address(1)),)


def test_overflow_starts_another_group_without_splitting_leaf():
    parts = leaves([3, 3, 2])
    assert group(parts) == (parts[0], (address(1), "0y", 5, None))


def test_maximum_address_whale_is_preserved():
    part = (address(0), "0y", 1000, address(2**160 - 1))
    assert group([part], 1) == (part,)


def test_output_is_immutable_and_does_not_alias_mutable_input():
    parts = [list(p) for p in leaves([2, 2])]
    result = group(parts)
    assert isinstance(result, tuple) and all(isinstance(p, tuple) for p in result)
    parts[0][2] = 100
    assert result == ((address(0), "0y", 4, None),)


@pytest.mark.parametrize("limit", [None, True, False, 0, -1, 250001, 1.5])
def test_invalid_requested_bound(limit):
    with pytest.raises(ValueError):
        group(leaves([1]), limit)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "generator",
        "too_many",
        "shape",
        "count_bool",
        "negative",
        "count_float",
        "empty_witness",
        "wallet_outside",
        "wallet_type",
        "oversized_without_witness",
        "missing_start",
        "missing_end",
        "gap",
        "overlap",
        "reversed",
        "uppercase",
        "early_sentinel",
    ],
)
def test_invalid_complete_plan_rejected(fault):
    parts = [list(p) for p in leaves([2, 2])]
    if fault == "empty":
        parts = []
    elif fault == "generator":
        parts = iter(parts)
    elif fault == "too_many":
        parts = leaves([0] * 5001)
    elif fault == "shape":
        parts[1].append("extra")
    elif fault == "count_bool":
        parts[1][2] = True
    elif fault == "negative":
        parts[1][2] = -1
    elif fault == "count_float":
        parts[1][2] = 2.0
    elif fault == "empty_witness":
        parts[1][2] = 0
    elif fault == "wallet_outside":
        parts[0][3] = address(1)
    elif fault == "wallet_type":
        parts[0][3] = 123
    elif fault == "oversized_without_witness":
        parts[1][2:4] = [6, None]
    elif fault == "missing_start":
        parts[0][0] = address(1)
    elif fault == "missing_end":
        parts[-1][1] = address(2)
    elif fault == "gap":
        parts[1][0] = address(2)
    elif fault == "overlap":
        parts[1][0] = address(0)
    elif fault == "reversed":
        parts.reverse()
    elif fault == "uppercase":
        parts[0][0] = "0X" + "0" * 40
    else:
        parts[0][1] = "0y"
    with pytest.raises(ValueError):
        group(parts)


def test_generated_source_full_order_coverage_and_counts_match(tmp_path):
    from arblab.hyperliquid_copy.wallet_partition_plan import counted_plan

    users = ["0x" + sha256(str(i).encode()).hexdigest()[:40] for i in range(200)] * 3
    users += [address(0), address(2**160 - 1)] + [address(15)] * 50
    with duckdb.connect(
        config={
            "memory_limit": "256MB",
            "threads": 1,
            "max_temp_directory_size": "2GB",
            "temp_directory": str(tmp_path / "spill"),
        }
    ) as db:
        db.execute("CREATE TABLE source(user VARCHAR)")
        db.executemany("INSERT INTO source VALUES (?)", [(u,) for u in users])
        original = counted_plan(db, max_rows=12)
        combined = group(original, 12)
        assert len(combined) < len(original)
        assert combined[0][0] == address(0) and combined[-1][1] == "0y"
        assert all(a[1] == b[0] for a, b in zip(combined, combined[1:]))
        assert sum(p[2] for p in combined) == len(users)
        for low, high, count, wallet in original:
            assert sum(a <= low < high <= b for a, b, _, _ in combined) == 1
            if count > 12:
                assert (low, high, count, wallet) in combined
        actual = []
        for low, high, count, wallet in combined:
            assert count <= 12 or wallet is not None
            rows = db.execute(
                "SELECT user FROM source WHERE user>=? AND user<? ORDER BY user",
                [low, high],
            ).fetchall()
            assert len(rows) == count
            actual.extend(row[0] for row in rows)
        assert actual == sorted(users)
