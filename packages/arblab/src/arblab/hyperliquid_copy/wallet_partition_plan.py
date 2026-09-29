"""Exact physical-row radix planning on a caller-owned bounded SQL connection.

The caller supplies the qualified, cutoff-filtered ``source`` relation, reserves
query scratch and verifies source/context before accepting the returned plan.
Only bounded partition descriptors enter Python; all wallet counts stay in SQL.
"""


def counted_plan(db, *, max_rows=250_000):
    if type(max_rows) is not int or not 0 < max_rows <= 250_000:
        raise ValueError("Invalid ordered replay row bound")
    # Do not replace/drop a pre-existing relation if this CREATE fails.
    db.execute(
        "CREATE TEMP TABLE wallet_partition_counts AS "
        "SELECT user,count(*) AS n FROM source GROUP BY user"
    )
    try:
        pending, result = [""], []
        while pending:
            prefix = pending.pop()
            lower = int(prefix.ljust(40, "0"), 16)
            upper = lower + 16 ** (40 - len(prefix))
            low = f"0x{lower:040x}"
            high = f"0x{upper:040x}" if upper < 2**160 else "0y"
            count, first, last = db.execute(
                "SELECT coalesce(sum(n),0),min(user),max(user) "
                "FROM wallet_partition_counts WHERE user>=? AND user<?",
                [low, high],
            ).fetchone()
            count = int(count)
            if count <= max_rows or first == last:
                result.append(
                    (low, high, count, first if count and first == last else None)
                )
            else:
                if len(prefix) >= 40 or len(pending) + len(result) + 16 > 5000:
                    raise ValueError("Ordered replay partition/depth limit exceeded")
                pending.extend(prefix + d for d in reversed("0123456789abcdef"))
        return tuple(result)
    finally:
        db.execute("DROP TABLE wallet_partition_counts")
