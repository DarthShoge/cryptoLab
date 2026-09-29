"""Coalesce complete physical coverage into bounded sequential replay groups.

This validates descriptors, not source data. The caller must qualify/count the
source, check each written group's physical count and verify its source pins.
"""

import re

ADDRESS = re.compile(r"0x[0-9a-f]{40}")
FIRST = "0x" + "0" * 40
LAST = "0y"


def _address(value):
    return type(value) is str and ADDRESS.fullmatch(value) is not None


def coalesce_plan(parts, *, max_rows=250_000):
    """Return immutable intervals; never split a leaf or enlarge the row bound."""
    if type(max_rows) is not int or not 1 <= max_rows <= 250_000:
        raise ValueError("Invalid replay row bound")
    if type(parts) not in (list, tuple) or not 1 <= len(parts) <= 5000:
        raise ValueError("Expected bounded complete physical plan")
    checked, previous = [], FIRST
    for part in parts:
        if type(part) not in (list, tuple) or len(part) != 4:
            raise ValueError("Invalid physical partition descriptor")
        lower, upper, count, wallet = part
        if (
            not _address(lower)
            or not (_address(upper) or upper == LAST)
            or lower != previous
            or not lower < upper
            or type(count) is not int
            or count < 0
            or (not count and wallet is not None)
            or (
                wallet is not None
                and (not _address(wallet) or not lower <= wallet < upper)
            )
            or (count > max_rows and wallet is None)
        ):
            raise ValueError("Invalid physical plan coverage/count/wallet")
        checked.append((lower, upper, count, wallet))
        previous = upper
    if previous != LAST:
        raise ValueError("Incomplete physical plan coverage")

    result, active = [], checked[0]
    for part in checked[1:]:
        count = active[2] + part[2]
        if count <= max_rows:
            wallet = active[3] if not part[2] else part[3] if not active[2] else None
            active = (active[0], part[1], count, wallet)
        else:
            result.append(active)
            active = part
    result.append(active)
    return tuple(result)
