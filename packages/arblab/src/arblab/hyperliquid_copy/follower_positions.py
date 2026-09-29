"""Bounded cohort lookup with an explicit legacy scalar-reader fallback."""

from math import isfinite


def selected_positions(activity, users, coin, decision):
    if (
        type(users) not in (list, tuple)
        or len(users) > 250
        or any(type(user) is not str for user in users)
        or len(set(users)) != len(users)
    ):
        raise ValueError("Invalid selected trader cohort")
    if not users:
        return {}
    requested = tuple(users)
    batch = getattr(activity, "positions", None)
    result = (
        batch(list(requested), coin, decision)
        if batch is not None
        else {user: activity.position(user, coin, decision) for user in requested}
    )
    if type(result) is not dict or set(result) != set(requested):
        raise ValueError("Position result does not match selected trader cohort")
    if any(
        quantity is not None
        and (type(quantity) not in (int, float) or not isfinite(quantity))
        for quantity in result.values()
    ):
        raise ValueError("Invalid selected trader position quantity")
    # Preserve cohort order even if the SQL reader returns address order.
    return {user: result[user] for user in requested}
