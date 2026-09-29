"""Conservative ranking-event bounds following the actual replay evaluation grid."""

from .lab_config import day
from .lab_config_proxy import LabConfigProxy
from .lab_pipeline import is_decision
from .proxy_schedule import decision_times


def selection_ticks(config):
    if not isinstance(config, LabConfigProxy):
        raise ValueError("Proxy capacity requires a proxy configuration")
    start, end = day(config.start), day(config.end)
    if not 0 < (end - start).days <= 732:
        raise ValueError("Capacity schedule day bound exceeded")
    return tuple(
        at
        for at in decision_times(start, end, getattr(config, "rebalance", "hourly"))
        if is_decision(at, start, config.trader.reselection)
        or is_decision(at, start, config.market_universe.reselection)
    )


def ranking_upper_bound(config, coins, count_at):
    if (
        type(coins) not in (list, tuple)
        or any(type(c) is not str for c in coins)
        or len(set(coins)) != len(coins)
    ):
        raise ValueError("Distinct capacity market superset required")
    scopes = coins if config.trader.scope == "per_asset" else [None] if coins else []
    total = 0
    for at in selection_ticks(config):
        for scope in scopes:
            count = count_at(at, coins, scope)
            if type(count) is not int or count < 0:
                raise ValueError("Invalid complete candidate count")
            total += count
    return total
