from dataclasses import replace
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_pipeline import is_decision
from arblab.hyperliquid_copy.proxy_schedule import decision_times
from .test_qualified_scheduled_activity import scheduled_config


@pytest.mark.parametrize("scope", ["per_asset", "pooled"])
def test_bound_covers_real_selection_state_entries_exits_and_pooled_changes(scope):
    from arblab.hyperliquid_copy.lab_schedule import SelectionState
    from arblab.hyperliquid_copy.capacity_schedule import ranking_upper_bound

    base = scheduled_config()._legacy()
    config = replace(
        base,
        end="2026-09-03",
        trader=replace(base.trader, scope=scope, reselection="monthly"),
        market_universe=replace(base.market_universe, reselection="weekly"),
    )
    actual = []

    class Oracle(SelectionState):
        def select_markets(self, at):
            choices = [["BTC"], ["BTC", "ETH"], ["ETH"], [], ["BTC"]]
            members = choices[min(len(self.market_cohorts), len(choices) - 1)]
            return (
                [
                    dict(instrument_id=c, budget=1 / len(members), selected=True)
                    for c in members
                ],
                dict(members=members),
            )

        def rank_traders(self, at, effective, coin):
            if self.active:
                actual.append((at, coin))
            return []

    state = Oracle(None, config)
    for at in decision_times(day(config.start), day(config.end), "hourly"):
        state.advance(at)
    bounded = []

    def count(at, coins, coin):
        bounded.append((at, coin))
        return 123456

    bound = ranking_upper_bound(config, ["BTC", "ETH"], count)
    assert actual
    assert set(actual) <= set(bounded)
    assert bound >= len(actual) * 123456
    assert any(
        c["decision_trigger"]
        == ("market_entry" if scope == "per_asset" else "market_change")
        for c in state.trader_cohorts
    )


@pytest.mark.parametrize("frequency", ["weekly", "daily"])
def test_scheduled_capacity_uses_actual_selection_ticks(frequency):
    from arblab.hyperliquid_copy.capacity_schedule import (
        selection_ticks,
        ranking_upper_bound,
    )

    base = scheduled_config()
    config = replace(
        base,
        end="2026-08-25",
        rebalance=frequency,
        trader=replace(base.trader, reselection=frequency),
        market_universe=replace(base.market_universe, reselection=frequency),
    )
    expected = tuple(decision_times(day(config.start), day(config.end), frequency))
    assert selection_ticks(config) == expected
    calls = []

    def count(at, coins, scope):
        calls.append((at, scope))
        return 200001

    assert (
        ranking_upper_bound(config, ["BTC", "ETH"], count) == len(expected) * 2 * 200001
    )
    assert len(calls) == len(expected) * 2
    assert ranking_upper_bound(config, [], count) == 0


def test_legacy_independent_market_trader_ticks_are_unioned():
    from arblab.hyperliquid_copy.capacity_schedule import selection_ticks

    base = scheduled_config()._legacy()
    config = replace(
        base,
        end="2026-09-03",
        trader=replace(base.trader, reselection="weekly"),
        market_universe=replace(base.market_universe, reselection="monthly"),
    )
    start, end = day(config.start), day(config.end)
    expected = []
    at = start
    while at < end:
        if is_decision(at, start, "weekly") or is_decision(at, start, "monthly"):
            expected.append(at)
        at += timedelta(hours=1)
    assert selection_ticks(config) == tuple(expected)


def test_pooled_capacity_counts_once_per_tick():
    from arblab.hyperliquid_copy.capacity_schedule import ranking_upper_bound

    base = scheduled_config()
    config = replace(base, trader=replace(base.trader, scope="pooled"))

    def count(at, coins, scope):
        assert scope is None
        return 123456

    assert ranking_upper_bound(config, ["BTC", "ETH"], count) == 123456
