"""Market decisions precede independent trader decisions; shared by preview/replay."""

from datetime import timedelta
from math import ceil
from .lab_config import day
from .lab_pipeline import is_decision
from .lab_ranking import rank_universe, cohort_snapshot
from .lab_market_selection import select_markets


def target_tick(at, start, minutes):
    elapsed = int((at - start).total_seconds() / 60)
    return start + timedelta(minutes=((elapsed + minutes - 1) // minutes) * minutes)


class SelectionState:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.start = day(config.start)
        self.active = []
        self.budgets = {}
        self.ever = set()
        self.selected = {}
        self.market_time = None
        self.previous = {}
        self.market_rankings = []
        self.market_cohorts = []
        self.rankings = []
        self.trader_cohorts = []

    def advance(self, at):
        c = self.config
        old = set(self.active)
        market_changed = False
        if is_decision(at, self.start, c.market_universe.reselection):
            rows, cohort = select_markets(
                self.dataset.catalogue,
                self.dataset.volume,
                c.market_universe,
                at,
                None if not self.market_cohorts else self.active,
                c.follower.scale_lookback_days
                if c.follower.aggregation == "conviction_trimmed"
                else 0,
                target_tick(at, self.start, c.follower.update_minutes),
            )
            self.market_rankings.extend(rows)
            self.market_cohorts.append(cohort)
            self.active = cohort["members"]
            self.budgets = {
                r["instrument_id"]: r["budget"] for r in rows if r["selected"]
            }
            self.ever.update(self.active)
            self.market_time = at
            market_changed = old != set(self.active)
        scheduled = is_decision(at, self.start, c.trader.reselection)
        if c.trader.scope == "pooled":
            scopes = [None] if scheduled or market_changed else []
        else:
            scopes = sorted(
                (set(self.active) if scheduled else set(self.active) - old)
                | (old - set(self.active))
            )
        for scope in scopes:
            exiting = scope is not None and scope not in self.active
            effective = c.effective(self.active, self.budgets)
            rows = (
                []
                if exiting
                else rank_universe(
                    self.dataset.fills,
                    at,
                    effective,
                    scope,
                    self.dataset.manifest["fee_semantics"],
                    smoke=True,
                )
            )
            trigger = (
                "market_exit"
                if exiting
                else "scheduled"
                if scheduled
                else "market_change"
                if scope is None
                else "market_entry"
            )
            for row in rows:
                row.update(
                    market_decision_time=self.market_time, decision_trigger=trigger
                )
            snapshot = cohort_snapshot(rows, at, scope, self.previous.get(scope))
            snapshot.update(
                market_decision_time=self.market_time,
                decision_trigger=trigger,
                requested_count=c.trader.top_n
                if c.trader.selection == "n"
                else ceil(c.trader.top_fraction * snapshot["eligible_count"]),
            )
            self.rankings.extend(rows)
            self.trader_cohorts.append(snapshot)
            self.previous[scope] = snapshot["members"]
            for coin in self.active if scope is None else [scope]:
                self.selected[coin] = [r for r in rows if r["selected"]]
        for coin in old - set(self.active):
            self.selected[coin] = []
        return bool(scopes)


def preview_selection(dataset, config, decision, scope):
    state = SelectionState(dataset, config)
    at = day(config.start)
    while at <= decision:
        state.advance(at)
        at += timedelta(days=1)
    actual = any(
        r["decision_time"] == decision and r["coin"] == scope
        for r in state.trader_cohorts
    )
    if actual:
        rows = [
            r
            for r in state.rankings
            if r["decision_time"] == decision and r["coin"] == scope
        ]
    else:
        rows = (
            rank_universe(
                dataset.fills,
                decision,
                config.effective(state.active, state.budgets),
                scope,
                dataset.manifest["fee_semantics"],
                smoke=True,
            )
            if scope is None or scope in state.active
            else []
        )
        for row in rows:
            row.update(
                market_decision_time=state.market_time,
                decision_trigger="hypothetical_preview",
            )
    return rows, state, not actual
