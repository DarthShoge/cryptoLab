"""Market decisions precede independent trader decisions; shared by preview/replay."""

from datetime import timedelta
from math import ceil
from .lab_config import day
from .lab_pipeline import is_decision
from .lab_ranking import rank_universe, cohort_snapshot
from .lab_market_selection import select_markets
from .proxy_schedule import decision_times
from .disk_score_result import ScoredCohort
from .selection_disk_evidence import append_scored_cohort


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
            rows, cohort = self.select_markets(at)
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
            rows = [] if exiting else self.rank_traders(at, effective, scope)
            trigger = (
                "market_exit"
                if exiting
                else "scheduled"
                if scheduled
                else "market_change"
                if scope is None
                else "market_entry"
            )
            if isinstance(rows, ScoredCohort):
                consume = getattr(
                    getattr(self.dataset, "activity", None), "consume_ranking", None
                )
                snapshot, selected = append_scored_cohort(
                    rows,
                    self.rankings,
                    at,
                    scope,
                    self.previous.get(scope),
                    market_time=self.market_time,
                    trigger=trigger,
                    acknowledge=(
                        None
                        if consume is None
                        else lambda result, snapshot, selected: consume(
                            result,
                            snapshot,
                            selected,
                            config=effective,
                            semantics=self.dataset.manifest["fee_semantics"],
                        )
                    ),
                )
            else:
                for row in rows:
                    row.update(
                        market_decision_time=self.market_time, decision_trigger=trigger
                    )
                snapshot = cohort_snapshot(rows, at, scope, self.previous.get(scope))
                self.rankings.extend(rows)
                selected = [r for r in rows if r["selected"]]
            snapshot.update(
                market_decision_time=self.market_time,
                decision_trigger=trigger,
                requested_count=c.trader.top_n
                if c.trader.selection == "n"
                else ceil(c.trader.top_fraction * snapshot["eligible_count"]),
            )
            self.trader_cohorts.append(snapshot)
            self.previous[scope] = snapshot["members"]
            for coin in self.active if scope is None else [scope]:
                self.selected[coin] = list(selected)
        for coin in old - set(self.active):
            self.selected[coin] = []
        return bool(scopes)

    def select_markets(self, at):
        c = self.config
        return select_markets(
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

    def rank_traders(self, at, effective, scope):
        return rank_universe(
            self.dataset.fills,
            at,
            effective,
            scope,
            self.dataset.manifest["fee_semantics"],
            smoke=True,
        )


def preview_selection(dataset, config, decision, scope, *, state_type=SelectionState):
    state = state_type(dataset, config)
    for at in decision_times(
        day(config.start),
        decision + timedelta(days=1),
        getattr(config, "rebalance", "daily"),
    ):
        # A preview returns only its requested decision rankings. Historical
        # membership/market state stays available without retaining every score.
        state.rankings.clear()
        state.advance(at)
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
            state.rank_traders(
                decision,
                config.effective(state.active, state.budgets),
                scope,
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
