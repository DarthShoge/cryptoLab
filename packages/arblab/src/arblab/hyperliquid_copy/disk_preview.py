"""Stream complete requested ranking evidence while replaying bounded membership."""

from datetime import timedelta

from .contracts import utc
from .disk_score_result import ScoredCohort
from .lab_config import day
from .proxy_schedule import decision_times
from .ranking_artifact import RankingSink
from .selection_disk_evidence import append_scored_cohort


class _PreviewSink(RankingSink):
    def __init__(self, path, decision, scope):
        super().__init__(path)
        self.decision, self.scope = decision, scope

    def extend(self, rows):
        for offset in range(0, len(rows), 4096):
            selected = [
                row
                for row in rows[offset : offset + 4096]
                if row["decision_time"] == self.decision and row["coin"] == self.scope
            ]
            if selected:
                super().extend(selected)


def preview_selection_disk(
    dataset, config, decision, scope, *, rankings_path, state_type
):
    decision = utc(decision)
    start, end = day(config.start), day(config.end)
    if (
        not start <= decision < end
        or decision.hour
        or decision.minute
        or decision.second
        or decision.microsecond
        or not 0 < (end - start).days <= 366
    ):
        raise ValueError("Invalid bounded preview decision/range")
    state = state_type(dataset, config)
    with _PreviewSink(rankings_path, decision, scope) as sink:
        state.rankings = sink
        for at in decision_times(
            start, decision + timedelta(days=1), getattr(config, "rebalance", "daily")
        ):
            state.advance(at)
        actual = any(
            row["decision_time"] == decision and row["coin"] == scope
            for row in state.trader_cohorts
        )
        if not actual and (scope is None or scope in state.active):
            rows = state.rank_traders(
                decision, config.effective(state.active, state.budgets), scope
            )
            if isinstance(rows, ScoredCohort):
                append_scored_cohort(
                    rows,
                    sink,
                    decision,
                    scope,
                    state.previous.get(scope),
                    market_time=state.market_time,
                    trigger="hypothetical_preview",
                )
            else:
                for offset in range(0, len(rows), 4096):
                    sink.extend(
                        [
                            row
                            | dict(
                                market_decision_time=state.market_time,
                                decision_trigger="hypothetical_preview",
                            )
                            for row in rows[offset : offset + 4096]
                        ]
                    )
    return sink.artifact, state, not actual
