"""Stream complete disk rankings into run evidence; retain only selected rows."""

from contextlib import closing

from .bound_scoring_context import BoundScoredCohort
from .disk_score_result import ScoredCohort
from .lab_ranking import cohort_snapshot
from .ranking_artifact import RankingSink


def append_scored_cohort(
    result,
    sink,
    decision,
    scope,
    previous,
    *,
    market_time,
    trigger,
    acknowledge=None,
):
    if (
        not isinstance(result, ScoredCohort)
        or not isinstance(sink, RankingSink)
        or acknowledge is not None
        and not callable(acknowledge)
    ):
        raise ValueError("ScoredCohort requires an open RankingSink")
    if isinstance(result, BoundScoredCohort):
        if result.bound_decision != decision or result.bound_scope != scope:
            raise ValueError("Bound cohort decision/scope context mismatch")
    elif result.candidate_count == 0:
        raise ValueError("Unbound empty cohort context is not supported")
    if (
        type(result.candidate_count) is not int
        or type(result.eligible_count) is not int
        or not 0 <= result.eligible_count <= result.candidate_count
        or len(result._selected_json) > 250
    ):
        raise ValueError("Invalid scored cohort counts")
    try:
        return _append_scored_cohort(
            result,
            sink,
            decision,
            scope,
            previous,
            market_time=market_time,
            trigger=trigger,
            acknowledge=acknowledge,
        )
    except BaseException:
        sink.abort()
        raise


def _append_scored_cohort(
    result,
    sink,
    decision,
    scope,
    previous,
    *,
    market_time,
    trigger,
    acknowledge,
):
    selected = list(result.selected)

    def context(row):
        if row["decision_time"] != decision or row["coin"] != scope:
            raise ValueError("Scored cohort decision/scope context mismatch")

    for row in selected:
        context(row)
    count = eligible = chosen = 0
    with closing(result.iter_batches()) as batches:
        for batch in batches:
            if len(batch) > 4096:
                raise ValueError("Ranking evidence batch limit exceeded")
            copied = []
            for row in batch:
                context(row)
                count += 1
                eligible += int(row["eligible"])
                if row["selected"]:
                    bare = {
                        k: v
                        for k, v in row.items()
                        if k not in ("market_decision_time", "decision_trigger")
                    }
                    if chosen >= len(selected) or bare != selected[chosen]:
                        raise ValueError("Scored cohort selected metadata mismatch")
                    chosen += 1
                copied.append(
                    row
                    | dict(market_decision_time=market_time, decision_trigger=trigger)
                )
            sink.extend(copied)
    if (count, eligible, chosen) != (
        result.candidate_count,
        result.eligible_count,
        len(selected),
    ):
        raise ValueError("Scored cohort evidence count mismatch")
    selected = [
        row | dict(market_decision_time=market_time, decision_trigger=trigger)
        for row in selected
    ]
    snapshot = cohort_snapshot(selected, decision, scope, previous)
    snapshot.update(candidate_count=count, eligible_count=eligible)
    if acknowledge is not None:
        acknowledge(result, snapshot, selected)
    return snapshot, selected
