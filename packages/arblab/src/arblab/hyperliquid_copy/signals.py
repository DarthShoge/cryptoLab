"""Ensemble aggregation with explicit missing-state shrinkage."""
from dataclasses import dataclass
from math import isfinite

SIGNALS = ("direction_equal", "direction_score_weighted", "conviction_trimmed")


@dataclass(frozen=True)
class Signal:
    value: float
    known_traders: int
    known_weight: float
    reason: str


def conviction_value(*, position_qty, mid, scale):
    if scale is None or scale <= 0:
        return None
    return max(-1.0, min(1.0, position_qty * mid / scale))


def aggregate_with_weights(values, scores, strategy, *, min_known=5, min_weight=.6, trim=.1):
    if strategy not in SIGNALS:
        raise ValueError("unknown signal")
    if set(values) != set(scores) or not 0 <= trim < .5:
        raise ValueError("invalid cohort/trim")
    if not values:
        return Signal(0, 0, 0, "insufficient_coverage"), {}
    if any(not isfinite(s) or s < 0 for s in scores.values()):
        raise ValueError("invalid score")
    total = sum(scores.values())
    weights = {u: (scores[u] / total if total and strategy != "direction_equal" else 1 / len(values)) for u in values}
    known = [(max(-1., min(1., v)), u, weights[u]) for u, v in values.items() if v is not None]
    if any(not isfinite(v) for v in values.values() if v is not None):
        raise ValueError("non-finite signal input")
    coverage = sum(row[2] for row in known)
    if len(known) < min_known or coverage + 1e-12 < min_weight:
        return Signal(0, len(known), coverage, "insufficient_coverage"), {u:0. for u in values}
    rows = [[v, u, w] for v, u, w in sorted(known)]
    if strategy == "conviction_trimmed":
        if coverage < 2 * trim:
            return Signal(0, len(known), coverage, "insufficient_coverage"), {u:0. for u in values}
        for tail in (rows, list(reversed(rows))):
            remaining = trim
            for row in tail:
                removed = min(row[2], remaining)
                row[2] -= removed
                remaining -= removed
        value = sum(v * w for v, _, w in rows) / (1 - 2 * trim)
    else:
        value = sum((1 if v > 0 else -1 if v < 0 else 0) * w for v, _, w in rows)
    effective = {u:0. for u in values}
    effective.update({u:w/(1-2*trim) if strategy == "conviction_trimmed" else w for _,u,w in rows})
    return Signal(max(-1., min(1., value)), len(known), coverage, "ok"), effective


def aggregate(values, scores, strategy, **options):
    return aggregate_with_weights(values,scores,strategy,**options)[0]
