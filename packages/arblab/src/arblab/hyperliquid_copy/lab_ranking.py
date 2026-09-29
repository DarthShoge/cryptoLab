"""Historical configurable selection; no performance or account-return inference."""

from collections import defaultdict
from datetime import timedelta
from math import ceil, isfinite

from .contracts import utc
from .lab_config import day
from .episodes import closing_size
from .ranking import RankingConfig, percentiles, wallet_metrics


def validate_ranking_bound(fill_count, config):
    # Each fill can introduce at most one wallet-scope pair. Daily decisions
    # conservatively bound weekly/monthly schedules without reading row data.
    decisions = (day(config.end) - day(config.start)).days
    if fill_count * decisions > 1_000_000:
        raise ValueError(
            "Run exceeds bounded ranking-history ceiling; no wallet sampling"
        )


def rank_universe(fills, decision, config, scope, semantics, *, smoke=False):
    decision = utc(decision)
    start = decision - timedelta(days=config.lookback_days)
    grouped = defaultdict(list)
    for fill in sorted(fills, key=lambda f: f.order_key):
        if (
            fill.exchange_time < decision
            and fill.coin in config.coins
            and (scope is None or fill.coin == scope)
        ):
            activity = grouped[fill.user]
            if fill.exchange_time >= start:
                activity.append(fill)
    return rank_activity(
        sorted(grouped.items()), decision, config, scope, semantics, smoke=smoke
    )


def rank_activity(grouped, decision, config, scope, semantics, *, smoke=False):
    """Score ordered wallet groups; callers may stream bounded histories from disk."""
    legacy = RankingConfig(
        lookback_days=config.lookback_days,
        min_active_days=config.min_active_days,
        min_episodes=config.min_episodes,
        min_notional=config.min_notional,
        min_minutes=config.min_minutes,
    )
    rows = []
    for user, activity in grouped:
        if activity:
            metrics, reasons = wallet_metrics(activity, legacy, semantics, smoke)
            if not sum(closing_size(f) * f.px for f in activity):
                metrics["pnl_efficiency"] = None
        else:
            metrics, reasons = (
                {k: None for k in config.metric_weights},
                ["no_activity_in_lookback"],
            )
        metrics["gross_volume"] = sum(f.sz * f.px for f in activity)
        reasons = list(reasons)
        if metrics["gross_volume"] < config.min_volume:
            reasons.append("insufficient_gross_volume")
        if any(
            metrics.get(k) is None or not isfinite(metrics[k])
            for k in config.metric_weights
        ):
            reasons.append("missing_ranking_metric")
        rows.append(
            dict(
                user=user,
                decision_time=decision,
                coin=scope,
                metrics=metrics,
                percentiles={},
                score=None,
                rank=None,
                selected=False,
                eligible=not reasons,
                reasons=reasons,
                exclusions=list(reasons),
                weight=0.0,
            )
        )
    eligible = [r for r in rows if r["eligible"]]
    for metric, weight in config.metric_weights.items():
        ranked = percentiles([r["metrics"][metric] for r in eligible])
        for row, percentile in zip(eligible, ranked):
            percentile = (
                1 - percentile
                if config.metric_directions[metric] == "asc"
                else percentile
            )
            row["percentiles"][metric] = percentile
            row["score"] = (row["score"] or 0) + weight * percentile
    eligible.sort(key=lambda r: (-r["score"], r["user"]))
    requested = (
        ceil(config.top_fraction * len(eligible))
        if config.selection == "fraction"
        else config.top_n
    )
    size = (
        min(config.max_cohort, max(config.min_cohort, requested))
        if len(eligible) >= config.min_cohort
        else 0
    )
    for index, row in enumerate(eligible):
        row["rank"] = index + 1
        row["selected"] = index < size
        if not row["selected"]:
            row["reasons"] = [
                "insufficient_cohort" if not size else "rank_below_cutoff"
            ]
    selected = [r for r in eligible if r["selected"]]
    total = sum(r["score"] for r in selected)
    for row in selected:
        row["weight"] = (
            row["score"] / total
            if total and config.aggregation != "direction_equal"
            else 1 / len(selected)
        )
    return sorted(
        rows, key=lambda r: (r["score"] is None, -(r["score"] or 0), r["user"])
    )


def cohort_snapshot(rows, decision, scope, previous=None):
    members = sorted(r["user"] for r in rows if r["selected"])
    old, new = set(previous or []), set(members)
    entries, exits = sorted(new - old), sorted(old - new)
    return dict(
        decision_time=decision,
        coin=scope,
        members=members,
        entries=entries,
        exits=exits,
        candidate_count=len(rows),
        eligible_count=sum(r["eligible"] for r in rows),
        selected_count=len(members),
        retention=len(old & new) / len(old) if old else None,
        membership_turnover=(len(entries) + len(exits)) / (len(old) + len(new))
        if previous is not None and old | new
        else (0.0 if previous is not None else None),
        cutoff_address=next((r["user"] for r in reversed(rows) if r["selected"]), None),
    )
