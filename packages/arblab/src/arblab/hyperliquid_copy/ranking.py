"""Point-in-time scoring. Only fills strictly before the decision are visible."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from math import ceil, log1p
from statistics import median

from .contracts import utc
from .episodes import build_episodes, closing_size, leader_net_pnl


@dataclass(frozen=True)
class RankingConfig:
    lookback_days: int = 90
    min_active_days: int = 30
    min_episodes: int = 20
    min_notional: float = 100000
    min_minutes: float = 15
    profit_factor_cap: float = 4
    drawdown_floor: float = 1
    drawdown_cap: float = 10
    duration_cap_minutes: float = 1440
    fragmentation_floor: float = 5
    min_cohort: int = 5
    max_cohort: int = 25
    top_fraction: float = .05


@dataclass(frozen=True)
class TraderScore:
    user: str
    decision_time: object
    coin: str | None
    metrics: dict
    score: float | None
    selected: bool
    exclusions: tuple


def percentiles(values):
    if len(values) == 1:
        return [.5]
    ranks = {}
    ordered = sorted(values)
    for i, value in enumerate(ordered):
        ranks.setdefault(value, []).append(i)
    return [sum(ranks[v])/len(ranks[v])/(len(values)-1) for v in values]


def _metrics(fills, config, semantics, smoke):
    episodes = [e for e in build_episodes(fills, semantics, smoke=smoke) if e.complete]
    pnl = [leader_net_pnl(f, semantics, smoke=smoke) for f in fills]
    daily = defaultdict(float)
    for f, value in zip(fills, pnl):
        daily[f.exchange_time.date()] += value
    notional = sum(closing_size(f)*f.px for f in fills)
    duration = median([e.minutes for e in episodes]) if episodes else 0
    excluded = []
    for condition, reason in [(len(daily) < config.min_active_days, "insufficient_active_days"),
                              (len(episodes) < config.min_episodes, "insufficient_episodes"),
                              (notional < config.min_notional, "insufficient_notional"),
                              (duration < config.min_minutes, "too_short_holding_period")]:
        if condition:
            excluded.append(reason)
    positive = sum(max(e.pnl, 0) for e in episodes)
    negative = -sum(min(e.pnl, 0) for e in episodes)
    pf = positive/negative if negative else (config.profit_factor_cap if positive else 0)
    peak, curve, drawdown = 0., 0., 0.
    for day in sorted(daily):
        curve += daily[day]
        peak = max(peak, curve)
        drawdown = max(drawdown, peak-curve)
    fragmentation = median([e.fill_count for e in episodes]) if episodes else config.fragmentation_floor
    copyability = (min(1., log1p(duration)/log1p(config.duration_cap_minutes)) +
                   config.fragmentation_floor/max(fragmentation, config.fragmentation_floor) +
                   sum(f.crossed for f in fills)/len(fills))/3
    metrics = dict(pnl_efficiency=sum(pnl)/notional if notional else 0,
                   profit_factor=min(config.profit_factor_cap, pf),
                   positive_day_rate=(sum(v > 0 for v in daily.values())+1)/(len(daily)+2),
                   drawdown_efficiency=max(-config.drawdown_cap, min(config.drawdown_cap, sum(pnl)/max(drawdown, config.drawdown_floor))),
                   copyability=copyability)
    return metrics, tuple(excluded)


def rank_traders(fills, decision_time, config, semantics, *, coin=None, smoke=False):
    decision_time = utc(decision_time)
    start = decision_time - timedelta(days=config.lookback_days)
    grouped = defaultdict(list)
    for fill in sorted(fills, key=lambda f: f.order_key):
        if start <= fill.exchange_time < decision_time and (coin is None or fill.coin == coin):
            grouped[fill.user].append(fill)
    candidates = {user: _metrics(rows, config, semantics, smoke) for user, rows in sorted(grouped.items())}
    eligible = [user for user, (_, exclusions) in candidates.items() if not exclusions]
    scores = {user: 0. for user in eligible}
    if eligible:
        for metric in candidates[eligible[0]][0]:
            for user, percentile in zip(eligible, percentiles([candidates[u][0][metric] for u in eligible])):
                scores[user] += percentile/5
    order = sorted(eligible, key=lambda u: (-scores[u], u))
    size = min(config.max_cohort, max(config.min_cohort, ceil(config.top_fraction*len(eligible))))
    selected = set(order[:size]) if len(order) >= config.min_cohort else set()
    rows = [TraderScore(u, decision_time, coin, metrics, scores.get(u), u in selected, exclusions)
            for u, (metrics, exclusions) in candidates.items()]
    return sorted(rows, key=lambda r: (r.score is None, -(r.score or 0), r.user))
