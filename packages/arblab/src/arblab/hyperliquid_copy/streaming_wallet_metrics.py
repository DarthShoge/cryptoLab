"""Reference-equivalent wallet metrics from a qualified, ordered fill stream.

The caller owns source validation/deduplication and the shared resource lease.
Source iterators must release query resources on exhaustion, before disk medians.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from math import log1p

from .contracts import utc
from .episodes import PositionEpisode, closing_size, leader_net_pnl
from .ranking import RankingConfig
from .wallet_metric_spool import MetricSpool


@dataclass(frozen=True)
class WalletMetricResult:
    metrics: dict
    exclusions: tuple
    fill_count: int
    closing_notional: float
    gross_volume: float
    complete_episodes: int
    active_days: int
    spool_bytes: int
    peak_buffered_rows: int


def _episode(active, fill, net, semantics, spool):
    episode = active.get(fill.coin)
    if episode is None:
        episode = PositionEpisode(
            fill.user,
            fill.coin,
            fill.exchange_time,
            left_censored=abs(fill.start_position) > 1e-9,
        )
        active[fill.coin] = episode
    flipping = fill.start_position * fill.post_position < 0
    opening_fee = (
        fill.fee * (fill.sz - closing_size(fill)) / fill.sz if flipping else 0.0
    )
    episode.pnl += net + (
        opening_fee if flipping and semantics == "gross_excludes_fee" else 0
    )
    episode.fill_count += 1
    episode.taker_count += int(fill.crossed)
    episode.fees += fill.fee - opening_fee
    episode.peak_notional = max(
        episode.peak_notional,
        abs(fill.start_position) * fill.px,
        0 if flipping else abs(fill.post_position) * fill.px,
    )
    if abs(fill.post_position) <= 1e-9 or flipping:
        episode.closed_at = fill.exchange_time
        if episode.complete:
            spool.add("episode", episode.pnl, episode.minutes, episode.fill_count)
        del active[fill.coin]
    if flipping:
        active[fill.coin] = PositionEpisode(
            fill.user,
            fill.coin,
            fill.exchange_time,
            pnl=-opening_fee if semantics == "gross_excludes_fee" else 0.0,
            fill_count=1,
            taker_count=int(fill.crossed),
            fees=opening_fee,
            peak_notional=abs(fill.post_position) * fill.px,
        )


def _finish(spool, daily, takers, config):
    count, episodes = spool.counts["fill"], spool.counts["episode"]
    if not count:
        metrics = dict.fromkeys(
            (
                "pnl_efficiency",
                "profit_factor",
                "positive_day_rate",
                "drawdown_efficiency",
                "copyability",
            )
        )
        return WalletMetricResult(
            metrics, ("no_activity_in_lookback",), 0, 0, 0, 0, 0, 0, 0
        )
    pnl = spool.total("fill", "a")
    notional = spool.total("fill", "b")
    volume = spool.total("fill", "c")
    duration = spool.median("episode", "b") if episodes else 0
    fragmentation = (
        spool.median("episode", "c") if episodes else config.fragmentation_floor
    )
    excluded = tuple(
        reason
        for condition, reason in (
            (len(daily) < config.min_active_days, "insufficient_active_days"),
            (episodes < config.min_episodes, "insufficient_episodes"),
            (notional < config.min_notional, "insufficient_notional"),
            (duration < config.min_minutes, "too_short_holding_period"),
        )
        if condition
    )
    positive = spool.total("episode", "a", transform=lambda value: max(value, 0))
    negative = -spool.total("episode", "a", transform=lambda value: min(value, 0))
    pf = (
        positive / negative
        if negative
        else (config.profit_factor_cap if positive else 0)
    )
    peak, curve, drawdown = 0.0, 0.0, 0.0
    for day in sorted(daily):
        curve += daily[day]
        peak = max(peak, curve)
        drawdown = max(drawdown, peak - curve)
    copyability = (
        min(1.0, log1p(duration) / log1p(config.duration_cap_minutes))
        + config.fragmentation_floor / max(fragmentation, config.fragmentation_floor)
        + takers / count
    ) / 3
    metrics = dict(
        pnl_efficiency=pnl / notional if notional else 0,
        profit_factor=min(config.profit_factor_cap, pf),
        positive_day_rate=(sum(v > 0 for v in daily.values()) + 1) / (len(daily) + 2),
        drawdown_efficiency=max(
            -config.drawdown_cap,
            min(config.drawdown_cap, pnl / max(drawdown, config.drawdown_floor)),
        ),
        copyability=copyability,
    )
    return WalletMetricResult(
        metrics,
        excluded,
        count,
        notional,
        volume,
        episodes,
        len(daily),
        spool.disk_bytes,
        spool.peak_buffered_rows,
    )


def stream_wallet_metrics(
    fills, decision, config, semantics, *, temp_root, smoke=False, buffer_rows=4096
):
    if (
        not isinstance(config, RankingConfig)
        or type(config.lookback_days) is not int
        or not 1 <= config.lookback_days <= 732
    ):
        raise ValueError("Expected RankingConfig with bounded lookback")
    decision = utc(decision)
    start = decision - timedelta(days=config.lookback_days)
    active, daily, coins = {}, defaultdict(float), set()
    user, previous, takers = None, None, 0
    with MetricSpool(temp_root, buffer_rows=buffer_rows) as spool:
        for fill in fills:
            if user is None:
                user = fill.user
            if fill.user != user or not start <= fill.exchange_time < decision:
                raise ValueError(
                    "Expected one wallet and strictly causal lookback fills"
                )
            key = fill.order_key
            if previous is not None and key < previous:
                raise ValueError("Expected ordered wallet fills")
            previous = key
            coins.add(fill.coin)
            if len(coins) > 50:
                raise ValueError("Wallet market limit exceeded")
            net = leader_net_pnl(fill, semantics, smoke=smoke)
            spool.add("fill", net, closing_size(fill) * fill.px, fill.sz * fill.px)
            daily[fill.exchange_time.date()] += net
            takers += int(fill.crossed)
            _episode(active, fill, net, semantics, spool)
        return _finish(spool, daily, takers, config)
