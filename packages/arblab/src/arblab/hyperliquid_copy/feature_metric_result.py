"""Candidate validation and metric-row projection shared by compact reducers."""

from math import isfinite, log1p
import re

import pyarrow as pa

from .lab_config import METRICS
from .ranking import RankingConfig


def candidate_table(candidates):
    values, previous = [], None
    for user in candidates:
        if (
            type(user) is not str
            or not re.fullmatch(r"0x[0-9a-f]{40}", user)
            or previous is not None
            and user <= previous
        ):
            raise ValueError("Invalid or unordered candidate identity")
        values.append(user)
        previous = user
    return pa.table({"user": pa.array(values, type=pa.string())})


def metric_row(row, config):
    (
        user,
        fill_count,
        pnl,
        notional,
        volume,
        takers,
        active_days,
        positive_days,
        drawdown,
        episodes,
        positive_episode_pnl,
        negative_episode_pnl,
        duration,
        fragmentation,
    ) = row
    fill_count = fill_count or 0
    if not fill_count:
        metrics = dict.fromkeys(METRICS)
        metrics["gross_volume"] = 0.0
        return dict(
            user=user,
            metrics=metrics,
            exclusions=["no_activity_in_lookback"],
        )
    legacy = RankingConfig(
        lookback_days=config.lookback_days,
        min_active_days=config.min_active_days,
        min_episodes=config.min_episodes,
        min_notional=config.min_notional,
        min_minutes=config.min_minutes,
    )
    episodes, active_days = episodes or 0, active_days or 0
    duration = duration or 0.0
    fragmentation = fragmentation or legacy.fragmentation_floor
    positive_episode_pnl = positive_episode_pnl or 0.0
    negative_episode_pnl = negative_episode_pnl or 0.0
    profit_factor = (
        positive_episode_pnl / negative_episode_pnl
        if negative_episode_pnl
        else legacy.profit_factor_cap
        if positive_episode_pnl
        else 0.0
    )
    copyability = (
        min(1.0, log1p(duration) / log1p(legacy.duration_cap_minutes))
        + legacy.fragmentation_floor / max(fragmentation, legacy.fragmentation_floor)
        + takers / fill_count
    ) / 3
    metrics = dict(
        pnl_efficiency=pnl / notional if notional else None,
        profit_factor=min(legacy.profit_factor_cap, profit_factor),
        positive_day_rate=(positive_days + 1) / (active_days + 2),
        drawdown_efficiency=max(
            -legacy.drawdown_cap,
            min(
                legacy.drawdown_cap,
                pnl / max(drawdown or 0.0, legacy.drawdown_floor),
            ),
        ),
        copyability=copyability,
        gross_volume=volume,
    )
    exclusions = [
        reason
        for condition, reason in (
            (active_days < config.min_active_days, "insufficient_active_days"),
            (episodes < config.min_episodes, "insufficient_episodes"),
            (notional < config.min_notional, "insufficient_notional"),
            (duration < config.min_minutes, "too_short_holding_period"),
        )
        if condition
    ]
    if config.min_volume > volume:
        exclusions.append("insufficient_gross_volume")
    if any(
        metrics[key] is None or not isfinite(metrics[key])
        for key in config.metric_weights
    ):
        exclusions.append("missing_ranking_metric")
    return dict(user=user, metrics=metrics, exclusions=exclusions)
