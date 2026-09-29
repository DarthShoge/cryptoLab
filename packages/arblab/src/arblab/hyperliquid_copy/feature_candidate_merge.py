"""Complete candidate/feature merge; rows remain unpublished staging evidence.

The row projection deliberately matches the preserved raw metric producer.
Caller owns iterator closure, source qualification and shared scratch accounting.
"""

from itertools import groupby
from math import isfinite
import re

from .feature_wallet_metrics import feature_wallet_metrics
from .lab_config import METRICS
from .ranking import RankingConfig


def merge_feature_metric_rows(
    candidates, observations, decision, config, semantics, *, temp_root
):
    legacy = RankingConfig(
        lookback_days=config.lookback_days,
        min_active_days=config.min_active_days,
        min_episodes=config.min_episodes,
        min_notional=config.min_notional,
        min_minutes=config.min_minutes,
    )
    groups = groupby(observations, key=lambda row: row.user)
    current, previous = next(groups, None), None
    for user in candidates:
        if (
            type(user) is not str
            or not re.fullmatch(r"0x[0-9a-f]{40}", user)
            or previous is not None
            and user <= previous
        ):
            raise ValueError("Invalid or unordered candidate identity")
        previous = user
        if current is not None and current[0] < user:
            raise ValueError("Feature wallet absent from complete candidate index")
        active = current is not None and current[0] == user
        rows = current[1] if active else iter(())
        result = feature_wallet_metrics(
            rows, decision, legacy, semantics, temp_root=temp_root
        )
        if next(rows, None) is not None:
            raise ValueError("Incomplete wallet feature consumption")
        metrics = {key: result.metrics.get(key) for key in METRICS}
        if not result.closing_notional:
            metrics["pnl_efficiency"] = None
        metrics["gross_volume"] = result.gross_volume
        reasons = list(result.exclusions)
        if result.gross_volume < config.min_volume:
            reasons.append("insufficient_gross_volume")
        if any(
            metrics[k] is None or not isfinite(metrics[k])
            for k in config.metric_weights
        ):
            reasons.append("missing_ranking_metric")
        yield dict(user=user, metrics=metrics, exclusions=reasons)
        if active:
            current = next(groups, None)
    if current is not None:
        raise ValueError("Feature wallet absent from complete candidate index")
