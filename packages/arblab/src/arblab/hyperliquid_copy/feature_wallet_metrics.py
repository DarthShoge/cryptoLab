"""Exact arithmetic over a caller-qualified, single-wallet feature window.

The caller owns source/fee identity, strict range selection and accounted scratch.
This reducer is not a source qualification certificate or a cache range reader.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from .contracts import utc
from .feature_order import ObservationOrder
from .feature_records import encode_observation
from .ranking import RankingConfig
from .streaming_wallet_metrics import _episode, _finish
from .wallet_day_features import EpisodeObservation
from .wallet_metric_spool import MetricSpool


@dataclass(frozen=True)
class _ReplayFill:
    observation: object

    @property
    def exchange_time(self):
        return self.observation.order_key[0]

    def __getattr__(self, name):
        return getattr(self.observation, name)


def feature_wallet_metrics(
    observations, decision, config, semantics, *, temp_root, buffer_rows=4096
):
    if (
        not isinstance(config, RankingConfig)
        or type(config.lookback_days) is not int
        or not 1 <= config.lookback_days <= 732
    ):
        raise ValueError("Expected RankingConfig with bounded lookback")
    if semantics not in ("gross_excludes_fee", "net_includes_fee"):
        raise ValueError("Expected resolved feature fee semantics")
    decision = utc(decision)
    start = decision - timedelta(days=config.lookback_days)
    order = ObservationOrder()
    active, daily = {}, defaultdict(float)
    coins, synchronized = set(), set()
    user, pending, takers = None, None, 0
    with MetricSpool(temp_root, buffer_rows=buffer_rows) as spool:
        for observation in observations:
            order.add(encode_observation(observation))
            if user is None:
                user = observation.user
            at = observation.order_key[0]
            if observation.user != user or not start <= at < decision:
                raise ValueError(
                    "Expected one wallet and strictly causal feature window"
                )
            coins.add(observation.coin)
            if len(coins) > 50:
                raise ValueError("Wallet feature market limit exceeded")
            if isinstance(observation, EpisodeObservation):
                pending = observation
                continue
            spool.add(
                "fill",
                observation.net_pnl,
                observation.closed_notional,
                observation.gross_volume,
            )
            daily[at.date()] += observation.net_pnl
            takers += int(observation.crossed)
            if observation.coin not in synchronized:
                _episode(
                    active,
                    _ReplayFill(observation),
                    observation.net_pnl,
                    semantics,
                    spool,
                )
                if (
                    abs(observation.post_position) <= 1e-9
                    or observation.start_position * observation.post_position < 0
                ):
                    synchronized.add(observation.coin)
                    active.pop(observation.coin, None)
            elif pending is not None:
                spool.add("episode", pending.pnl, pending.minutes, pending.fragments)
            pending = None
        order.finish()
        return _finish(spool, daily, takers, config)
