"""Streaming arithmetic core; emitted rows are unpublished staging evidence.

The caller owns qualification, native deduplication, resource accounting and
atomic publication. Exhaustion here is not proof of complete source coverage.
"""

from dataclasses import dataclass
from datetime import timedelta

from .contracts import FillEvent, symbol
from .episode_state import (
    _number,
    _timestamp,
    decode_episode_state,
    encode_episode_state,
)
from .episodes import closing_size, leader_net_pnl
from .streaming_wallet_metrics import _episode


@dataclass(frozen=True)
class FillObservation:
    user: str
    coin: str
    order_key: tuple
    px: int | float
    sz: int | float
    side: str
    start_position: int | float
    post_position: int | float
    fee: int | float
    crossed: bool
    net_pnl: int | float
    closed_notional: int | float
    gross_volume: int | float


@dataclass(frozen=True)
class EpisodeObservation:
    user: str
    coin: str
    order_key: tuple
    pnl: int | float
    minutes: float
    fragments: int


@dataclass(frozen=True)
class DayFeatureResult:
    checkpoint: dict
    fill_count: int
    episode_count: int


class _EpisodeSink:
    def __init__(self, callback):
        self.callback, self.count, self.fill = callback, 0, None

    def add(self, kind, pnl, minutes, fragments):
        if kind != "episode":
            raise ValueError("Expected completed episode observation")
        for number in (pnl, minutes, fragments):
            _number(number)
        self.callback(
            EpisodeObservation(
                self.fill.user,
                self.fill.coin,
                self.fill.order_key,
                pnl,
                minutes,
                fragments,
            )
        )
        self.count += 1


def derive_wallet_day(fills, *, user, day, semantics, checkpoint, on_fill, on_episode):
    """Emit each row synchronously; return next state only after full exhaustion."""
    day = _timestamp(day)
    active = decode_episode_state(
        checkpoint, user=user, cutoff=day, semantics=semantics
    )
    try:
        end = day + timedelta(days=1)
    except OverflowError as exc:
        raise ValueError("Feature day outside timestamp range") from exc
    if not callable(on_fill) or not callable(on_episode):
        raise ValueError("Expected synchronous observation sinks")
    coins, previous, count = set(active), None, 0
    sink = _EpisodeSink(on_episode)
    for fill in fills:
        if not isinstance(fill, FillEvent) or fill.user != user:
            raise ValueError("Expected qualified single-wallet fills")
        at = _timestamp(fill.exchange_time)
        key = fill.order_key
        if not day <= at < end or previous is not None and key <= previous:
            raise ValueError("Expected strictly ordered fills within feature day")
        coin = symbol(fill.coin)
        coins.add(coin)
        if len(coins) > 50:
            raise ValueError("Wallet feature market limit exceeded")
        net = leader_net_pnl(fill, semantics, smoke=False)
        notional, volume = closing_size(fill) * fill.px, fill.sz * fill.px
        for number in (
            fill.px,
            fill.sz,
            fill.start_position,
            fill.post_position,
            fill.fee,
            net,
            notional,
            volume,
        ):
            _number(number)
        observation = FillObservation(
            user,
            coin,
            key,
            fill.px,
            fill.sz,
            fill.side,
            fill.start_position,
            fill.post_position,
            fill.fee,
            fill.crossed,
            net,
            notional,
            volume,
        )
        sink.fill = fill
        _episode(active, fill, net, semantics, sink)
        on_fill(observation)
        previous, count = key, count + 1
    return DayFeatureResult(
        encode_episode_state(active, user=user, cutoff=end, semantics=semantics),
        count,
        sink.count,
    )
