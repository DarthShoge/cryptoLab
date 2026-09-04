"""Observed position episodes and explicitly resolved leader fee accounting."""
from dataclasses import dataclass


def leader_net_pnl(fill, semantics, *, smoke=False):
    if semantics == "unknown" and smoke:
        return fill.closed_pnl
    if semantics == "net_includes_fee":
        return fill.closed_pnl
    if semantics == "gross_excludes_fee":
        if fill.fee_token != "USDC":
            raise ValueError("unsupported fee token")
        return fill.closed_pnl - fill.fee
    raise ValueError("unresolved fee semantics")


def closing_size(fill):
    delta = fill.sz if fill.side == "B" else -fill.sz
    return min(fill.sz, abs(fill.start_position)) if delta * fill.start_position < 0 else 0.0


@dataclass
class PositionEpisode:
    user: str
    coin: str
    opened_at: object
    closed_at: object = None
    pnl: float = 0.0
    fill_count: int = 0
    taker_count: int = 0
    fees: float = 0.0
    peak_notional: float = 0.0
    left_censored: bool = False

    @property
    def complete(self):
        return self.closed_at is not None and not self.left_censored

    @property
    def minutes(self):
        return (self.closed_at - self.opened_at).total_seconds() / 60


def build_episodes(fills, semantics, *, smoke=False):
    active, episodes = {}, []
    for fill in sorted(fills, key=lambda f: f.order_key):
        key = fill.user, fill.coin
        episode = active.get(key)
        if episode is None:
            episode = PositionEpisode(*key, fill.exchange_time,
                                       left_censored=abs(fill.start_position) > 1e-9)
            active[key] = episode
        flipping = fill.start_position * fill.post_position < 0
        net = leader_net_pnl(fill, semantics, smoke=smoke)
        opening_fee = fill.fee * (fill.sz - closing_size(fill)) / fill.sz if flipping else 0.0
        episode.pnl += net + (opening_fee if flipping and semantics == "gross_excludes_fee" else 0)
        episode.fill_count += 1
        episode.taker_count += int(fill.crossed)
        episode.fees += fill.fee - opening_fee
        episode.peak_notional = max(episode.peak_notional, abs(fill.start_position) * fill.px,
                                   0 if flipping else abs(fill.post_position) * fill.px)
        if abs(fill.post_position) <= 1e-9 or flipping:
            episode.closed_at = fill.exchange_time
            episodes.append(episode)
            del active[key]
        if flipping:
            active[key] = PositionEpisode(*key, fill.exchange_time,
                pnl=-opening_fee if semantics == "gross_excludes_fee" else 0.0,
                fill_count=1, taker_count=int(fill.crossed), fees=opening_fee,
                peak_notional=abs(fill.post_position) * fill.px)
    return episodes + list(active.values())
