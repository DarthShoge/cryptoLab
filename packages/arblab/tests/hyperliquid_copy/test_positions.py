from datetime import timedelta

import pytest

from .test_archive import parse, raw_fill


def test_replay_cutoff_and_unknown():
    from arblab.hyperliquid_copy.positions import PositionReplay
    a = parse().events[0]
    b = parse(raw_fill(time=1753574400900, startPosition="1", sz="1", side="A")).events[0]
    replay = PositionReplay([b, a])
    assert replay.snapshot(a.exchange_time - timedelta(microseconds=1)) == {}
    assert replay.snapshot(a.exchange_time)[(a.user, a.coin)] == 1
    assert replay.snapshot(b.exchange_time)[(a.user, a.coin)] == 0
    assert replay.snapshot(b.exchange_time, inclusive=False)[(a.user, a.coin)] == 1


def test_episodes_flip_and_censoring_and_fee_semantics():
    from arblab.hyperliquid_copy.episodes import build_episodes, leader_net_pnl
    start = parse(raw_fill(startPosition="0", sz="1")).events[0]
    flip = parse(raw_fill(time=1753574459900, startPosition="1", sz="2", side="A", dir="Long > Short")).events[0]
    close = parse(raw_fill(time=1753574519900, startPosition="-1", sz="1", dir="Close Short")).events[0]
    result = build_episodes([close, start, flip], "gross_excludes_fee")
    assert len(result) == 2
    assert all(e.complete for e in result)
    assert sum(e.pnl for e in result) == pytest.approx(sum(f.closed_pnl - f.fee for f in [start, flip, close]))
    censored = build_episodes([flip, close], "gross_excludes_fee")
    assert censored[0].left_censored
    with pytest.raises(ValueError, match="unresolved"):
        leader_net_pnl(start, "unknown")
    assert leader_net_pnl(start, "unknown", smoke=True) == start.closed_pnl
