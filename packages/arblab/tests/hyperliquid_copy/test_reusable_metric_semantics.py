"""Characterize reference constraints before adding reusable metric derivations."""

from dataclasses import replace
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.episodes import build_episodes
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from arblab.hyperliquid_copy.ranking import RankingConfig, wallet_metrics
from .test_proxy_activity import partition
from .test_ranking import history


def test_global_episode_filter_loses_valid_window_local_episode(tmp_path):
    base = history(1)[0]
    start = base.exchange_time + timedelta(days=1)
    end = start + timedelta(hours=2)
    rows = [
        replace(base, tid=0, oid=0, event_id="before", fee=0.0),
        replace(base, tid=1, oid=1, event_id="reset", fee=0.0, exchange_time=start),
        replace(
            base,
            tid=2,
            oid=2,
            event_id="close",
            fee=0.0,
            exchange_time=start + timedelta(hours=1),
            side="A",
            start_position=1.0,
            post_position=0.0,
            direction="Close Long",
            closed_pnl=2.0,
        ),
    ]
    # Proxy validation does not assert continuity between separate native fills.
    # Do not silently introduce that stronger condition to justify episode reuse.
    with ProxyActivity([partition(tmp_path, rows)], temp_root=tmp_path) as activity:
        activity.validate_registered_scope(["BTC"], base.exchange_time, end)
        assert activity.count == 3
    global_episodes = build_episodes(rows, "gross_excludes_fee")
    naive = [
        e
        for e in global_episodes
        if e.complete and start <= e.opened_at and e.closed_at < end
    ]
    local = [e for e in build_episodes(rows[1:], "gross_excludes_fee") if e.complete]
    assert naive == []
    assert len(local) == 1
    assert local[0].pnl == 2.0 and local[0].fill_count == 2
    assert local[0].minutes == 60 and not local[0].left_censored


def test_daily_notional_regrouping_can_change_reference_eligibility():
    base = history(1)[0]
    values = [1e12, 0.0001, 0.0001, 0.0001]
    rows = [
        replace(
            base,
            px=value,
            fee=0.0,
            sz=1.0,
            side="A",
            start_position=1.0,
            post_position=0.0,
            direction="Close Long",
            tid=i,
            oid=i,
            event_id=str(i),
            exchange_time=base.exchange_time + timedelta(days=i // 2, minutes=i),
        )
        for i, value in enumerate(values)
    ]
    direct = sum(values)
    grouped = sum([sum(values[:2]), sum(values[2:])])
    threshold = max(direct, grouped)
    config = RankingConfig(
        min_active_days=0, min_episodes=0, min_minutes=0, min_notional=threshold
    )
    _, exclusions = wallet_metrics(rows, config, "gross_excludes_fee", False)
    assert direct < threshold <= grouped
    assert "insufficient_notional" in exclusions


def test_daily_pnl_totals_are_not_reference_fill_order_total():
    values = [1e8, 1e-6, 1e-6, -1e8]
    direct = sum(values)
    day_totals = sum([sum(values[:2]), sum(values[2:])])
    assert direct == 2e-6
    assert direct != day_totals


@pytest.mark.parametrize("semantics", ["gross_excludes_fee", "net_includes_fee"])
@pytest.mark.parametrize("offset", [0, 1, 3, 11, 27])
@pytest.mark.parametrize("late_close", [False, True])
def test_first_boundary_replay_then_global_suffix_matches_window_episodes(
    tmp_path, semantics, offset, late_close
):
    """Characterization only: no persistent cache or production reuse is added.

    Completion ordinals, not timestamps alone, distinguish multiple closures at
    one timestamp. Prefix replay ends independently for each wallet/asset.
    """
    base = history(1)[0]
    # Each row is internally valid, but consecutive native positions may reset.
    # Include long/short closes and both flip directions, in interleaved markets.
    shapes = [
        ("B", 0, 1, 1),
        ("A", 1, 0, 1),
        ("A", 1, -1, 2),
        ("B", -1, 1, 2),
        ("B", -1, 0, 1),
        ("A", 0, -1, 1),
    ]
    rows = []
    for i in range(1200):
        side, start_position, post_position, size = shapes[
            (i // 2 + i % 2) % len(shapes)
        ]
        if late_close and i % 2 == 0:
            # ETH's only closure is after every tested window. Global cached
            # knowledge of that future closure must not create a local episode.
            side, start_position, post_position, size = (
                ("A", 1, 0, 1) if i == 1198 else ("B", 0, 1, 1)
            )
        rows.append(
            replace(
                base,
                coin="BTC" if i % 2 else "ETH",
                side=side,
                start_position=float(start_position),
                post_position=float(post_position),
                sz=float(size),
                fee=0.01 * (i % 3 + 1),
                closed_pnl=float(i % 7 - 3),
                exchange_time=base.exchange_time + timedelta(minutes=20 * (i // 4)),
                tid=i,
                oid=i,
                event_id=f"event-{i:04d}",
                source_line=i,
                event_index=i,
            )
        )

    begin = base.exchange_time + timedelta(minutes=20 * offset)
    boundary, cached_suffix, _, _ = check_reuse(
        rows, begin, begin + timedelta(days=1), semantics, tmp_path
    )
    assert cached_suffix
    if offset == 1:
        assert any(
            rows[row[0]].exchange_time == rows[boundary[row[1]]].exchange_time
            for row in cached_suffix
        ), "Timestamp-only suffix filtering would lose episodes"


def check_reuse(rows, begin, end, semantics, tmp_path):
    """Test-only splice model; one already ordered wallet, not a cache reader."""
    from collections import defaultdict
    from arblab.hyperliquid_copy.episodes import leader_net_pnl, closing_size
    from arblab.hyperliquid_copy.streaming_wallet_metrics import (
        _episode,
        _finish,
        stream_wallet_metrics,
    )
    from arblab.hyperliquid_copy.wallet_metric_spool import MetricSpool

    assert len({fill.user for fill in rows}) <= 1

    class Observations:
        def __init__(self):
            self.rows = []

        def add(self, kind, pnl, minutes, fragments):
            assert kind == "episode"
            self.rows.append((self.ordinal, self.coin, pnl, minutes, fragments))

    global_spool, global_active, numeric = Observations(), {}, []
    for index, fill in enumerate(rows):
        global_spool.ordinal, global_spool.coin = index, fill.coin
        net = leader_net_pnl(fill, semantics)
        numeric.append((net, closing_size(fill) * fill.px, fill.sz * fill.px))
        _episode(global_active, fill, net, semantics, global_spool)
    window = [
        (i, fill) for i, fill in enumerate(rows) if begin <= fill.exchange_time < end
    ]
    boundary, local_active, prefix_spool = {}, {}, Observations()
    for index, fill in window:
        if fill.coin in boundary:
            continue
        prefix_spool.ordinal, prefix_spool.coin = index, fill.coin
        _episode(
            local_active, fill, leader_net_pnl(fill, semantics), semantics, prefix_spool
        )
        if (
            abs(fill.post_position) <= 1e-9
            or fill.start_position * fill.post_position < 0
        ):
            boundary[fill.coin] = index
    cached_suffix = [
        row
        for row in global_spool.rows
        if row[1] in boundary
        and row[0] > boundary[row[1]]
        and begin <= rows[row[0]].exchange_time < end
    ]
    actual = [
        (coin, pnl, minutes, fragments)
        for _, coin, pnl, minutes, fragments in sorted(
            prefix_spool.rows + cached_suffix
        )
    ]
    expected = [
        (episode.coin, episode.pnl, episode.minutes, episode.fill_count)
        for episode in build_episodes([fill for _, fill in window], semantics)
        if episode.complete
    ]
    assert actual == expected
    # Lossless per-fill numbers remain ordered; never sum daily subtotals to
    # compute the all-window totals. Episode observations can use the splice.
    config = RankingConfig(
        lookback_days=1,
        min_active_days=0,
        min_episodes=0,
        min_notional=0,
        min_minutes=0,
    )
    daily, takers = defaultdict(float), 0
    with MetricSpool(tmp_path) as spool:
        for index, fill in window:
            net, notional, volume = numeric[index]
            spool.add("fill", net, notional, volume)
            daily[fill.exchange_time.date()] += net
            takers += int(fill.crossed)
        for _, pnl, minutes, fragments in actual:
            spool.add("episode", pnl, minutes, fragments)
        reused = _finish(spool, daily, takers, config)
    reference = stream_wallet_metrics(
        [fill for _, fill in window], end, config, semantics, temp_root=tmp_path
    )
    for field in (
        "metrics",
        "exclusions",
        "fill_count",
        "closing_notional",
        "gross_volume",
        "complete_episodes",
        "active_days",
    ):
        assert getattr(reused, field) == getattr(reference, field), field
    return boundary, cached_suffix, actual, reused


@pytest.mark.parametrize("epsilon", [0.0, 1e-10, 1e-9, 2e-9])
@pytest.mark.parametrize("semantics", ["gross_excludes_fee", "net_includes_fee"])
def test_splice_near_zero_cancellation_and_exact_cutoff(tmp_path, epsilon, semantics):
    base = history(1)[0]
    begin = base.exchange_time + timedelta(days=1)
    end = begin + timedelta(days=1)
    specs = [
        (-1, 0.0, 1.0, 0.0),
        (0, epsilon, 1.0, 0.0),
        (1, 1.0, epsilon, 0.0),
        (2, 0.0, 1.0, 1e16),
        (3, 1.0, 2.0, 1.0),
        (4, 2.0, 0.0, -1e16),
        (5, 0.0, 1.0, 0.0),
        (24, 1.0, 0.0, 100.0),
    ]
    rows = [
        replace(
            base,
            exchange_time=begin + timedelta(hours=hour),
            side="B" if post > start else "A",
            sz=abs(post - start),
            start_position=start,
            post_position=post,
            closed_pnl=pnl,
            fee=0.0,
            tid=i,
            oid=i,
            source_line=i,
            event_index=i,
            event_id=str(i),
        )
        for i, (hour, start, post, pnl) in enumerate(specs)
    ]
    boundary, suffix, episodes, result = check_reuse(
        rows, begin, end, semantics, tmp_path
    )
    assert boundary[base.coin] == (2 if epsilon <= 1e-9 else 5)
    assert result.fill_count == 6  # Neither the pre-window fill nor cutoff closure.
    assert result.complete_episodes == (2 if epsilon <= 1e-9 else 0)
    assert all(pnl == 0.0 for _, pnl, _, _ in episodes)
    if epsilon <= 1e-9:
        # Episode accumulation is incremental +=, unlike compensated fill sum.
        assert len(suffix) == 1 and suffix[0][2] == 0.0
        assert sum([1e16, 1.0, -1e16]) == 1.0


def test_splice_empty_window_does_not_reuse_past_episodes(tmp_path):
    rows = history(1)
    begin = max(fill.exchange_time for fill in rows) + timedelta(days=2)
    boundary, suffix, episodes, result = check_reuse(
        rows, begin, begin + timedelta(days=1), "gross_excludes_fee", tmp_path
    )
    assert boundary == {} and suffix == [] and episodes == []
    assert result.fill_count == result.complete_episodes == 0
    assert result.exclusions == ("no_activity_in_lookback",)


def test_splice_state_and_observations_are_isolated_per_wallet(tmp_path):
    base = history(1)[0]
    begin, end = base.exchange_time, base.exchange_time + timedelta(days=1)
    all_rows = []
    for user, pnl in [("0x" + "1" * 40, 3.0), ("0x" + "2" * 40, -5.0)]:
        rows = [
            replace(base, user=user, fee=0.0, closed_pnl=0.0),
            replace(
                base,
                user=user,
                fee=0.0,
                closed_pnl=pnl,
                side="A",
                start_position=1.0,
                post_position=0.0,
                exchange_time=begin + timedelta(hours=1),
                tid=2,
                oid=2,
                event_id="close",
                source_line=2,
            ),
        ]
        all_rows.extend(rows)
        _, _, episodes, result = check_reuse(
            rows, begin, end, "gross_excludes_fee", tmp_path
        )
        assert len(episodes) == result.complete_episodes == 1
        assert episodes[0][1] == pnl
    completed = [
        e for e in build_episodes(all_rows, "gross_excludes_fee") if e.complete
    ]
    assert {(e.user, e.pnl) for e in completed} == {
        ("0x" + "1" * 40, 3.0),
        ("0x" + "2" * 40, -5.0),
    }


def test_reference_active_peak_can_be_a_lossless_large_integer():
    from arblab.hyperliquid_copy.episodes import leader_net_pnl
    from arblab.hyperliquid_copy.streaming_wallet_metrics import _episode

    class NoCompletedEpisode:
        def add(self, *args):
            pytest.fail("left-censored first flip cannot complete an episode")

    fill = replace(
        history(1)[0],
        px=2**53 + 1,
        start_position=-1,
        post_position=1,
        sz=2,
        side="B",
        fee=0.0,
    )
    active = {}
    _episode(
        active,
        fill,
        leader_net_pnl(fill, "gross_excludes_fee"),
        "gross_excludes_fee",
        NoCompletedEpisode(),
    )
    peak = active[fill.coin].peak_notional
    assert type(peak) is int and peak == 2**53 + 1
    assert int(float(peak)) != peak
