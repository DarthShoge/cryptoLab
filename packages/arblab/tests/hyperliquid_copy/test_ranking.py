from dataclasses import replace
from datetime import timedelta

import pytest

from .test_archive import parse, raw_fill


def history(n=6):
    opening = parse(raw_fill(startPosition="0", sz="1", closedPnl="0", fee="0.01")).events[0]
    rows = []
    for i in range(n):
        user = "0x" + f"{i:040x}"
        rows.extend([replace(opening, user=user, event_id=f"{i}:0", direction="Open Long"),
                     replace(opening, user=user, event_id=f"{i}:1", side="A", start_position=1,
                             post_position=0, px=101+i, closed_pnl=1+i, direction="Close Long",
                             exchange_time=opening.exchange_time + timedelta(hours=1))])
    return rows


def test_rank_cutoff_ties_eligibility_and_determinism():
    from arblab.hyperliquid_copy.ranking import RankingConfig, rank_traders
    fills = history()
    decision = max(f.exchange_time for f in fills) + timedelta(seconds=1)
    config = RankingConfig(min_active_days=1, min_episodes=1, min_notional=0, min_minutes=0)
    rows = rank_traders(fills, decision, config, "gross_excludes_fee")
    assert rows == rank_traders(list(reversed(fills)), decision, config, "gross_excludes_fee")
    assert len([r for r in rows if r.selected]) == 5
    assert rows[0].user == "0x" + f"{5:040x}"
    assert all(r.score is not None and 0 <= r.score <= 1 for r in rows)
    at_close = rank_traders(fills, decision-timedelta(seconds=1), config, "gross_excludes_fee")
    assert all("insufficient_episodes" in r.exclusions for r in at_close)
    one = rank_traders(history(1), decision, config, "gross_excludes_fee")
    assert one[0].score == .5 and not one[0].selected
    for field, value, reason in [("min_active_days", 3, "insufficient_active_days"),
                                  ("min_notional", 1e6, "insufficient_notional"),
                                  ("min_minutes", 100, "too_short_holding_period")]:
        assert all(reason in r.exclusions for r in rank_traders(fills, decision, replace(config, **{field:value}), "gross_excludes_fee"))


def test_percentiles_average_ties():
    from arblab.hyperliquid_copy.ranking import percentiles
    assert percentiles([0, 0, 10]) == [.25, .25, 1]


def test_scope_filter_before_core_validation():
    import json
    from arblab.hyperliquid_copy.archive import parse_archive_line
    from .test_archive import USER
    result = parse_archive_line(json.dumps([USER, raw_fill(coin="@107")]).encode(), "key", 0, coins={"BTC"})
    assert not result.events and not result.issues
