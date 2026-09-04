import json
from datetime import datetime, timezone

import pytest

USER = "0x" + "ab" * 20


def raw_fill(**changes):
    data = dict(coin="BTC", px="100", sz="2", side="B", time=1753574399900,
                startPosition="-1", dir="Short > Long", closedPnl="3", fee="0.09",
                feeToken="USDC", crossed=True, tid=10, hash="0xabc", oid=20)
    return data | changes


def parse(fill=None, **kwargs):
    from arblab.hyperliquid_copy.archive import parse_archive_line
    return parse_archive_line(json.dumps([USER, fill or raw_fill()]).encode(), "legacy/20250726/0", 0, **kwargs)


def test_legacy_and_block_preserve_position_and_identity():
    from arblab.hyperliquid_copy.archive import parse_archive_line
    first = parse()
    assert not first.issues
    f = first.events[0]
    assert (f.start_position, f.post_position, f.fee, f.direction) == (-1, 1, .09, "Short > Long")
    assert f.event_id == parse().events[0].event_id
    block = dict(block_number=123, block_time="2025-07-26T23:59:59.900Z", events=[[USER, raw_fill()]])
    other = parse_archive_line(json.dumps(block).encode(), "blocks/20250726/0", 7)
    assert not other.issues
    assert other.events[0].source_line == 7
    assert other.events[0].block_number == 123
    assert other.events[0].post_position == 1
    assert other.events[0].event_id != f.event_id


@pytest.mark.parametrize("change", [dict(side="X"), dict(px="nan"), dict(sz="-1"), dict(crossed="false"), dict(startPosition=None)])
def test_invalid_required_fields_are_issues(change):
    result = parse(raw_fill(**change))
    assert not result.events
    assert result.issues


def test_no_default_zero_and_maker_taker_distinct():
    missing = raw_fill()
    del missing["startPosition"]
    assert parse(missing).issues
    from arblab.hyperliquid_copy.archive import parse_archive_line
    block = {"block_number": 1, "events": [[USER, raw_fill()], ["0x" + "cd" * 20, raw_fill(side="A", startPosition="1", crossed=False)]]}
    result = parse_archive_line(json.dumps(block).encode(), "x", 0)
    assert len({f.event_id for f in result.events}) == 2
    assert [f.post_position for f in result.events] == [1, -1]


def test_archive_prefix_handoff():
    from arblab.hyperliquid_copy.archive import archive_keys
    old = archive_keys("2025-07-26")
    new = archive_keys("2025-07-27")
    assert len(old) == len(new) == 24
    assert old[0] == "node_fills/hourly/20250726/0.lz4"
    assert new[-1] == "node_fills_by_block/hourly/20250727/23.lz4"
