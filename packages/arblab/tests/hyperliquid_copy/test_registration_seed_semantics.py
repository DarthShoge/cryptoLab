"""Characterize why an interior-only dataset cannot replace source seed history."""

from dataclasses import replace
from datetime import timedelta

from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from arblab.hyperliquid_copy.registration_partitions import publish_interior
from .test_proxy_activity import fills, partition
from .test_lab_ranking import settings


def test_interior_projection_loses_dormant_candidate_and_known_position(tmp_path):
    opening = fills()[0]
    begin = opening.exchange_time.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    decision = begin + timedelta(days=1)
    other = replace(
        opening,
        user="0x" + "f" * 40,
        tid=999,
        oid=999,
        event_id="interior-wallet",
        exchange_time=begin + timedelta(hours=1),
    )
    path = partition(tmp_path, [opening, other])
    stage = tmp_path / "interior"
    stage.mkdir()
    entries = publish_interior(
        [
            dict(
                path=str(path),
                bytes=path.stat().st_size,
                sha256=file_hash(path),
                rows=2,
            )
        ],
        begin,
        decision + timedelta(days=1),
        stage,
    )
    config = settings(lookback_days=1)
    with ProxyActivity([path], temp_root=tmp_path) as source:
        source.validate_registered_scope(
            {"BTC"}, begin - timedelta(days=1), decision + timedelta(days=1)
        )
        source_rows = source.rank(decision, config, "BTC", "gross_excludes_fee")
        assert source.position(opening.user, "BTC", decision) == opening.post_position
    with ProxyActivity(
        [stage / row["name"] for row in entries], temp_root=tmp_path
    ) as interior:
        interior.validate_registered_scope({"BTC"}, begin, decision + timedelta(days=1))
        interior_rows = interior.rank(decision, config, "BTC", "gross_excludes_fee")
        assert interior.position(opening.user, "BTC", decision) is None
    assert {row["user"] for row in source_rows} == {opening.user, other.user}
    assert {row["user"] for row in interior_rows} == {other.user}
    dormant = next(row for row in source_rows if row["user"] == opening.user)
    assert not dormant["eligible"]
    assert "no_activity_in_lookback" in dormant["exclusions"]
    # Interior rows themselves remain identical: this is missing seed context,
    # not corruption or permission to extend the metric lookback.
    assert (
        next(row for row in source_rows if row["user"] == other.user)
        == interior_rows[0]
    )
