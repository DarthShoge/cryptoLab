from datetime import datetime, timedelta, timezone
from itertools import groupby

import pytest

from arblab.hyperliquid_copy.contracts import FillEvent
from arblab.hyperliquid_copy.episode_state import encode_episode_state
from arblab.hyperliquid_copy.episodes import leader_net_pnl
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity, ORDER
from arblab.hyperliquid_copy.proxy_compact import COLUMNS
from arblab.hyperliquid_copy.qualified_window import QualifiedWindow
from arblab.hyperliquid_copy.streaming_wallet_metrics import _episode
from .test_candidate_day import resources
from .test_qualified_day import qualified

DAY = datetime(2026, 8, 1, tzinfo=timezone.utc)
SEMANTICS = "gross_excludes_fee"


def test_contiguous_builder_resumes_under_fresh_cache_lease(qualified, tmp_path):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.feature_publication import FeatureDay

    root = tmp_path / "resume_cache"
    root.mkdir()
    with CacheLease(root) as lease:
        original = CacheResources.create(lease, "feature-resume-test")
        first = build(original, qualified)
        inputs = first.inputs
        expected = list(first.checkpoints())
        before = original.audit()
    with CacheLease(root) as lease:
        reopened = CacheResources(lease, "feature-resume-test")
        previous = FeatureDay(reopened, inputs)
        assert list(previous.checkpoints()) == expected
        assert reopened.audit() == before
        second = build(reopened, qualified, DAY + timedelta(days=1), previous)
        third = build(reopened, qualified, DAY + timedelta(days=2), second)
        assert third.inputs["previous"] == second.publication.key
        assert list(third.observations()) == []
        assert list(third.checkpoints())
        assert publication_count(reopened) == 3
        assert reopened.audit()["reserved_bytes"] == 0


def test_qualified_whale_feature_publication_has_every_fill(tmp_path, resources):
    import json
    import lz4.frame
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    _, archive, _, _ = inputs(tmp_path, days=1)
    count = 100_002
    for key, body in list(archive.bodies.items()):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        data = json.loads(lz4.frame.decompress(body))
        user, fill = data["events"][0]
        data["events"] = [
            [user, dict(fill, tid=i)]
            for i in range(hour * 4167, min((hour + 1) * 4167, count))
        ]
        archive.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    raw = download_archive(archive, "2026-08-01", "2026-08-02", tmp_path / "raw")
    normalized = import_archive(
        raw, ["BTC"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    result = build(resources, qualify([compact], tmp_path))
    seen = 0
    previous_key = None
    for observation in result.observations():
        if not hasattr(observation, "net_pnl"):
            continue
        assert observation.user == user
        assert previous_key is None or previous_key < observation.order_key
        previous_key = observation.order_key
        seen += 1
    assert seen == count
    assert publication_count(resources) == 1
    assert resources.audit()["reserved_bytes"] == 0
    assert not list((resources.root / "scratch").iterdir())


def build(resources, pin, day=DAY, previous=None):
    from arblab.hyperliquid_copy.feature_day_builder import build_feature_day

    return build_feature_day(resources, pin, day, ["BTC"], SEMANTICS, previous)


def publication_count(resources):
    with resources._connect() as db:
        return db.execute(
            "SELECT count(*) FROM publications WHERE json_extract(descriptor,'$.kind')='qualified_features_day'"
        ).fetchone()[0]


def test_three_qualified_days_match_full_reference_and_preserve_dormant_state(
    qualified, resources, tmp_path
):
    view = QualifiedWindow(qualified, DAY, DAY + timedelta(days=3))
    with ProxyActivity([e.path for e in view.entries], temp_root=tmp_path) as reference:
        fills = [
            FillEvent(**dict(zip(COLUMNS, row)))
            for row in reference.db.execute(
                f"SELECT {','.join(COLUMNS)} FROM fills WHERE exchange_time>=? AND exchange_time<? ORDER BY user,{ORDER}",
                [view.start, view.end],
            ).fetchall()
        ]
    expected_state = []
    expected_episodes = []

    class Sink:
        def add(self, kind, pnl, minutes, fragments):
            expected_episodes.append(
                (
                    self.fill.user,
                    self.fill.coin,
                    self.fill.order_key,
                    pnl,
                    minutes,
                    fragments,
                )
            )

    sink = Sink()
    for user, rows in groupby(fills, key=lambda f: f.user):
        active = {}
        for fill in rows:
            sink.fill = fill
            _episode(active, fill, leader_net_pnl(fill, SEMANTICS), SEMANTICS, sink)
        if active:
            expected_state.append(
                encode_episode_state(
                    active,
                    user=user,
                    cutoff=DAY + timedelta(days=3),
                    semantics=SEMANTICS,
                )
            )
    previous = None
    actual = []
    for offset in range(3):
        previous = build(resources, qualified, DAY + timedelta(days=offset), previous)
        actual.extend(previous.observations())
    observed = [o for o in actual if hasattr(o, "net_pnl")]
    episodes = [
        (o.user, o.coin, o.order_key, o.pnl, o.minutes, o.fragments)
        for o in actual
        if hasattr(o, "fragments")
    ]
    observed.sort(key=lambda o: (o.user, *o.order_key))
    assert [(o.user, o.coin, o.order_key, o.net_pnl) for o in observed] == [
        (f.user, f.coin, f.order_key, leader_net_pnl(f, SEMANTICS)) for f in fills
    ]
    assert sorted(episodes) == sorted(expected_episodes)
    assert list(previous.checkpoints()) == expected_state
    assert list(previous.observations()) == []  # Third source day has no events.
    assert expected_state and publication_count(resources) == 3
    assert resources.audit()["reserved_bytes"] == 0
    assert not list((resources.root / "scratch").iterdir())


def test_reuse_has_no_sort_or_new_allocation(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    first = build(resources, qualified)
    second = build(resources, qualified, DAY + timedelta(days=1), first)
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("cache reuse must not sort")

    monkeypatch.setattr(OrderedWalletPartitions, "plan", forbidden)
    assert build(resources, qualified, DAY + timedelta(days=1), first) == second
    assert resources.audit() == before


def test_missing_or_noncontiguous_previous_day_rejected(qualified, resources):
    before = resources.audit()
    with pytest.raises(ValueError):
        build(resources, qualified, DAY + timedelta(days=1))
    assert resources.audit() == before
    first = build(resources, qualified)
    for at in (DAY, DAY + timedelta(days=2)):
        with pytest.raises(ValueError):
            build(resources, qualified, at, first)


def test_source_mutation_after_output_finish_blocks_publication(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy.feature_writer import FeatureWriter

    source = QualifiedWindow(qualified, DAY, DAY + timedelta(days=1)).entries[0].path
    original = FeatureWriter.finish

    def finish(writer):
        tokens = original(writer)
        with source.open("ab") as output:
            output.write(b"changed")
        return tokens

    monkeypatch.setattr(FeatureWriter, "finish", finish)
    with pytest.raises(ValueError):
        build(resources, qualified)
    assert publication_count(resources) == 0
    assert resources.audit()["retained_bytes"] > 0


def test_interrupted_wallet_replay_cannot_publish(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import feature_day_builder as module

    def interrupted(fills, **kwargs):
        next(iter(fills))
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "derive_wallet_day", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        build(resources, qualified)
    assert publication_count(resources) == 0
    assert resources.audit()["reserved_bytes"] > 0


def test_all_working_space_reserved_before_first_source_query(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    original = OrderedWalletPartitions.plan
    calls = []

    def plan(reader, *args, **kwargs):
        calls.append(resources.audit()["reserved_bytes"])
        assert calls[-1] == 3 * 1024**3 + 256 * 1024**2
        return original(reader, *args, **kwargs)

    monkeypatch.setattr(OrderedWalletPartitions, "plan", plan)
    build(resources, qualified)
    assert len(calls) == 1


def test_previous_payload_changed_after_output_finish_blocks_next_day(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy.feature_writer import FeatureWriter

    first = build(resources, qualified)
    path = resources.root / first.publication.artifacts[-1].path
    original = FeatureWriter.finish

    def finish(writer):
        result = original(writer)
        with path.open("ab") as output:
            output.write(b"changed")
        return result

    monkeypatch.setattr(FeatureWriter, "finish", finish)
    with pytest.raises(ValueError):
        build(resources, qualified, DAY + timedelta(days=1), first)
    assert publication_count(resources) == 1


def test_missing_fill_callback_is_detected_before_publication(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import feature_day_builder as module

    original = module.derive_wallet_day

    def omit(fills, **kwargs):
        kwargs["on_fill"] = lambda row: None
        # Suppress episodes too so the earlier pairing guard does not mask the
        # independent source-vs-fill-observation count check under test.
        kwargs["on_episode"] = lambda row: None
        return original(fills, **kwargs)

    monkeypatch.setattr(module, "derive_wallet_day", omit)
    with pytest.raises(ValueError, match="count mismatch"):
        build(resources, qualified)
    assert publication_count(resources) == 0


@pytest.mark.parametrize(
    "coins,semantics",
    [(["ETH"], SEMANTICS), (["BTC"], "net_includes_fee"), (["BTC", "BTC"], SEMANTICS)],
)
def test_changed_previous_scope_or_semantics_rejected(
    qualified, resources, coins, semantics
):
    from arblab.hyperliquid_copy.feature_day_builder import build_feature_day

    first = build(resources, qualified)
    before = resources.audit()
    with pytest.raises(ValueError):
        build_feature_day(
            resources, qualified, DAY + timedelta(days=1), coins, semantics, first
        )
    assert resources.audit() == before


def test_changed_builder_engine_after_finish_blocks_publication(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import feature_day_builder as module
    from arblab.hyperliquid_copy.feature_writer import FeatureWriter

    original = FeatureWriter.finish

    def finish(writer):
        result = original(writer)
        monkeypatch.setattr(module, "feature_engine", lambda: {"changed": True})
        return result

    monkeypatch.setattr(FeatureWriter, "finish", finish)
    with pytest.raises(ValueError, match="engine"):
        build(resources, qualified)
    assert publication_count(resources) == 0


@pytest.mark.parametrize("coins", [["BTC"], ["BTC", "xyz:GOLD"]])
def test_cross_market_qualified_source_matches_selected_scope(
    tmp_path, resources, coins
):
    import json
    import lz4.frame
    from arblab.hyperliquid_copy.feature_day_builder import build_feature_day
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    _, archive, _, _ = inputs(tmp_path, days=1)
    for key, body in list(archive.bodies.items()):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        data = json.loads(lz4.frame.decompress(body))
        for event in data["events"]:
            event[1]["coin"] = "BTC" if hour % 2 else "xyz:GOLD"
        archive.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    raw = download_archive(archive, "2026-08-01", "2026-08-02", tmp_path / "raw")
    normalized = import_archive(
        raw, ["BTC", "xyz:GOLD"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    pin = qualify([compact], tmp_path)
    source = QualifiedWindow(pin, DAY, DAY + timedelta(days=1))
    with ProxyActivity(
        [e.path for e in source.entries], temp_root=tmp_path
    ) as reference:
        expected = reference.db.execute(
            f"SELECT user,coin,exchange_time,closed_pnl-fee FROM fills WHERE coin IN (SELECT unnest(?)) ORDER BY user,{ORDER}",
            [coins],
        ).fetchall()
    result = build_feature_day(resources, pin, DAY, coins, SEMANTICS)
    actual = [
        (o.user, o.coin, o.order_key[0], o.net_pnl)
        for o in result.observations()
        if hasattr(o, "net_pnl")
    ]
    assert actual == expected
    assert {row[1] for row in actual} == set(coins)
    assert all(e["coin"] in coins for p in result.checkpoints() for e in p["episodes"])
