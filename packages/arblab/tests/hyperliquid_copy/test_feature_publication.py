from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from arblab.hyperliquid_copy.feature_records import (
    OBSERVATION_SCHEMA,
    CHECKPOINT_SCHEMA,
    encode_observation,
)
from arblab.hyperliquid_copy.feature_writer import FeatureWriter
from arblab.hyperliquid_copy.qualified_window import QualifiedWindow
from arblab.hyperliquid_copy.episode_state import encode_episode_state
from arblab.hyperliquid_copy.episodes import PositionEpisode
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_feature_records import example

DAY = datetime(2026, 8, 1, tzinfo=timezone.utc)


def context(pin):
    from arblab.hyperliquid_copy.feature_publication import feature_engine

    source = QualifiedWindow(pin, DAY, DAY + timedelta(days=1))
    return dict(
        schema=1,
        source=source.inputs(),
        origin="2026-08-01",
        day="2026-08-01",
        coins=["BTC"],
        semantics="gross_excludes_fee",
        previous=None,
        engine=feature_engine(),
    )


def opened(resources, inputs):
    from arblab.hyperliquid_copy.feature_publication import FeatureDay

    return FeatureDay(resources, inputs)


def event(kind):
    row = example(kind)
    return replace(row, order_key=(DAY, *row.order_key[1:]))


def publish(resources, inputs, rows=()):
    with FeatureWriter(
        resources, day=DAY, semantics=inputs["semantics"], buffer_rows=1, shard_rows=1
    ) as writer:
        for row in rows:
            writer.add_observation(row)
        tokens = writer.finish()
    return PublishedArtifacts(resources).publish(
        "qualified_features_day", inputs, tokens
    )


def test_reopen_pair_across_shards_and_empty_state(qualified, resources):
    inputs = context(qualified)
    rows = [event("episode"), event("fill")]
    publication = publish(resources, inputs, rows)
    before = resources.audit()
    day = opened(resources, inputs)
    assert day.publication == publication
    assert list(day.observations()) == rows
    assert list(day.checkpoints()) == []
    inputs["coins"].append("ETH")
    assert day.inputs["coins"] == ["BTC"]
    assert resources.audit() == before


def test_empty_publication_remains_explicit(qualified, resources):
    inputs = context(qualified)
    publish(resources, inputs)
    day = opened(resources, inputs)
    assert list(day.observations()) == list(day.checkpoints()) == []


def test_unpublished_or_corrupt_artifacts_rejected(qualified, resources):
    inputs = context(qualified)
    with pytest.raises(ValueError):
        opened(resources, inputs)
    publication = publish(resources, inputs)
    path = resources.root / publication.artifacts[0].path
    with path.open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError):
        opened(resources, inputs)


@pytest.mark.parametrize(
    "kind", ["duplicate", "reversed", "dangling", "coin", "cutoff"]
)
def test_reader_rejects_bad_cross_shard_order_or_cutoff(qualified, resources, kind):
    inputs = context(qualified)
    fill, episode = event("fill"), event("episode")
    rows = {
        "duplicate": [fill, fill],
        "reversed": [fill, episode],
        "dangling": [episode],
        "coin": [episode, replace(fill, coin="ETH")],
        "cutoff": [
            replace(fill, order_key=(DAY + timedelta(days=1), *fill.order_key[1:]))
        ],
    }[kind]
    tokens = []
    for row in rows:
        relative = f"artifacts/{uuid.uuid4().hex}.parquet"
        token = resources.reserve(relative, 128 * 1024**2, "payload")
        pq.write_table(
            pa.Table.from_pylist([encode_observation(row)], schema=OBSERVATION_SCHEMA),
            resources.root / relative,
        )
        resources.settle(token)
        tokens.append(token)
    relative = f"artifacts/{uuid.uuid4().hex}.parquet"
    token = resources.reserve(relative, 128 * 1024**2, "payload")
    pq.write_table(
        pa.Table.from_pylist([], schema=CHECKPOINT_SCHEMA), resources.root / relative
    )
    resources.settle(token)
    tokens.append(token)
    PublishedArtifacts(resources).publish("qualified_features_day", inputs, tokens)
    with pytest.raises(ValueError):
        list(opened(resources, inputs).observations())


def test_engine_changes_invalidate_existing_handle(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import feature_publication as module

    inputs = context(qualified)
    publish(resources, inputs)
    day = opened(resources, inputs)
    monkeypatch.setattr(module, "feature_engine", lambda: {"changed": True})
    with pytest.raises(ValueError, match="engine"):
        list(day.observations())


def test_changed_file_is_detected_when_iterator_closed_early(qualified, resources):
    inputs = context(qualified)
    publication = publish(resources, inputs, [event("episode"), event("fill")])
    iterator = opened(resources, inputs).observations()
    next(iterator)
    with (resources.root / publication.artifacts[-1].path).open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError):
        iterator.close()


def test_verified_handle_cannot_drop_shards_or_rebind_context(qualified, resources):
    inputs = context(qualified)
    publish(resources, inputs)
    day = opened(resources, inputs)
    for field, value in (
        ("_observations", ()),
        ("_checkpoints", ()),
        ("day", DAY + timedelta(days=1)),
        ("_encoded", b"{}"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(day, field, value)


@pytest.mark.parametrize("iterator_end", [False, True])
def test_engine_change_during_final_lookup_is_rejected(
    qualified, resources, monkeypatch, iterator_end
):
    from arblab.hyperliquid_copy import feature_publication as module

    inputs = context(qualified)
    publish(resources, inputs, [event("fill")])
    day = opened(resources, inputs)
    original = PublishedArtifacts.lookup
    stable = module.feature_engine()
    armed = changed = False

    def lookup(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if armed:
            changed = True
        return result

    monkeypatch.setattr(PublishedArtifacts, "lookup", lookup)
    monkeypatch.setattr(
        module, "feature_engine", lambda: {"changed": True} if changed else stable
    )
    if iterator_end:
        iterator = day.observations()
        next(iterator)
        armed = True
        with pytest.raises(ValueError, match="engine"):
            next(iterator)
    else:
        armed = True
        with pytest.raises(ValueError, match="engine"):
            day.verify()


def test_nonempty_checkpoint_shards_preserve_dormant_wallet_state(qualified, resources):
    inputs = context(qualified)
    expected = []
    with FeatureWriter(
        resources, day=DAY, semantics=inputs["semantics"], shard_rows=1
    ) as writer:
        for digit in ("0", "f"):
            user = "0x" + digit * 40
            state = {
                "BTC": PositionEpisode(
                    user,
                    "BTC",
                    DAY - timedelta(days=400),
                    pnl=-0.0,
                    fill_count=1,
                    peak_notional=2**53 + 1,
                )
            }
            payload = encode_episode_state(
                state,
                user=user,
                cutoff=DAY + timedelta(days=1),
                semantics=inputs["semantics"],
            )
            expected.append(payload)
            writer.add_checkpoint(payload, user=user)
        tokens = writer.finish()
    PublishedArtifacts(resources).publish("qualified_features_day", inputs, tokens)
    actual = list(opened(resources, inputs).checkpoints())
    assert actual == expected
    for payload in actual:
        episode = payload["episodes"][0]
        assert episode["pnl"].hex() == (-0.0).hex()
        assert (
            type(episode["peak_notional"]) is int
            and episode["peak_notional"] == 2**53 + 1
        )


def test_reversed_checkpoint_shards_are_rejected(qualified, resources):
    inputs = context(qualified)
    with FeatureWriter(
        resources, day=DAY, semantics=inputs["semantics"], shard_rows=1
    ) as writer:
        for digit in ("0", "f"):
            user = "0x" + digit * 40
            payload = encode_episode_state(
                {},
                user=user,
                cutoff=DAY + timedelta(days=1),
                semantics=inputs["semantics"],
            )
            writer.add_checkpoint(payload, user=user)
        tokens = writer.finish()
    assert len(tokens) == 3
    PublishedArtifacts(resources).publish(
        "qualified_features_day", inputs, [tokens[0], tokens[2], tokens[1]]
    )
    with pytest.raises(ValueError, match="order"):
        list(opened(resources, inputs).checkpoints())
