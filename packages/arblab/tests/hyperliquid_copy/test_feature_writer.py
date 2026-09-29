# ruff: noqa: F811

from dataclasses import replace
from datetime import timedelta

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.feature_records import (
    OBSERVATION_SCHEMA,
    CHECKPOINT_SCHEMA,
)
from arblab.hyperliquid_copy.episode_state import encode_episode_state
from .test_candidate_day import resources  # noqa: F401,F811
from .test_feature_records import example
from .test_wallet_day_features import BASE, DAY, SEMANTICS


def writer(resources, **kwargs):
    from arblab.hyperliquid_copy.feature_writer import FeatureWriter

    return FeatureWriter(resources, day=DAY, semantics=SEMANTICS, **kwargs)


def files(resources, tokens):
    with resources._connect() as db:
        return [
            resources.root
            / db.execute("SELECT path FROM allocations WHERE token=?", (t,)).fetchone()[
                0
            ]
            for t in tokens
        ]


def test_empty_outputs_reserved_together_then_settled_without_publication(resources):
    with writer(resources) as output:
        assert resources.audit()["reserved_bytes"] == 256 * 1024**2
        tokens = output.finish()
    paths = files(resources, tokens)
    assert len(paths) == 2
    assert [pq.read_schema(p) for p in paths] == [OBSERVATION_SCHEMA, CHECKPOINT_SCHEMA]
    assert all(pq.read_metadata(p).num_rows == 0 for p in paths)
    assert resources.audit()["reserved_bytes"] == 0

    assert resources.audit()["retained_bytes"] == sum(p.stat().st_size for p in paths)
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0


def test_optional_day_budget_is_shared_across_both_streams(resources):
    maximum = 2 * 1024**2
    with writer(resources, max_shard_bytes=1024**2, max_day_bytes=maximum) as output:
        assert resources.audit()["reserved_bytes"] == maximum
        tokens = output.finish()

    assert sum(path.stat().st_size for path in files(resources, tokens)) <= maximum


def test_rollover_reserves_only_uncommitted_day_allowance(resources):
    with writer(
        resources,
        max_shard_bytes=1024**2,
        max_day_bytes=2 * 1024**2,
        shard_rows=1,
        buffer_rows=1,
    ) as output:
        output.add_observation(example("episode"))
        output.add_observation(example())
        output.finish()

    with resources._connect() as db:
        maxima = [row[0] for row in db.execute("SELECT maximum FROM allocations")]
    assert len(maxima) == 3
    assert sum(maxima) < 3 * 1024**2


def test_observations_can_use_unused_checkpoint_reservation(resources):
    # Small shards reproduce a busy day whose observations exceed half the day
    # budget while its checkpoint is small. Footers count toward the same cap.
    from arblab.hyperliquid_copy.feature_records import decode_observation

    expected = [
        replace(
            example(), order_key=(DAY + timedelta(seconds=i), *example().order_key[1:])
        )
        for i in range(8)
    ]
    with writer(
        resources,
        max_shard_bytes=32_000,
        max_day_bytes=64_000,
        shard_rows=1,
        buffer_rows=1,
    ) as output:
        for row in expected:
            output.add_observation(row)
        tokens = output.finish()
    paths = files(resources, tokens)
    assert sum(p.stat().st_size for p in paths[:-1]) > 32_000
    assert sum(p.stat().st_size for p in paths) <= 64_000
    assert [
        decode_observation(row)
        for p in paths[:-1]
        for row in pq.read_table(p).to_pylist()
    ] == expected
    assert resources.audit()["reserved_bytes"] == 0
    with writer(
        resources, max_shard_bytes=32_000, shard_rows=1, buffer_rows=1
    ) as baseline:
        for row in expected:
            baseline.add_observation(row)
        baseline_tokens = baseline.finish()
    assert [p.read_bytes() for p in paths] == [
        p.read_bytes() for p in files(resources, baseline_tokens)
    ]


def test_rebalancing_still_enforces_actual_day_limit(resources):
    with pytest.raises(ValueError, match="byte limit"):
        with writer(
            resources,
            max_shard_bytes=32_000,
            max_day_bytes=64_000,
            shard_rows=1,
            buffer_rows=1,
        ) as output:
            for i in range(30):
                output.add_observation(
                    replace(
                        example(),
                        order_key=(
                            DAY + timedelta(seconds=i),
                            *example().order_key[1:],
                        ),
                    )
                )
            output.finish()
    assert resources.audit()["reserved_bytes"] > 0
    assert (
        sum(p.stat().st_size for p in (resources.root / "artifacts").iterdir())
        <= 64_000
    )


def test_checkpoint_can_reclaim_capacity_after_observation_borrows(resources):
    import hashlib

    users = sorted(
        "0x" + hashlib.sha256(str(i).encode()).hexdigest()[:40] for i in range(8)
    )
    with writer(
        resources,
        max_shard_bytes=32_000,
        max_day_bytes=64_000,
        shard_rows=1,
        buffer_rows=1,
        max_artifacts=5000,
    ) as output:
        for i in range(8):
            output.add_observation(
                replace(
                    example(),
                    order_key=(DAY + timedelta(seconds=i), *example().order_key[1:]),
                )
            )
        # Force the observation footer to borrow, then leave its new shard open.
        output.add_observation(
            replace(
                example(),
                order_key=(DAY + timedelta(seconds=8), *example().order_key[1:]),
            )
        )
        for user in users:
            output.add_checkpoint(
                encode_episode_state(
                    {}, user=user, cutoff=DAY + timedelta(days=1), semantics=SEMANTICS
                ),
                user=user,
            )
        tokens = output.finish()
    checkpoints = [
        p for p in files(resources, tokens) if pq.read_schema(p) == CHECKPOINT_SCHEMA
    ]
    assert [
        row["user"] for p in checkpoints for row in pq.read_table(p).to_pylist()
    ] == users


def test_aggregate_day_budget_rejects_impossible_initial_pair(resources):
    with pytest.raises(ValueError, match="aggregate"):
        writer(
            resources,
            max_shard_bytes=1024**2,
            max_day_bytes=2 * 1024**2 - 1,
        )
    assert resources.audit()["reserved_bytes"] == 0


def test_episode_pair_survives_batch_and_shard_boundary(resources):
    with writer(resources, shard_rows=1, buffer_rows=1) as output:
        output.add_observation(example("episode"))
        output.add_observation(example("fill"))
        state = encode_episode_state(
            {}, user=BASE.user, cutoff=DAY + timedelta(days=1), semantics=SEMANTICS
        )
        output.add_checkpoint(state, user=BASE.user)
        tokens = output.finish()
    paths = files(resources, tokens)
    assert len(paths) == 3
    assert [pq.read_table(p).to_pylist()[0].get("kind") for p in paths] == [
        "episode",
        "fill",
        None,
    ]
    assert resources.audit()["reserved_bytes"] == 0


@pytest.mark.parametrize(
    "case",
    ["duplicate_fill", "duplicate_episode", "reversed", "coin", "dangling", "next_key"],
)
def test_invalid_pairing_never_finishes(resources, case):
    fill, episode = example(), example("episode")
    rows = {
        "duplicate_fill": [fill, fill],
        "duplicate_episode": [episode, episode],
        "reversed": [fill, episode],
        "coin": [episode, replace(fill, coin="ETH")],
        "dangling": [episode],
        "next_key": [
            episode,
            replace(fill, order_key=(DAY + timedelta(seconds=1), *fill.order_key[1:])),
        ],
    }[case]
    with pytest.raises(ValueError):
        with writer(resources, buffer_rows=1, shard_rows=1) as output:
            for row in rows:
                output.add_observation(row)
            output.finish()
    assert resources.audit()["reserved_bytes"] > 0


def test_footer_byte_exhaustion_leaves_charged_output(resources):
    with pytest.raises(ValueError, match="byte limit"):
        with writer(resources, max_shard_bytes=10) as output:
            output.finish()
    assert resources.audit()["reserved_bytes"] == 20


def test_combined_artifact_limit_rejects_before_new_reservation(resources):
    with pytest.raises(ValueError, match="artifact"):
        with writer(resources, shard_rows=1, buffer_rows=1, max_artifacts=2) as output:
            output.add_observation(example("episode"))
            output.add_observation(example())
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM allocations").fetchone()[0] == 2


def test_caller_failure_does_not_settle_or_publish(resources):
    with pytest.raises(RuntimeError, match="source"):
        with writer(resources, buffer_rows=1) as output:
            output.add_observation(example())
            raise RuntimeError("source")
    assert resources.audit()["reserved_bytes"] == 256 * 1024**2


def test_checkpoint_order_and_context_rejected(resources):
    state = encode_episode_state(
        {}, user=BASE.user, cutoff=DAY + timedelta(days=1), semantics=SEMANTICS
    )
    with pytest.raises(ValueError):
        with writer(resources) as output:
            output.add_checkpoint(state, user=BASE.user)
            output.add_checkpoint(state, user=BASE.user)


def test_invalid_bounds_do_not_allocate(resources):
    for kwargs in (
        {"buffer_rows": 257},
        {"shard_rows": 250001},
        {"max_artifacts": 1},
        {"buffer_bytes": 0},
        {"max_shard_bytes": True},
    ):
        with pytest.raises(ValueError):
            writer(resources, **kwargs)
    assert resources.audit()["reserved_bytes"] == 0


def test_caught_validation_error_still_prevents_finalization(resources):
    with writer(resources) as output:
        output.add_observation(example())
        with pytest.raises(ValueError):
            output.add_observation(example())
        with pytest.raises(ValueError, match="not open"):
            output.finish()
    assert resources.audit()["reserved_bytes"] == 256 * 1024**2


def test_payload_replacement_is_rejected_before_settling(resources, tmp_path):
    with writer(resources, buffer_rows=1) as output:
        output.add_observation(example())
        path = next((resources.root / "artifacts").iterdir())
        path.rename(tmp_path / "moved-owned-output.parquet")
        path.touch()
        with pytest.raises(ValueError, match="identity"):
            output.finish()
    assert resources.audit()["reserved_bytes"] == 256 * 1024**2


def test_buffer_byte_bound_rejects_single_oversized_row(resources):
    with pytest.raises(ValueError, match="byte limit"):
        with writer(resources, buffer_bytes=10) as output:
            output.add_observation(example())
    assert resources.audit()["reserved_bytes"] == 256 * 1024**2


def test_buffer_rows_and_shard_rows_remain_bounded(resources):
    with writer(resources, buffer_rows=2, shard_rows=3) as output:
        for i in range(10):
            row = example()
            output.add_observation(
                replace(row, order_key=(DAY + timedelta(seconds=i), *row.order_key[1:]))
            )
        tokens = output.finish()
    paths = files(resources, tokens)
    assert [pq.read_metadata(p).num_rows for p in paths] == [3, 3, 3, 1, 0]
    assert all(
        pq.read_metadata(p).row_group(i).num_rows <= 2
        for p in paths
        for i in range(pq.read_metadata(p).num_row_groups)
    )


def test_shared_budget_rejects_two_outputs_before_any_payload_write(resources):
    resources.reserve(
        "scratch/" + "a" * 32, resources.limit - 32 * 1024**2 - 128 * 1024**2, "scratch"
    )
    with pytest.raises(ValueError, match="budget"):
        with writer(resources):
            pytest.fail("combined reservation must fail")
    assert not list((resources.root / "artifacts").iterdir())
    assert resources.audit()["total_bytes"] == resources.limit


def test_descriptor_remains_open_through_settlement(resources, monkeypatch):
    original = resources.settle
    with writer(resources) as output:

        def settle(token):
            stream = next(s for s in output.streams if s.token == token)
            assert stream.handle is not None and not stream.handle.closed
            return original(token)

        monkeypatch.setattr(resources, "settle", settle)
        output.finish()


@pytest.mark.parametrize("when", ["before", "after"])
def test_replacement_during_settlement_rejects_final_result(
    resources, monkeypatch, tmp_path, when
):
    original = resources.settle
    with writer(resources) as output:
        changed = False

        def settle(token):
            nonlocal changed
            stream = next(s for s in output.streams if s.token == token)
            if changed:
                return original(token)
            changed = True
            if when == "after":
                original(token)
            stream.path.rename(tmp_path / "replaced-final-output.parquet")
            stream.path.touch()
            if when == "before":
                original(token)

        monkeypatch.setattr(resources, "settle", settle)
        with pytest.raises(ValueError, match="identity"):
            output.finish()
        assert all(not stream.tokens for stream in output.streams)
