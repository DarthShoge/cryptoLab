from types import SimpleNamespace

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_pipeline_proxy import _signals_at
from .test_lab_pipeline_proxy import config, dataset
from .test_candidate_day import resources
from .test_qualified_day import qualified


def positions(activity, users):
    from arblab.hyperliquid_copy.follower_positions import selected_positions

    return selected_positions(activity, users, "BTC", day("2026-08-03"))


def test_batch_capability_uses_one_query_and_preserves_order():
    calls = []

    def batch(users, coin, at):
        calls.append((users, coin, at))
        return {"b": None, "a": -2.0}

    def single(*args):
        pytest.fail("batch-capable reader used scalar fallback")

    result = positions(SimpleNamespace(positions=batch, position=single), ["a", "b"])
    assert list(result) == ["a", "b"] and result == {"a": -2.0, "b": None}
    assert calls == [(["a", "b"], "BTC", day("2026-08-03"))]


def test_legacy_scalar_path_retains_known_zero_short_and_unknown():
    expected = {"a": 0.0, "b": -5.0, "c": None}
    calls = []

    def single(user, coin, at):
        calls.append(user)
        return expected[user]

    assert positions(SimpleNamespace(position=single), list(expected)) == expected
    assert calls == list(expected)


def test_empty_selection_never_queries_reader():
    assert positions(object(), []) == {}


@pytest.mark.parametrize("users", [["a", "a"], list(map(str, range(251)))])
def test_invalid_cohort_rejects_before_query(users):
    with pytest.raises(ValueError, match="selected"):
        positions(object(), users)


@pytest.mark.parametrize(
    "returned",
    [
        {},
        {"a": 1, "extra": 2},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": True},
        {"a": "1"},
        [],
    ],
)
def test_bad_batch_cannot_be_interpreted_as_unknown_or_flat(returned):
    reader = SimpleNamespace(positions=lambda *args: returned)
    with pytest.raises(ValueError):
        positions(reader, ["a"])


def test_batch_failure_does_not_retry_as_scalar():
    def batch(*args):
        raise OSError("native query failed")

    def single(*args):
        pytest.fail("failed batch was retried")

    with pytest.raises(OSError, match="native query failed"):
        positions(SimpleNamespace(positions=batch, position=single), ["a"])


def test_actual_signal_path_uses_batch_and_preserves_contributions(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import ProxySelectionState

    data, c = dataset(tmp_path), config()
    at = day(c.start)
    original = data.activity
    try:
        state = ProxySelectionState(data, c)
        state.advance(at)
        expected = _signals_at(data, state, c, at, None)
        calls = []

        def batch(users, coin, decision):
            calls.append((list(users), coin, decision))
            return {u: original.position(u, coin, decision) for u in users}

        def single(*args):
            pytest.fail("signal path ignored batched positions")

        data.activity = SimpleNamespace(positions=batch, position=single)
        assert _signals_at(data, state, c, at, None) == expected
        assert calls == [([r["user"] for r in state.selected["BTC"]], "BTC", at)]
    finally:
        original.close()


@pytest.mark.parametrize("conviction_enabled", [False, True])
def test_qualified_native_batch_matches_reference_signals(
    qualified, resources, tmp_path, conviction_enabled
):
    import json
    from pathlib import Path
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions
    from .test_archive import USER

    users = [USER, "0x" + "cd" * 20, "0x" + "ef" * 20]
    at = day("2026-08-03")
    c = config(
        follower=dict(
            aggregation="conviction_trimmed"
            if conviction_enabled
            else "direction_equal",
            update_minutes=60,
            min_known=1,
            trim=0,
        )
    )
    rows = [
        dict(user=u, score=3 - i, weight=1 / 3, decision_time=at)
        for i, u in enumerate(users)
    ]
    state = SimpleNamespace(
        ever={"BTC", "xyz:TSLA"},
        active=["BTC"],
        selected={"BTC": rows},
        budgets={"BTC": 1},
        market_time=at,
    )
    values = dict(zip(users, [0.4, -0.2, None]))
    conviction = (
        SimpleNamespace(value=lambda key, at: values[key[0]])
        if conviction_enabled
        else None
    )
    report = json.loads(Path(qualified["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        expected = _signals_at(SimpleNamespace(activity=ref), state, c, at, conviction)
    calls = []

    def batch(selected, coin, decision):
        calls.append((selected, coin, decision))
        return native_positions(resources, qualified, decision, coin, selected)

    actual = _signals_at(
        SimpleNamespace(activity=SimpleNamespace(positions=batch)),
        state,
        c,
        at,
        conviction,
    )
    assert actual == expected
    assert calls == [(users, "BTC", at)]
    assert resources.audit()["reserved_bytes"] == 0
