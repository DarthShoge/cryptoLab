from datetime import timedelta

import pytest

from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS, publication_count
from .test_qualified_day import qualified


def history(resources, qualified):
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    return FeatureHistory(resources, qualified, ["BTC"], SEMANTICS)


def test_first_request_builds_from_origin_but_returns_only_lookback(
    resources, qualified
):
    controller = history(resources, qualified)
    result = controller.window(DAY + timedelta(days=2), DAY + timedelta(days=3))
    assert publication_count(resources) == 3
    assert len(result.days) == 1
    assert result.days[0].day == DAY + timedelta(days=2)
    assert result.days[0].inputs["previous"] is not None
    assert list(result.days[0].observations()) == []
    assert list(result.days[0].checkpoints())  # Dormant open episodes survive.
    result.verify()
    assert resources.audit()["reserved_bytes"] == 0


def test_advancement_builds_only_missing_days_and_reuses_same_decision(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history as module

    controller = history(resources, qualified)
    original = module.build_feature_day
    calls = []

    def build(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_feature_day", build)
    first = controller.window(DAY, DAY + timedelta(days=1))
    second = controller.window(DAY, DAY + timedelta(days=2))
    before = resources.audit()
    shorter = controller.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    assert calls == [DAY, DAY + timedelta(days=1)]
    assert first.days == second.days[:1]
    assert shorter.days == second.days[1:]
    assert resources.audit() == before


def test_intraday_window_reopens_without_sort_under_new_lease(
    tmp_path, qualified, monkeypatch
):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    root = tmp_path / "history_cache"
    root.mkdir()
    start, end = DAY + timedelta(hours=3), DAY + timedelta(days=1, hours=4)
    with CacheLease(root) as lease:
        original = CacheResources.create(lease, "history-reopen")
        window = history(original, qualified).window(start, end)
        expected = window.inputs()
        before = original.audit()
        assert window.start == start and window.end == end
        assert len(window.days) == 2
    with CacheLease(root) as lease:
        reopened = CacheResources(lease, "history-reopen")

        def forbidden(*args, **kwargs):
            pytest.fail("reopened history must not sort published days")

        monkeypatch.setattr(OrderedWalletPartitions, "plan", forbidden)
        result = history(reopened, qualified).window(start, end)
        assert result.inputs() == expected
        assert reopened.audit() == before


@pytest.mark.parametrize(
    "start,end",
    [
        (DAY - timedelta(hours=1), DAY + timedelta(days=1)),
        (DAY, DAY + timedelta(days=4)),
        (DAY + timedelta(days=1), DAY + timedelta(days=1)),
        (DAY + timedelta(days=2), DAY + timedelta(days=1)),
        (DAY, DAY + timedelta(days=733, hours=1)),
    ],
)
def test_invalid_interval_does_not_build_or_advance(resources, qualified, start, end):
    controller = history(resources, qualified)
    before = resources.audit()
    with pytest.raises(ValueError):
        controller.window(start, end)
    assert resources.audit() == before
    assert controller._days == () and controller._last_end is None


@pytest.mark.parametrize(
    "coins,semantics",
    [
        ([], SEMANTICS),
        (["ETH"], SEMANTICS),
        (["BTC", "BTC"], SEMANTICS),
        (["BTC"], "unknown"),
        ("BTC", SEMANTICS),
    ],
)
def test_invalid_context_rejected_at_construction(
    resources, qualified, coins, semantics
):
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    before = resources.audit()
    with pytest.raises(ValueError):
        FeatureHistory(resources, qualified, coins, semantics)
    assert resources.audit() == before


def test_reversed_decision_rejected_without_changing_state(resources, qualified):
    controller = history(resources, qualified)
    controller.window(DAY, DAY + timedelta(days=2))
    before, days = resources.audit(), controller._days
    with pytest.raises(ValueError, match="reversed"):
        controller.window(DAY, DAY + timedelta(days=1))
    assert resources.audit() == before
    assert controller._days == days and controller._last_end == DAY + timedelta(days=2)


@pytest.mark.parametrize("target", ["pin", "coins", "engine"])
def test_context_mutation_after_builder_refuses_cursor_commit(
    resources, qualified, monkeypatch, target
):
    from arblab.hyperliquid_copy import feature_history as module

    coins = ["BTC"]
    controller = module.FeatureHistory(resources, qualified, coins, SEMANTICS)
    original = module.build_feature_day

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "pin":
            qualified["sha256"] = "0" * 64
        elif target == "coins":
            coins.append("ETH")
        else:
            monkeypatch.setattr(
                module, "_engine", lambda: {"changed": True}, raising=False
            )
        return result

    monkeypatch.setattr(module, "build_feature_day", changed)
    with pytest.raises(ValueError, match="context"):
        controller.window(DAY, DAY + timedelta(days=1))
    assert controller._days == () and controller._last_end is None
    assert publication_count(resources) == 1  # Completed output is not deleted.


def test_failure_preserves_published_day_and_explicit_retry_reuses_it(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history as module
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    controller = history(resources, qualified)
    original = module.build_feature_day

    def interrupted(*args, **kwargs):
        if args[2] == DAY + timedelta(days=1):
            raise RuntimeError("interrupted before next day")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_feature_day", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        controller.window(DAY, DAY + timedelta(days=2))
    assert controller._days == () and controller._last_end is None
    assert publication_count(resources) == 1
    monkeypatch.setattr(module, "build_feature_day", original)
    plan = OrderedWalletPartitions.plan

    def checked(reader, *args, **kwargs):
        assert reader.window.start != DAY, "published first day must not sort again"
        return plan(reader, *args, **kwargs)

    monkeypatch.setattr(OrderedWalletPartitions, "plan", checked)
    result = controller.window(DAY, DAY + timedelta(days=2))
    assert len(result.days) == 2 and publication_count(resources) == 2


def test_reentrant_request_rejected_then_busy_state_cleared(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history as module

    controller = history(resources, qualified)
    original = module.build_feature_day

    def nested(*args, **kwargs):
        controller.window(DAY, DAY + timedelta(days=1))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_feature_day", nested)
    with pytest.raises(ValueError, match="busy"):
        controller.window(DAY, DAY + timedelta(days=1))
    assert publication_count(resources) == 0
    monkeypatch.setattr(module, "build_feature_day", original)
    assert len(controller.window(DAY, DAY + timedelta(days=1)).days) == 1


def test_context_mutation_during_final_window_check_refuses_commit(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy.feature_window import FeatureWindow

    controller = history(resources, qualified)
    original = FeatureWindow.verify

    def changed(window):
        original(window)
        controller.semantics = "net_includes_fee"

    monkeypatch.setattr(FeatureWindow, "verify", changed)
    with pytest.raises(ValueError, match="context"):
        controller.window(DAY, DAY + timedelta(days=1))
    assert controller._days == () and controller._last_end is None
    assert publication_count(resources) == 1


def test_expired_lease_rejects_even_cached_window(tmp_path, qualified):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

    root = tmp_path / "expired_history"
    root.mkdir()
    with CacheLease(root) as lease:
        resources = CacheResources.create(lease, "expired-history")
        controller = history(resources, qualified)
        controller.window(DAY, DAY + timedelta(days=1))
    with pytest.raises(ValueError):
        controller.window(DAY, DAY + timedelta(days=1))


@pytest.mark.parametrize("target", ["pin", "coins", "resources"])
def test_caller_mutation_during_final_report_hash_is_rejected(
    resources, qualified, monkeypatch, target
):
    from pathlib import Path
    from arblab.hyperliquid_copy import feature_history as module
    from arblab.hyperliquid_copy.feature_window import FeatureWindow

    coins = ["BTC"]
    controller = module.FeatureHistory(resources, qualified, coins, SEMANTICS)
    original, verify = module.file_hash, FeatureWindow.verify
    final_check = False

    def completed(window):
        nonlocal final_check
        verify(window)
        final_check = True

    def changed(path):
        digest = original(path)
        if final_check and Path(path) == Path(qualified["path"]):
            if target == "pin":
                qualified["sha256"] = "0" * 64
            elif target == "coins":
                coins.append("ETH")
            else:
                controller.resources = object()
        return digest

    monkeypatch.setattr(FeatureWindow, "verify", completed)
    monkeypatch.setattr(module, "file_hash", changed)
    with pytest.raises(ValueError, match="context"):
        controller.window(DAY, DAY + timedelta(days=1))
    assert controller._days == () and controller._last_end is None
