from datetime import timedelta

import pytest

from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS, publication_count
from .test_feature_history import history
from .test_qualified_day import qualified


def test_resume_advances_only_missing_days_and_preserves_original_origin(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history as module
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.feature_publication import FeatureDay
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=2)).days
    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    expected_checkpoint = list(days[-1].checkpoints())
    anchor_inputs = anchor.inputs
    resources.lease.__exit__(None)
    original_verify, original_build = FeatureDay.verify, module.build_feature_day
    calls = []

    def verify(day):
        assert day.day != DAY, "must not revisit pre-boundary day"
        return original_verify(day)

    def build(*args, **kwargs):
        calls.append(args[2])
        assert args[2] == DAY + timedelta(days=2)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(FeatureDay, "verify", verify)
    monkeypatch.setattr(module, "build_feature_day", build)
    with CacheLease(resources.root) as lease:
        current = CacheResources(lease, resources.identity)
        controller = module.FeatureHistory(
            current, qualified, ["BTC"], SEMANTICS, anchor_inputs=anchor_inputs
        )
        assert controller.origin == DAY
        result = controller.window(DAY + timedelta(days=1), DAY + timedelta(days=3))
        assert calls == [DAY + timedelta(days=2)]
        assert result.days[-1].inputs["previous"] == result.days[0].publication.key
        assert list(result.days[0].checkpoints()) == expected_checkpoint
        assert list(result.days[-1].checkpoints())
        assert list(result.days[-1].observations()) == []
        assert publication_count(current) == 3
        assert current.audit()["reserved_bytes"] == 0


def make_anchor(resources, qualified, count=3, tail=True):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=count)).days
    return publish_feature_anchor(
        resources, qualified, days[1:] if tail else days, ["BTC"], SEMANTICS
    ).inputs


def resumed(resources, qualified, inputs):
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    return FeatureHistory(
        resources, qualified, ["BTC"], SEMANTICS, anchor_inputs=inputs
    )


def test_intraday_query_excludes_later_cached_days_and_reuses(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history as module

    controller = resumed(resources, qualified, make_anchor(resources, qualified))
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("cached interval must not rebuild")

    monkeypatch.setattr(module, "build_feature_day", forbidden)
    start, end = DAY + timedelta(days=1, hours=3), DAY + timedelta(days=1, hours=4)
    result = controller.window(start, end)
    assert result.start == start and result.end == end
    assert len(result.days) == 1 and result.days[0].day == DAY + timedelta(days=1)
    assert controller.window(start, end).inputs() == result.inputs()
    assert resources.audit() == before


def test_before_retained_boundary_and_reversed_decisions_reject_without_work(
    resources, qualified
):
    controller = resumed(resources, qualified, make_anchor(resources, qualified))
    before, days = resources.audit(), controller._days
    with pytest.raises(ValueError, match="retained"):
        controller.window(DAY, DAY + timedelta(days=2))
    assert controller._last_end is None and controller._days == days
    controller.window(DAY + timedelta(days=1), DAY + timedelta(days=3))
    with pytest.raises(ValueError, match="reversed"):
        controller.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    assert resources.audit() == before


@pytest.mark.parametrize(
    "fault", ["schema_type", "coins", "empty", "source", "semantics"]
)
def test_changed_or_incompatible_anchor_is_rejected(resources, qualified, fault):
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    inputs = make_anchor(resources, qualified)
    before = resources.audit()
    if fault == "semantics":
        with pytest.raises(ValueError, match="anchor"):
            FeatureHistory(
                resources, qualified, ["BTC"], "net_includes_fee", anchor_inputs=inputs
            )
    elif fault in ("schema_type", "coins"):
        controller = resumed(resources, qualified, inputs)
        if fault == "schema_type":
            inputs["schema"] = True
        else:
            inputs["coins"].append("ETH")
        with pytest.raises(ValueError, match="context"):
            controller.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    else:
        if fault == "empty":
            inputs = {}
        else:
            inputs["report_sha256"] = "0" * 64
        with pytest.raises(ValueError):
            resumed(resources, qualified, inputs)
    assert resources.audit() == before


@pytest.mark.parametrize("target", ["caller", "engine"])
def test_constructor_detects_changes_during_anchor_verification(
    resources, qualified, monkeypatch, target
):
    from arblab.hyperliquid_copy import feature_history as module
    from arblab.hyperliquid_copy.feature_resume_anchor import FeatureAnchor

    inputs = make_anchor(resources, qualified)
    original = FeatureAnchor.verify

    def changed(anchor):
        original(anchor)
        if target == "caller":
            inputs["coins"].append("ETH")
        else:
            monkeypatch.setattr(module, "_engine", lambda: dict(changed=True))

    monkeypatch.setattr(FeatureAnchor, "verify", changed)
    with pytest.raises(ValueError, match="context|engine"):
        resumed(resources, qualified, inputs)


def test_anchor_mutation_during_final_window_check_does_not_commit(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy.feature_window import FeatureWindow

    inputs = make_anchor(resources, qualified)
    controller = resumed(resources, qualified, inputs)
    before = controller._days
    original = FeatureWindow.verify

    def changed(window):
        original(window)
        inputs["coins"].append("ETH")

    monkeypatch.setattr(FeatureWindow, "verify", changed)
    with pytest.raises(ValueError, match="context"):
        controller.window(DAY + timedelta(days=1), DAY + timedelta(days=2))
    assert controller._last_end is None and controller._days == before


def test_missing_day_failure_preserves_anchor_and_allows_explicit_retry(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history as module

    controller = resumed(
        resources, qualified, make_anchor(resources, qualified, count=2)
    )
    before, original = resources.audit(), module.build_feature_day

    def interrupted(*args, **kwargs):
        raise RuntimeError("before missing day")

    monkeypatch.setattr(module, "build_feature_day", interrupted)
    with pytest.raises(RuntimeError, match="before missing"):
        controller.window(DAY + timedelta(days=1), DAY + timedelta(days=3))
    assert len(controller._days) == 1 and controller._last_end is None
    assert resources.audit() == before
    monkeypatch.setattr(module, "build_feature_day", original)
    assert (
        len(controller.window(DAY + timedelta(days=1), DAY + timedelta(days=3)).days)
        == 2
    )


def test_future_retained_descriptors_are_included_in_limit(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history as module

    controller = resumed(
        resources, qualified, make_anchor(resources, qualified, tail=False)
    )
    before = resources.audit()
    monkeypatch.setattr(module, "MAX_DAYS", 2)
    with pytest.raises(ValueError, match="Retained.*limit"):
        controller.window(DAY, DAY + timedelta(days=1))
    assert resources.audit() == before


def test_expired_lease_rejects_cached_anchor_window(resources, qualified):
    controller = resumed(resources, qualified, make_anchor(resources, qualified))
    resources.lease.__exit__(None)
    with pytest.raises(ValueError, match="lease"):
        controller.window(DAY + timedelta(days=1), DAY + timedelta(days=2))


def test_anchor_substitution_during_initial_source_hash_is_rejected(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import feature_history as module

    requested = make_anchor(resources, qualified)
    other = make_anchor(resources, qualified, tail=False)
    original = module._previous

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        requested.clear()
        requested.update(other)
        return result

    monkeypatch.setattr(module, "_previous", changed)
    with pytest.raises(ValueError, match="context"):
        resumed(resources, qualified, requested)
