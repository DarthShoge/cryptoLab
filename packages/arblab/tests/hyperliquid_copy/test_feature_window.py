from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.qualified_window import QualifiedWindow
from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS, build
from .test_qualified_day import qualified


@pytest.fixture
def days(qualified, resources):
    previous, result = None, []
    for offset in range(3):
        previous = build(resources, qualified, DAY + timedelta(days=offset), previous)
        result.append(previous)
    return result


def test_complete_and_intraday_windows_match_exact_daily_source(
    qualified, resources, days
):
    from arblab.hyperliquid_copy.feature_window import FeatureWindow

    before = resources.audit()
    for start, end, selected in (
        (DAY, DAY + timedelta(days=3), days),
        (DAY + timedelta(hours=1), DAY + timedelta(days=1, hours=1), days[:2]),
        (DAY + timedelta(days=2), DAY + timedelta(days=3), days[2:]),
    ):
        window = FeatureWindow(
            resources, qualified, selected, start, end, ["BTC"], SEMANTICS
        )
        assert window.start == start and window.end == end
        assert window.days == tuple(selected)
        for day in selected:
            expected = QualifiedWindow(qualified, day.day, day.cutoff).inputs()
            assert window._day_source(day.day) == expected == day.inputs["source"]
        window.verify()
        inputs = window.inputs()
        inputs["coins"].append("ETH")
        assert window.inputs()["coins"] == ["BTC"]
        with pytest.raises(FrozenInstanceError):
            window.start = DAY
    assert resources.audit() == before


def test_invalid_chain_scope_and_fee_rejected_without_allocation(
    qualified, resources, days
):
    from arblab.hyperliquid_copy.feature_window import FeatureWindow

    before = resources.audit()
    for selected, coins, semantics in (
        (days[:2], ["BTC"], SEMANTICS),
        (list(reversed(days)), ["BTC"], SEMANTICS),
        ([days[0], days[0], days[2]], ["BTC"], SEMANTICS),
        (days, ["ETH"], SEMANTICS),
        (days, ["BTC"], "net_includes_fee"),
    ):
        with pytest.raises(ValueError):
            FeatureWindow(
                resources,
                qualified,
                selected,
                DAY,
                DAY + timedelta(days=3),
                coins,
                semantics,
            )
    assert resources.audit() == before


@pytest.mark.parametrize("target", ["feature", "source"])
def test_changed_window_payload_cannot_verify(qualified, resources, days, target):
    from arblab.hyperliquid_copy.feature_window import FeatureWindow

    window = FeatureWindow(
        resources, qualified, days, DAY, DAY + timedelta(days=3), ["BTC"], SEMANTICS
    )
    path = (
        resources.root / days[0].publication.artifacts[0].path
        if target == "feature"
        else window.source.entries[0].path
    )
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        window.verify()


@pytest.mark.parametrize("target", ["qualification", "feature"])
def test_engine_mutation_after_last_day_check_rejected(
    qualified, resources, days, monkeypatch, target
):
    from arblab.hyperliquid_copy.feature_window import FeatureWindow
    from arblab.hyperliquid_copy import feature_publication, qualified_window

    window = FeatureWindow(
        resources, qualified, days, DAY, DAY + timedelta(days=3), ["BTC"], SEMANTICS
    )
    original = feature_publication.FeatureDay.verify

    def verify(day):
        original(day)
        if day is days[-1]:
            if target == "qualification":
                monkeypatch.setattr(
                    qualified_window, "_engine", lambda: {"changed": True}
                )
            else:
                monkeypatch.setattr(
                    feature_publication, "feature_engine", lambda: {"changed": True}
                )

    monkeypatch.setattr(feature_publication.FeatureDay, "verify", verify)
    with pytest.raises(ValueError):
        window.verify()


@pytest.mark.parametrize("target", ["source", "earlier_feature"])
def test_mutation_during_final_day_verification_rejected(
    qualified, resources, days, monkeypatch, target
):
    from arblab.hyperliquid_copy.feature_window import FeatureWindow
    from arblab.hyperliquid_copy.feature_publication import FeatureDay

    window = FeatureWindow(
        resources, qualified, days, DAY, DAY + timedelta(days=3), ["BTC"], SEMANTICS
    )
    path = (
        window.source.entries[0].path
        if target == "source"
        else resources.root / days[0].publication.artifacts[0].path
    )
    original = FeatureDay.verify

    def verify(day):
        original(day)
        if day is days[-1]:
            with path.open("ab") as handle:
                handle.write(b"changed after earlier verification")

    monkeypatch.setattr(FeatureDay, "verify", verify)
    with pytest.raises(ValueError):
        window.verify()
