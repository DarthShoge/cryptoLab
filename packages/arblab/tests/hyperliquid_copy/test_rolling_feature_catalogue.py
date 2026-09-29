# ruff: noqa: F401,F811

from datetime import timedelta

import pytest

from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS
from .test_feature_history import history
from .test_qualified_day import qualified


@pytest.mark.parametrize("hit", [False, True])
def test_final_selection_callback_cannot_change_catalogue(resources, qualified, hit):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy.rolling_feature_catalogue import select_anchor

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=2)).days
    anchor = publish_feature_anchor(resources, qualified, days, ["BTC"], SEMANTICS)
    args = (
        resources,
        qualified,
        ["BTC"],
        SEMANTICS,
        DAY,
        DAY + timedelta(days=2 if hit else 1),
    )
    calls = 0

    def count():
        nonlocal calls
        calls += 1

    result = select_anchor(*args, validation=count)
    assert (result is not None) == hit
    final_call, calls = calls, 0
    changed = False

    def validation():
        nonlocal calls, changed
        calls += 1
        if calls == final_call:
            with resources._connect() as db:
                db.execute(
                    "DELETE FROM publications WHERE key = ?", (anchor.publication.key,)
                )
                db.commit()
            changed = True

    with pytest.raises(ValueError, match="catalogue|changed"):
        select_anchor(*args, validation=validation)
    assert changed


def test_selection_uses_latest_compatible_anchor_without_allocating(
    resources, qualified
):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy.rolling_feature_catalogue import select_anchor

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days
    publish_feature_anchor(resources, qualified, days[:2], ["BTC"], SEMANTICS)
    selected = publish_feature_anchor(
        resources, qualified, days[1:], ["BTC"], SEMANTICS
    )
    before = resources.audit()
    result = select_anchor(
        resources,
        qualified,
        ["BTC"],
        SEMANTICS,
        DAY + timedelta(days=1),
        DAY + timedelta(days=3),
    )
    assert result.publication == selected.publication
    assert resources.audit() == before


def test_anchor_selection_isolated_by_execution_policy(resources, qualified):
    from arblab.hyperliquid_copy.annual_execution_policy import POLICY
    from arblab.hyperliquid_copy.feature_history import FeatureHistory
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy.rolling_feature_catalogue import select_anchor

    days = (
        FeatureHistory(
            resources,
            qualified,
            ["BTC"],
            SEMANTICS,
            execution_policy_name=POLICY,
        )
        .window(DAY, DAY + timedelta(days=2))
        .days
    )
    annual = publish_feature_anchor(resources, qualified, days, ["BTC"], SEMANTICS)

    legacy = select_anchor(
        resources,
        qualified,
        ["BTC"],
        SEMANTICS,
        DAY,
        DAY + timedelta(days=2),
    )
    selected = select_anchor(
        resources,
        qualified,
        ["BTC"],
        SEMANTICS,
        DAY,
        DAY + timedelta(days=2),
        execution_policy_name=POLICY,
    )

    assert legacy is None
    assert selected.publication == annual.publication


def test_duplicate_day_variants_are_rejected_without_allocating(resources, qualified):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy.rolling_feature_catalogue import expired_feature_days

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days
    anchor = publish_feature_anchor(resources, qualified, days[2:], ["BTC"], SEMANTICS)
    variant = days[1].inputs
    variant["previous"] = "b" * 64
    PublishedArtifacts(resources).publish(
        "qualified_features_day",
        variant,
        [p.token for p in days[1].publication.artifacts],
    )
    before = resources.audit()
    with pytest.raises(ValueError, match="Ambiguous"):
        expired_feature_days(resources, qualified, anchor)
    assert resources.audit() == before


@pytest.mark.parametrize("limit", ["MAX_DAYS", "MAX_INPUT_BYTES", "MAX_TARGET_FILES"])
def test_target_inventory_limits_fail_without_allocating(
    resources, qualified, monkeypatch, limit
):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy import rolling_feature_catalogue as module

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=2)).days
    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    monkeypatch.setattr(module, limit, 0, raising=False)
    before = resources.audit()
    with pytest.raises(ValueError, match="resource limit"):
        module.expired_feature_days(resources, qualified, anchor)
    assert resources.audit() == before


def test_corrupt_selected_anchor_has_no_fallback(resources, qualified):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy.rolling_feature_catalogue import select_anchor

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days
    publish_feature_anchor(resources, qualified, days[:2], ["BTC"], SEMANTICS)
    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    with (resources.root / anchor.publication.artifacts[0].path).open("ab") as handle:
        handle.write(b"corrupt selected anchor")
    with pytest.raises(ValueError):
        select_anchor(
            resources,
            qualified,
            ["BTC"],
            SEMANTICS,
            DAY + timedelta(days=1),
            DAY + timedelta(days=3),
        )


def test_scope_freezes_pin_before_its_first_hash(
    resources, qualified, monkeypatch, tmp_path
):
    import shutil
    from pathlib import Path
    from arblab.hyperliquid_copy import rolling_feature_catalogue as module

    alias = tmp_path / "catalogue-alias" / "manifest.json"
    alias.parent.mkdir()
    shutil.copyfile(qualified["path"], alias)
    original = module.file_hash

    def changed(path):
        value = original(path)
        if Path(path) == Path(module.__file__):
            qualified["path"] = str(alias)
        return value

    monkeypatch.setattr(module, "file_hash", changed)
    before = resources.audit()
    with pytest.raises(ValueError, match="context|caller"):
        module.select_anchor(
            resources, qualified, ["BTC"], SEMANTICS, DAY, DAY + timedelta(days=1)
        )
    assert resources.audit() == before


@pytest.mark.parametrize("target", ["anchor", "day"])
def test_selected_anchor_guard_spans_final_validation(
    resources, qualified, monkeypatch, target
):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy import rolling_feature_catalogue as module

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=2)).days
    anchor = publish_feature_anchor(resources, qualified, days, ["BTC"], SEMANTICS)
    original = module.FeatureAnchor
    armed = False
    changed = False

    def constructed(*args):
        nonlocal armed
        result = original(*args)
        armed = True
        return result

    def validation():
        nonlocal changed
        if armed and not changed:
            changed = True
            publication = (
                anchor.publication if target == "anchor" else days[-1].publication
            )
            with (resources.root / publication.artifacts[0].path).open("ab") as handle:
                handle.write(b"changed during final selection validation")

    monkeypatch.setattr(module, "FeatureAnchor", constructed)
    with pytest.raises(ValueError):
        module.select_anchor(
            resources,
            qualified,
            ["BTC"],
            SEMANTICS,
            DAY,
            DAY + timedelta(days=2),
            validation=validation,
        )
    assert changed


def test_expired_target_guard_spans_final_validation(resources, qualified, monkeypatch):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor
    from arblab.hyperliquid_copy import rolling_feature_catalogue as module

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=2)).days
    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    original = module.FeatureDay
    armed = False
    changed = False

    def constructed(*args):
        nonlocal armed
        result = original(*args)
        armed = True
        return result

    def validation():
        nonlocal changed
        if armed and not changed:
            changed = True
            with (resources.root / days[0].publication.artifacts[0].path).open(
                "ab"
            ) as handle:
                handle.write(b"changed during final target validation")

    monkeypatch.setattr(module, "FeatureDay", constructed)
    with pytest.raises(ValueError):
        module.expired_feature_days(resources, qualified, anchor, validation=validation)
    assert changed
