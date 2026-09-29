from dataclasses import replace
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.ranking_staging_policy import POLICY
from arblab.hyperliquid_copy.qualified_scheduled_activity import (
    QualifiedScheduledActivity,
)
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_qualified_scheduled_activity import reader, scheduled_config
from .test_scheduled_feature_history import rows


@pytest.mark.parametrize("policy", ["unknown", False, {}, 1])
def test_unknown_staging_policy_rejected_before_work(resources, qualified, policy):
    before = resources.audit()
    with pytest.raises(ValueError, match="policy"):
        reader(resources, qualified, ranking_staging_policy=policy)
    assert resources.audit() == before


def arm_final_cleanup_engine(monkeypatch, module, mutate):
    from arblab.hyperliquid_copy import ranking_staging_cleanup as cleanup

    original_unlink, original_engine = cleanup._unlink_owned, module._engine
    state = dict(paths=[], counting=False, calls=0, final=None, changed=False)

    def arm(resources, directory_fd, row, identity, guard):
        state["paths"].append(resources._path(row[1]))
        state.update(counting=True, calls=0)
        guard()
        state.update(final=state["calls"], calls=0, counting=False)
        return original_unlink(resources, directory_fd, row, identity, guard)

    def engine():
        result = original_engine()
        if state["paths"]:
            state["calls"] += 1
        if (
            state["paths"]
            and not state["counting"]
            and not state["changed"]
            and state["calls"] == state["final"]
        ):
            state["changed"] = True
            mutate()
        return result

    monkeypatch.setattr(cleanup, "_unlink_owned", arm)
    monkeypatch.setattr(module, "_engine", engine)
    return state


@pytest.mark.parametrize("fault", ["policy", "effective", "decision"])
def test_final_cleanup_context_change_preserves_temporaries(
    resources, qualified, monkeypatch, fault
):
    from arblab.hyperliquid_copy import qualified_scheduled_activity as module

    actual = reader(resources, qualified, ranking_staging_policy=POLICY)
    at = day("2026-08-03")
    actual.prepare(at)
    effective = scheduled_config().effective(["BTC"], {"BTC": 1})

    def mutate():
        if fault == "policy":
            actual.ranking_staging_policy = None
        elif fault == "effective":
            effective.min_volume += 1
        else:
            actual.last_decision += timedelta(hours=1)

    state = arm_final_cleanup_engine(monkeypatch, module, mutate)
    try:
        with pytest.raises(ValueError):
            actual.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
        assert state["changed"] and state["paths"]
        assert all(path.exists() for path in state["paths"])
        assert len(list((resources.root / "staging").glob("*.parquet"))) == 2
        assert resources.audit()["reserved_bytes"] > 0
    finally:
        actual.close()


def test_default_and_explicit_policy_are_frozen(resources, qualified):
    with reader(resources, qualified) as actual:
        assert actual.ranking_staging_policy is None
    with reader(resources, qualified, ranking_staging_policy=POLICY) as actual:
        actual.ranking_staging_policy = None
        with pytest.raises(ValueError, match="changed"):
            actual.prepare(day("2026-08-03"))


def test_staged_monday_reuse_precedes_pending_gate(resources, qualified, monkeypatch):
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    at = day("2026-08-03")
    weekly = scheduled_config()
    effective = weekly.effective(["BTC"], {"BTC": 1})
    with reader(resources, qualified, ranking_staging_policy=POLICY) as actual:
        actual.prepare(at)
        first = actual.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
        expected = rows(first)
    assert resources.audit()["reserved_bytes"] == 0
    with resources._connect() as db:
        kinds = {
            r[0]
            for r in db.execute(
                "SELECT json_extract(descriptor,'$.kind') FROM publications"
            )
        }
    assert "staged_cohort_rankings" in kinds
    assert not kinds & {"candidate_metrics", "candidate_scores"}
    resources.reserve("staging/" + "a" * 32, 100, purpose="payload")
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("saved hit or pending miss derived feature history")

    monkeypatch.setattr(FeatureHistory, "window", forbidden)
    daily = replace(
        weekly,
        rebalance="daily",
        trader=replace(weekly.trader, reselection="daily"),
        market_universe=replace(weekly.market_universe, reselection="daily"),
    )
    effective = daily.effective(["BTC"], {"BTC": 1})
    with QualifiedScheduledActivity(
        resources,
        qualified,
        daily,
        coverage_start="2026-08-01",
        coverage_end="2026-08-04",
        semantics="gross_excludes_fee",
        ranking_staging_policy=POLICY,
    ) as actual:
        actual.prepare(at)
        result = actual.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
        assert rows(result) == expected
        assert result.selected == first.selected
        assert actual._features is None
        later = at + timedelta(hours=1)
        actual.prepare(later)
        with pytest.raises(ValueError, match="[Pp]ending"):
            actual.rank(later, effective, "BTC", "gross_excludes_fee", smoke=True)
    assert resources.audit() == before
