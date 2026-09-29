from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxyScheduled
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from arblab.hyperliquid_copy import feature_metric_producer as producer_module
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_archive import USER
from .test_lab_pipeline_proxy import config


def scheduled_config():
    raw = config().to_dict()
    raw.pop("schema_version")
    raw["trader"].update(
        reselection="weekly", metric_weights={"gross_volume": 1}, metric_directions={}
    )
    raw["market_universe"]["reselection"] = "weekly"
    return LabConfigProxyScheduled(**raw)


def reader(resources, qualified, **kwargs):
    from arblab.hyperliquid_copy.qualified_scheduled_activity import (
        QualifiedScheduledActivity,
    )

    return QualifiedScheduledActivity(
        resources,
        qualified,
        scheduled_config(),
        coverage_start=kwargs.pop("coverage_start", "2026-08-01"),
        coverage_end=kwargs.pop("coverage_end", "2026-08-04"),
        semantics=kwargs.pop("semantics", "gross_excludes_fee"),
        **kwargs,
    )


def reference(qualified, tmp_path):
    report = json.loads(Path(qualified["path"]).read_text())
    return ProxyActivity([Path(f["path"]) for f in report["files"]], temp_root=tmp_path)


def test_real_native_queries_match_reference_and_borrow_lease(
    resources, qualified, tmp_path
):
    at = day("2026-08-03")
    users = [USER, "0x" + "cd" * 20, "0x" + "ef" * 20]
    with reference(qualified, tmp_path) as ref:
        with reader(resources, qualified) as actual:
            actual.prepare(at)
            assert actual.observed(at) == ref.observed(at)
            assert actual.positions(users, "BTC", at) == {
                u: ref.position(u, "BTC", at) for u in users
            }
            assert actual.position(USER, "BTC", at) == ref.position(USER, "BTC", at)
            assert actual.volume("BTC", at - timedelta(days=1), at) == pytest.approx(
                ref.volume("BTC", at - timedelta(days=1), at)
            )
            args = (USER, "BTC", at - timedelta(days=1), at + timedelta(hours=1))
            assert actual.hourly_exposure(
                *args, max_price_age_seconds=86400
            ) == ref.hourly_exposure(*args, max_price_age_seconds=86400)
        resources.lease.check()
        assert resources.audit()["reserved_bytes"] == 0


def test_actual_disk_ranking_and_reopen_reuse(
    resources, qualified, tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as producer
    from arblab.hyperliquid_copy.disk_score_result import ScoredCohort

    c, at = scheduled_config(), day("2026-08-03")
    effective = c.effective(["BTC"], {"BTC": 1})
    with reference(qualified, tmp_path) as ref:
        expected = ref.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
    with reader(resources, qualified) as actual:
        actual.prepare(at)
        result = actual.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
        assert isinstance(result, ScoredCohort)
        rows = [r for batch in result.iter_batches() for r in batch]
        assert [r["user"] for r in rows] == [r["user"] for r in expected]
        for row, old in zip(rows, expected):
            for key in ("selected", "eligible", "reasons", "exclusions", "rank"):
                assert row[key] == old[key]
            for key, value in old["metrics"].items():
                assert (
                    row["metrics"][key] == pytest.approx(value)
                    if value is not None
                    else row["metrics"][key] is None
                )
    before = resources.audit()
    monkeypatch.setattr(
        producer,
        "partition_metric_rows",
        lambda *a, **k: pytest.fail("reuse recomputed metrics"),
    )
    with reader(resources, qualified) as reopened:
        reopened.prepare(at)
        assert (
            reopened.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
            == result
        )
    assert resources.audit() == before


def test_selection_and_signal_integration_never_opens_full_reader(
    resources, qualified, tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import (
        ProxySelectionState,
        _signals_at,
    )
    from arblab.hyperliquid_copy.ranking_artifact import RankingSink
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
    from .test_proxy_selection import mapping

    monkeypatch.setattr(
        ProxyActivity,
        "__init__",
        lambda *a, **k: pytest.fail("full-prefix reader opened"),
    )
    c, at = scheduled_config(), day("2026-08-03")
    with reader(resources, qualified) as actual:
        data = SimpleNamespace(
            activity=actual,
            mappings=ProxyMappings([mapping()]),
            coverage_start=day("2026-08-01"),
            coverage_end=day("2026-08-04"),
            manifest={"fee_semantics": "gross_excludes_fee"},
        )
        state = ProxySelectionState(data, c)
        with RankingSink(tmp_path / "run-rankings.parquet") as sink:
            state.rankings = sink
            assert state.advance(at)
            signals, contributions = _signals_at(data, state, c, at, None)
        assert sink.artifact.rows == 2
        assert state.trader_cohorts[0]["candidate_count"] == 2
        assert len(signals) == 1 and contributions


@pytest.mark.parametrize("at", ["2026-08-02", "2026-08-04", "2026-08-03T00:01:00"])
def test_prepare_rejects_outside_or_nonhour_decisions(resources, qualified, at):
    from datetime import datetime, timezone

    with reader(resources, qualified) as actual:
        with pytest.raises(ValueError):
            actual.prepare(datetime.fromisoformat(at).replace(tzinfo=timezone.utc))
        with pytest.raises(ValueError):
            actual.observed(day("2026-08-03"))


def test_lifecycle_monotonicity_and_idempotent_close(resources, qualified):
    actual = reader(resources, qualified)
    at = day("2026-08-03")
    with pytest.raises(ValueError):
        actual.observed(at)
    with pytest.raises(ValueError):
        actual.prepare(at.replace(tzinfo=None))
    actual.prepare(at)
    actual.prepare(at)
    actual.prepare(at + timedelta(hours=1))
    with pytest.raises(ValueError):
        actual.prepare(at)
    actual.close()
    actual.close()
    with pytest.raises(ValueError):
        actual.prepare(at)
    resources.lease.check()


@pytest.mark.parametrize(
    "change",
    [
        dict(coverage_start="2026-08-02"),
        dict(coverage_end="2026-08-03"),
        dict(semantics="unresolved"),
    ],
)
def test_invalid_declared_context_rejects_before_queries(resources, qualified, change):
    before = resources.audit()
    with pytest.raises(ValueError):
        reader(resources, qualified, **change)
    assert resources.audit() == before


@pytest.mark.parametrize(
    "method", ["observed", "positions", "rank", "volume", "hourly"]
)
def test_future_queries_reject_before_delegate(
    resources, qualified, monkeypatch, method
):
    from arblab.hyperliquid_copy import qualified_scheduled_activity as module

    at = day("2026-08-03")
    with reader(resources, qualified) as actual:
        actual.prepare(at)
        for name in (
            "observed_markets",
            "native_positions",
            "market_volume",
            "hourly_exposure",
        ):
            monkeypatch.setattr(
                module, name, lambda *a, **k: pytest.fail("future helper ran")
            )
        monkeypatch.setattr(
            producer_module,
            "build_and_score_features",
            lambda *a, **k: pytest.fail("future producer ran"),
        )
        future = at + timedelta(hours=1)
        calls = {
            "observed": lambda: actual.observed(future),
            "positions": lambda: actual.positions([USER], "BTC", future),
            "rank": lambda: actual.rank(
                future,
                scheduled_config().effective(["BTC"], {"BTC": 1}),
                "BTC",
                "gross_excludes_fee",
                smoke=True,
            ),
            "volume": lambda: actual.volume("BTC", at, future),
            "hourly": lambda: actual.hourly_exposure(
                USER,
                "BTC",
                at,
                future + timedelta(hours=1),
                max_price_age_seconds=86400,
            ),
        }
        with pytest.raises(ValueError):
            calls[method]()


@pytest.mark.parametrize(
    "fault", ["pin", "config", "engine", "decision", "reentry", "interrupt"]
)
def test_final_context_and_reentrant_guards(resources, qualified, monkeypatch, fault):
    from arblab.hyperliquid_copy import qualified_scheduled_activity as module

    at = day("2026-08-03")
    actual = reader(resources, qualified)
    actual.prepare(at)
    original = module.native_positions

    def changed(*args):
        if fault == "reentry":
            return actual.observed(at)
        if fault == "interrupt":
            raise OSError("interrupted")
        value = original(*args)
        if fault == "pin":
            qualified["sha256"] = "0" * 64
        elif fault == "config":
            actual.config.trader.metric_weights["gross_volume"] = 0.5
        elif fault == "decision":
            actual.last_decision += timedelta(hours=1)
        else:
            previous = module._engine()
            monkeypatch.setattr(module, "_engine", lambda: previous | {"changed": True})
        return value

    monkeypatch.setattr(module, "native_positions", changed)
    with pytest.raises((ValueError, OSError)):
        actual.positions([USER], "BTC", at)
    actual.close()
    resources.lease.check()


@pytest.mark.parametrize("change", ["lookback", "coins", "scope", "semantics", "smoke"])
def test_effective_rank_context_rejects_before_producer(
    resources, qualified, monkeypatch, change
):
    from arblab.hyperliquid_copy import qualified_scheduled_activity as module

    with reader(resources, qualified) as actual:
        at = day("2026-08-03")
        actual.prepare(at)
        effective = scheduled_config().effective(["BTC"], {"BTC": 1})
        scope, semantics, smoke = "BTC", "gross_excludes_fee", True
        if change == "lookback":
            effective.lookback_days = 2
        elif change == "coins":
            effective.coins = ["ETH"]
            scope = "ETH"
        elif change == "scope":
            scope = "ETH"
        elif change == "semantics":
            semantics = "net_includes_fee"
        else:
            smoke = False
        monkeypatch.setattr(
            producer_module,
            "build_and_score_features",
            lambda *a, **k: pytest.fail("invalid rank invoked producer"),
        )
        with pytest.raises(ValueError):
            actual.rank(at, effective, scope, semantics, smoke=smoke)


def test_empty_configured_universe_is_explicit_cash_not_empty_artifact(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import qualified_scheduled_activity as module

    with reader(resources, qualified) as actual:
        at = day("2026-08-03")
        actual.prepare(at)
        effective = scheduled_config().effective([], {})
        monkeypatch.setattr(
            producer_module,
            "build_and_score_features",
            lambda *a, **k: pytest.fail("empty configured universe invoked producer"),
        )
        assert actual.rank(at, effective, None, "gross_excludes_fee", smoke=True) == []


def test_failed_prepare_does_not_publish_cutoff(resources, qualified, monkeypatch):
    with reader(resources, qualified) as actual:
        at = day("2026-08-03")
        actual.prepare(at)
        verify = actual._verify
        calls = []

        def final_failure():
            verify()
            calls.append(True)
            if len(calls) == 2:
                raise ValueError("final verification failed")

        monkeypatch.setattr(actual, "_verify", final_failure)
        with pytest.raises(ValueError, match="final verification failed"):
            actual.prepare(at + timedelta(hours=1))
        assert actual.last_decision == at
        actual.prepare(at)
