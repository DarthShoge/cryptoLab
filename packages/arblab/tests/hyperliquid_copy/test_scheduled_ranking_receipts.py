from dataclasses import replace

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.qualified_scheduled_activity import (
    QualifiedScheduledActivity,
)
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_qualified_scheduled_activity import reader, scheduled_config
from .test_scheduled_feature_history import rows


@pytest.mark.parametrize("retirement", ["logical", "physical"])
def test_scheduled_receipt_reopens_without_intermediate_history(
    resources, qualified, monkeypatch, retirement
):
    from arblab.hyperliquid_copy import feature_metric_producer as producer
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    at = day("2026-08-03")
    weekly = scheduled_config()
    with reader(resources, qualified) as actual:
        actual.prepare(at)
        result = actual.rank(
            at,
            weekly.effective(["BTC"], {"BTC": 1}),
            "BTC",
            "gross_excludes_fee",
            smoke=True,
        )
        expected_rows, expected_selected = rows(result), result.selected
    before = resources.audit()
    if retirement == "logical":
        with resources._connect() as db, db:
            # Fixture-only metadata retirement; every payload stays intact/accounted.
            db.execute(
                "DELETE FROM publications WHERE json_extract(descriptor,'$.kind') "
                "<> 'saved_feature_ranking'"
            )
        expected_publications = 1
    else:
        import json
        from arblab.hyperliquid_copy.cache_retirement import (
            prepare_retirement,
            begin_retirement,
            finish_retirement,
        )

        with resources._connect() as db:
            targets = [
                json.loads(raw)
                for (raw,) in db.execute("SELECT descriptor FROM publications")
                if json.loads(raw)["kind"] != "saved_feature_ranking"
            ]
        freed = 0
        for target in targets:
            intent = prepare_retirement(
                resources,
                target["kind"],
                target["inputs"],
                reason="Fixture saved ranking is independent of intermediates",
            )
            begin_retirement(resources, intent)
            freed += finish_retirement(resources, intent)["retired_bytes"]
        assert freed > 0
        before = resources.audit()
        # Completed retirement compacts its intent and retains only the
        # authenticated committed receipt for each target.
        expected_publications = 1 + len(targets)
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)

    def forbidden(*args, **kwargs):
        pytest.fail("saved scheduled ranking attempted intermediate replay")

    monkeypatch.setattr(FeatureHistory, "window", forbidden)
    monkeypatch.setattr(producer, "build_and_score_features", forbidden)
    monkeypatch.setattr(producer.OrderedFeaturePartitions, "plan", forbidden)
    daily = replace(
        weekly,
        rebalance="daily",
        trader=replace(weekly.trader, reselection="daily"),
        market_universe=replace(weekly.market_universe, reselection="daily"),
    )
    with CacheLease(root) as lease:
        reopened = CacheResources(lease, identity)
        for config in (weekly, daily):
            with QualifiedScheduledActivity(
                reopened,
                qualified,
                config,
                coverage_start="2026-08-01",
                coverage_end="2026-08-04",
                semantics="gross_excludes_fee",
            ) as actual:
                actual.prepare(at)
                result = actual.rank(
                    at,
                    config.effective(["BTC"], {"BTC": 1}),
                    "BTC",
                    "gross_excludes_fee",
                    smoke=True,
                )
                assert rows(result) == expected_rows
                assert result.selected == expected_selected
                assert result.bound_decision == at and result.bound_scope == "BTC"
                assert actual._features is None
            assert reopened.audit() == before
            with reopened._connect() as db:
                assert (
                    db.execute("SELECT count(*) FROM publications").fetchone()[0]
                    == expected_publications
                )


@pytest.mark.parametrize(
    "fault", ["effective", "facade", "decision", "source", "ranking"]
)
def test_receipt_hit_rechecks_scheduled_context(
    resources, qualified, monkeypatch, fault
):
    from datetime import timedelta
    from arblab.hyperliquid_copy.saved_feature_rankings import SavedFeatureRankings

    at = day("2026-08-03")
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    with reader(resources, qualified) as actual:
        actual.prepare(at)
        actual.rank(at, config, "BTC", "gross_excludes_fee", smoke=True)
        original = SavedFeatureRankings.find

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            assert result is not None
            if fault == "effective":
                config.min_volume += 1
            elif fault == "facade":
                actual.config.trader.metric_weights["gross_volume"] = 0.5
            elif fault == "decision":
                actual.last_decision += timedelta(hours=1)
            else:
                import json
                from pathlib import Path

                path = (
                    Path(
                        json.loads(Path(qualified["path"]).read_text())["files"][0][
                            "path"
                        ]
                    )
                    if fault == "source"
                    else resources.root / result.publication.artifacts[0].path
                )
                with path.open("ab") as handle:
                    handle.write(b"late mutation")
            return result

        monkeypatch.setattr(SavedFeatureRankings, "find", changed)
        with pytest.raises(ValueError):
            actual.rank(at, config, "BTC", "gross_excludes_fee", smoke=True)
