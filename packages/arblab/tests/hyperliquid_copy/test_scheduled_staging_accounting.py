from datetime import timedelta
import json

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.ranking_staging_policy import POLICY
from arblab.hyperliquid_copy.ranking_staging_manifest import ROLES
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_qualified_scheduled_activity import reader, scheduled_config


def test_multiple_decisions_retain_only_full_rankings_and_bound_admission(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner

    original = RankingStagingOwner.create
    admissions = []

    def measured(cache, context):
        before = cache.audit()
        owner = original(cache, context)
        after = cache.audit()
        assert before["reserved_bytes"] == 0
        assert after["reserved_bytes"] == sum(role[2] for role in ROLES.values())
        assert after["total_bytes"] == before["total_bytes"] + after["reserved_bytes"]
        assert after["total_bytes"] <= cache.limit
        admissions.append(after["total_bytes"])
        return owner

    monkeypatch.setattr(RankingStagingOwner, "create", measured)
    effective = scheduled_config().effective(["BTC"], {"BTC": 1})
    with reader(resources, qualified, ranking_staging_policy=POLICY) as actual:
        ranking_bytes = 0
        for hour in range(3):
            at = day("2026-08-03") + timedelta(hours=hour)
            actual.prepare(at)
            result = actual.rank(at, effective, "BTC", "gross_excludes_fee", smoke=True)
            ranking_bytes += result.publication.artifacts[0].bytes
            with resources._connect() as db:
                publications = [
                    json.loads(r[0])
                    for r in db.execute("SELECT descriptor FROM publications")
                ]
            ranks = [p for p in publications if p["kind"] == "staged_cohort_rankings"]
            assert len(ranks) == hour + 1
            assert sum(p["artifacts"][0]["bytes"] for p in ranks) == ranking_bytes
            assert not any(
                p["kind"] in ("candidate_metrics", "candidate_scores")
                for p in publications
            )
            assert resources.audit()["reserved_bytes"] == 0
            assert not list((resources.root / "staging").iterdir())
            assert not list((resources.root / "scratch").iterdir())
    assert len(admissions) == 3 and ranking_bytes > 0
