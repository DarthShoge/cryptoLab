import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
from arblab.hyperliquid_copy.saved_feature_rankings import SavedFeatureRankings
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_qualified_scheduled_activity import scheduled_config
from .test_scheduled_feature_history import rows


@pytest.mark.parametrize(
    "fault", ["source", "candidate_origin", "candidate_scope", "engine", "bound"]
)
def test_raw_capture_rejects_authentic_but_wrong_provenance(
    resources, qualified, monkeypatch, fault
):
    from copy import deepcopy
    from arblab.hyperliquid_copy import candidate_metric_producer as raw
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.bound_scoring_context import _read_publication
    from arblab.hyperliquid_copy.disk_cohort_scoring import score_metric_artifact

    at = day("2026-08-03")
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    inputs = raw.build_candidate_metrics(
        resources, qualified, "2026-08-01", at, config, "BTC", "gross_excludes_fee"
    )
    publications = PublishedArtifacts(resources)
    metric = publications.lookup("candidate_metrics", inputs)
    forged = deepcopy(inputs)
    provenance = forged["provenance"]
    if fault == "source":
        provenance["source"]["report_sha256"] = "f" * 64
    elif fault.startswith("candidate"):
        candidate, context = _read_publication(
            resources, provenance["candidates"], "candidate_history"
        )
        context["source_start" if fault == "candidate_origin" else "scope"] = (
            "2026-08-02" if fault == "candidate_origin" else None
        )
        variant = publications.publish(
            "candidate_history", context, [p.token for p in candidate.artifacts]
        )
        provenance["candidates"] = variant.key
    elif fault == "engine":
        provenance["engine"] = {"wrong": True}
    else:
        provenance["max_partition_rows"] = True
    publications.publish(
        "candidate_metrics", forged, [p.token for p in metric.artifacts]
    )
    result = score_metric_artifact(
        resources,
        forged,
        config,
        at,
        "BTC",
        "gross_excludes_fee",
        verify_source=lambda: deepcopy(provenance),
    )
    monkeypatch.setattr(
        raw, "build_and_score_candidates", lambda *args, **kwargs: result
    )
    receipts = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    with pytest.raises(ValueError, match="provenance|source|producer"):
        receipts.capture_raw(at, config, "BTC", "gross_excludes_fee")
    assert receipts.find(at, config, "BTC", "gross_excludes_fee") is None


def test_existing_capture_checks_artifact_after_final_verification(
    resources, qualified, monkeypatch
):
    at = day("2026-08-03")
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    receipts = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    inputs = receipts.capture_raw(at, config, "BTC", "gross_excludes_fee")
    result = receipts.load(inputs, at, config, "BTC", "gross_excludes_fee")
    original = SavedFeatureRankings._verify
    calls = 0

    def count(self):
        nonlocal calls
        original(self)
        calls += 1

    monkeypatch.setattr(SavedFeatureRankings, "_verify", count)
    receipts.capture_raw(at, config, "BTC", "gross_excludes_fee")
    final_call, calls = calls, 0

    def mutate(self):
        count(self)
        if calls == final_call:
            with result.artifact.path.open("ab") as stream:
                stream.write(b"late capture corruption")

    monkeypatch.setattr(SavedFeatureRankings, "_verify", mutate)
    with pytest.raises(ValueError):
        receipts.capture_raw(at, config, "BTC", "gross_excludes_fee")


@pytest.mark.parametrize("scope", ["BTC", None])
@pytest.mark.parametrize("empty", [False, True])
def test_raw_receipt_reopens_through_normal_lookup(
    resources, qualified, monkeypatch, scope, empty
):
    from arblab.hyperliquid_copy import candidate_metric_producer as raw
    from arblab.hyperliquid_copy import feature_metric_producer as feature
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

    at = day("2026-08-03")
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    if empty:
        config.min_volume = 1e15
    reference = raw.build_and_score_candidates(
        resources, qualified, "2026-08-01", at, config, scope, "gross_excludes_fee"
    )
    expected_rows, selected = rows(reference), reference.selected
    if empty:
        assert reference.eligible_count == 0 and not selected
    receipts = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    inputs = receipts.capture_raw(at, config, scope, "gross_excludes_fee")
    result = receipts.load(inputs, at, config, scope, "gross_excludes_fee")
    assert rows(result) == expected_rows and result.selected == selected
    before = resources.audit()
    resources.lease.__exit__(None, None, None)

    def forbidden(*args, **kwargs):
        pytest.fail("saved raw receipt reopened a producer")

    monkeypatch.setattr(raw, "build_and_score_candidates", forbidden)
    monkeypatch.setattr(feature, "build_and_score_features", forbidden)
    with CacheLease(resources.root) as lease:
        reopened = CacheResources(lease, resources.identity)
        receipts = SavedFeatureRankings(reopened, QualifiedSourceSession(qualified))
        assert receipts.capture([], at, config, scope, "gross_excludes_fee") == inputs
        result = receipts.find(at, config, scope, "gross_excludes_fee")
        assert rows(result) == expected_rows and result.selected == selected
        assert result.bound_decision == at and result.bound_scope == scope
        assert reopened.audit() == before


def test_raw_capture_reuses_exact_feature_receipt_without_duplicates(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_metric_producer as raw
    from arblab.hyperliquid_copy.feature_history import FeatureHistory

    at = day("2026-08-03")
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    window = FeatureHistory(resources, qualified, ["BTC"], "gross_excludes_fee").window(
        day("2026-08-02"), at
    )
    receipts = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    expected = receipts.capture(window.days, at, config, "BTC", "gross_excludes_fee")
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("alternate producer recomputed an exact saved query")

    monkeypatch.setattr(raw, "build_and_score_candidates", forbidden)
    assert receipts.capture_raw(at, config, "BTC", "gross_excludes_fee") == expected
    assert receipts.find(at, config, "BTC", "gross_excludes_fee") is not None
    assert resources.audit() == before
