import json
from copy import deepcopy
from dataclasses import replace

import pytest

from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
from arblab.hyperliquid_copy.saved_feature_rankings import SavedFeatureRankings
from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from .test_candidate_day import resources
from .test_feature_window import days
from .test_qualified_day import qualified
from .test_feature_metric_producer import args


def assert_metric_equivalent(actual, expected):
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        observed, reference = dict(observed), dict(reference)
        observed_metrics = observed.pop("metrics")
        reference_metrics = reference.pop("metrics")
        assert observed == reference
        for key, value in observed_metrics.items():
            if value is None or reference_metrics[key] is None:
                assert value is reference_metrics[key]
            else:
                assert value == pytest.approx(
                    reference_metrics[key], rel=1e-14, abs=1e-14
                )


from .test_scheduled_feature_history import rows

POLICY = "bounded_ranking_staging_v1"


@pytest.mark.parametrize(
    "route,scope", [("raw", "BTC"), ("features", "BTC"), ("raw", None)]
)
def test_saved_staged_ranking_survives_cleanup_and_fresh_lease(
    resources, qualified, days, tmp_path, monkeypatch, route, scope
):
    from arblab.hyperliquid_copy import ranking_staging_sources as sources
    from arblab.hyperliquid_copy.candidate_metric_producer import (
        build_and_score_candidates,
    )

    values = list(args(resources, qualified, days))
    values[6] = scope
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    with CacheLease(reference_root) as lease:
        reference_cache = CacheResources.create(lease, "staged-reference")
        reference = build_and_score_candidates(
            reference_cache, *values[1:3], *values[4:]
        )
        expected, selected = rows(reference), reference.selected
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    inputs = receipts.capture_staged(
        values[4],
        values[5],
        scope,
        values[7],
        days=values[3] if route == "features" else None,
    )
    actual = receipts.find(values[4], values[5], scope, values[7])
    observed = rows(actual)
    if route == "features":
        assert_metric_equivalent(observed, expected)
        assert_metric_equivalent(actual.selected, selected)
    else:
        assert observed == expected
        assert actual.selected == selected
    assert resources.audit()["reserved_bytes"] == 0
    assert not list((resources.root / "staging").iterdir())
    assert not list((resources.root / "scratch").iterdir())
    with resources._connect() as db:
        kinds = [
            json.loads(row[0])["kind"]
            for row in db.execute("SELECT descriptor FROM publications")
        ]
    assert "candidate_metrics" not in kinds and "candidate_scores" not in kinds
    before = resources.audit()
    resources.lease.__exit__(None, None, None)

    def forbidden(*args, **kwargs):
        pytest.fail("saved staged lookup rebuilt source inputs")

    monkeypatch.setattr(sources, "prepare_staging_source", forbidden)
    with CacheLease(resources.root) as lease:
        reopened = CacheResources(lease, resources.identity)
        receipts = SavedFeatureRankings(
            reopened, QualifiedSourceSession(qualified), staging_policy=POLICY
        )
        assert (
            receipts.capture_staged(values[4], values[5], scope, values[7], days=[])
            == inputs
        )
        actual = receipts.load(inputs, values[4], values[5], scope, values[7])
        if route == "features":
            assert_metric_equivalent(rows(actual), expected)
            assert_metric_equivalent(actual.selected, selected)
        else:
            assert rows(actual) == expected and actual.selected == selected
        assert reopened.audit() == before


def test_staged_capture_requires_explicit_policy(resources, qualified):
    from .test_lab_ranking import settings
    from arblab.hyperliquid_copy.lab_config import day

    receipts = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    before = resources.audit()
    with pytest.raises(ValueError, match="policy"):
        receipts.capture_staged(
            day("2026-08-03"), settings(lookback_days=1), "BTC", "gross_excludes_fee"
        )
    assert resources.audit() == before


def test_unknown_staged_policy_is_rejected(resources, qualified):
    with pytest.raises(ValueError, match="policy"):
        SavedFeatureRankings(
            resources, QualifiedSourceSession(qualified), staging_policy="guess"
        )


@pytest.mark.parametrize(
    "boundary", ["settle", "staged_cohort_rankings", "saved_feature_ranking", "cleanup"]
)
def test_partial_publication_keeps_charges_and_saved_receipts_readable(
    resources, qualified, days, monkeypatch, boundary
):
    from arblab.hyperliquid_copy import ranking_staging_binding as binding
    from arblab.hyperliquid_copy import ranking_staging_capture as capture
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    original_bind, original_settle, original_publish = (
        binding.bind_pending,
        CacheResources.settle,
        PublishedArtifacts.publish,
    )
    owned = {}

    def bind(owner, *args):
        owned["token"] = owner._allocations["ranking"]["token"]
        return original_bind(owner, *args)

    def settle(self, token):
        result = original_settle(self, token)
        if boundary == "settle" and token == owned.get("token"):
            raise RuntimeError("publication boundary")
        return result

    def publish(self, kind, *args, **kwargs):
        result = original_publish(self, kind, *args, **kwargs)
        if kind == boundary:
            raise RuntimeError("publication boundary")
        return result

    def cleanup(*args, **kwargs):
        raise RuntimeError("publication boundary")

    with monkeypatch.context() as patch:
        patch.setattr(binding, "bind_pending", bind)
        patch.setattr(CacheResources, "settle", settle)
        patch.setattr(PublishedArtifacts, "publish", publish)
        if boundary == "cleanup":
            patch.setattr(capture, "finish_staging", cleanup)
        with pytest.raises(RuntimeError, match="publication boundary"):
            receipts.capture_staged(values[4], values[5], values[6], values[7])
    before = resources.audit()
    assert before["reserved_bytes"] > 0
    result = receipts.find(values[4], values[5], values[6], values[7])
    if boundary in ("saved_feature_ranking", "cleanup"):
        assert result is not None and len(rows(result)) == 2
        receipts.capture_staged(values[4], values[5], values[6], values[7])
    else:
        assert result is None
    assert resources.audit() == before
    with pytest.raises(ValueError, match="Pending"):
        receipts.capture_staged(
            values[4], replace(values[5], min_volume=1), values[6], values[7]
        )
    assert resources.audit() == before


@pytest.mark.parametrize("route", ["raw", "features"])
@pytest.mark.parametrize(
    "fault",
    [
        "summary",
        "policy",
        "window",
        "candidate",
        "producer_engine",
        "artifact",
        "config",
        "bound",
    ],
)
def test_saved_load_rejects_authenticated_wrong_staged_provenance(
    resources, qualified, days, fault, route
):
    from arblab.hyperliquid_copy.bound_scoring_context import _read_publication
    from arblab.hyperliquid_copy.contracts import semantic_hash
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    inputs = receipts.capture_staged(
        values[4],
        values[5],
        values[6],
        values[7],
        days=values[3] if route == "features" else None,
    )
    publication, body = _read_publication(
        resources, inputs["ranking"], "staged_cohort_rankings"
    )
    body = deepcopy(body)
    if fault == "summary":
        body["summary"]["selected_count"] += 1
    elif fault == "policy":
        body["policy"] = "wrong"
    elif fault == "artifact":
        body["ranking"]["sha256"] = "f" * 64
    else:
        source = body["source"]
        provenance = source["inputs"]["provenance"]
        if fault == "window":
            window = (
                provenance["source"]
                if route == "raw"
                else provenance["features"]["source"]
            )
            window["start"] = "2026-08-01T00:00:00+00:00"
        elif fault == "candidate":
            provenance["candidates"] = "f" * 64
        elif fault == "config":
            source["config"]["min_volume"] = 1e12
        elif fault == "bound":
            provenance["max_partition_rows"] = True
        else:
            provenance["engine"] = {"wrong": True}
        source["key"] = semantic_hash(
            {key: value for key, value in source.items() if key != "key"}
        )
    publications = PublishedArtifacts(resources)
    forged = publications.publish(
        "staged_cohort_rankings", body, [publication.artifacts[0].token]
    )
    forged_inputs = dict(query=inputs["query"], ranking=forged.key)
    publications.publish(
        "saved_feature_ranking", forged_inputs, [publication.artifacts[0].token]
    )
    with pytest.raises(ValueError):
        receipts.load(forged_inputs, values[4], values[5], values[6], values[7])


@pytest.mark.parametrize("fault", ["artifact", "receipt"])
def test_final_external_validation_cannot_corrupt_saved_result(
    resources, qualified, days, fault
):
    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    changed = []

    def validate():
        with resources._connect() as db, db:
            pending = db.execute(
                "SELECT 1 FROM allocations WHERE state='pending' LIMIT 1"
            ).fetchone()
            row = db.execute(
                "SELECT key,descriptor FROM publications WHERE json_extract(descriptor,'$.kind')='saved_feature_ranking'"
            ).fetchone()
            if row is not None and pending is None:
                if fault == "artifact":
                    path = resources.root / json.loads(row[1])["artifacts"][0]["path"]
                    with path.open("ab") as stream:
                        stream.write(b"late external validation")
                else:
                    db.execute("DELETE FROM publications WHERE key=?", (row[0],))
                changed.append(True)

    with pytest.raises(ValueError):
        receipts.capture_staged(
            values[4], values[5], values[6], values[7], validation=validate
        )
    assert changed == [True]


def test_invalid_validation_callback_rejects_before_admission(
    resources, qualified, days
):
    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    before = resources.audit()
    with pytest.raises(ValueError, match="validation"):
        receipts.capture_staged(
            values[4], values[5], values[6], values[7], validation=7
        )
    assert resources.audit() == before


@pytest.mark.parametrize("kind", ["staged_cohort_rankings", "saved_feature_ranking"])
def test_direct_staged_load_guards_catalogue_through_last_callback(
    resources, qualified, days, monkeypatch, kind
):
    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    inputs = receipts.capture_staged(values[4], values[5], values[6], values[7])
    original, calls = SavedFeatureRankings._verify, 0

    def mutate(self):
        nonlocal calls
        original(self)
        calls += 1
        if calls == 2:
            with resources._connect() as db, db:
                db.execute(
                    "DELETE FROM publications WHERE json_extract(descriptor,'$.kind')=?",
                    (kind,),
                )

    monkeypatch.setattr(SavedFeatureRankings, "_verify", mutate)
    with pytest.raises(ValueError):
        receipts.load(inputs, values[4], values[5], values[6], values[7])


def test_exact_staged_hit_still_honours_external_validation(resources, qualified, days):
    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    receipts.capture_staged(values[4], values[5], values[6], values[7])
    before = resources.audit()

    def invalid():
        raise ValueError("registered inputs changed")

    with pytest.raises(ValueError, match="registered inputs"):
        receipts.capture_staged(
            values[4], values[5], values[6], values[7], validation=invalid
        )
    assert resources.audit() == before


@pytest.mark.parametrize("method", ["capture", "capture_raw"])
def test_staged_policy_rejects_permanent_intermediate_capture(
    resources, qualified, days, method
):
    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    before = resources.audit()
    with pytest.raises(ValueError, match="policy"):
        if method == "capture":
            receipts.capture(values[3], values[4], values[5], values[6], values[7])
        else:
            receipts.capture_raw(values[4], values[5], values[6], values[7])
    assert resources.audit() == before


@pytest.mark.parametrize("kind", ["candidate_history", "qualified_features_day"])
def test_last_cleanup_validation_cannot_change_consumed_inputs(
    resources, qualified, days, kind
):
    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    calls = 0

    def validate():
        nonlocal calls
        calls += 1
        if calls == 6:  # Last callback inside first owned unlink guard.
            with resources._connect() as db:
                row = db.execute(
                    "SELECT descriptor FROM publications WHERE json_extract(descriptor,'$.kind')=? ORDER BY key",
                    (kind,),
                ).fetchone()
            assert row is not None
            path = resources.root / (
                days[1].publication.artifacts[0].path
                if kind == "qualified_features_day"
                else json.loads(row[0])["artifacts"][0]["path"]
            )
            with path.open("ab") as stream:
                stream.write(b"late consumed input mutation")

    with pytest.raises(ValueError):
        receipts.capture_staged(
            values[4],
            values[5],
            values[6],
            values[7],
            days=values[3],
            validation=validate,
        )
    # All successful temporary removals must be prevented by the final guard.
    assert len(list((resources.root / "staging").glob("*.parquet"))) == 2


def test_wrong_prepared_scope_rejects_before_staging_admission(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import ranking_staging_sources as sources

    values = args(resources, qualified, days)
    receipts = SavedFeatureRankings(
        resources, QualifiedSourceSession(qualified), staging_policy=POLICY
    )
    original = sources.prepare_staging_source

    def wrong(*args, **kwargs):
        args = list(args)
        args[5] = None
        return original(*args, **kwargs)

    monkeypatch.setattr(sources, "prepare_staging_source", wrong)
    with pytest.raises(ValueError):
        receipts.capture_staged(values[4], values[5], values[6], values[7])
    assert resources.audit()["reserved_bytes"] == 0
    assert not list((resources.root / "staging").iterdir())
