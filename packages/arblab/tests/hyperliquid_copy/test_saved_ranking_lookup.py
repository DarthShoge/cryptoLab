from dataclasses import replace

import pytest

from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
from arblab.hyperliquid_copy.saved_feature_rankings import SavedFeatureRankings
from arblab.hyperliquid_copy.saved_ranking_lookup import find_receipt_inputs
from .test_candidate_day import resources
from .test_feature_window import days
from .test_qualified_day import qualified
from .test_feature_metric_producer import args
from .test_saved_feature_rankings import rows


def test_find_exact_receipt_without_building_or_writing(
    resources, qualified, days, monkeypatch
):
    values = args(resources, qualified, days)
    store = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    before = resources.audit()
    assert store.find(*values[4:]) is None
    assert resources.audit() == before
    receipt = store.capture(values[3], *values[4:])
    expected = store.load(receipt, *values[4:])
    before = resources.audit()
    from arblab.hyperliquid_copy import feature_metric_producer as producer

    def forbidden(*a, **kw):
        pytest.fail("receipt discovery invoked producer")

    monkeypatch.setattr(producer, "build_and_score_features", forbidden)
    assert rows(store.find(*values[4:])) == rows(expected)
    weekly = (values[4], replace(values[5], reselection="weekly"), *values[6:])
    assert rows(store.find(*weekly)) == rows(expected)
    changed = (values[4], replace(values[5], min_volume=10), *values[6:])
    assert store.find(*changed) is None
    assert resources.audit() == before


def test_find_exact_annual_receipt_with_execution_policy(resources, qualified, days):
    values = args(resources, qualified, days)
    store = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    receipt = store.capture(values[3], *values[4:])
    publication = PublishedArtifacts(resources).lookup("saved_feature_ranking", receipt)
    query = receipt["query"] | {"execution_policy": "annual_bounded_rolling_v1"}
    annual_receipt = {"query": query, "ranking": receipt["ranking"]}
    PublishedArtifacts(resources).publish(
        "saved_feature_ranking", annual_receipt, [publication.artifacts[0].token]
    )

    assert find_receipt_inputs(resources, query) == annual_receipt


@pytest.mark.parametrize(
    "fault", ["duplicate", "digest", "kind", "oversized", "malformed", "payload"]
)
def test_find_does_not_turn_bad_receipts_into_misses(resources, qualified, days, fault):
    values = args(resources, qualified, days)
    store = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    receipt = store.capture(values[3], *values[4:])
    publications = PublishedArtifacts(resources)
    publication = publications.lookup("saved_feature_ranking", receipt)
    if fault == "duplicate":
        publications.publish(
            "saved_feature_ranking",
            receipt | {"ranking": "0" * 64},
            [publication.artifacts[0].token],
        )
    elif fault == "payload":
        with (resources.root / publication.artifacts[0].path).open("ab") as handle:
            handle.write(b"corrupt")
    else:
        with resources._connect() as db, db:
            if fault == "digest":
                db.execute(
                    "UPDATE publications SET sha256=? WHERE key=?",
                    ("0" * 64, publication.key),
                )
            elif fault == "kind":
                raw = db.execute(
                    "SELECT descriptor FROM publications WHERE key=?",
                    (publication.key,),
                ).fetchone()[0]
                raw = raw.replace(
                    '"kind":"saved_feature_ranking"', '"kind":"another_kind"'
                )
                db.execute(
                    "UPDATE publications SET descriptor=? WHERE key=?",
                    (raw, publication.key),
                )
            else:
                raw = (
                    '{"kind":"saved_feature_ranking","padding":"' + "x" * 1024**2 + '"}'
                    if fault == "oversized"
                    else "{invalid"
                )
                db.execute(
                    "UPDATE publications SET descriptor=? WHERE key=?",
                    (raw, publication.key),
                )
    with pytest.raises(ValueError):
        store.find(*values[4:])


def test_find_bounds_metadata_before_any_json_classification(
    resources, qualified, days
):
    from arblab.hyperliquid_copy.derived_publication import MAX_DESCRIPTOR_BYTES

    values = args(resources, qualified, days)
    store = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    receipt = store.capture(values[3], *values[4:])
    publication = PublishedArtifacts(resources).lookup("saved_feature_ranking", receipt)
    with resources._connect() as db, db:
        db.execute(
            "UPDATE publications SET descriptor=? WHERE key=?",
            ("x" * (MAX_DESCRIPTOR_BYTES + 1), publication.key),
        )
    with pytest.raises(ValueError, match="oversized"):
        store.find(*values[4:])


def test_find_guards_ranking_bytes_after_loading(
    resources, qualified, days, monkeypatch
):
    values = args(resources, qualified, days)
    store = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    store.capture(values[3], *values[4:])
    original = SavedFeatureRankings.load

    def load(*a, **kw):
        result = original(*a, **kw)
        with result.artifact.path.open("ab") as handle:
            handle.write(b"late ranking mutation")
        return result

    monkeypatch.setattr(SavedFeatureRankings, "load", load)
    with pytest.raises(ValueError):
        store.find(*values[4:])


@pytest.mark.parametrize("hit", [True, False])
@pytest.mark.parametrize("fault", ["query", "catalogue", "source", "engine", "lease"])
def test_find_rechecks_hit_and_miss_context(
    resources, qualified, days, monkeypatch, hit, fault
):
    from arblab.hyperliquid_copy import saved_feature_rankings as module
    from pathlib import Path
    import json

    values = args(resources, qualified, days)
    store = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    receipt = store.capture(values[3], *values[4:])
    publication = PublishedArtifacts(resources).lookup("saved_feature_ranking", receipt)
    query = list(values[4:])
    if not hit:
        query[1] = replace(query[1], min_volume=10)
    original = module.find_receipt_inputs
    if fault == "catalogue":
        # Some filesystems coalesce metadata timestamps for rapid same-size
        # writes. Catalogue content changes must not depend on stat precision.
        original_identity = module._identity
        fixed = original_identity(resources.path)
        monkeypatch.setattr(
            module,
            "_identity",
            lambda path: fixed if path == resources.path else original_identity(path),
        )

    def find(*a, **kw):
        result = original(*a, **kw)
        assert (result is not None) == hit
        if fault == "query":
            query[1].metric_weights["pnl_efficiency"] = 1
        elif fault == "catalogue":
            with resources._connect() as db, db:
                db.execute(
                    "UPDATE publications SET sha256=? WHERE key=?",
                    ("0" * 64, publication.key),
                )
        elif fault == "source":
            path = Path(
                json.loads(Path(qualified["path"]).read_text())["files"][0]["path"]
            )
            with path.open("ab") as handle:
                handle.write(b"late source mutation")
        elif fault == "engine":
            monkeypatch.setattr(module, "_engine", lambda: "changed")
        else:
            resources.lease.__exit__(None, None, None)
        return result

    monkeypatch.setattr(module, "find_receipt_inputs", find)
    with pytest.raises(ValueError):
        store.find(*query)
