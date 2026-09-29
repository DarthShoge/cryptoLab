from dataclasses import replace

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
from .test_candidate_day import resources
from .test_feature_window import days
from .test_qualified_day import qualified
from .test_feature_metric_producer import args


def rows(result):
    return [row for batch in result.iter_batches() for row in batch]


def test_capture_reuses_payload_and_survives_logical_intermediate_retirement(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_metric_producer as producer
    from arblab.hyperliquid_copy.saved_feature_rankings import SavedFeatureRankings

    values = args(resources, qualified, days)
    expected = producer.build_and_score_features(*values)
    expected_rows, expected_selected = rows(expected), expected.selected
    before = resources.audit()
    store = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    query = values[4:]
    receipt = store.capture(values[3], *query)
    publication = PublishedArtifacts(resources).lookup("saved_feature_ranking", receipt)
    assert publication.artifacts == expected.publication.artifacts
    assert resources.audit() == before
    assert rows(store.load(receipt, *query)) == expected_rows

    def forbidden(*a, **kw):
        pytest.fail("saved ranking reuse attempted producer sorting/replay")

    monkeypatch.setattr(producer, "stage_metric_summaries", forbidden)
    weekly = (query[0], replace(query[1], reselection="weekly"), *query[2:])
    assert store.capture(values[3], *weekly) == receipt
    assert resources.audit() == before
    with resources._connect() as db, db:
        assert (
            db.execute(
                "SELECT count(*) FROM publications WHERE json_extract(descriptor,'$.kind')='saved_feature_ranking'"
            ).fetchone()[0]
            == 1
        )
        # Fixture-only logical retirement. Payloads remain accounted and intact.
        # This tests read independence, NOT a physical deletion protocol.
        db.execute(
            "DELETE FROM publications WHERE json_extract(descriptor,'$.kind') <> 'saved_feature_ranking'"
        )
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)
    monkeypatch.setattr(producer, "build_and_score_features", forbidden)
    with CacheLease(root) as lease:
        reopened = CacheResources(lease, identity)
        restored = SavedFeatureRankings(reopened, QualifiedSourceSession(qualified))
        actual = restored.load(receipt, *weekly)
        assert rows(actual) == expected_rows
        assert actual.selected == expected_selected
        assert actual.bound_decision == query[0]
        assert actual.bound_scope == query[2]
        assert reopened.audit() == before


@pytest.mark.parametrize(
    "fault",
    [
        "decision",
        "scope",
        "metrics",
        "selection",
        "semantics",
        "receipt",
        "source",
        "ranking",
    ],
)
def test_saved_rankings_reject_changed_request_or_evidence(
    qualified, resources, days, fault
):
    from arblab.hyperliquid_copy.saved_feature_rankings import SavedFeatureRankings
    from datetime import timedelta
    from pathlib import Path
    import json

    values = args(resources, qualified, days)
    store = SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    decision, config, scope, semantics = values[4:]
    receipt = store.capture(values[3], decision, config, scope, semantics)
    if fault == "decision":
        decision += timedelta(days=1)
    elif fault == "scope":
        scope = None
    elif fault == "metrics":
        config = replace(
            config, metric_weights={"pnl_efficiency": 1}, metric_directions={}
        )
    elif fault == "selection":
        config = replace(config, top_n=1)
    elif fault == "semantics":
        semantics = "net_includes_fee"
    elif fault == "receipt":
        receipt["unexpected"] = True
    else:
        if fault == "source":
            path = Path(
                json.loads(Path(qualified["path"]).read_text())["files"][0]["path"]
            )
        else:
            publication = PublishedArtifacts(resources).lookup(
                "saved_feature_ranking", receipt
            )
            path = resources.root / publication.artifacts[0].path
        with path.open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(ValueError):
        store.load(receipt, decision, config, scope, semantics)


@pytest.mark.parametrize("fault", ["caller", "engine", "lease", "source", "ranking"])
def test_saved_rankings_recheck_after_final_publication_lookup(
    qualified, resources, days, monkeypatch, fault
):
    from arblab.hyperliquid_copy import saved_feature_rankings as module
    from pathlib import Path
    import json

    values = args(resources, qualified, days)
    store = module.SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    receipt = store.capture(values[3], *values[4:])
    original, calls = PublishedArtifacts.lookup, []

    def lookup(publications, kind, inputs):
        result = original(publications, kind, inputs)
        if kind == "saved_feature_ranking":
            calls.append(kind)
            if len(calls) == 2:
                if fault == "caller":
                    receipt["ranking"] = "0" * 64
                elif fault == "engine":
                    monkeypatch.setattr(module, "_engine", lambda: "changed")
                elif fault == "lease":
                    resources.lease.__exit__(None, None, None)
                elif fault == "source":
                    path = Path(
                        json.loads(Path(qualified["path"]).read_text())["files"][0][
                            "path"
                        ]
                    )
                    with path.open("ab") as handle:
                        handle.write(b"late source change")
                else:
                    path = resources.root / result.artifacts[0].path
                    with path.open("ab") as handle:
                        handle.write(b"late ranking change")
        return result

    monkeypatch.setattr(PublishedArtifacts, "lookup", lookup)
    with pytest.raises(ValueError):
        store.load(receipt, *values[4:])
    assert len(calls) == 2


def test_saved_rankings_recheck_source_after_final_engine_hash(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import saved_feature_rankings as module
    from pathlib import Path
    import json

    values = args(resources, qualified, days)
    store = module.SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    receipt = store.capture(values[3], *values[4:])
    original, calls = module._engine, []

    def engine():
        result = original()
        calls.append(True)
        if len(calls) == 2:
            path = Path(
                json.loads(Path(qualified["path"]).read_text())["files"][0]["path"]
            )
            with path.open("ab") as handle:
                handle.write(b"late source change")
        return result

    monkeypatch.setattr(module, "_engine", engine)
    with pytest.raises(ValueError):
        store.load(receipt, *values[4:])
    assert len(calls) == 2


def test_saved_rankings_recheck_engine_after_final_source_verification(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import saved_feature_rankings as module

    values = args(resources, qualified, days)
    store = module.SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    receipt = store.capture(values[3], *values[4:])
    original, calls = QualifiedSourceSession.verify, []

    def verify(source):
        original(source)
        calls.append(True)
        if len(calls) == 2:
            monkeypatch.setattr(module, "_engine", lambda: "changed")

    monkeypatch.setattr(QualifiedSourceSession, "verify", verify)
    with pytest.raises(ValueError, match="engine"):
        store.load(receipt, *values[4:])
    assert len(calls) == 2


def test_capture_rejects_substituted_producer_artifact(
    qualified, resources, days, monkeypatch
):
    from arblab.hyperliquid_copy import saved_feature_rankings as module

    values = args(resources, qualified, days)
    result = module.producer.build_and_score_features(*values)
    forged = replace(
        result, artifact=replace(result.artifact, rows=result.artifact.rows + 1)
    )
    monkeypatch.setattr(module.producer, "build_and_score_features", lambda *a: forged)
    store = module.SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    with pytest.raises(ValueError, match="artifact mismatch"):
        store.capture(values[3], *values[4:])
    with resources._connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM publications WHERE json_extract(descriptor,'$.kind')='saved_feature_ranking'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("fault", ["engine", "source_pin"])
def test_new_store_cannot_adopt_receipt_from_another_source_or_engine(
    qualified, resources, days, tmp_path, monkeypatch, fault
):
    from arblab.hyperliquid_copy import saved_feature_rankings as module
    import shutil

    values = args(resources, qualified, days)
    store = module.SavedFeatureRankings(resources, QualifiedSourceSession(qualified))
    receipt = store.capture(values[3], *values[4:])
    pin = dict(qualified)
    if fault == "engine":
        monkeypatch.setattr(module, "_engine", lambda: "changed")
    else:
        copied = tmp_path / "copied_report" / "manifest.json"
        copied.parent.mkdir()
        shutil.copyfile(pin["path"], copied)
        pin["path"] = str(copied)
    other = module.SavedFeatureRankings(resources, QualifiedSourceSession(pin))
    with pytest.raises(ValueError, match="receipt/query mismatch"):
        other.load(receipt, *values[4:])
