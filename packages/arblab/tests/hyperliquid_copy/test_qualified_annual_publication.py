import json
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.download import file_hash
from .test_annual_registration import publication_inputs


@pytest.mark.parametrize("feature_policy", [None, "rolling_feature_anchor_v1"])
@pytest.mark.parametrize("staging_policy", [None, "bounded_ranking_staging_v1"])
def test_qualified_publication_retains_source_and_reopens_after_rename(
    tmp_path, feature_policy, staging_policy
):
    from arblab.hyperliquid_copy.annual_registration import register_annual_dataset
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    job, _, config, target, args = publication_inputs(tmp_path)
    cache = tmp_path / "derived"
    cache.mkdir()
    with CacheLease(cache) as lease:
        resources = CacheResources.create(lease, "publication-fixture")
        before = resources.audit()
    reference = dict(path=str(cache), identity="publication-fixture")
    records = job.store.records()
    path = register_annual_dataset(
        *args,
        config=config,
        name="Full source annual fixture",
        cache_reference=reference,
        feature_history_policy=feature_policy,
        ranking_staging_policy=staging_policy,
    )
    metadata = json.loads(path.read_text())
    assert metadata["validation_mode"] == "qualified_v1"
    assert metadata["history_policy"] == "qualified_source_seed_v1"
    assert metadata["derived_cache"] == reference
    assert metadata.get("feature_history_policy") == feature_policy
    assert metadata.get("ranking_staging_policy") == staging_policy
    report = json.loads((target / "activity_qualification.json").read_text())
    for index, entry in enumerate(report["files"]):
        assert file_hash(target / f"fills-{index:04d}.parquet") == entry["sha256"]
        assert file_hash(Path(entry["path"])) == entry["sha256"]
    assert not (target / ".activity_checkpoints").exists()
    assert job.store.records() == records
    with CacheLease(cache) as lease:
        current = CacheResources(lease, reference["identity"])
        capacity = json.loads((target / "candidate_capacity.json").read_text())
        retained = capacity["payload"]["bytes"]
        assert current.audit() == dict(
            before,
            retained_bytes=retained,
            total_bytes=before["total_bytes"] + retained,
        )
        with current._connect() as db:
            assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 1
    renamed = target.with_name("renamed")
    target.rename(renamed)
    manifest = ProxyDatasetManifest(renamed)
    with manifest.load(
        temp_root=tmp_path, expected_hash=manifest.identity, config=config
    ) as loaded:
        assert loaded.activity._activity.ranking_staging_policy == staging_policy
        run = run_configured_proxy(
            loaded, config, rankings_path=tmp_path / "rankings.parquet"
        )
    assert len(next(iter(run.result.strategies.values())).equity) == 25


@pytest.mark.parametrize("fault", ["busy", "validation"])
def test_failed_qualified_publication_preserves_cache_and_source(
    tmp_path, monkeypatch, fault
):
    from arblab.hyperliquid_copy import annual_registration as module
    from arblab.hyperliquid_copy.derived_cache_lease import CacheBusyError

    job, _, config, target, args = publication_inputs(tmp_path)
    cache = tmp_path / "derived"
    cache.mkdir()
    identity = "failed-publication-fixture"
    owner = CacheLease(cache)
    owner.__enter__()
    resources = CacheResources.create(owner, identity)
    before = resources.audit()
    records = job.store.records()
    reference = dict(path=str(cache), identity=identity)
    try:
        if fault == "validation":
            owner.__exit__(None, None, None)

            def reject(*args, **kwargs):
                raise ValueError("Injected qualified validation rejection")

            monkeypatch.setattr(module, "validate_proxy_run", reject)
        error = CacheBusyError if fault == "busy" else ValueError
        with pytest.raises(error, match="busy|Injected"):
            module.register_annual_dataset(
                *args, config=config, name="Must not publish", cache_reference=reference
            )
        assert not target.exists()
        assert list(target.parent.iterdir()) == []
        assert job.store.records() == records
        if fault == "busy":
            assert resources.audit() == before
    finally:
        owner.__exit__(None, None, None)
    with CacheLease(cache) as lease:
        current = CacheResources(lease, identity)
        if fault == "busy":
            assert current.audit() == before
        else:
            # A completed capacity derivation remains reusable and charged even
            # if subsequent dataset validation fails; no implicit cache refund.
            from arblab.hyperliquid_copy.candidate_capacity import (
                CandidateCapacity,
                KIND,
            )
            from arblab.hyperliquid_copy.qualified_source_session import (
                QualifiedSourceSession,
            )

            with current._connect() as db:
                descriptors = [
                    json.loads(row[0])
                    for row in db.execute("SELECT descriptor FROM publications")
                ]
            assert len(descriptors) == 1 and descriptors[0]["kind"] == KIND
            inputs = descriptors[0]["inputs"]
            evidence = CandidateCapacity(
                current, QualifiedSourceSession(inputs["source"]["pin"]), inputs
            )
            retained = sum(p.bytes for p in evidence.publication.artifacts)
            assert retained > 0
            assert current.audit() == dict(
                before,
                retained_bytes=retained,
                total_bytes=before["total_bytes"] + retained,
            )
