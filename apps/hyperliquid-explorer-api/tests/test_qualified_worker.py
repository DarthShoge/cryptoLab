"""Qualified archive publication through catalogue, workers and saved evidence."""

from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest


@pytest.mark.parametrize(
    "cache_level", [8, 16, 64], ids=["legacy_8gib", "approved_16gib", "approved_64gib"]
)
@pytest.mark.parametrize("feature_policy", [None, "rolling_feature_anchor_v1"])
@pytest.mark.parametrize("staging_policy", [None, "bounded_ranking_staging_v1"])
@pytest.mark.parametrize("verification_policy", ["full", "guarded-session-v1"])
def test_qualified_saved_weekly_daily_and_preview_reuse_cache(
    tmp_path,
    monkeypatch,
    cache_level,
    feature_policy,
    staging_policy,
    verification_policy,
):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "packages/arblab/tests"))
    from hyperliquid_copy.test_annual_registration import publication_inputs
    from arblab.hyperliquid_copy.annual_registration import register_annual_dataset
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from hyperliquid_explorer_api.lab_datasets import DatasetCatalog
    from hyperliquid_explorer_api.lab_jobs import LabJobs
    from hyperliquid_explorer_api.lab_models import Submission, Preview
    from hyperliquid_explorer_api.lab_worker import execute
    from hyperliquid_explorer_api.lab_queries import universe, compare
    from hyperliquid_explorer_api.repository import Repository

    job, _, config, target, args = publication_inputs(tmp_path)
    cache = tmp_path / "derived"
    cache.mkdir()
    identity = "qualified-worker-fixture"
    reference = dict(path=str(cache), identity=identity)
    with CacheLease(cache) as lease:
        resources = CacheResources.create(lease, identity)
        if cache_level >= 16:
            from arblab.hyperliquid_copy.derived_cache_expansion import (
                prepare_expansion,
                apply_expansion,
            )
            from arblab.hyperliquid_copy.derived_cache_policy import (
                open_expanded_cache,
            )

            receipt = prepare_expansion(
                resources,
                approved_from=8 * 1024**3,
                approved_to=16 * 1024**3,
                approval="worker test approval",
            )
            apply_expansion(lease, identity, receipt)
            reference["expansion_receipt"] = receipt
        if cache_level == 64:
            from arblab.hyperliquid_copy.derived_cache_expansion_64 import (
                apply_expansion_64,
                prepare_expansion_64,
            )

            resources = open_expanded_cache(lease, identity, receipt)
            receipt_64 = prepare_expansion_64(
                resources,
                receipt,
                approved_from=16 * 1024**3,
                approved_to=64 * 1024**3,
                approval="worker test 64-GiB approval",
            )
            apply_expansion_64(lease, identity, receipt, receipt_64)
            reference["expansion_64gib_receipt"] = receipt_64
    register_annual_dataset(
        *args,
        config=config,
        name="Qualified worker fixture",
        cache_reference=reference,
        feature_history_policy=feature_policy,
        ranking_staging_policy=staging_policy,
    )
    from arblab.hyperliquid_copy import lab_pipeline_proxy
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    expected_funding_starts = ProxyDatasetManifest(target).funding_starts
    simulation_funding_starts = []
    original_simulate_proxy = lab_pipeline_proxy.simulate_proxy

    def capture_simulation_funding_starts(*args, **kwargs):
        simulation_funding_starts.append(kwargs.get("funding_starts"))
        return original_simulate_proxy(*args, **kwargs)

    monkeypatch.setattr(
        lab_pipeline_proxy, "simulate_proxy", capture_simulation_funding_starts
    )
    records = job.store.records()
    monkeypatch.setattr(
        ProxyActivity, "__init__", lambda *a, **k: pytest.fail("full reader opened")
    )
    catalog = DatasetCatalog(tmp_path)
    assert catalog.list()[0]["available"]
    import duckdb

    with monkeypatch.context() as guard:
        guard.setattr(
            duckdb, "connect", lambda *a, **k: pytest.fail("preflight opened DuckDB")
        )
        preflight = catalog.inspect("registered", config)
    assert preflight["ready"]
    assert any("source-origin" in note for note in preflight["estimate_notes"])
    jobs = LabJobs(tmp_path, verification_policy=verification_policy)
    saved = []
    baseline = None
    for kind, cadence in [
        ("backtest", "weekly"),
        ("backtest", "daily"),
        ("cohort_preview", "weekly"),
    ]:
        matched = replace(
            config,
            rebalance=cadence,
            trader=replace(config.trader, reselection=cadence),
            market_universe=replace(config.market_universe, reselection=cadence),
        )
        body = dict(
            name=f"Qualified {cadence} {kind}",
            dataset_id="registered",
            config=matched.to_dict(),
        )
        request = (
            Preview(**body, decision_date=config.start, scope="BTC")
            if kind == "cohort_preview"
            else Submission(**body)
        )
        item = jobs.submit(request, kind=kind)
        jobs.store.transition(item["id"], "queued", "running")
        execute(tmp_path, item["id"])
        run_id, hashes = jobs.publish(jobs.store.get(item["id"]))
        jobs.store.transition(
            item["id"], "running", "completed", run_id=run_id, artifact_hashes=hashes
        )
        saved.append(jobs.store.get(item["id"]))
        directory = tmp_path / "results" / item["id"]
        rankings = pq.read_table(directory / "rankings.parquet")
        assert rankings.num_rows > 0
        assert (
            catalog.inspect("registered", matched)["estimates"]["ranking_rows"]
            >= rankings.num_rows
        )
        if baseline is None:
            baseline = rankings
        else:
            assert rankings.equals(baseline)
        page = universe(jobs, item["id"], "rankings", page_size=1)
        assert page.total == rankings.num_rows and len(page.rows) == 1
        if kind == "backtest":
            report = tmp_path / "reports" / run_id
            assert pq.read_metadata(report / "equity_curve.parquet").num_rows == 25
            assert (report / "trader_scores.parquet").read_bytes() == (
                directory / "rankings.parquet"
            ).read_bytes()
        assert not (tmp_path / "scratch" / item["id"]).exists()
        with CacheLease(cache) as lease:
            if cache_level == 64:
                from arblab.hyperliquid_copy.derived_cache_policy_64 import (
                    open_expanded_64_cache,
                )

                audit = open_expanded_64_cache(
                    lease, identity, receipt, receipt_64
                ).audit()
            elif cache_level == 16:
                audit = open_expanded_cache(lease, identity, receipt).audit()
            else:
                audit = CacheResources(lease, identity).audit()
        if len(saved) == 1:
            first_audit = audit
        else:
            assert audit == first_audit
    assert len({item["id"] for item in saved}) == 3
    assert all(item["status"] == "completed" for item in saved)
    # Each saved backtest runs strategy, BTC benchmark and cash in that order.
    assert simulation_funding_starts == [
        expected_funding_starts,
        {"BTC": expected_funding_starts["BTC"]},
        None,
        expected_funding_starts,
        {"BTC": expected_funding_starts["BTC"]},
        None,
    ]
    comparison = compare(
        jobs,
        Repository(tmp_path / "external_reports", lab_root=tmp_path),
        [item["id"] for item in saved[:2]],
        "usd",
    )
    assert len(comparison.series) == 2
    assert [series.config["rebalance"] for series in comparison.series] == [
        "weekly",
        "daily",
    ]
    assert all(len(series.curve.rows) == 25 for series in comparison.series)
    assert job.store.records() == records
    assert not (target / ".activity_checkpoints").exists()
