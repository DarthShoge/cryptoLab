from dataclasses import replace
from pathlib import Path
import json

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
from .test_qualified_registration import registered_source, resources


def test_registered_qualified_mode_runs_and_reopens_at_final_path(
    registered_source, resources, tmp_path, monkeypatch
):
    fixture, config = registered_source
    root, identity = resources.root, resources.identity
    resources.lease.__exit__(None, None, None)
    moved = fixture.directory.with_name("final-renamed")
    fixture.directory.rename(moved)
    manifest = ProxyDatasetManifest(moved)
    assert manifest.validation_mode == "qualified_v1"
    report = json.loads((moved / "activity_qualification.json").read_text())

    with manifest.load(
        temp_root=tmp_path, expected_hash=manifest.identity, config=config
    ) as loaded:
        with ProxyActivity(
            [Path(row["path"]) for row in report["files"]], temp_root=tmp_path
        ) as ref:
            expected = run_configured_proxy(replace(loaded, activity=ref), config)
    monkeypatch.setattr(
        ProxyActivity,
        "__init__",
        lambda *a, **k: pytest.fail("registered qualified load opened full reader"),
    )
    with manifest.load(
        temp_root=tmp_path, expected_hash=manifest.identity, config=config
    ) as loaded:
        actual = run_configured_proxy(
            loaded, config, rankings_path=tmp_path / "weekly.parquet"
        )
    assert actual.result.cohorts == expected.result.cohorts
    assert actual.result.signals == expected.result.signals
    assert actual.result.strategies == expected.result.strategies
    assert len(actual.result.scores) == len(expected.result.scores)
    with CacheLease(root) as lease:
        before = CacheResources(lease, identity).audit()
    daily = replace(
        config,
        rebalance="daily",
        trader=replace(config.trader, reselection="daily"),
        market_universe=replace(config.market_universe, reselection="daily"),
    )
    with manifest.load(
        temp_root=tmp_path, expected_hash=manifest.identity, config=daily
    ) as loaded:
        comparison = run_configured_proxy(
            loaded, daily, rankings_path=tmp_path / "daily.parquet"
        )
    assert comparison.result.cohorts == actual.result.cohorts
    assert pq.read_table(comparison.result.scores.path).equals(
        pq.read_table(actual.result.scores.path)
    )
    with CacheLease(root) as lease:
        assert CacheResources(lease, identity).audit() == before
    assert not (moved / ".activity_checkpoints").exists()


def test_qualified_manifest_requires_scheduled_configuration(
    registered_source, resources, tmp_path
):
    from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxy

    fixture, config = registered_source
    manifest = ProxyDatasetManifest(fixture.directory)
    raw = config.to_dict()
    raw.pop("schema_version")
    raw.pop("rebalance")
    with pytest.raises(ValueError, match="scheduled"):
        manifest.load(
            temp_root=tmp_path,
            expected_hash=manifest.identity,
            config=LabConfigProxy(**raw),
        )
