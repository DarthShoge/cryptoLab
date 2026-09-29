import json

from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
from arblab.hyperliquid_copy.derived_cache_resources import CacheResources

from test_lab_proxy import write_proxy_fixture


def test_qualified_mode_requires_scheduled_config_in_preflight(tmp_path):
    from hyperliquid_explorer_api.lab_proxy_datasets import inspect
    from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxy

    directory = tmp_path / "dataset"
    config = write_proxy_fixture(directory)
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    cache = tmp_path / "derived"
    cache.mkdir()
    with CacheLease(cache) as lease:
        CacheResources.create(lease, "preflight-fixture")
    data.update(
        validation_mode="qualified_v1",
        history_policy="qualified_source_seed_v1",
        derived_cache=dict(path=str(cache), identity="preflight-fixture"),
    )
    path.write_text(json.dumps(data))
    assert isinstance(config, LabConfigProxy)
    _, result = inspect(directory, config)
    assert not result["ready"]
    assert "scheduled_config_required" in {issue["code"] for issue in result["issues"]}
    assert "capacity_evidence_required" in {issue["code"] for issue in result["issues"]}
    assert "ranking_rows" not in result["estimates"]
