from dataclasses import asdict

import pytest


def test_saved_identity_annotations_and_recovery(tmp_path):
    from hyperliquid_explorer_api.lab_store import ExperimentStore
    from arblab.hyperliquid_copy.lab_config import LabConfig

    store = ExperimentStore(tmp_path)
    item = store.create(
        "example", "dataset", LabConfig().to_dict(), {"dataset_hash": "abc"}
    )
    assert item["status"] == "queued"
    original = item["config_hash"]
    store.annotate(item["id"], "New name", "Notes")
    again = ExperimentStore(tmp_path).get(item["id"])
    assert again["name"] == "New name" and again["config_hash"] == original
    assert again["config"] == item["config"]
    store.transition(item["id"], "queued", "running")
    store.recover()
    assert store.get(item["id"])["status"] == "failed"
    assert "Interrupted" in store.get(item["id"])["error"]
    with pytest.raises(ValueError):
        store.transition(item["id"], "running", "completed")


def test_queue_cap_and_new_ids(tmp_path):
    from hyperliquid_explorer_api.lab_store import ExperimentStore
    from arblab.hyperliquid_copy.lab_config import LabConfig

    store = ExperimentStore(tmp_path)
    ids = [
        store.create("Run", "dataset", LabConfig().to_dict(), {})["id"]
        for _ in range(32)
    ]
    assert len(set(ids)) == 32
    with pytest.raises(ValueError, match="32"):
        store.create("Run", "dataset", LabConfig().to_dict(), {})


def test_dataset_hashes_scope_and_resource_gate(tmp_path):
    from hyperliquid_explorer_api.lab_datasets import DatasetCatalog
    from arblab.hyperliquid_copy.lab_fixture import write_fixture
    from arblab.hyperliquid_copy.lab_config import LabConfig

    write_fixture(tmp_path / "datasets" / "demo")
    catalog = DatasetCatalog(tmp_path)
    info = catalog.list()[0]
    assert info["synthetic"] and info["available"]
    config = LabConfig.from_dict(info["default_config"])
    frozen = catalog.preflight("demo", config)
    assert frozen["dataset_hash"]
    with pytest.raises(ValueError, match="lookback"):
        catalog.preflight(
            "demo", LabConfig.from_dict(config.to_dict() | {"lookback_days": 90})
        )
    with pytest.raises(ValueError):
        catalog.preflight("../demo", config)
    path = tmp_path / "datasets" / "demo" / "fills.parquet"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        catalog.preflight("demo", config)
