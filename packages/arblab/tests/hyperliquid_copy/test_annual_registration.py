import json
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.archive_job import ArchiveJob
from .test_archive_job import inputs


def test_registration_json_preserves_numeric_configuration(tmp_path):
    from arblab.hyperliquid_copy.annual_registration import _json_file

    config = {"weight": 0.2, "enabled": True, "count": 5}
    path = tmp_path / "manifest.json"
    _json_file(path, {"default_config": config})
    assert json.loads(path.read_bytes())["default_config"] == config


def qualified(root, start="2026-08-02", end="2026-08-03"):
    from arblab.hyperliquid_copy.annual_registration import completed_job_inputs

    return completed_job_inputs(root, start=start, end=end)


def test_partial_qualified_prefix_cannot_be_registered(tmp_path):
    inventory, source, total, daily = inputs(tmp_path, days=3)
    job = ArchiveJob.create(
        inventory,
        tmp_path / "job",
        ["BTC"],
        max_download_bytes=total,
        max_batch_bytes=daily,
    )
    job.run_next(source=source)
    calls = len(source.gets)
    with pytest.raises(ValueError, match="complete|unfinished"):
        qualified(job.store.root)
    assert len(source.gets) == calls


def test_completed_job_provenance_survives_raw_cleanup(tmp_path):
    inventory, source, total, _ = inputs(tmp_path, days=3)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    run = job.run_next(source=source)
    assert run["completed_batches"] == run["batch_count"]
    calls = len(source.gets)
    result = qualified(job.store.root)
    assert result["source_start"] == "2026-08-01"
    assert result["source_end"] == "2026-08-04"
    assert result["coins"] == ["BTC"]
    assert len(result["files"]) == 3
    assert len(result["source_keys"]) == 24
    assert len(result["padding_source_keys"]) == 48
    assert result["qualification"] == run["qualification_report"]
    assert len(source.gets) == calls
    with pytest.raises(ValueError, match="padding"):
        qualified(job.store.root, start="2026-08-01")
    path = Path(result["qualification"]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="identity|changed"):
        qualified(job.store.root)


def publication_inputs(tmp_path):
    from dataclasses import replace
    from arblab.hyperliquid_copy.download import file_hash
    from .test_proxy_market_bundle import segment, bundle
    from .test_proxy_weekly import weekly_config

    inventory, source, total, _ = inputs(tmp_path, days=4)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", ["BTC"], max_download_bytes=total
    )
    job.run_next(source=source)
    pins = {}
    for kind in ("prices", "funding"):
        original = segment(tmp_path, kind, kind, range(-48, 48))
        path = bundle(tmp_path, kind, [original], begin=-48, end=48)
        pins[kind] = dict(path=str(path), sha256=file_hash(path))
    config = weekly_config(end="2026-08-04")
    config = replace(config, trader=replace(config.trader, lookback_days=1))
    target = tmp_path / "datasets" / "registered"
    args = (job.store.root, pins["prices"], pins["funding"], target)
    return job, pins, config, target, args


def test_published_dataset_reopens_and_runs_without_staging_cache_paths(tmp_path):
    from arblab.hyperliquid_copy.annual_registration import register_annual_dataset
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
    from arblab.hyperliquid_copy.download import file_hash

    job, pins, config, target, args = publication_inputs(tmp_path)
    path = register_annual_dataset(
        *args, config=config, name="Synthetic publication integration"
    )
    assert path == target / "manifest.json"
    assert not (target / ".activity_checkpoints").exists()
    manifest = ProxyDatasetManifest(target)
    assert manifest.validation_mode == "sharded_v1"
    for kind in ("prices", "funding"):
        filename = "price_source.json" if kind == "prices" else "funding_source.json"
        assert file_hash(target / filename) == pins[kind]["sha256"]
    report = job.store.records()[0, "qualified"]
    assert (target / "activity_qualification.json").read_bytes() == Path(
        report["path"]
    ).read_bytes()
    assert manifest.metadata["default_config"] == config.to_dict()
    with manifest.load(
        temp_root=tmp_path, expected_hash=manifest.identity, config=config
    ) as loaded:
        run = run_configured_proxy(loaded, config)
        assert (
            sum(
                e["rows"]
                for e in manifest.metadata["files"]
                if e["name"].startswith("fills-")
            )
            == 96
        )
    assert len(next(iter(run.result.strategies.values())).equity) == 25
    assert (target / ".activity_checkpoints").is_dir()
    with pytest.raises(ValueError, match="exists"):
        register_annual_dataset(*args, config=config, name="No overwrite")
    sidecar = target / "price_source.json"
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="provenance"):
        manifest.verify()


def test_late_validation_failure_leaves_no_published_dataset_or_staging(
    tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import annual_registration as module

    job, _, config, target, args = publication_inputs(tmp_path)
    records = job.store.records()

    def reject(*args, **kwargs):
        raise ValueError("Injected late coverage rejection")

    monkeypatch.setattr(module, "validate_proxy_run", reject)
    with pytest.raises(ValueError, match="Injected late coverage"):
        module.register_annual_dataset(*args, config=config, name="Must not publish")
    assert not target.exists()
    assert list(target.parent.iterdir()) == []
    assert job.store.records() == records


@pytest.mark.parametrize("fault", [None, "changed", "missing", "binding", "inventory"])
def test_loader_verifies_registered_provenance_copies(tmp_path, fault):
    from arblab.hyperliquid_copy.download import file_hash
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from .test_proxy_dataset import registered

    metadata = registered(tmp_path)
    names = (
        "price_source.json",
        "funding_source.json",
        "activity_qualification.json",
        "activity_source.json",
    )
    inventory = {}
    for name in names:
        path = tmp_path / name
        path.write_text(json.dumps({"fixture": name}))
        inventory[name] = file_hash(path)
    metadata.update(
        registration_engine={"fixture": "a" * 64}, registration_provenance=inventory
    )
    metadata["price_source_manifest_hash"] = inventory["price_source.json"]
    metadata["funding_source_manifest_hash"] = inventory["funding_source.json"]
    metadata["activity_provenance"]["source_manifest_hash"] = inventory[
        "activity_qualification.json"
    ]
    sidecar = tmp_path / "activity_source.json"
    if fault == "changed":
        sidecar.write_text("{}")
    elif fault == "missing":
        sidecar.unlink()
    elif fault == "binding":
        metadata["price_source_manifest_hash"] = "a" * 64
    elif fault == "inventory":
        metadata.pop("registration_provenance")
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    if fault:
        with pytest.raises(ValueError, match="provenance"):
            ProxyDatasetManifest(tmp_path).verify()
    else:
        ProxyDatasetManifest(tmp_path).verify()
