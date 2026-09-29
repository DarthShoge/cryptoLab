from dataclasses import asdict
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.download import file_hash
from .test_lab_pipeline_proxy import dataset, config


def scheduled_config():
    from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxyScheduled

    data = config().to_dict()
    data.pop("schema_version")
    data["trader"]["reselection"] = "weekly"
    data["market_universe"]["reselection"] = "weekly"
    return LabConfigProxyScheduled(**data)


def test_sharded_registration_matches_pipeline_and_reuses_verified_seed(
    tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
    import arblab.hyperliquid_copy.registered_activity as bridge

    metadata = registered(tmp_path)
    original = ProxyDatasetManifest(tmp_path)
    c = scheduled_config()
    with original.load(temp_root=tmp_path, expected_hash=original.identity) as loaded:
        expected = run_configured_proxy(loaded, c)
    metadata["validation_mode"] = "sharded_v1"
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    manifest = ProxyDatasetManifest(tmp_path)
    calls = []
    original_qualify = bridge.qualified_seed

    def qualify(*args, **kwargs):
        calls.append(True)
        return original_qualify(*args, **kwargs)

    monkeypatch.setattr(bridge, "qualified_seed", qualify)
    for _ in range(2):
        with manifest.load(
            temp_root=tmp_path, expected_hash=manifest.identity, config=c
        ) as loaded:
            assert run_configured_proxy(loaded, c) == expected
            assert (
                "sharded_validation.py"
                in loaded.activity.store.metadata(loaded.activity.identity)["engine"][
                    "code"
                ]
            )
    assert len(calls) == 1


def test_sharded_dataset_refuses_legacy_materializing_load(tmp_path):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    metadata = registered(tmp_path) | {"validation_mode": "sharded_v1"}
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    manifest = ProxyDatasetManifest(tmp_path)
    for c in (None, config()):
        with pytest.raises(ValueError, match="scheduled"):
            manifest.load(temp_root=tmp_path, expected_hash=manifest.identity, config=c)


def test_only_explicit_sharded_mode_admits_larger_corpus(tmp_path, monkeypatch):
    import arblab.hyperliquid_copy.proxy_dataset as module

    metadata = registered(tmp_path)
    monkeypatch.setattr(module, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="byte ceiling"):
        module.ProxyDatasetManifest(tmp_path)
    metadata["validation_mode"] = "sharded_v1"
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    module.ProxyDatasetManifest(tmp_path)
    monkeypatch.setattr(module, "MAX_CORPUS_BYTES", 1)
    with pytest.raises(ValueError, match="byte ceiling"):
        module.ProxyDatasetManifest(tmp_path)


def test_sharded_qualification_includes_empty_source_partition_types(tmp_path):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    metadata = registered(tmp_path) | {"validation_mode": "sharded_v1"}
    empty = pq.read_table(tmp_path / "fills-0000.parquet").slice(0, 0)
    empty = empty.set_column(
        empty.schema.get_field_index("tid"), "tid", pa.array([], type=pa.float64())
    )
    path = tmp_path / "fills-empty.parquet"
    pq.write_table(empty, path)
    metadata["files"].append(
        dict(name=path.name, bytes=path.stat().st_size, rows=0, sha256=file_hash(path))
    )
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    manifest = ProxyDatasetManifest(tmp_path)
    with pytest.raises(ValueError, match="identity types"):
        manifest.load(
            temp_root=tmp_path,
            expected_hash=manifest.identity,
            config=scheduled_config(),
        )
    assert not list((tmp_path / ".activity_checkpoints").glob("*/seeds.parquet"))


def test_sharded_future_invalid_fill_does_not_publish_checkpoint(tmp_path):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    metadata = registered(tmp_path) | {"validation_mode": "sharded_v1"}
    path = tmp_path / "fills-0000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[-1]["post_position"] = 999
    pq.write_table(pa.Table.from_pylist(rows), path)
    for e in metadata["files"]:
        if e["name"] == path.name:
            e.update(bytes=path.stat().st_size, sha256=file_hash(path))
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    manifest = ProxyDatasetManifest(tmp_path)
    with pytest.raises(ValueError, match="economic"):
        manifest.load(
            temp_root=tmp_path,
            expected_hash=manifest.identity,
            config=scheduled_config(),
        )
    assert not list((tmp_path / ".activity_checkpoints").glob("*/seeds.parquet"))


def test_scheduled_registration_uses_reusable_checkpoint_reader(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from arblab.hyperliquid_copy.scheduled_activity import ScheduledActivity
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    registered(tmp_path)
    manifest = ProxyDatasetManifest(tmp_path)
    c = scheduled_config()
    with manifest.load(temp_root=tmp_path, expected_hash=manifest.identity) as full:
        expected = run_configured_proxy(full, c)
    with manifest.load(
        temp_root=tmp_path, expected_hash=manifest.identity, config=c
    ) as loaded:
        assert isinstance(loaded.activity, ScheduledActivity)
        identity = loaded.activity.identity
        assert run_configured_proxy(loaded, c) == expected
    import arblab.hyperliquid_copy.registered_activity as bridge

    def unexpected_build(*args, **kwargs):
        raise AssertionError("Reusable cache rebuilt the complete fill reader")

    monkeypatch.setattr(bridge, "build_initial", unexpected_build)
    with manifest.load(
        temp_root=tmp_path, expected_hash=manifest.identity, config=c
    ) as loaded:
        assert loaded.activity.identity == identity
        assert run_configured_proxy(loaded, c) == expected
    assert not list(tmp_path.glob("proxy_activity_*"))


def test_scheduled_registration_rejects_corrupt_cached_seed(tmp_path):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    registered(tmp_path)
    manifest = ProxyDatasetManifest(tmp_path)
    c = scheduled_config()
    with manifest.load(
        temp_root=tmp_path, expected_hash=manifest.identity, config=c
    ) as loaded:
        root, identity = loaded.activity.store.root, loaded.activity.identity
    (root / identity / "seeds.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        manifest.load(temp_root=tmp_path, expected_hash=manifest.identity, config=c)


def registered(tmp_path):
    data = dataset(tmp_path)
    data.activity.close()
    (tmp_path / "fills.parquet").rename(tmp_path / "fills-0000.parquet")
    for name, rows in [("bars", data.bars), ("funding", data.funding)]:
        pq.write_table(
            pa.Table.from_pylist([asdict(r) for r in rows]),
            tmp_path / f"{name}.parquet",
        )
    metadata = dict(
        schema="hyperliquid_lab_proxy_dataset_v1",
        name="Synthetic proxy loader check",
        synthetic=True,
        coverage_start="2026-08-01",
        coverage_end="2026-08-04",
        coins=["BTC"],
        fee_semantics="gross_excludes_fee",
        coverage_note="Fabricated integration fixture",
        mappings=[asdict(m) for m in data.mappings.records],
        default_config=config().to_dict(),
        activity_provenance=dict(scope="synthetic", complete=False, source_keys=[]),
        price_source_manifest_hash="a" * 64,
        funding_source_manifest_hash="b" * 64,
        price_policy=dict(
            adjustment="raw", corporate_actions="none_detected", rolls="not_applicable"
        ),
        files=[
            dict(
                name=p.name,
                bytes=p.stat().st_size,
                rows=pq.ParquetFile(p).metadata.num_rows,
                sha256=file_hash(p),
            )
            for p in sorted(tmp_path.glob("*.parquet"))
        ],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    return metadata


def test_registered_proxy_load_freezes_identity_and_keeps_fills_disk_backed(tmp_path):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy

    metadata = registered(tmp_path)
    manifest = ProxyDatasetManifest(tmp_path)
    assert manifest.metadata == metadata
    with manifest.load(temp_root=tmp_path, expected_hash=manifest.identity) as loaded:
        assert not hasattr(loaded, "fills")
        assert loaded.activity.count > 0
        output = run_configured_proxy(loaded, config())
        assert next(iter(output.result.strategies.values())).fills
    assert loaded.activity.db is None
    assert not list(tmp_path.glob("proxy_activity_*"))


def test_registered_proxy_rejects_changed_files_and_manifest_identity(tmp_path):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    registered(tmp_path)
    manifest = ProxyDatasetManifest(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        manifest.load(temp_root=tmp_path, expected_hash="0" * 64)
    path = tmp_path / "bars.parquet"
    table = pq.read_table(path).set_column(2, "end", pq.read_table(path).column("end"))
    pq.write_table(table, path, compression="gzip")
    with pytest.raises(ValueError, match="changed"):
        manifest.load(temp_root=tmp_path, expected_hash=manifest.identity)


@pytest.mark.parametrize(
    "change",
    [
        {"validation_mode": "unknown"},
        {"synthetic": False},
        {"coins": ["BTC", "BTC"]},
        {
            "price_policy": {
                "adjustment": "raw",
                "corporate_actions": "unchecked",
                "rolls": "not_applicable",
            }
        },
    ],
)
def test_invalid_registration_or_unqualified_real_coverage_rejects(tmp_path, change):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    metadata = registered(tmp_path) | change
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        ProxyDatasetManifest(tmp_path)


def test_registered_files_cannot_escape_dataset_directory(tmp_path):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    metadata = registered(tmp_path)
    metadata["files"][0]["name"] = "../bars.parquet"
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="file"):
        ProxyDatasetManifest(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"coin": "ETH"},
        {"px": float("nan")},
        {"post_position": 999},
        {"tid": None},
        {"tid": 0.5},
        {"oid": None},
        {"crossed": None},
        {"source_line": None},
    ],
)
@pytest.mark.parametrize("scheduled", [False, True])
def test_hashes_do_not_replace_fill_semantic_validation(tmp_path, change, scheduled):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    metadata = registered(tmp_path)
    path = tmp_path / "fills-0000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0].update(change)
    pq.write_table(pa.Table.from_pylist(rows), path)
    for entry in metadata["files"]:
        if entry["name"] == path.name:
            entry.update(bytes=path.stat().st_size, sha256=file_hash(path))
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    manifest = ProxyDatasetManifest(tmp_path)
    with pytest.raises(ValueError, match="invalid registered fill"):
        manifest.load(
            temp_root=tmp_path,
            expected_hash=manifest.identity,
            config=scheduled_config() if scheduled else None,
        )
    assert not list(tmp_path.glob("proxy_activity_*"))
