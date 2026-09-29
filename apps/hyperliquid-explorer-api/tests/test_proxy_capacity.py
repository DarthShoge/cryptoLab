import json
from types import SimpleNamespace
from datetime import timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from test_lab_proxy import write_proxy_fixture


def test_uncovered_count_interval_fails_closed(tmp_path, monkeypatch):
    from hyperliquid_explorer_api import lab_proxy_datasets as module
    from arblab.hyperliquid_copy.lab_config import day

    directory = tmp_path / "dataset"
    config = write_proxy_fixture(directory)
    evidence = SimpleNamespace(
        source=SimpleNamespace(
            origin=day(config.start) + timedelta(days=1), finish=day(config.end)
        )
    )
    monkeypatch.setattr(module, "load_registration_capacity", lambda *_: evidence)
    _, result = module.inspect(directory, config)
    assert not result["ready"]
    assert "capacity_evidence_required" in {i["code"] for i in result["issues"]}
    assert "ranking_rows" not in result["estimates"]


def test_upper_bound_above_limit_is_unproven_not_actual_overflow(tmp_path, monkeypatch):
    from hyperliquid_explorer_api import lab_proxy_datasets as module
    from test_lab_proxy import write_scheduled_fixture

    directory = tmp_path / "dataset"
    config = write_scheduled_fixture(directory)
    _, baseline = module.inspect(directory, config)
    bound = baseline["estimates"]["ranking_rows"]
    assert bound > 0
    monkeypatch.setattr(module, "MAX_RANKING_ROWS", bound)
    _, exact = module.inspect(directory, config)
    assert "capacity_unproven" not in {i["code"] for i in exact["issues"]}
    monkeypatch.setattr(module, "MAX_RANKING_ROWS", bound - 1)
    _, over = module.inspect(directory, config)
    issue = next(i for i in over["issues"] if i["code"] == "capacity_unproven")
    assert not over["ready"]
    assert "does not prove actual overflow" in issue["message"]


def test_legacy_fill_upper_bound_is_not_clamped(tmp_path):
    from hyperliquid_explorer_api.lab_proxy_datasets import inspect, candidate_ids
    from arblab.hyperliquid_copy.capacity_schedule import selection_ticks
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from arblab.hyperliquid_copy.download import file_hash

    directory = tmp_path / "dataset"
    config = write_proxy_fixture(directory)
    path = directory / "fills-0000.parquet"
    table = pq.read_table(path)
    repeated = pa.concat_tables([table] * (100001 // table.num_rows + 1))
    pq.write_table(repeated, path)
    metadata_path = directory / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    entry = next(row for row in metadata["files"] if row["name"] == path.name)
    entry.update(
        rows=repeated.num_rows, bytes=path.stat().st_size, sha256=file_hash(path)
    )
    metadata_path.write_text(json.dumps(metadata))
    manifest = ProxyDatasetManifest(directory)
    ids = candidate_ids(manifest, config)
    _, result = inspect(directory, config)
    expected = repeated.num_rows * len(selection_ticks(config)) * len(ids)
    assert result["estimates"]["ranking_rows"] == expected
    assert any("fill-count" in note for note in result["estimate_notes"])
