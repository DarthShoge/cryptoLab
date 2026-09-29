import json
from datetime import datetime, timedelta

import lz4.frame
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_archive_import import import_archive
from arblab.hyperliquid_copy.proxy_compact import compact_history
from .test_proxy_archive_import import archive


def normalized(tmp_path):
    source = archive(tmp_path, days=2)
    data = json.loads(source.read_text())
    obj = data["objects"][0]
    path = source.parent / obj["file"]
    payload = json.loads(lz4.frame.decompress(path.read_bytes()))
    for _, fill in payload["events"]:
        fill["time"] -= 1
    payload["block_time"] = (
        datetime.fromisoformat(payload["block_time"]) - timedelta(milliseconds=1)
    ).isoformat()
    path.write_bytes(lz4.frame.compress(json.dumps(payload).encode() + b"\n"))
    obj.update(bytes=path.stat().st_size, sha256=file_hash(path))
    source.write_text(json.dumps(data))
    return import_archive(
        source, ["BTC"], tmp_path / "normalized", retain_boundary_spill=True
    )


def test_daily_compaction_preserves_exact_rows_types_and_source_day_spill(tmp_path):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    manifest = normalized(tmp_path)
    original = json.loads(manifest.read_text())
    paths = [manifest.parent / e["name"] for e in original["files"]]
    expected = pa.concat_tables(
        [pq.read_table(p).drop(["raw_details_json"]) for p in paths]
    )
    output = compact_history(manifest, tmp_path / "daily", partitioning="source_day")
    data = json.loads(output.read_text())
    assert data["partitioning"] == "source_day"
    assert [e["source_day"] for e in data["files"]] == ["2026-08-01", "2026-08-02"]
    daily = [output.parent / e["name"] for e in data["files"]]
    assert pa.concat_tables([pq.read_table(p) for p in daily]).equals(expected)
    assert (
        str(pq.read_table(daily[0])["exchange_time"][0].as_py().date()) == "2026-07-31"
    )
    assert data["rows"] == 96
    assert data["source_evidence"]["boundary_spill_retained"] is True
    CompactCatalog(tmp_path / "catalog.sqlite3").register(output)
    with (
        ProxyActivity(paths, temp_root=tmp_path) as a,
        ProxyActivity(daily, temp_root=tmp_path) as b,
    ):
        assert a.count == b.count == 96


def test_daily_compaction_rejects_undeclared_row_source_keys(tmp_path):
    manifest = normalized(tmp_path)
    data = json.loads(manifest.read_text())
    e = data["files"][0]
    path = manifest.parent / e["name"]
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("source_key"),
        "source_key",
        pa.array(["unknown"] * table.num_rows),
    )
    pq.write_table(table, path)
    old_size = e["bytes"]
    e.update(bytes=path.stat().st_size, sha256=file_hash(path))
    data["output_bytes"] += e["bytes"] - old_size
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source key"):
        compact_history(manifest, tmp_path / "daily", partitioning="source_day")
    assert not list((tmp_path / "daily").glob("*/manifest.json"))


def test_invalid_partitioning_rejected(tmp_path):
    manifest = normalized(tmp_path)
    with pytest.raises(ValueError, match="partitioning"):
        compact_history(manifest, tmp_path / "daily", partitioning="exchange_week")


def test_shared_output_cap_is_global_and_checked_before_writing():
    from io import BytesIO
    from arblab.hyperliquid_copy.daily_compact import _SharedOutput

    first, second, used = BytesIO(), BytesIO(), [0]
    a, b = _SharedOutput(first, 5, used), _SharedOutput(second, 5, used)
    assert a.write(b"abc") == 3
    with pytest.raises(ValueError, match="total byte limit"):
        b.write(b"def")
    assert second.getvalue() == b""
    assert b.write(b"de") == 2
    assert used == [5]


def test_daily_output_cap_prevents_publication(tmp_path, monkeypatch):
    import arblab.hyperliquid_copy.daily_compact as module

    manifest = normalized(tmp_path)
    monkeypatch.setattr(module, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="byte limit"):
        compact_history(manifest, tmp_path / "daily", partitioning="source_day")
    assert not list((tmp_path / "daily").glob("*/manifest.json"))
    assert manifest.is_file()


def test_daily_handoff_keeps_duplicates_for_activity_validation(tmp_path):
    from datetime import datetime, timezone
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    source = archive(tmp_path, days=7, start=datetime(2025, 7, 24, tzinfo=timezone.utc))
    manifest = import_archive(source, ["BTC"], tmp_path / "normalized")
    output = compact_history(manifest, tmp_path / "daily", partitioning="source_day")
    data = json.loads(output.read_text())
    assert len(data["files"]) == 7
    assert data["rows"] == 338
    with ProxyActivity(
        [output.parent / e["name"] for e in data["files"]], temp_root=tmp_path
    ) as activity:
        assert activity.count == 336


def test_daily_compaction_keeps_empty_source_day(tmp_path):
    source = archive(tmp_path, days=2)
    data = json.loads(source.read_text())
    for obj in data["objects"][24:]:
        path = source.parent / obj["file"]
        payload = json.loads(lz4.frame.decompress(path.read_bytes()))
        for _, fill in payload["events"]:
            fill["coin"] = "ETH"
        path.write_bytes(lz4.frame.compress(json.dumps(payload).encode() + b"\n"))
        obj.update(bytes=path.stat().st_size, sha256=file_hash(path))
    source.write_text(json.dumps(data))
    manifest = import_archive(source, ["BTC"], tmp_path / "normalized")
    output = compact_history(manifest, tmp_path / "daily", partitioning="source_day")
    result = json.loads(output.read_text())
    assert [entry["rows"] for entry in result["files"]] == [48, 0]
    empty = pq.read_table(output.parent / result["files"][1]["name"])
    assert empty.num_rows == 0


def test_daily_compaction_rejects_schema_coercion(tmp_path):
    manifest = normalized(tmp_path)
    data = json.loads(manifest.read_text())
    entry = data["files"][0]
    path = manifest.parent / entry["name"]
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("px"),
        "px",
        pa.array([100] * table.num_rows, type=pa.int64()),
    )
    pq.write_table(table, path)
    old_size = entry["bytes"]
    entry.update(bytes=path.stat().st_size, sha256=file_hash(path))
    data["output_bytes"] += entry["bytes"] - old_size
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identical canonical schemas"):
        compact_history(manifest, tmp_path / "daily", partitioning="source_day")
    assert not list((tmp_path / "daily").glob("*/manifest.json"))
