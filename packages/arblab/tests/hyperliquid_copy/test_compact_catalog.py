from datetime import timedelta
import json
import sqlite3

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from arblab.hyperliquid_copy.proxy_compact import compact_history
from arblab.hyperliquid_copy.download import file_hash
from .test_proxy_compact import source


def compact(tmp_path):
    manifest, _, rows = source(tmp_path)
    return compact_history(manifest, tmp_path / "compact"), rows


def test_registration_reopens_idempotently_and_prunes_by_event_time(tmp_path):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog

    manifest, rows = compact(tmp_path)
    db = tmp_path / "catalog.sqlite3"
    catalog = CompactCatalog(db)
    identity = catalog.register(manifest)
    assert CompactCatalog(db).register(manifest) == identity
    start = min(r.exchange_time for r in rows)
    end = max(r.exchange_time for r in rows)
    found = catalog.partitions([identity], start, end + timedelta(microseconds=1))
    assert len(found) == 1
    assert found[0]["rows"] == len(rows)
    assert found[0]["sha256"] == json.loads(manifest.read_text())["files"][0]["sha256"]
    assert catalog.partitions([identity], start - timedelta(days=1), start) == []
    assert (
        catalog.partitions(
            [identity], end + timedelta(microseconds=1), end + timedelta(days=1)
        )
        == []
    )
    # Inclusive lower boundary retains an event exactly at the beginning.
    assert catalog.partitions([identity], end, end + timedelta(microseconds=1))
    with sqlite3.connect(db) as connection:
        assert connection.execute("select count(*) from manifests").fetchone()[0] == 1
        assert connection.execute("select count(*) from partitions").fetchone()[0] == 1


@pytest.mark.parametrize(
    "fault", ["incomplete", "projection", "hash", "rows", "path", "missing"]
)
def test_invalid_input_leaves_no_registered_manifest(tmp_path, fault):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog

    manifest, _ = compact(tmp_path)
    data = json.loads(manifest.read_text())
    if fault == "incomplete":
        data["complete"] = False
    elif fault == "projection":
        data["projection_version"] = 999
    elif fault == "hash":
        data["files"][0]["sha256"] = "0" * 64
    elif fault == "rows":
        data["rows"] += 1
    elif fault == "path":
        data["files"][0]["name"] = "../bad.parquet"
    else:
        data["files"][0]["name"] = "absent.parquet"
    manifest.write_text(json.dumps(data))
    db = tmp_path / "catalog.sqlite3"
    catalog = CompactCatalog(db)
    with pytest.raises(ValueError):
        catalog.register(manifest)
    with sqlite3.connect(db) as connection:
        assert connection.execute("select count(*) from manifests").fetchone()[0] == 0


def test_bad_final_partition_does_not_publish_partial_catalog(tmp_path):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog

    manifest, _ = compact(tmp_path)
    data = json.loads(manifest.read_text())
    data["files"].append(data["files"][0] | {"name": "missing.parquet"})
    manifest.write_text(json.dumps(data))
    db = tmp_path / "catalog.sqlite3"
    catalog = CompactCatalog(db)
    with pytest.raises(ValueError):
        catalog.register(manifest)
    with sqlite3.connect(db) as connection:
        assert connection.execute("select count(*) from partitions").fetchone()[0] == 0


def test_reuse_checks_files_and_unknown_window_ids_fail(tmp_path):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog

    manifest, rows = compact(tmp_path)
    catalog = CompactCatalog(tmp_path / "catalog.sqlite3")
    identity = catalog.register(manifest)
    with pytest.raises(ValueError, match="unknown"):
        catalog.partitions(
            ["unknown"],
            rows[0].exchange_time,
            rows[0].exchange_time + timedelta(days=1),
        )
    with pytest.raises(ValueError):
        catalog.partitions([identity], rows[0].exchange_time, rows[0].exchange_time)
    path = manifest.parent / json.loads(manifest.read_text())["files"][0]["name"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        catalog.register(manifest)


def test_unrelated_database_not_modified(tmp_path):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog

    db = tmp_path / "other.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("create table other(x)")
    with pytest.raises(ValueError, match="catalog"):
        CompactCatalog(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "select name from sqlite_master where type='table'"
        ).fetchall() == [("other",)]


@pytest.mark.parametrize("column", ["px", "tid", "user"])
def test_incompatible_canonical_types_rejected(tmp_path, column):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog

    manifest, _ = compact(tmp_path)
    data = json.loads(manifest.read_text())
    path = manifest.parent / data["files"][0]["name"]
    table = pq.read_table(path)
    array = (
        pa.array(["bad"] * table.num_rows)
        if column != "user"
        else pa.array([1] * table.num_rows)
    )
    table = table.set_column(table.schema.get_field_index(column), column, array)
    pq.write_table(table, path)
    data["files"][0].update(bytes=path.stat().st_size, sha256=file_hash(path))
    data["output_bytes"] = path.stat().st_size
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="type"):
        CompactCatalog(tmp_path / "catalog.sqlite3").register(manifest)


def test_multiple_sources_are_explicit_and_empty_files_do_not_overlap(tmp_path):
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog

    manifest, rows = compact(tmp_path)
    data = json.loads(manifest.read_text())
    original = manifest.parent / data["files"][0]["name"]
    empty = manifest.parent / "empty.parquet"
    pq.write_table(pq.read_table(original).slice(0, 0), empty)
    data["files"].append(
        dict(
            name=empty.name, rows=0, bytes=empty.stat().st_size, sha256=file_hash(empty)
        )
    )
    data["output_bytes"] += empty.stat().st_size
    manifest.write_text(json.dumps(data))
    second_root = tmp_path / "second"
    second_root.mkdir()
    second, _ = compact(second_root)
    catalog = CompactCatalog(tmp_path / "catalog.sqlite3")
    first_id, second_id = catalog.register(manifest), catalog.register(second)
    start = rows[0].exchange_time
    assert len(catalog.partitions([first_id], start, start + timedelta(days=2))) == 1
    assert (
        len(catalog.partitions([first_id, second_id], start, start + timedelta(days=2)))
        == 2
    )
    with pytest.raises(ValueError):
        catalog.partitions([first_id] * 1001, start, start + timedelta(days=2))
