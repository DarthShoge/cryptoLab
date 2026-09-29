from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import sys

import lz4.frame
import pyarrow.parquet as pq
import pytest

from .test_archive import USER, raw_fill
from arblab.hyperliquid_copy.archive import archive_keys
from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_archive_import import import_archive


def archive(tmp_path, days=1, start=datetime(2026, 8, 1, tzinfo=timezone.utc)):
    source = tmp_path / "raw"
    source.mkdir()
    objects = []
    keys = [
        key
        for d in range(days)
        for key in archive_keys((start + timedelta(days=d)).date().isoformat())
    ]
    for index, key in enumerate(keys):
        day, filename = key.split("/")[-2:]
        event_time = datetime.strptime(day, "%Y%m%d").replace(
            tzinfo=timezone.utc
        ) + timedelta(hours=int(filename.split(".")[0]))
        hour = int((event_time - start).total_seconds() // 3600)
        at = int(event_time.timestamp() * 1000)
        payload = dict(
            block_number=hour,
            block_time=event_time.isoformat(),
            events=[
                [USER, raw_fill(time=at, tid=hour)],
                [
                    "0x" + "cd" * 20,
                    raw_fill(
                        time=at, tid=hour, side="A", startPosition="1", crossed=False
                    ),
                ],
                [USER, raw_fill(coin="ETH", time=at, tid=hour + 100)],
            ],
        )
        lines = payload["events"] if key.startswith("node_fills/") else [payload]
        path = source / f"fills_{index:04d}.lz4"
        path.write_bytes(
            lz4.frame.compress(
                b"".join(json.dumps(line).encode() + b"\n" for line in lines)
            )
        )
        objects.append(
            dict(
                key=key,
                file=path.name,
                bytes=path.stat().st_size,
                sha256=file_hash(path),
                status="downloaded",
            )
        )
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="hyperliquid_proxy_archive_v1",
                bucket="hl-mainnet-node-data",
                start=start.date().isoformat(),
                end=(start + timedelta(days=days)).date().isoformat(),
                complete=True,
                objects=objects,
            )
        )
    )
    return manifest


def test_handoff_week_compacts_and_indexes_both_formats_without_double_counting(
    tmp_path,
):
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    source = archive(tmp_path, days=7, start=datetime(2025, 7, 24, tzinfo=timezone.utc))
    normalized = import_archive(source, ["BTC"], tmp_path / "out")
    data = json.loads(normalized.read_text())
    assert len(data["files"]) == 169
    assert data["rows"] == 338
    compact = compact_history(normalized, tmp_path / "compact")
    catalog = CompactCatalog(tmp_path / "catalog.sqlite3")
    catalog.register(compact)
    files = [
        compact.parent / entry["name"]
        for entry in json.loads(compact.read_text())["files"]
    ]
    with ProxyActivity(files, temp_root=tmp_path) as activity:
        assert activity.count == 336


@pytest.mark.parametrize("prefix", ["node_fills", "node_fills_by_block"])
def test_handoff_requires_both_hour_eight_objects(tmp_path, prefix):
    source = archive(tmp_path, start=datetime(2025, 7, 27, tzinfo=timezone.utc))
    data = json.loads(source.read_text())
    missing = f"{prefix}/hourly/20250727/8.lz4"
    data["objects"] = [obj for obj in data["objects"] if obj["key"] != missing]
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="partition|objects|keys"):
        import_archive(source, ["BTC"], tmp_path / "out")


def test_import_preserves_all_wallets_and_native_fields(tmp_path):
    source = archive(tmp_path)
    result = import_archive(source, ["BTC"], tmp_path / "out")
    manifest = json.loads(result.read_text())
    assert manifest["complete"] and manifest["rows"] == 48
    assert (
        manifest["coverage_basis"]
        == "complete_source_partitions_not_exchange_time_boundaries"
    )
    assert len(manifest["source_keys"]) == 24
    first = pq.read_table(result.parent / manifest["files"][0]["name"]).to_pylist()
    assert {row["user"] for row in first} == {USER, "0x" + "cd" * 20}
    assert all(row["raw_details_json"] for row in first)
    assert all(row["coin"] == "BTC" for row in first)


@pytest.mark.parametrize("failure", ["hash", "missing", "incomplete"])
def test_import_rejects_unqualified_archive_without_success(tmp_path, failure):
    source = archive(tmp_path)
    data = json.loads(source.read_text())
    if failure == "hash":
        data["objects"][0]["sha256"] = "a" * 64
    elif failure == "missing":
        data["objects"].pop()
    else:
        data["complete"] = False
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        import_archive(source, ["BTC"], tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/manifest.json"))


def test_import_bounds_decoded_lines_before_json_parsing(tmp_path):
    source = archive(tmp_path)
    with pytest.raises(ValueError, match="line"):
        import_archive(source, ["BTC"], tmp_path / "out", max_line_bytes=10)
    assert not list((tmp_path / "out").glob("*/manifest.json"))


def test_import_records_boundary_spill_without_inventing_in_range_fills(tmp_path):
    source = archive(tmp_path)
    data = json.loads(source.read_text())
    obj = data["objects"][0]
    path = source.parent / obj["file"]
    payload = json.loads(lz4.frame.decompress(path.read_bytes()))
    payload["events"][0][1]["time"] -= 1
    path.write_bytes(lz4.frame.compress(json.dumps(payload).encode() + b"\n"))
    obj.update(bytes=path.stat().st_size, sha256=file_hash(path))
    source.write_text(json.dumps(data))
    result = import_archive(source, ["BTC"], tmp_path / "out")
    manifest = json.loads(result.read_text())
    assert manifest["rows"] == 47
    assert manifest["objects"][0]["outside_window"] == 1
    assert manifest["objects"][0]["first_event"].startswith("2026-07-31")


def test_import_rejects_malformed_mapped_fill_instead_of_dropping_wallet(tmp_path):
    source = archive(tmp_path)
    data = json.loads(source.read_text())
    obj = data["objects"][0]
    path = source.parent / obj["file"]
    payload = json.loads(lz4.frame.decompress(path.read_bytes()))
    payload["events"][0][1]["px"] = "nan"
    path.write_bytes(lz4.frame.compress(json.dumps(payload).encode() + b"\n"))
    obj.update(bytes=path.stat().st_size, sha256=file_hash(path))
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid archive"):
        import_archive(source, ["BTC"], tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/manifest.json"))


def test_import_flushes_by_payload_bytes_not_only_row_count(tmp_path):
    source = archive(tmp_path)
    result = import_archive(source, ["BTC"], tmp_path / "out", max_batch_bytes=100)
    manifest = json.loads(result.read_text())
    first = pq.ParquetFile(result.parent / manifest["files"][0]["name"])
    assert first.num_row_groups == 2


def test_retained_spill_survives_adjacent_batches_compaction_and_combined_queries(
    tmp_path,
):
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    compact_files = []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        source = archive(root, start=datetime(2026, 8, 1 + index, tzinfo=timezone.utc))
        data = json.loads(source.read_text())
        for hour, obj in enumerate(data["objects"]):
            path = source.parent / obj["file"]
            payload = json.loads(lz4.frame.decompress(path.read_bytes()))
            for _, fill in payload["events"]:
                fill["tid"] += 24 * index
                if (index, hour) == (0, 23):
                    fill["time"] += 3_600_001
                elif (index, hour) == (1, 0):
                    fill["time"] -= 1
            payload["block_time"] = datetime.fromtimestamp(
                payload["events"][0][1]["time"] / 1000, timezone.utc
            ).isoformat()
            path.write_bytes(lz4.frame.compress(json.dumps(payload).encode() + b"\n"))
            obj.update(bytes=path.stat().st_size, sha256=file_hash(path))
        source.write_text(json.dumps(data))
        filtered = import_archive(source, ["BTC"], root / "filtered")
        assert json.loads(filtered.read_text())["rows"] == 46
        retained = import_archive(
            source, ["BTC"], root / "retained", retain_boundary_spill=True
        )
        metadata = json.loads(retained.read_text())
        assert metadata["rows"] == 48
        assert metadata["boundary_spill_retained"] is True
        assert sum(o["outside_window"] for o in metadata["objects"]) == 2
        compact = compact_history(retained, root / "compact")
        result = json.loads(compact.read_text())
        assert result["source_evidence"]["boundary_spill_retained"] is True
        compact_files.extend(compact.parent / f["name"] for f in result["files"])
    with ProxyActivity(compact_files, temp_root=tmp_path) as activity:
        assert activity.count == 96
        activity.validate_registered_scope(
            {"BTC"},
            datetime(2026, 8, 1, tzinfo=timezone.utc),
            datetime(2026, 8, 3, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("target", ["manifest", "raw"])
def test_mutated_import_sources_never_publish_normalized_manifest(tmp_path, target):
    source = archive(tmp_path)
    data = json.loads(source.read_text())

    def mutate(item):
        if item["imported"] == item["objects"]:
            path = (
                source
                if target == "manifest"
                else source.parent / data["objects"][0]["file"]
            )
            path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="changed"):
        import_archive(source, ["BTC"], tmp_path / "out", progress=mutate)
    assert not list((tmp_path / "out").glob("*/manifest.json"))


def test_boundary_retention_requires_boolean(tmp_path):
    source = archive(tmp_path)
    with pytest.raises(ValueError, match="boundary"):
        import_archive(source, ["BTC"], tmp_path / "out", retain_boundary_spill="false")


def test_cli_retains_boundary_rows_when_explicitly_requested(tmp_path):
    source = archive(tmp_path)
    data = json.loads(source.read_text())
    obj = data["objects"][0]
    path = source.parent / obj["file"]
    payload = json.loads(lz4.frame.decompress(path.read_bytes()))
    for _, fill in payload["events"]:
        fill["time"] -= 1
    path.write_bytes(lz4.frame.compress(json.dumps(payload).encode() + b"\n"))
    obj.update(bytes=path.stat().st_size, sha256=file_hash(path))
    source.write_text(json.dumps(data))
    tool = (
        Path(__file__).resolve().parents[4]
        / "tools/import_hyperliquid_proxy_archive.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--manifest",
            str(source),
            "--coins",
            "BTC",
            "--output-root",
            str(tmp_path / "out"),
            "--retain-boundary-spill",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(Path(result.stdout.splitlines()[-1]).read_text())
    assert metadata["rows"] == 48
    assert metadata["boundary_spill_retained"] is True
