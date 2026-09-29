from dataclasses import asdict, replace
from datetime import timedelta
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.download import file_hash
from .test_proxy_activity import fills
from .test_lab_ranking import settings


def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    rows = [replace(f, raw_details_json='{"unused":"payload"}') for f in fills()]
    table = pa.Table.from_pylist([asdict(f) for f in rows])
    path = root / "fills.parquet"
    pq.write_table(table, path)
    data = dict(
        schema="hyperliquid_proxy_activity_v1",
        complete=True,
        start="2026-01-01",
        end="2026-01-08",
        coins=["BTC"],
        scope="all_wallets_for_declared_markets",
        coverage_basis="complete_source_partitions_not_exchange_time_boundaries",
        rows=len(rows),
        output_bytes=path.stat().st_size,
        files=[
            dict(
                name=path.name,
                rows=len(rows),
                bytes=path.stat().st_size,
                sha256=file_hash(path),
            )
        ],
        source_keys=["example"],
        objects=[{"key": "example", "outside_window": 2}],
    )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest, path, rows


def test_compact_preserves_all_canonical_values_and_source(tmp_path):
    from arblab.hyperliquid_copy.proxy_compact import compact_history

    manifest, original, rows = source(tmp_path)
    before = file_hash(original)
    output = compact_history(manifest, tmp_path / "out")
    data = json.loads(output.read_text())
    compact = output.parent / data["files"][0]["name"]
    assert pq.read_table(compact).equals(
        pq.read_table(original).drop(["raw_details_json"])
    )
    assert file_hash(original) == before
    assert data["complete"] is True
    assert data["rows"] == len(rows)
    assert data["source_manifest_sha256"] == file_hash(manifest)
    assert data["source_evidence"] == json.loads(manifest.read_text())
    assert data["files"][0]["sha256"] == file_hash(compact)
    assert not list(output.parent.parent.glob("*.partial"))
    second = compact_history(manifest, tmp_path / "out")
    assert second != output and output.exists()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"complete": False}, "complete"),
        ({"rows": 999}, "row"),
        ({"output_bytes": 1}, "byte"),
        ({"scope": "selected_wallets"}, "scope"),
    ],
)
def test_bad_manifest_rejected(tmp_path, change, match):
    from arblab.hyperliquid_copy.proxy_compact import compact_history

    manifest, _, _ = source(tmp_path)
    data = json.loads(manifest.read_text()) | change
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        compact_history(manifest, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/manifest.json"))


def test_partition_bound_stops_at_169_for_compaction_and_catalog(tmp_path):
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from arblab.hyperliquid_copy.compact_catalog import CompactCatalog

    manifest, _, _ = source(tmp_path)
    compact = compact_history(manifest, tmp_path / "out")
    for path in (manifest, compact):
        data = json.loads(path.read_text())
        data["files"] *= 170
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="partitions"):
        compact_history(manifest, tmp_path / "other")
    with pytest.raises(ValueError, match="partition count"):
        CompactCatalog(tmp_path / "catalog.sqlite3").register(compact)


@pytest.mark.parametrize(
    "fault", ["hash", "rowcount", "duplicate", "escape", "symlink", "extra", "missing"]
)
def test_bad_partitions_never_publish(tmp_path, fault):
    from arblab.hyperliquid_copy.proxy_compact import compact_history

    manifest, path, _ = source(tmp_path)
    data = json.loads(manifest.read_text())
    if fault == "hash":
        data["files"][0]["sha256"] = "0" * 64
    elif fault == "rowcount":
        data["files"][0]["rows"] += 1
    elif fault == "duplicate":
        data["files"] *= 2
    elif fault == "escape":
        data["files"][0]["name"] = "../source/fills.parquet"
    elif fault == "symlink":
        link = path.parent / "link.parquet"
        link.symlink_to(path)
        data["files"][0]["name"] = link.name
    else:
        table = pq.read_table(path)
        table = (
            table.append_column("extra", pa.array([1] * table.num_rows))
            if fault == "extra"
            else table.drop(["post_position"])
        )
        pq.write_table(table, path)
        data["files"][0].update(bytes=path.stat().st_size, sha256=file_hash(path))
        data["output_bytes"] = path.stat().st_size
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        compact_history(manifest, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/manifest.json"))


def test_compact_query_equivalence(tmp_path):
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    manifest, original, rows = source(tmp_path)
    output = compact_history(manifest, tmp_path / "out")
    files = [output.parent / f["name"] for f in json.loads(output.read_text())["files"]]
    with (
        ProxyActivity([original], temp_root=tmp_path) as a,
        ProxyActivity(files, temp_root=tmp_path) as b,
    ):
        assert a.count == b.count
        for at in [
            rows[0].exchange_time,
            max(r.exchange_time for r in rows) + timedelta(seconds=1),
            max(r.exchange_time for r in rows) + timedelta(days=5),
        ]:
            assert a.observed(at) == b.observed(at)
            for user in {r.user for r in rows}:
                assert a.position(user, "BTC", at) == b.position(user, "BTC", at)
            for days in [1, 3]:
                assert a.rank(
                    at, settings(lookback_days=days), "BTC", "gross_excludes_fee"
                ) == b.rank(
                    at, settings(lookback_days=days), "BTC", "gross_excludes_fee"
                )
                assert a.volume("BTC", at - timedelta(days=days), at) == b.volume(
                    "BTC", at - timedelta(days=days), at
                )
        start = rows[0].exchange_time
        args = (rows[0].user, "BTC", start, start + timedelta(hours=5))
        assert a.hourly_exposure(
            *args, max_price_age_seconds=3600
        ) == b.hourly_exposure(*args, max_price_age_seconds=3600)


def test_cli_is_local_and_preserves_input(tmp_path):
    manifest, original, _ = source(tmp_path)
    repo = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "tools/compact_hyperliquid_proxy_history.py"),
            "--manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)["manifest"]).exists()
    assert original.exists()


def test_empty_partition_and_raw_optional_schema_are_preserved(tmp_path):
    from arblab.hyperliquid_copy.proxy_compact import compact_history

    manifest, path, _ = source(tmp_path)
    table = pq.read_table(path).drop(["raw_details_json"])
    pq.write_table(table, path)
    empty = path.parent / "empty.parquet"
    pq.write_table(table.slice(0, 0), empty)
    data = json.loads(manifest.read_text())
    data["files"] = [
        dict(
            name=p.name,
            rows=pq.read_metadata(p).num_rows,
            bytes=p.stat().st_size,
            sha256=file_hash(p),
        )
        for p in [path, empty]
    ]
    data["output_bytes"] = sum(p.stat().st_size for p in [path, empty])
    manifest.write_text(json.dumps(data))
    output = compact_history(manifest, tmp_path / "out")
    result = json.loads(output.read_text())
    assert len(result["files"]) == 2
    assert pq.read_table(output.parent / result["files"][1]["name"]).equals(
        table.slice(0, 0)
    )


def test_order_duplicates_and_future_events_keep_same_meaning(tmp_path):
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    manifest, path, rows = source(tmp_path)
    first = rows[0]
    last = replace(
        first,
        tid=999,
        event_id="last",
        source_line=first.source_line + 1,
        start_position=1,
        post_position=2,
    )
    duplicate = replace(first, event_id="duplicate", source_key="later")
    future = replace(
        first,
        tid=1000,
        event_id="future",
        exchange_time=first.exchange_time + timedelta(days=2),
        post_position=3,
    )
    pq.write_table(
        pa.Table.from_pylist([asdict(f) for f in [future, last, duplicate, first]]),
        path,
        row_group_size=1,
    )
    data = json.loads(manifest.read_text())
    data.update(rows=4, output_bytes=path.stat().st_size)
    data["files"][0].update(rows=4, bytes=path.stat().st_size, sha256=file_hash(path))
    manifest.write_text(json.dumps(data))
    output = compact_history(manifest, tmp_path / "out")
    compact = output.parent / json.loads(output.read_text())["files"][0]["name"]
    with (
        ProxyActivity([path], temp_root=tmp_path) as a,
        ProxyActivity([compact], temp_root=tmp_path) as b,
    ):
        assert a.count == b.count == 3
        assert b.position(first.user, "BTC", first.exchange_time) is None
        at = first.exchange_time + timedelta(seconds=1)
        assert a.position(first.user, "BTC", at) == b.position(first.user, "BTC", at)


@pytest.mark.parametrize("limit", ["MAX_BYTES", "MAX_BATCH_BYTES"])
def test_resource_failure_does_not_publish_or_remove_input(
    tmp_path, monkeypatch, limit
):
    from arblab.hyperliquid_copy import proxy_compact

    manifest, original, _ = source(tmp_path)
    before = file_hash(original)
    monkeypatch.setattr(proxy_compact, limit, 1)
    with pytest.raises(ValueError, match="byte"):
        proxy_compact.compact_history(manifest, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/manifest.json"))
    assert file_hash(original) == before


def test_source_change_after_validation_is_not_published(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import proxy_compact

    manifest, original, _ = source(tmp_path)
    real_sync = proxy_compact._sync
    mutated = False

    def sync_and_mutate(path):
        nonlocal mutated
        real_sync(path)
        if not mutated:
            table = pq.read_table(original)
            table = table.set_column(
                table.schema.get_field_index("px"),
                "px",
                pa.array([999.0] * table.num_rows),
            )
            pq.write_table(table, original)
            mutated = True

    monkeypatch.setattr(proxy_compact, "_sync", sync_and_mutate)
    with pytest.raises(ValueError, match="projection|changed"):
        proxy_compact.compact_history(manifest, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/manifest.json"))
    assert original.exists()
