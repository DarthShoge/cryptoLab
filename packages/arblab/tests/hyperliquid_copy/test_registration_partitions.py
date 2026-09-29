from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.download import file_hash

START = datetime(2026, 8, 2, tzinfo=timezone.utc)


def source(tmp_path, hours, name="input.parquet"):
    path = tmp_path / name
    rows = [
        dict(exchange_time=START + timedelta(hours=h), coin="BTC", value=i)
        for i, h in enumerate(hours)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return dict(
        path=str(path),
        sha256=file_hash(path),
        bytes=path.stat().st_size,
        rows=len(rows),
    )


def publish(entries, stage):
    from arblab.hyperliquid_copy.registration_partitions import publish_interior

    return publish_interior(entries, START, START + timedelta(days=1), stage)


def test_boundary_filter_preserves_columns_order_and_duplicates(tmp_path):
    entry = source(tmp_path, [-1, 0, 24, 5, 0, 23, -2])
    stage = tmp_path / "stage"
    stage.mkdir()
    before = pq.read_table(entry["path"]).to_pylist()
    outputs = publish([entry], stage)
    actual = pq.read_table(stage / outputs[0]["name"]).to_pylist()
    assert actual == [
        r for r in before if START <= r["exchange_time"] < START + timedelta(days=1)
    ]
    assert file_hash(Path(entry["path"])) == entry["sha256"]


def test_wholly_interior_file_is_linked_without_extra_storage(tmp_path):
    entry = source(tmp_path, [0, 1, 2])
    stage = tmp_path / "stage"
    stage.mkdir()
    outputs = publish([entry], stage)
    assert (stage / outputs[0]["name"]).stat().st_ino == Path(
        entry["path"]
    ).stat().st_ino
    assert outputs[0]["sha256"] == entry["sha256"]


def test_empty_retained_partition_keeps_schema(tmp_path):
    entry = source(tmp_path, [-2, -1, 24])
    stage = tmp_path / "stage"
    stage.mkdir()
    outputs = publish([entry], stage)
    saved = pq.read_table(stage / outputs[0]["name"])
    assert saved.num_rows == 0 and saved.schema == pq.read_schema(entry["path"])


def test_corrupt_source_rejected_without_output(tmp_path):
    entry = source(tmp_path, [0, 1])
    with Path(entry["path"]).open("ab") as handle:
        handle.write(b"bad")
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(ValueError, match="identity|changed"):
        publish([entry], stage)
    assert not list(stage.iterdir())


def test_existing_output_is_never_overwritten(tmp_path):
    entry = source(tmp_path, [0, 1])
    stage = tmp_path / "stage"
    stage.mkdir()
    outputs = publish([entry], stage)
    with pytest.raises((ValueError, FileExistsError), match="exist"):
        publish([entry], stage)
    assert file_hash(stage / outputs[0]["name"]) == entry["sha256"]


def test_cross_filesystem_fallback_copies_identically(tmp_path, monkeypatch):
    import errno
    from arblab.hyperliquid_copy import registration_partitions as module

    entry = source(tmp_path, [0, 1])
    stage = tmp_path / "stage"
    stage.mkdir()

    def unavailable(*args):
        raise OSError(errno.EXDEV, "different filesystem")

    monkeypatch.setattr(module.os, "link", unavailable)
    outputs = publish([entry], stage)
    saved = stage / outputs[0]["name"]
    assert saved.stat().st_ino != Path(entry["path"]).stat().st_ino
    assert file_hash(saved) == entry["sha256"]


def test_byte_bound_rejects_before_output(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import registration_partitions as module

    entry = source(tmp_path, [0, 1])
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(module, "MAX_BYTES", entry["bytes"] - 1)
    with pytest.raises(ValueError, match="byte bound"):
        publish([entry], stage)
    assert not list(stage.iterdir())


def test_changed_source_during_projection_rejects(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import registration_partitions as module

    entry = source(tmp_path, [-1, 0, 1])
    stage = tmp_path / "stage"
    stage.mkdir()
    original = module.filter_partition

    def mutate(*args):
        original(*args)
        with Path(entry["path"]).open("ab") as handle:
            handle.write(b"bad")

    monkeypatch.setattr(module, "filter_partition", mutate)
    with pytest.raises(ValueError, match="identity"):
        publish([entry], stage)


def test_timestamp_precision_must_not_be_truncated(tmp_path):
    entry = source(tmp_path, [0, 1])
    path = Path(entry["path"])
    table = pq.read_table(path)
    field = table.schema.field("exchange_time").with_type(pa.timestamp("ns", tz="UTC"))
    table = table.cast(table.schema.set(0, field))
    pq.write_table(table, path)
    entry.update(bytes=path.stat().st_size, sha256=file_hash(path))
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(ValueError, match="microsecond"):
        publish([entry], stage)


def test_footer_cannot_write_past_retained_byte_limit(tmp_path):
    from arblab.hyperliquid_copy.registration_partitions import filter_partition

    entry = source(tmp_path, [-1, 0, 1, 24])
    source_path = Path(entry["path"])
    baseline = tmp_path / "baseline.parquet"
    schema = pq.read_schema(source_path)
    filter_partition(
        source_path, baseline, schema, START, START + timedelta(days=1), 1000000
    )
    ceiling = baseline.stat().st_size - 1
    output = tmp_path / "limited.parquet"
    with pytest.raises(ValueError, match="byte|limit|ceiling"):
        filter_partition(
            source_path, output, schema, START, START + timedelta(days=1), ceiling
        )
    assert output.stat().st_size <= ceiling
