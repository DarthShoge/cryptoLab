from pathlib import Path

import pytest

from arblab.hyperliquid_copy.download import file_hash
from .test_registration_partitions import source


def retain(entries, stage):
    from arblab.hyperliquid_copy.registration_partitions import retain_full_source

    return retain_full_source(entries, stage)


def test_full_source_retention_preserves_both_boundary_spills(tmp_path):
    entries = [
        source(tmp_path, [-100, -1, 0, 24, 100], "a.parquet"),
        source(tmp_path, [-200, 200], "b.parquet"),
    ]
    stage = tmp_path / "stage"
    stage.mkdir()
    outputs = retain(entries, stage)
    assert outputs == [
        dict(
            name=f"fills-{i:04d}.parquet",
            **{k: entry[k] for k in ("bytes", "rows", "sha256")},
        )
        for i, entry in enumerate(entries)
    ]
    for entry, output in zip(entries, outputs):
        assert file_hash(stage / output["name"]) == entry["sha256"]
        assert file_hash(Path(entry["path"])) == entry["sha256"]


@pytest.mark.parametrize("fault", ["bytes", "rows", "duplicate", "existing", "limit"])
def test_full_source_prechecks_before_any_output(tmp_path, monkeypatch, fault):
    from arblab.hyperliquid_copy import registration_partitions as module

    entries = [source(tmp_path, [0], "a.parquet"), source(tmp_path, [1], "b.parquet")]
    stage = tmp_path / "stage"
    stage.mkdir()
    if fault == "bytes":
        entries[1]["bytes"] += 1
    elif fault == "rows":
        entries[1]["rows"] += 1
    elif fault == "duplicate":
        entries[1] = entries[0]
    elif fault == "existing":
        (stage / "fills-0001.parquet").write_bytes(b"owned existing data")
    else:
        monkeypatch.setattr(module, "MAX_BYTES", sum(e["bytes"] for e in entries) - 1)
    before = {p.name: p.read_bytes() for p in stage.iterdir()}
    with pytest.raises(ValueError):
        retain(entries, stage)
    assert {p.name: p.read_bytes() for p in stage.iterdir()} == before


def test_full_source_copy_fallback_and_final_source_recheck(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import registration_partitions as module
    import shutil

    entries = [
        source(tmp_path, [-1, 25], "a.parquet"),
        source(tmp_path, [1], "b.parquet"),
    ]
    stage = tmp_path / "stage"
    stage.mkdir()

    def copy_then_mutate(source_path, destination, **kwargs):
        shutil.copyfile(source_path, destination)
        if source_path.name == "b.parquet":
            with Path(entries[0]["path"]).open("ab") as stream:
                stream.write(b"changed after first copy")

    monkeypatch.setattr(module, "link_or_copy", copy_then_mutate)
    with pytest.raises(ValueError, match="identity|changed"):
        retain(entries, stage)


def test_full_source_copy_cannot_exceed_pinned_bytes(tmp_path, monkeypatch):
    import errno
    from arblab.hyperliquid_copy import registration_partitions as module

    entry = source(tmp_path, [-1, 25])
    stage = tmp_path / "stage"
    stage.mkdir()

    def grow_then_fallback(source_path, destination):
        with source_path.open("ab") as stream:
            stream.write(b"extra bytes after verification")
        raise OSError(errno.EXDEV, "force copy after source growth")

    monkeypatch.setattr(module.os, "link", grow_then_fallback)
    with pytest.raises(ValueError):
        retain([entry], stage)
    assert (stage / "fills-0000.parquet").stat().st_size <= entry["bytes"]
