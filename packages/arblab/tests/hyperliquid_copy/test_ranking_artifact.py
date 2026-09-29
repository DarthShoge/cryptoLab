import shutil

import pyarrow.parquet as pq
import pytest


def row(**changes):
    return dict(
        user="0x" + "a" * 40,
        metrics={"gross_volume": 10},
        percentiles={},
        eligible=False,
        reasons=["missing"],
        **changes,
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_approved_annual_report_ceiling_without_large_allocation(tmp_path, explicit):
    from arblab.hyperliquid_copy.ranking_artifact import RankingSink

    path = tmp_path / "annual.parquet"
    limits = {"max_rows": 250_000_000, "max_bytes": 16 * 1024**3} if explicit else {}
    sink = RankingSink(path, **limits)
    assert sink.max_rows == 250_000_000
    assert sink.max_bytes == 16 * 1024**3
    assert not path.exists()


@pytest.mark.parametrize("name", ["max_rows", "max_bytes"])
@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "1", None, "above_cap"])
def test_report_constructor_rejects_invalid_or_above_approved_limits(
    tmp_path, name, invalid
):
    from arblab.hyperliquid_copy.ranking_artifact import RankingSink

    if invalid == "above_cap":
        invalid = (250_000_000 if name == "max_rows" else 16 * 1024**3) + 1
    path = tmp_path / "rejected.parquet"
    with pytest.raises(ValueError, match="Invalid ranking artifact limits"):
        RankingSink(path, **{name: invalid})
    assert not path.exists()


def test_fixed_schema_preserves_metrics_that_appear_in_later_batches(tmp_path):
    from arblab.hyperliquid_copy.ranking_artifact import RankingSink

    with RankingSink(tmp_path / "rows.parquet") as sink:
        sink.extend([row()])
        active = row()
        active.update(
            metrics={"gross_volume": 20, "pnl_efficiency": 0.5},
            percentiles={"pnl_efficiency": 1},
            eligible=True,
        )
        sink.extend([active])
    artifact = sink.artifact
    assert len(artifact) == 2
    rows = pq.read_table(artifact.path).to_pylist()
    assert rows[0]["metrics"]["pnl_efficiency"] is None
    assert rows[1]["metrics"]["pnl_efficiency"] == 0.5
    with pytest.raises(TypeError):
        list(artifact)  # Publication must copy, never materialize all rows.
    artifact.copy_to(tmp_path / "copied.parquet")
    assert (tmp_path / "copied.parquet").read_bytes() == artifact.path.read_bytes()


@pytest.mark.parametrize("fault", ["rows", "bytes", "field", "metric"])
def test_bounds_and_unknown_evidence_reject(tmp_path, fault):
    from arblab.hyperliquid_copy.ranking_artifact import RankingSink

    limits = (
        {"max_rows": 1}
        if fault == "rows"
        else {"max_bytes": 1}
        if fault == "bytes"
        else {}
    )
    value = row()
    if fault == "field":
        value["unexpected"] = 123
    if fault == "metric":
        value["metrics"]["unrecognized"] = 1
    with pytest.raises(ValueError, match="limit|field|metric"):
        with RankingSink(tmp_path / "rows.parquet", **limits) as sink:
            sink.extend([value, value])


def test_empty_output_corruption_and_no_overwrite(tmp_path):
    from arblab.hyperliquid_copy.ranking_artifact import RankingSink

    with RankingSink(tmp_path / "empty.parquet") as sink:
        pass
    assert len(sink.artifact) == 0
    assert pq.read_table(sink.artifact.path).num_rows == 0
    existing = tmp_path / "existing.parquet"
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        sink.artifact.copy_to(existing)
    assert existing.read_bytes() == b"preserve"
    sink.artifact.path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        sink.artifact.copy_to(tmp_path / "bad.parquet")


def test_publication_rejects_insufficient_disk_before_writing(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy.ranking_artifact import RankingSink

    with RankingSink(tmp_path / "rows.parquet") as sink:
        sink.extend([row()])
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
    with pytest.raises(ValueError, match="space"):
        sink.artifact.copy_to(tmp_path / "copy.parquet")
    assert not (tmp_path / "copy.parquet").exists()
