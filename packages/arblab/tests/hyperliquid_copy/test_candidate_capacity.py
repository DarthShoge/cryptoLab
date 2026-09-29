from datetime import timedelta
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession
from .test_candidate_day import resources
from .test_qualified_day import qualified


def test_capacity_counts_all_source_origin_candidates_and_reuses(
    resources, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_capacity as module

    source = QualifiedSourceSession(qualified)
    inputs = module.build_candidate_capacity(resources, source)
    actual = module.CandidateCapacity(resources, source, inputs)
    report = json.loads(Path(qualified["path"]).read_text())
    rows = [
        row
        for entry in report["files"]
        for row in pq.read_table(entry["path"]).to_pylist()
    ]
    for offset in range(4):
        at = day("2026-08-01") + timedelta(days=offset)
        expected = {
            row["user"] for row in rows if source.origin <= row["exchange_time"] < at
        }
        assert actual.upper_bound(at, ["BTC"], "BTC") == len(expected)
        assert actual.upper_bound(at, ["BTC"], None) == len(expected)
        assert actual.upper_bound(at, [], None) == 0
    # Aug3 has no new fills, but all earlier dormant candidates remain counted.
    assert actual.upper_bound(day("2026-08-04"), ["BTC"], "BTC") == 2
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("capacity reuse reran raw query")

    monkeypatch.setattr(module, "_write", forbidden)
    assert module.build_candidate_capacity(resources, source) == inputs
    assert resources.audit() == before


@pytest.mark.parametrize("fault", ["source", "artifact", "engine", "query"])
def test_capacity_rejects_changed_evidence(resources, qualified, monkeypatch, fault):
    from arblab.hyperliquid_copy import candidate_capacity as module
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts

    source = QualifiedSourceSession(qualified)
    inputs = module.build_candidate_capacity(resources, source)
    actual = module.CandidateCapacity(resources, source, inputs)
    if fault in ("source", "artifact"):
        path = (
            Path(json.loads(Path(qualified["path"]).read_text())["files"][0]["path"])
            if fault == "source"
            else resources.root
            / PublishedArtifacts(resources)
            .lookup("candidate_capacity", inputs)
            .artifacts[0]
            .path
        )
        with path.open("ab") as handle:
            handle.write(b"changed")
    elif fault == "engine":
        monkeypatch.setattr(module, "_engine", lambda: "changed")
    with pytest.raises(ValueError):
        actual.upper_bound(
            day("2026-08-03"), ["ETH"] if fault == "query" else ["BTC"], None
        )


@pytest.mark.parametrize("kwargs", [{"max_bytes": 10}, {"max_rows": 1}])
def test_capacity_output_bounds_do_not_publish_partial_evidence(
    resources, qualified, kwargs
):
    from arblab.hyperliquid_copy import candidate_capacity as module

    with pytest.raises(ValueError):
        module.build_candidate_capacity(
            resources, QualifiedSourceSession(qualified), **kwargs
        )
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0


def test_capacity_per_market_and_pooled_overlap(resources, tmp_path):
    from arblab.hyperliquid_copy.candidate_capacity import (
        build_candidate_capacity,
        CandidateCapacity,
    )
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    _, archive, _, _ = inputs(tmp_path, days=3)
    raw = download_archive(archive, "2026-08-01", "2026-08-04", tmp_path / "downloaded")
    normalized = import_archive(
        raw, ["BTC", "ETH"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    pin = qualify([compact], tmp_path)
    source = QualifiedSourceSession(pin)
    counts = CandidateCapacity(
        resources, source, build_candidate_capacity(resources, source)
    )
    at = day("2026-08-02")
    assert counts.upper_bound(at, ["BTC", "ETH"], "BTC") == 2
    assert counts.upper_bound(at, ["BTC", "ETH"], "ETH") == 1
    assert counts.upper_bound(at, ["BTC", "ETH"], None) == 2
    assert counts.upper_bound(at, ["ETH"], None) == 1
    assert counts.upper_bound(source.origin, ["BTC", "ETH"], None) == 0


def test_capacity_source_span_rejects_before_reservation(resources, qualified):
    from arblab.hyperliquid_copy.candidate_capacity import build_candidate_capacity

    source = QualifiedSourceSession(qualified)
    object.__setattr__(source, "finish", source.origin + timedelta(days=733))
    before = resources.audit()
    with pytest.raises(ValueError, match="bounds"):
        build_candidate_capacity(resources, source)
    assert resources.audit() == before


@pytest.mark.parametrize(
    "fault", ["zero", "missing_pool", "duplicate", "date", "coin", "pooled_high"]
)
def test_malformed_producer_output_is_not_published(
    resources, qualified, monkeypatch, fault
):
    from arblab.hyperliquid_copy import candidate_capacity as module
    import pyarrow as pa

    original = module._write

    def malformed(source, scratch, target, max_bytes, max_rows):
        original(source, scratch, target, max_bytes, max_rows)
        rows = pq.read_table(target).to_pylist()
        if fault == "zero":
            rows[0]["entrants"] = 0
        elif fault == "missing_pool":
            rows = [row for row in rows if row["coin"] is not None]
        elif fault == "duplicate":
            rows.insert(0, rows[0])
        elif fault == "date":
            rows[0]["day"] = source.finish.date()
        elif fault == "coin":
            rows[-1]["coin"] = "UNKNOWN"
        else:
            rows[0]["entrants"] = 999
        pq.write_table(pa.Table.from_pylist(rows, schema=module.SCHEMA), target)

    monkeypatch.setattr(module, "_write", malformed)
    with pytest.raises(ValueError):
        module.build_candidate_capacity(resources, QualifiedSourceSession(qualified))
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0


@pytest.mark.parametrize("fault", ["caller", "lease", "source"])
def test_capacity_checks_after_engine_hashing(resources, qualified, monkeypatch, fault):
    from arblab.hyperliquid_copy import candidate_capacity as module

    source = QualifiedSourceSession(qualified)
    inputs = module.build_candidate_capacity(resources, source)
    actual = module.CandidateCapacity(resources, source, inputs)
    original = module._engine
    calls = []

    def changed():
        result = original()
        calls.append(True)
        # _verify_artifact verifies context before and after publication lookup.
        if len(calls) == 2:
            if fault == "caller":
                inputs["engine"] = "0" * 64
            elif fault == "lease":
                resources.lease.__exit__(None, None, None)
            else:
                path = Path(
                    json.loads(Path(qualified["path"]).read_text())["files"][0]["path"]
                )
                with path.open("ab") as handle:
                    handle.write(b"late mutation")
        return result

    monkeypatch.setattr(module, "_engine", changed)
    with pytest.raises(ValueError):
        actual.upper_bound(day("2026-08-03"), ["BTC"], None)


def test_capacity_rejects_oversized_decoding_before_batches(
    tmp_path, qualified, monkeypatch
):
    from arblab.hyperliquid_copy import candidate_capacity_records as records
    import pyarrow as pa

    source = QualifiedSourceSession(qualified)
    path = tmp_path / "oversized.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(day=source.origin.date(), coin="x" * 4096, entrants=1),
            ],
            schema=records.SCHEMA,
        ),
        path,
        compression="zstd",
    )
    monkeypatch.setattr(records, "MAX_DECODED_BYTES", 1024)
    monkeypatch.setattr(
        pq.ParquetFile,
        "iter_batches",
        lambda *a, **k: pytest.fail("decoded oversized column"),
    )
    with pytest.raises(ValueError, match="decoded"):
        records.read_counts(path, source, max_bytes=8 * 1024**2, max_rows=37332)


def test_staged_capacity_identity_spans_validation(resources, qualified, monkeypatch):
    from arblab.hyperliquid_copy import candidate_capacity as module
    import pyarrow as pa

    original, calls = module.read_counts, []

    def changed(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        calls.append(True)
        if len(calls) == 1:
            rows = pq.read_table(path).to_pylist()
            rows[0]["entrants"] = 0
            pq.write_table(pa.Table.from_pylist(rows, schema=module.SCHEMA), path)
        return result

    monkeypatch.setattr(module, "read_counts", changed)
    with pytest.raises(ValueError):
        module.build_candidate_capacity(resources, QualifiedSourceSession(qualified))
    with resources._connect() as db:
        assert db.execute("SELECT count(*) FROM publications").fetchone()[0] == 0


@pytest.mark.parametrize("maximum", [2, 37332])
def test_capacity_bounds_actual_decoded_rows(qualified, tmp_path, monkeypatch, maximum):
    from arblab.hyperliquid_copy import candidate_capacity_records as records
    import pyarrow as pa

    source = QualifiedSourceSession(qualified)
    path = tmp_path / "counts.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(day=source.origin.date(), coin=coin, entrants=1)
                for coin in (None, "BTC")
            ],
            schema=records.SCHEMA,
        ),
        path,
    )
    dates = [source.origin.date() + timedelta(days=i) for i in range(3)] * 2
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(dates, type=pa.date32()),
            pa.array([None] * 3 + ["BTC"] * 3, type=pa.string()).dictionary_encode(),
            pa.array([1] * 6, type=pa.int64()),
        ],
        names=["day", "coin", "entrants"],
    )
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", lambda *a, **k: iter([batch]))
    with pytest.raises(ValueError, match="row"):
        records.read_counts(path, source, max_bytes=8 * 1024**2, max_rows=maximum)
