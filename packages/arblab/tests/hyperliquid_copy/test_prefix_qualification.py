from datetime import datetime, timedelta
import json
from pathlib import Path

import lz4.frame
import pytest

from arblab.hyperliquid_copy.proxy_archive_download import download_archive
from arblab.hyperliquid_copy.proxy_archive_import import import_archive
from arblab.hyperliquid_copy.proxy_compact import compact_history
from .test_archive_job import inputs


def compact_batches(tmp_path, *, repeat=False, conflict=None):
    _, source, _, _ = inputs(tmp_path, days=3)
    if repeat:
        for key in list(source.bodies):
            if "20260803" in key:
                source.bodies[key] = source.bodies[key.replace("20260803", "20260801")]
        if conflict:
            key = "node_fills_by_block/hourly/20260803/0.lz4"
            data = json.loads(lz4.frame.decompress(source.bodies[key]))
            for i, pair in enumerate(data["events"][:2]):
                pair[1]["px"] = str(float(pair[1]["px"]) + 1)
                if conflict == "counterparty":
                    pair[0] = "0x" + ("ef" if i else "fe") * 20
            source.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    result = []
    begin = datetime(2026, 8, 1)
    for n in range(3):
        a, b = [(begin + timedelta(days=d)).date().isoformat() for d in (n, n + 1)]
        raw = download_archive(source, a, b, tmp_path / "downloaded")
        normalized = import_archive(
            raw, ["BTC"], tmp_path / "normalized", retain_boundary_spill=True
        )
        result.append(
            compact_history(normalized, tmp_path / "compact", partitioning="source_day")
        )
    return result


def qualify(paths, tmp_path, previous=None):
    from arblab.hyperliquid_copy.prefix_qualification import qualify_prefix

    return qualify_prefix(
        paths,
        previous=previous,
        output_root=tmp_path / "qualified",
        temp_root=tmp_path,
        buckets=1,
    )


def test_incremental_prefix_prunes_only_disjoint_trade_ranges(tmp_path):
    paths = compact_batches(tmp_path)
    first = qualify(paths[:1], tmp_path)
    second = qualify(paths[:2], tmp_path, previous=first)
    report = json.loads(Path(second["path"]).read_text())
    assert report["status"] == "canonical_prefix_validated"
    assert report["research_eligible"] is False
    assert report["raw_disposal_authorized"] is False
    assert report["candidate_files"] == 1
    assert report["skipped_old_files"] == 1
    assert report["rows"] == 96
    assert len(report["manifests"]) == 2
    assert report["source_end"] == "2026-08-03"
    assert len(report["raw_sources"]) == 2
    assert len(report["raw_sources"][0]["objects"]) == 24


def test_distant_exact_duplicates_and_spill_are_retained(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity

    paths = compact_batches(tmp_path, repeat=True)
    prior = qualify(paths[:2], tmp_path)
    result = qualify(paths, tmp_path, previous=prior)
    report = json.loads(Path(result["path"]).read_text())
    assert report["rows"] == 144
    assert report["candidate_files"] == 2  # First and third days, not adjacent day two.
    assert report["skipped_old_files"] == 1
    full = qualify(paths, tmp_path)
    assert json.loads(Path(full["path"]).read_text())["rows"] == report["rows"]
    files = [Path(e["path"]) for e in report["files"]]
    with ProxyActivity(files, temp_root=tmp_path) as activity:
        assert activity.count == 96


@pytest.mark.parametrize(
    "kind,match", [("economics", "economic"), ("counterparty", "counterpart")]
)
def test_distant_conflicts_reject_without_publishing(tmp_path, kind, match):
    paths = compact_batches(tmp_path, repeat=True, conflict=kind)
    prior = qualify(paths[:2], tmp_path)
    before = list((tmp_path / "qualified").glob("*/manifest.json"))
    with pytest.raises(ValueError, match=match):
        qualify(paths, tmp_path, previous=prior)
    assert list((tmp_path / "qualified").glob("*/manifest.json")) == before


def test_previous_certificate_must_match_pinned_identity(tmp_path):
    paths = compact_batches(tmp_path)
    prior = qualify(paths[:1], tmp_path)
    p = Path(prior["path"])
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="previous|identity"):
        qualify(paths[:2], tmp_path, previous=prior)


def test_previous_prefix_cannot_be_replaced_or_repeated(tmp_path):
    paths = compact_batches(tmp_path)
    prior = qualify(paths[:1], tmp_path)
    for replacement in (paths[:1], paths[1:]):
        with pytest.raises(ValueError, match="prefix|extend"):
            qualify(replacement, tmp_path, previous=prior)


@pytest.mark.parametrize("target", ["compact", "raw"])
def test_manifest_snapshot_cannot_change_between_read_and_validation(
    tmp_path, monkeypatch, target
):
    from arblab.hyperliquid_copy import qualification_bounds as bounds

    paths = compact_batches(tmp_path)
    original = bounds.read_json
    compact = Path(paths[0])
    raw = Path(json.loads(compact.read_text())["source_evidence"]["source_manifest"])
    chosen = compact if target == "compact" else raw

    def changed(path, *args):
        data = original(path, *args)
        if Path(path) == chosen:
            chosen.write_text(chosen.read_text() + " ")
            if target == "raw":
                # Simulate a newly pinned provenance identity after the old read.
                from arblab.hyperliquid_copy.download import file_hash

                evidence = json.loads(compact.read_text())
                evidence["source_evidence"]["source_manifest_sha256"] = file_hash(raw)
                compact.write_text(json.dumps(evidence))
        return data

    monkeypatch.setattr(bounds, "read_json", changed)
    with pytest.raises(ValueError, match="identity|changed"):
        qualify(paths[:1], tmp_path)
    assert not list((tmp_path / "qualified").glob("*/manifest.json"))


def test_old_disjoint_file_is_rehashed_before_reuse(tmp_path):
    paths = compact_batches(tmp_path)
    prior = qualify(paths[:1], tmp_path)
    report = json.loads(Path(prior["path"]).read_text())
    with Path(report["files"][0]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity|hash"):
        qualify(paths[:2], tmp_path, previous=prior)


def test_schema_change_requires_new_full_validation_version(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from arblab.hyperliquid_copy.download import file_hash

    paths = compact_batches(tmp_path)
    prior = qualify(paths[:1], tmp_path)
    manifest = Path(paths[1])
    data = json.loads(manifest.read_text())
    entry = data["files"][0]
    file = manifest.parent / entry["name"]
    table = pq.read_table(file)
    index = table.schema.get_field_index("px")
    table = table.set_column(index, "px", table.column(index).cast(pa.int64()))
    pq.write_table(table, file)
    entry.update(bytes=file.stat().st_size, sha256=file_hash(file))
    data["output_bytes"] = entry["bytes"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema changed"):
        qualify(paths[:2], tmp_path, previous=prior)


def test_incomplete_raw_membership_rejects_even_with_updated_digest(tmp_path):
    from arblab.hyperliquid_copy.download import file_hash

    paths = compact_batches(tmp_path)
    manifest = Path(paths[0])
    data = json.loads(manifest.read_text())
    raw = Path(data["source_evidence"]["source_manifest"])
    source = json.loads(raw.read_text())
    source["objects"].pop()
    raw.write_text(json.dumps(source))
    data["source_evidence"]["source_manifest_sha256"] = file_hash(raw)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source-key membership"):
        qualify(paths[:1], tmp_path)


def test_mutation_during_validation_prevents_publication(tmp_path):
    from arblab.hyperliquid_copy.prefix_qualification import qualify_prefix

    paths = compact_batches(tmp_path)
    manifest = Path(paths[0])

    def mutate(_):
        manifest.write_text(manifest.read_text() + " ")

    with pytest.raises(ValueError, match="identity changed"):
        qualify_prefix(
            paths[:1],
            output_root=tmp_path / "qualified",
            temp_root=tmp_path,
            buckets=1,
            progress=mutate,
        )
    assert not list((tmp_path / "qualified").glob("*/manifest.json"))


def test_interval_overlap_is_per_coin_and_inclusive():
    from arblab.hyperliquid_copy.qualification_bounds import overlaps

    assert overlaps({"BTC": [1, 9]}, {"BTC": [9, 20]})
    assert not overlaps({"BTC": [1, 9]}, {"ETH": [1, 9]})
    assert not overlaps(
        {"BTC": [1, 9], "ETH": [20, 30]}, {"BTC": [10, 30], "ETH": [1, 19]}
    )


def test_previous_report_cannot_be_restored_after_reading_unpinned_bytes(
    tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import prefix_qualification as prefix

    paths = compact_batches(tmp_path)
    pin = qualify(paths[:1], tmp_path)
    path = Path(pin["path"])
    pinned = path.read_text()
    path.write_text(pinned + " ")
    original = prefix.read_json

    def restore(path, *args):
        data = original(path, *args)
        Path(path).write_text(pinned)
        return data

    monkeypatch.setattr(prefix, "read_json", restore)
    with pytest.raises(ValueError, match="identity"):
        prefix._previous(pin, prefix._engine())
