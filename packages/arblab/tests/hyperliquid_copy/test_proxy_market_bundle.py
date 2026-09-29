from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.proxy_sessions import calendar_provenance
from .test_proxy_download import mapping

START = datetime(2026, 8, 3, tzinfo=timezone.utc)


def segment(tmp_path, name, kind, hours, *, rate=0.001, missing=(), complete=True):
    root = tmp_path / name
    root.mkdir()
    raw = root / "source_0000.raw"
    raw.write_bytes(b"synthetic fixture")
    times = [START + timedelta(hours=h) for h in hours]
    rows = []
    for at in times:
        if at.hour in missing:
            continue
        if kind == "funding":
            rows.append(
                dict(
                    instrument_id="BTC",
                    hour=at,
                    time=at + timedelta(milliseconds=80),
                    rate=rate,
                    premium=0.0,
                )
            )
        else:
            rows.append(
                dict(
                    instrument_id="BTC",
                    start=at,
                    end=at + timedelta(hours=1),
                    open=100.0,
                    high=110.0,
                    low=90.0,
                    close=105.0,
                )
            )
    filename = "funding.parquet" if kind == "funding" else "bars.parquet"
    pq.write_table(pa.Table.from_pylist(rows), root / filename)
    metadata = dict(
        schema=f"hyperliquid_proxy_{kind}_v1",
        complete=complete,
        start=min(times).isoformat(),
        end=(max(times) + timedelta(hours=1)).isoformat(),
        sources=[dict(file=raw.name, bytes=raw.stat().st_size, sha256=file_hash(raw))],
        coverage=[dict(instrument_id="BTC", missing_hours=[])]
        if kind == "funding"
        else [
            dict(
                instrument_id="BTC",
                missing_starts=[
                    (START + timedelta(hours=h)).isoformat() for h in missing
                ],
            )
        ],
    )
    metadata["funding_sha256" if kind == "funding" else "bars_sha256"] = file_hash(
        root / filename
    )
    if kind == "prices":
        sessions = root / "sessions.json"
        sessions.write_text(
            json.dumps(
                {
                    "BTC": {
                        t.isoformat(): (t + timedelta(hours=1)).isoformat()
                        for t in times
                    }
                }
            )
        )
        metadata.update(
            mappings=[asdict(mapping())],
            calendar=calendar_provenance(),
            price_policy=dict(
                adjustment="raw",
                corporate_actions="none_detected",
                rolls="not_applicable",
            ),
            sessions_sha256=file_hash(sessions),
        )
    path = root / "manifest.json"
    path.write_text(json.dumps(metadata))
    return path


def bundle(tmp_path, kind, paths, begin=0, end=4):
    from arblab.hyperliquid_copy.proxy_market_bundle import bundle_market_inputs

    return bundle_market_inputs(
        kind,
        paths,
        {"BTC": (START + timedelta(hours=begin), START + timedelta(hours=end))},
        tmp_path / "bundles",
    )


@pytest.mark.parametrize("kind", ["funding", "prices"])
def test_bundle_combines_segments_and_preserves_source_pins(tmp_path, kind):
    paths = [
        segment(tmp_path, "a", kind, range(2)),
        segment(tmp_path, "b", kind, range(2, 4)),
    ]
    result = bundle(tmp_path, kind, paths)
    data = json.loads(result.read_text())
    assert data["research_eligible"] is False
    assert data["file"]["rows"] == 4
    assert [e["sha256"] for e in data["source_manifests"]] == [
        file_hash(p) for p in paths
    ]
    rows = pq.read_table(result.parent / data["file"]["name"]).to_pylist()
    if kind == "funding":
        assert rows[0]["time"] == START + timedelta(milliseconds=80)


@pytest.mark.parametrize("kind", ["funding", "prices"])
def test_exact_overlap_is_deduplicated(tmp_path, kind):
    paths = [
        segment(tmp_path, "a", kind, range(3)),
        segment(tmp_path, "b", kind, range(2, 4)),
    ]
    data = json.loads(bundle(tmp_path, kind, paths).read_text())
    assert data["file"]["rows"] == 4


def test_conflicting_funding_rejects_before_output(tmp_path):
    paths = [
        segment(tmp_path, "a", "funding", range(3)),
        segment(tmp_path, "b", "funding", range(2, 4), rate=0.002),
    ]
    with pytest.raises(ValueError, match="conflict"):
        bundle(tmp_path, "funding", paths)
    assert not list((tmp_path / "bundles").glob("*/manifest.json"))


@pytest.mark.parametrize("kind", ["funding", "prices"])
def test_missing_required_hour_is_not_zero_or_forward_filled(tmp_path, kind):
    paths = [segment(tmp_path, "a", kind, [0, 1, 3])]
    with pytest.raises(ValueError, match="coverage|missing"):
        bundle(tmp_path, kind, paths)


def test_price_subinterval_preserves_original_incomplete_source_evidence(tmp_path):
    path = segment(tmp_path, "a", "prices", range(4), missing=[0], complete=False)
    data = json.loads(bundle(tmp_path, "prices", [path], begin=1).read_text())
    assert data["file"]["rows"] == 3
    assert data["source_manifests"][0]["complete"] is False
    assert data["source_manifests"][0]["coverage"][0]["missing_starts"] == [
        START.isoformat()
    ]
    assert data["intervals"]["BTC"][0] == (START + timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError, match="coverage|missing"):
        bundle(tmp_path, "prices", [path])


def test_raw_corruption_rejects(tmp_path):
    path = segment(tmp_path, "a", "funding", range(4))
    (path.parent / "source_0000.raw").write_bytes(b"bad")
    with pytest.raises(ValueError, match="identity|hash"):
        bundle(tmp_path, "funding", [path])


def test_inconsistent_price_policy_rejects(tmp_path):
    paths = [
        segment(tmp_path, "a", "prices", range(2)),
        segment(tmp_path, "b", "prices", range(2, 4)),
    ]
    d = json.loads(paths[1].read_text())
    d["price_policy"]["corporate_actions"] = "unknown"
    paths[1].write_text(json.dumps(d))
    with pytest.raises(ValueError, match="policy"):
        bundle(tmp_path, "prices", paths)


def rewrite_rows(path, kind, mutate):
    filename = "bars.parquet" if kind == "prices" else "funding.parquet"
    rows = pq.read_table(path.parent / filename).to_pylist()
    mutate(rows)
    pq.write_table(pa.Table.from_pylist(rows), path.parent / filename)
    data = json.loads(path.read_text())
    data["bars_sha256" if kind == "prices" else "funding_sha256"] = file_hash(
        path.parent / filename
    )
    path.write_text(json.dumps(data))


def test_conflicting_prices_outside_retained_interval_reject(tmp_path):
    paths = [
        segment(tmp_path, "a", "prices", range(4)),
        segment(tmp_path, "b", "prices", range(4)),
    ]
    rewrite_rows(paths[1], "prices", lambda rows: rows[0].update(close=106.0))
    with pytest.raises(ValueError, match="conflict"):
        bundle(tmp_path, "prices", paths, begin=1)


def test_invalid_prices_outside_retained_interval_reject(tmp_path):
    path = segment(tmp_path, "a", "prices", range(4))
    rewrite_rows(
        path, "prices", lambda rows: rows[0].update(end=START + timedelta(minutes=30))
    )
    with pytest.raises(ValueError, match="calendar"):
        bundle(tmp_path, "prices", [path], begin=1)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"instrument_id": "ETH"}, "instrument"),
        ({"time": START + timedelta(hours=1)}, "Settlement"),
        ({"rate": "0.001"}, "numeric"),
    ],
)
def test_invalid_funding_records_reject(tmp_path, change, match):
    path = segment(tmp_path, "a", "funding", range(4))
    if "rate" in change:
        rewrite_rows(path, "funding", lambda rows: [r.update(change) for r in rows])
    else:
        rewrite_rows(path, "funding", lambda rows: rows[0].update(change))
    with pytest.raises(ValueError, match=match):
        bundle(tmp_path, "funding", [path])


def test_input_row_bound_checked_before_decode(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import market_bundle_inputs

    path = segment(tmp_path, "a", "funding", range(4))
    monkeypatch.setattr(market_bundle_inputs, "MAX_ROWS", 3)
    with pytest.raises(ValueError, match="row bound"):
        bundle(tmp_path, "funding", [path])


def test_repeat_bundle_never_overwrites(tmp_path):
    path = segment(tmp_path, "a", "funding", range(4))
    first = bundle(tmp_path, "funding", [path])
    digest = file_hash(first)
    second = bundle(tmp_path, "funding", [path])
    assert first != second and file_hash(first) == digest


def test_changed_input_before_publish_rejects(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import proxy_market_bundle

    path = segment(tmp_path, "a", "funding", range(4))
    original = proxy_market_bundle.validated_rows

    def mutate(*args):
        result = original(*args)
        path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(proxy_market_bundle, "validated_rows", mutate)
    with pytest.raises(ValueError, match="identity"):
        bundle(tmp_path, "funding", [path])
    assert not list((tmp_path / "bundles").glob("*/manifest.json"))


def test_engine_pin_includes_input_validation_dependency(tmp_path):
    path = segment(tmp_path, "a", "funding", range(4))
    data = json.loads(bundle(tmp_path, "funding", [path]).read_text())
    assert any(p.endswith("market_bundle_inputs.py") for p in data["engine"])


def test_source_range_rejected_before_calendar_expansion(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import proxy_market_bundle

    path = segment(tmp_path, "a", "prices", range(4))
    data = json.loads(path.read_text())
    data["start"] = "1900-01-01T00:00:00+00:00"
    path.write_text(json.dumps(data))

    def forbidden(*args):
        pytest.fail("Calendar expansion happened before source bound check")

    monkeypatch.setattr(proxy_market_bundle, "windows", forbidden)
    with pytest.raises(ValueError, match="source interval"):
        bundle(tmp_path, "prices", [path])


def test_nanosecond_settlement_is_not_silently_truncated(tmp_path):
    path = segment(tmp_path, "a", "funding", range(4))
    parquet = path.parent / "funding.parquet"
    table = pq.read_table(parquet)
    nanos = [int(v.timestamp() * 1_000_000_000) + 1 for v in table["time"].to_pylist()]
    table = table.set_column(
        table.schema.get_field_index("time"),
        "time",
        pa.array(nanos, type=pa.timestamp("ns", tz="UTC")),
    )
    pq.write_table(table, parquet)
    data = json.loads(path.read_text())
    data["funding_sha256"] = file_hash(parquet)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="precision"):
        bundle(tmp_path, "funding", [path])


@pytest.mark.parametrize("field", ["calendar", "mappings"])
def test_inconsistent_price_metadata_rejects(tmp_path, field):
    paths = [
        segment(tmp_path, "a", "prices", range(2)),
        segment(tmp_path, "b", "prices", range(2, 4)),
    ]
    data = json.loads(paths[1].read_text())
    if field == "calendar":
        data[field]["version"] = "different"
    else:
        data[field][0]["unit"] = "different"
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="calendar|mapping"):
        bundle(tmp_path, "prices", paths)


def test_shortened_exchange_session_uses_actual_closing_bar(tmp_path):
    from arblab.hyperliquid_copy.proxy_market_bundle import bundle_market_inputs
    from arblab.hyperliquid_copy.proxy_sessions import hourly_windows

    path = segment(tmp_path, "a", "prices", range(4))
    begin = datetime(2026, 11, 27, tzinfo=timezone.utc)
    end = begin + timedelta(days=1)
    scheduled = hourly_windows("XNYS", begin, end)
    assert len(scheduled) == 4
    rows = [
        dict(
            instrument_id="xyz:TSLA",
            start=a,
            end=b,
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
        )
        for a, b in scheduled.items()
    ]
    pq.write_table(pa.Table.from_pylist(rows), path.parent / "bars.parquet")
    data = json.loads(path.read_text())
    data.update(
        start=begin.isoformat(),
        end=end.isoformat(),
        coverage=[dict(instrument_id="xyz:TSLA", missing_starts=[])],
    )
    data["mappings"][0].update(
        instrument_id="xyz:TSLA",
        provider="yahoo",
        ticker="TSLA",
        asset_class="equities",
        calendar="XNYS",
        valid_from="2026-11-01",
        valid_to="2026-12-01",
    )
    sessions = path.parent / "sessions.json"
    sessions.write_text(
        json.dumps(
            {"xyz:TSLA": {a.isoformat(): b.isoformat() for a, b in scheduled.items()}}
        )
    )
    data.update(
        bars_sha256=file_hash(path.parent / "bars.parquet"),
        sessions_sha256=file_hash(sessions),
    )
    path.write_text(json.dumps(data))
    result = bundle_market_inputs(
        "prices", [path], {"xyz:TSLA": (begin, end)}, tmp_path / "bundles"
    )
    saved = pq.read_table(result.parent / "bars.parquet").to_pylist()
    assert saved[-1]["end"] - saved[-1]["start"] == timedelta(minutes=30)
