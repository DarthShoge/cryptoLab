import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from .test_prefix_qualification import compact_batches, qualify


def scan(pin):
    from arblab.hyperliquid_copy.native_history_scan import scan_qualified_history

    return scan_qualified_history(pin)


@pytest.mark.parametrize("repeat", [False, True])
def test_scan_preserves_exact_event_bounds_and_physical_count(tmp_path, repeat):
    pin = qualify(compact_batches(tmp_path, repeat=repeat), tmp_path)
    original = json.loads(Path(pin["path"]).read_text())
    rows = [
        r
        for f in original["files"]
        for r in pq.read_table(f["path"], columns=["coin", "exchange_time"]).to_pylist()
    ]
    result = scan(pin)
    btc = result["markets"]["BTC"]
    assert btc["rows"] == len(rows) == 144
    assert btc["first_event"] == min(r["exchange_time"] for r in rows).isoformat()
    assert btc["last_event"] == max(r["exchange_time"] for r in rows).isoformat()
    assert result["source_end"] == "2026-08-04"
    assert result["qualification"] == pin
    assert result["native_availability_qualified"] is False
    assert "available_from" not in btc


@pytest.mark.parametrize("target", ["report", "file"])
def test_scan_rejects_changed_pinned_inputs(tmp_path, target):
    pin = qualify(compact_batches(tmp_path), tmp_path)
    path = Path(pin["path"])
    if target == "file":
        path = Path(json.loads(path.read_text())["files"][0]["path"])
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="identity|changed"):
        scan(pin)


def test_absent_declared_market_has_no_invented_dates(tmp_path):
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import inputs

    _, source, _, _ = inputs(tmp_path, days=1)
    raw = download_archive(source, "2026-08-01", "2026-08-02", tmp_path / "raw")
    normalized = import_archive(
        raw, ["BTC", "UNOBSERVED"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    result = scan(qualify([compact], tmp_path))
    assert result["markets"]["UNOBSERVED"] == dict(
        rows=0, first_event=None, last_event=None
    )
