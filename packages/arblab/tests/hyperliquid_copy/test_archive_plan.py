from datetime import datetime, timedelta, timezone
import json

import pytest

from arblab.hyperliquid_copy.archive import archive_keys


def inventory(tmp_path, *, start="2026-08-01", days=2, size=3):
    begin = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    objects = [
        dict(key=key, bytes=size, etag='"fixed"')
        for n in range(days)
        for key in archive_keys((begin + timedelta(days=n)).date().isoformat())
    ]
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            dict(
                schema="hyperliquid_annual_metadata_audit_v1",
                start=start,
                end=(begin + timedelta(days=days)).date().isoformat(),
                objects=objects,
                total_bytes=sum(o["bytes"] for o in objects),
                expected_objects=len(objects),
                listed_expected_objects=len(objects),
                missing=[],
            )
        )
    )
    return path


def test_plan_recomputes_chronological_bounded_batches(tmp_path):
    from arblab.hyperliquid_copy.archive_plan import plan_archive

    path = inventory(tmp_path, days=8)
    before = path.read_bytes()
    plan = plan_archive(path, max_batch_bytes=144)
    assert path.read_bytes() == before
    assert [b["bytes"] for b in plan["batches"]] == [144] * 4
    assert [b["start"] for b in plan["batches"]] == [
        "2026-08-01",
        "2026-08-03",
        "2026-08-05",
        "2026-08-07",
    ]
    assert plan["batches"][-1]["end"] == "2026-08-09"
    assert plan["total_bytes"] == 576
    assert plan["unsupported_objects"] == []
    assert len(plan["inventory_sha256"]) == 64
    assert [len(b["objects"]) for b in plan_archive(path)["batches"]] == [168, 24]
    assert plan_archive(path) == plan_archive(path)


def test_handoff_day_has_all_25_sources(tmp_path):
    from arblab.hyperliquid_copy.archive_plan import plan_archive

    plan = plan_archive(inventory(tmp_path, start="2025-07-27", days=1))
    assert len(plan["batches"][0]["objects"]) == 25
    assert plan["total_bytes"] == 75


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "total", "outside", "etag", "bool"]
)
def test_plan_rejects_inconsistent_or_invalid_inventory(tmp_path, change):
    from arblab.hyperliquid_copy.archive_plan import plan_archive

    path = inventory(tmp_path)
    data = json.loads(path.read_text())
    if change == "missing":
        data["objects"].pop()
    elif change == "duplicate":
        data["objects"][-1] = data["objects"][0]
    elif change == "total":
        data["total_bytes"] += 1
    elif change == "outside":
        data["objects"][0]["key"] = "node_fills_by_block/hourly/20260803/0.lz4"
    elif change == "etag":
        data["objects"][0]["etag"] = ""
    else:
        data["objects"][0]["bytes"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        plan_archive(path)


def test_plan_reports_oversized_sources_without_skipping_them(tmp_path):
    from arblab.hyperliquid_copy.archive_plan import plan_archive

    path = inventory(tmp_path, days=1)
    data = json.loads(path.read_text())
    data["objects"][0]["bytes"] = 385 * 1024**2
    data["total_bytes"] = sum(o["bytes"] for o in data["objects"])
    path.write_text(json.dumps(data))
    plan = plan_archive(path)
    assert len(plan["unsupported_objects"]) == 1
    assert len(plan["batches"][0]["objects"]) == 24
    assert not plan["transfer_ready"]
    with pytest.raises(ValueError, match="day"):
        plan_archive(path, max_batch_bytes=1024)


def test_read_only_planner_cli_reports_verified_reuse(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    from .test_archive_cache import cached

    path, cache = inventory(tmp_path), cached(tmp_path)
    tool = Path(__file__).resolve().parents[4] / "tools/plan_hyperliquid_archive_job.py"
    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--inventory",
            str(path),
            "--cache-manifest",
            str(cache),
            "--summary",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["reused_objects"] == 24
    assert data["reused_bytes"] == data["remaining_download_bytes"] == 72
    assert data["batch_count"] == 1
    assert data["authorization"] == "not_granted_by_plan"
    assert not list(tmp_path.rglob("*.sqlite3"))
