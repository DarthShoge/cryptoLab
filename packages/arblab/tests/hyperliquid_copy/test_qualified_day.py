import json
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.download import file_hash
from .test_prefix_qualification import compact_batches, qualify


@pytest.fixture
def qualified(tmp_path):
    return qualify(compact_batches(tmp_path, repeat=True), tmp_path)


def test_event_day_includes_distant_source_spill(qualified):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay

    day = QualifiedDay(qualified, "2026-08-01")
    assert len(day.entries) == 2
    assert day.coins == ("BTC",)
    assert day.start.isoformat() == "2026-08-01T00:00:00+00:00"
    assert sum(e.rows for e in day.entries) == 96
    day.verify()
    assert "path" not in day.inputs()


def test_empty_event_day_retains_verified_schema_witness(qualified):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay

    day = QualifiedDay(qualified, "2026-08-03")
    assert day.entries == ()
    assert day.witness.rows > 0
    day.verify()


@pytest.mark.parametrize("day", ["2026-07-31", "2026-08-04", "2026-08-01T01:00:00Z"])
def test_outside_source_day_rejected(qualified, day):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay

    with pytest.raises(ValueError):
        QualifiedDay(qualified, day)


@pytest.mark.parametrize("fault", ["pin", "engine", "bounds", "duplicate", "symlink"])
def test_invalid_report_or_source_rejected(qualified, tmp_path, fault):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay

    pin = dict(qualified)
    report = Path(pin["path"])
    data = json.loads(report.read_text())
    if fault == "pin":
        pin["sha256"] = "0" * 64
    else:
        if fault == "engine":
            data["engine"]["version"] = 999
        elif fault == "bounds":
            data["files"][0]["min_time"] = "not an integer"
        elif fault == "duplicate":
            data["files"].append(data["files"][0])
        else:
            alias = tmp_path / "alias.parquet"
            alias.symlink_to(data["files"][0]["path"])
            data["files"][0]["path"] = str(alias)
        report.write_text(json.dumps(data))
        pin["sha256"] = file_hash(report)
    with pytest.raises(ValueError):
        QualifiedDay(pin, "2026-08-01")


def test_changed_source_is_rechecked_on_existing_snapshot(qualified):
    from arblab.hyperliquid_copy.qualified_day import QualifiedDay

    day = QualifiedDay(qualified, "2026-08-01")
    with day.entries[0].path.open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError):
        day.verify()
