"""Exact selected-cohort quantities without a full-prefix activity reader."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from .test_archive import USER
from .test_candidate_day import resources
from .test_qualified_day import qualified

OTHER = "0x" + "cd" * 20
UNKNOWN = "0x" + "ef" * 20


def at(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "decision",
    [
        "2026-08-01",
        "2026-08-01T00:00:00.001",
        "2026-08-01T12:00",
        "2026-08-02",
        "2026-08-03T12:00",
        "2026-08-04",
    ],
)
def test_complete_mapping_matches_reference(qualified, resources, tmp_path, decision):
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions

    decision = at(decision)
    report = json.loads(Path(qualified["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        expected = {
            user: ref.position(user, "BTC", decision) for user in (USER, OTHER, UNKNOWN)
        }
    assert (
        native_positions(
            resources, qualified, decision, "BTC", ["0x" + "AB" * 20, OTHER, UNKNOWN]
        )
        == expected
    )
    assert resources.audit()["reserved_bytes"] == 0
    assert list((resources.root / "scratch").iterdir()) == []


@pytest.mark.parametrize(
    "decision,coin,users",
    [
        ("2026-07-31", "BTC", [USER]),
        ("2026-08-04T00:00:00.001", "BTC", [USER]),
        ("2026-08-02", "ETH", [USER]),
        ("2026-08-02", "BTC", [USER, USER]),
        ("2026-08-02", "BTC", [USER, "0x" + "AB" * 20]),
        ("2026-08-02", "BTC", ["bad"]),
        ("2026-08-02", "BTC", ["0x" + f"{i:040x}" for i in range(251)]),
    ],
)
def test_invalid_request_rejected_without_scratch(
    qualified, resources, decision, coin, users
):
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions

    with pytest.raises(ValueError):
        native_positions(resources, qualified, at(decision), coin, users)
    assert resources.audit()["reserved_bytes"] == 0


def test_empty_cohort_still_verifies_source_authority(qualified, resources):
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions

    assert native_positions(resources, qualified, at("2026-08-02"), "BTC", []) == {}
    invalid = dict(qualified, sha256="0" * 64)
    with pytest.raises(ValueError):
        native_positions(resources, invalid, at("2026-08-02"), "BTC", [])


def test_naive_decision_and_closed_lease_rejected(qualified, resources):
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions

    with pytest.raises(ValueError):
        native_positions(resources, qualified, datetime(2026, 8, 2), "BTC", [USER])
    resources.lease.__exit__(None, None, None)
    with pytest.raises(ValueError):
        native_positions(resources, qualified, at("2026-08-02"), "BTC", [USER])


@pytest.mark.parametrize(
    "fault", ["source", "engine", "pin", "users", "interrupt", "scratch"]
)
def test_midquery_changes_cannot_return_positions(
    qualified, resources, monkeypatch, fault
):
    from arblab.hyperliquid_copy import qualified_native_positions as module

    original = module._query_day
    users = [USER]
    owned = []

    def changed(db, day, *args):
        result = original(db, day, *args)
        if fault == "source":
            with day.entries[0].path.open("ab") as stream:
                stream.write(b"changed")
        elif fault == "engine":
            previous = module._engine()
            monkeypatch.setattr(module, "_engine", lambda: dict(previous, changed=True))
        elif fault == "pin":
            qualified["sha256"] = "0" * 64
        elif fault == "users":
            users.append(UNKNOWN)
        elif fault == "interrupt":
            raise RuntimeError("interrupted selected-position query")
        else:
            scratch = next((resources.root / "scratch").iterdir())
            saved = scratch.with_name(scratch.name + "_saved")
            scratch.rename(saved)
            scratch.mkdir()
            owned.append((scratch, saved))
        return result

    monkeypatch.setattr(module, "_query_day", changed)
    with pytest.raises((ValueError, RuntimeError), match="changed|interrupted"):
        module.native_positions(resources, qualified, at("2026-08-02"), "BTC", users)
    if fault == "scratch":
        # Restore the test-created moved directory, not a production recovery.
        scratch, saved = owned[0]
        scratch.rmdir()
        saved.rename(scratch)
        assert resources.audit()["reserved_bytes"] == module.SPILL_BYTES
    else:
        assert resources.audit()["reserved_bytes"] == 0


def test_shared_budget_rejects_before_connection(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_native_positions as module

    resources.reserve("staging/" + "a" * 32, 6 * 1024**3, "payload")
    before = resources.audit()

    def unexpected(*args, **kwargs):
        raise AssertionError("unbudgeted connection")

    monkeypatch.setattr(module.duckdb, "connect", unexpected)
    with pytest.raises(ValueError, match="budget"):
        module.native_positions(resources, qualified, at("2026-08-02"), "BTC", [USER])
    assert resources.audit() == before


def test_unknown_spill_stays_charged_after_failure(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_native_positions as module

    def interrupted(*args):
        scratch = next((resources.root / "scratch").iterdir())
        (scratch / "unfinished").write_bytes(b"unknown partial work")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "_query_day", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        module.native_positions(resources, qualified, at("2026-08-02"), "BTC", [USER])
    assert resources.audit()["reserved_bytes"] == module.SPILL_BYTES


def test_day_relation_remains_lazy(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_native_positions as module

    original = module._query_day
    plans = []

    def inspected(db, *args):
        result = original(db, *args)
        plans.append(db.execute("EXPLAIN SELECT * FROM native_day").fetchone()[1])
        return result

    monkeypatch.setattr(module, "_query_day", inspected)
    module.native_positions(resources, qualified, at("2026-08-02"), "BTC", [USER])
    assert plans and all(
        "PARQUET" in plan and "COLUMN_DATA_SCAN" not in plan for plan in plans
    )


def test_earlier_queried_file_changed_later_cannot_return(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import qualified_native_positions as module

    original = module._query_day
    seen = []

    def changed(db, day, *args):
        result = original(db, day, *args)
        if seen:
            assert seen[0] not in [entry.path for entry in day.entries]
            with seen[0].open("ab") as stream:
                stream.write(b"changed after earlier-day verification")
        seen.append(day.entries[0].path)
        return result

    monkeypatch.setattr(module, "_query_day", changed)
    with pytest.raises(ValueError, match="changed"):
        module.native_positions(
            resources, qualified, at("2026-08-03"), "BTC", [USER, UNKNOWN]
        )
    assert len(seen) == 2


def custom_history(tmp_path, change, *, days=3, start=None):
    import lz4.frame
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import Source
    from .test_proxy_archive_import import archive
    from .test_prefix_qualification import qualify

    start = start or at("2026-08-01")
    raw_fixture = archive(tmp_path, days=days, start=start)
    metadata = json.loads(raw_fixture.read_text())
    source = Source(
        {
            entry["key"]: (raw_fixture.parent / entry["file"]).read_bytes()
            for entry in metadata["objects"]
        }
    )
    for key, body in list(source.bodies.items()):
        block = json.loads(lz4.frame.decompress(body))
        change(key, block)
        source.bodies[key] = lz4.frame.compress(json.dumps(block).encode() + b"\n")
    from datetime import timedelta

    compacts = []
    for offset in range(0, days, 7):
        begin = (start + timedelta(days=offset)).date().isoformat()
        end = (start + timedelta(days=min(offset + 7, days))).date().isoformat()
        raw = download_archive(source, begin, end, tmp_path / "downloaded")
        normalized = import_archive(
            raw, ["BTC"], tmp_path / "normalized", retain_boundary_spill=True
        )
        compacts.append(
            compact_history(normalized, tmp_path / "compact", partitioning="source_day")
        )
    return qualify(compacts, tmp_path)


def test_native_dedup_precedes_latest_order_and_preserves_known_zero(
    tmp_path, resources
):
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions
    from .test_archive import raw_fill

    zero = "0x" + "12" * 20

    def change(key, block):
        block["events"] = []
        moment = int(at("2026-08-01T12:00").timestamp() * 1000)
        first = [
            USER,
            raw_fill(time=moment, tid=1000, startPosition="2", dir="Open Long"),
        ]
        if key.endswith("20260801/12.lz4"):
            block["events"] = [
                first,
                [
                    USER,
                    raw_fill(time=moment, tid=1001, startPosition="4", dir="Open Long"),
                ],
                [
                    OTHER,
                    raw_fill(
                        time=moment,
                        tid=1002,
                        startPosition="0",
                        side="A",
                        dir="Open Short",
                    ),
                ],
                [
                    zero,
                    raw_fill(
                        time=moment,
                        tid=1003,
                        startPosition="2",
                        side="A",
                        dir="Close Long",
                    ),
                ],
            ]
        elif key.endswith("20260803/0.lz4"):
            block.update(
                block_time=at("2026-08-01T12:00").isoformat(),
                block_number=999,
                events=[first],
            )
        elif key.endswith("20260802/23.lz4"):
            block["events"] = [
                [
                    UNKNOWN,
                    raw_fill(
                        time=int(at("2026-08-02T23:00").timestamp() * 1000),
                        tid=1004,
                        startPosition="9",
                        dir="Open Long",
                    ),
                ]
            ]

    pin = custom_history(tmp_path, change)
    decision = at("2026-08-02")
    expected = {USER: 6.0, OTHER: -2.0, zero: 0.0, UNKNOWN: None}
    assert native_positions(resources, pin, decision, "BTC", list(expected)) == expected
    report = json.loads(Path(pin["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        assert {
            user: ref.position(user, "BTC", decision) for user in expected
        } == expected


def test_overlapping_days_do_not_retain_day_objects_or_duplicate_file_pins(
    tmp_path, resources, monkeypatch
):
    import weakref
    from arblab.hyperliquid_copy import qualified_native_positions as module

    def change(key, block):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        # Every physical day spans the same three event days, legitimately.
        moment = at("2026-08-01") if hour < 12 else at("2026-08-03")
        block["block_time"] = moment.isoformat()
        for user, event in block["events"]:
            event["time"] = int(moment.timestamp() * 1000)

    pin = custom_history(tmp_path, change)
    original = module.QualifiedDay
    references = []

    def tracked(*args):
        assert not any(ref() is not None for ref in references)
        day = original(*args)
        references.append(weakref.ref(day))
        return day

    sizes = []
    remember = module._Source.remember

    def remembered(source, day):
        remember(source, day)
        sizes.append(len(source.files))

    monkeypatch.setattr(module, "QualifiedDay", tracked)
    monkeypatch.setattr(module._Source, "remember", remembered)
    assert module.native_positions(
        resources, pin, at("2026-08-04"), "BTC", [UNKNOWN]
    ) == {UNKNOWN: None}
    assert len(references) == 4
    assert sizes == [3, 3, 3, 3]
    assert not any(ref() is not None for ref in references)


def test_whale_above_100000_same_day_fills_returns_final_quantity(tmp_path, resources):
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions

    count = 100_002

    def change(key, block):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        user, fill = block["events"][0]
        block["events"] = [
            [user, dict(fill, tid=i, startPosition=str(2 * i), dir="Open Long")]
            for i in range(hour * 4167, min((hour + 1) * 4167, count))
        ]

    pin = custom_history(tmp_path, change, days=1)
    assert native_positions(resources, pin, at("2026-08-02"), "BTC", [USER]) == {
        USER: float(2 * count)
    }
    assert resources.audit()["reserved_bytes"] == 0


def test_dormant_position_before_90_day_lookback_survives(tmp_path, resources):
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions

    def change(key, block):
        user, fill = block["events"][0]
        block["events"] = [[USER if key.endswith("20250801/0.lz4") else OTHER, fill]]

    pin = custom_history(tmp_path, change, days=94, start=at("2025-08-01"))
    decision = at("2025-11-03")
    assert (decision - at("2025-08-01")).days > 90
    assert native_positions(resources, pin, decision, "BTC", [USER, UNKNOWN]) == {
        USER: 1.0,
        UNKNOWN: None,
    }
    assert resources.audit()["reserved_bytes"] == 0


def test_preorigin_spill_matches_reference_clipped_to_authorized_interval(
    tmp_path, resources
):
    from arblab.hyperliquid_copy.qualified_native_positions import native_positions
    from arblab.hyperliquid_copy.registration_partitions import publish_interior

    def change(key, block):
        user, fill = block["events"][0]
        block["events"] = [[OTHER, fill]]
        if key.endswith("20260801/0.lz4"):
            moment = at("2026-07-31T23:59:59.999")
            block["block_time"] = moment.isoformat()
            block["events"] = [[USER, dict(fill, time=int(moment.timestamp() * 1000))]]

    pin = custom_history(tmp_path, change, days=1)
    report = json.loads(Path(pin["path"]).read_text())
    stage = tmp_path / "interior"
    stage.mkdir()
    entries = publish_interior(
        report["files"], at("2026-08-01"), at("2026-08-02"), stage
    )
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as raw:
        assert raw.position(USER, "BTC", at("2026-08-01")) == 1.0
    with ProxyActivity(
        [stage / entry["name"] for entry in entries], temp_root=tmp_path
    ) as reference:
        for decision in (at("2026-08-01"), at("2026-08-02")):
            expected = {
                user: reference.position(user, "BTC", decision)
                for user in (USER, OTHER)
            }
            assert (
                native_positions(resources, pin, decision, "BTC", [USER, OTHER])
                == expected
            )
            assert expected[USER] is None


def test_busy_cache_cannot_obtain_second_query_lease(qualified, resources):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheBusyError, CacheLease

    before = resources.audit()
    with pytest.raises(CacheBusyError):
        with CacheLease(resources.root):
            pytest.fail("second selected-position query obtained busy lease")
    assert resources.audit() == before


def test_day_bound_checked_before_day_construction(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_native_positions as module
    from arblab.hyperliquid_copy.download import file_hash

    path = Path(qualified["path"])
    report = json.loads(path.read_text())
    report["source_start"] = "2020-01-01"
    path.write_text(json.dumps(report))
    qualified["sha256"] = file_hash(path)

    def unexpected(*args):
        raise AssertionError("day metadata allocated before span validation")

    monkeypatch.setattr(module, "QualifiedDay", unexpected)
    with pytest.raises(ValueError, match="day bound"):
        module.native_positions(resources, qualified, at("2026-08-02"), "BTC", [USER])
