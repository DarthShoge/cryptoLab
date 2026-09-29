"""Strict native hourly samples, with bounded seed and replay queries."""

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
    "start,end,age,user",
    [
        ("2026-08-01", "2026-08-01T01:00", 3600, USER),
        ("2026-08-01", "2026-08-04", 3600, USER),
        ("2026-08-01", "2026-08-03", 3599, USER),
        ("2026-08-01T00:30", "2026-08-02T23:30", 3600, USER),
        ("2026-08-02T12:00", "2026-08-04", 86400, OTHER),
        ("2026-08-01", "2026-08-04", 3600, UNKNOWN),
        ("2026-08-03", "2026-08-04", 7200, USER),
    ],
)
def test_all_sample_values_and_times_match_reference(
    qualified, resources, tmp_path, start, end, age, user
):
    from arblab.hyperliquid_copy.qualified_hourly_exposure import hourly_exposure

    report = json.loads(Path(qualified["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        expected = ref.hourly_exposure(
            user, "BTC", at(start), at(end), max_price_age_seconds=age
        )
    actual = hourly_exposure(
        resources, qualified, user, "BTC", at(start), at(end), max_price_age_seconds=age
    )
    assert actual == expected
    assert resources.audit()["reserved_bytes"] == 0
    assert list((resources.root / "scratch").iterdir()) == []


@pytest.mark.parametrize(
    "start,end,age,user,coin",
    [
        ("2026-07-31", "2026-08-02", 3600, USER, "BTC"),
        ("2026-08-01", "2026-08-05", 3600, USER, "BTC"),
        ("2026-08-02", "2026-08-02", 3600, USER, "BTC"),
        ("2026-08-02", "2026-08-01", 3600, USER, "BTC"),
        ("2026-08-01", "2026-08-01T00:30", 3600, USER, "BTC"),
        ("2026-08-01", "2026-08-02", 0, USER, "BTC"),
        ("2026-08-01", "2026-08-02", float("inf"), USER, "BTC"),
        ("2026-08-01", "2026-08-02", 3600, "bad", "BTC"),
        ("2026-08-01", "2026-08-02", 3600, USER, "ETH"),
    ],
)
def test_invalid_request_fails_without_scratch(
    qualified, resources, start, end, age, user, coin
):
    from arblab.hyperliquid_copy.qualified_hourly_exposure import hourly_exposure

    with pytest.raises(ValueError):
        hourly_exposure(
            resources,
            qualified,
            user,
            coin,
            at(start),
            at(end),
            max_price_age_seconds=age,
        )
    assert resources.audit()["reserved_bytes"] == 0


@pytest.fixture
def transitions(tmp_path):
    from .test_archive import raw_fill
    from .test_qualified_native_positions import custom_history

    def fill(moment, tid, **changes):
        return raw_fill(time=int(at(moment).timestamp() * 1000), tid=tid, **changes)

    opening = [
        USER,
        fill("2026-08-01", 1, startPosition="0", px="100", dir="Open Long"),
    ]
    before_tie = [
        USER,
        fill("2026-08-01T23:00", 2, startPosition="2", px="120", dir="Open Long"),
    ]
    after_tie = [
        USER,
        fill("2026-08-01T23:00", 22, startPosition="4", px="130", dir="Open Long"),
    ]
    events = {
        "20260801/0.lz4": [opening],
        "20260801/1.lz4": [[OTHER, fill("2026-08-01T01:00", 10, px="101")]],
        "20260801/23.lz4": [before_tie, after_tie],
        "20260802/0.lz4": [
            [
                USER,
                fill(
                    "2026-08-02",
                    3,
                    startPosition="6",
                    sz="6",
                    side="A",
                    px="80",
                    dir="Close Long",
                ),
            ]
        ],
        "20260802/1.lz4": [[OTHER, fill("2026-08-02T01:00", 11, px="75")]],
        "20260802/2.lz4": [
            [
                USER,
                fill(
                    "2026-08-02T02:00",
                    4,
                    startPosition="0",
                    side="A",
                    px="90",
                    dir="Open Short",
                ),
            ]
        ],
        "20260803/23.lz4": [
            [
                UNKNOWN,
                fill(
                    "2026-08-03T23:00", 12, startPosition="9", px="200", dir="Open Long"
                ),
            ]
        ],
    }

    def change(key, block):
        block["events"] = events.get("/".join(key.split("/")[-2:]), [])
        if key.endswith("20260803/0.lz4"):
            block.update(
                block_time=at("2026-08-01T23:00").isoformat(),
                block_number=999,
                events=[before_tie],
            )

    return custom_history(tmp_path, change)


def test_chunk_edges_native_order_zero_short_and_future_knownness(
    transitions, resources, tmp_path
):
    from arblab.hyperliquid_copy.qualified_hourly_exposure import hourly_exposure

    report = json.loads(Path(transitions["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        for user in (USER, UNKNOWN):
            expected = ref.hourly_exposure(
                user,
                "BTC",
                at("2026-08-01"),
                at("2026-08-04"),
                max_price_age_seconds=999999,
            )
            actual = hourly_exposure(
                resources,
                transitions,
                user,
                "BTC",
                at("2026-08-01"),
                at("2026-08-04"),
                max_price_age_seconds=999999,
            )
            assert actual == expected
            if user == USER:
                assert [actual[i][1] for i in (0, 1, 2, 23, 24, 25, 27)] == [
                    None,
                    200,
                    202,
                    202,
                    780,
                    0,
                    -180,
                ]
            else:
                assert all(value is None for _, value in actual)


def test_conviction_incremental_samples_and_scale_match_reference(
    transitions, resources, tmp_path
):
    from dataclasses import replace
    from types import SimpleNamespace
    from arblab.hyperliquid_copy.qualified_hourly_exposure import hourly_exposure
    from arblab.hyperliquid_copy.proxy_conviction import ProxyConviction
    from .test_lab_pipeline_proxy import config

    settings = config()
    settings = replace(
        settings,
        follower=replace(
            settings.follower, aggregation="conviction_trimmed", scale_lookback_days=1
        ),
    )
    facade = SimpleNamespace(
        hourly_exposure=lambda user, coin, start, end, **kw: hourly_exposure(
            resources, transitions, user, coin, start, end, **kw
        )
    )
    report = json.loads(Path(transitions["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        actual, expected = [
            ProxyConviction(activity, settings, at("2026-08-04"))
            for activity in (facade, ref)
        ]
        required = {(USER, "BTC"), (UNKNOWN, "BTC")}
        for decision in map(
            at, ["2026-08-02", "2026-08-02T01:00", "2026-08-02T03:00", "2026-08-03"]
        ):
            for conviction in (actual, expected):
                conviction.advance(required, decision)
            assert actual.samples == expected.samples
            assert {key: actual.value(key, decision) for key in required} == {
                key: expected.value(key, decision) for key in required
            }
            assert all(len(samples) == 25 for samples in actual.samples.values())
        actual.advance(set(), at("2026-08-03"))
        assert actual.samples == {}


@pytest.mark.parametrize(
    "fault",
    ["source", "pin", "engine", "interrupt", "scratch", "cardinality", "future_state"],
)
def test_bad_query_or_mutation_never_returns_partial_samples(
    qualified, resources, monkeypatch, fault
):
    from arblab.hyperliquid_copy import qualified_hourly_exposure as module

    original = module._sample_rows
    moved = []

    def changed(db, window, *args):
        rows = original(db, window, *args)
        if fault == "source":
            with window.entries[0].path.open("ab") as stream:
                stream.write(b"changed")
        elif fault == "pin":
            qualified["sha256"] = "0" * 64
        elif fault == "engine":
            engine = module._engine()
            monkeypatch.setattr(module, "_engine", lambda: dict(engine, changed=True))
        elif fault == "interrupt":
            raise RuntimeError("interrupted")
        elif fault == "cardinality":
            rows.append(rows[-1])
        elif fault == "future_state":
            row = list(rows[-1])
            row[2] = row[0]
            rows[-1] = tuple(row)
        else:
            # Keep attacking the same owned path, not whichever moved test
            # directory happens to appear first on the next iteration.
            scratch = (
                moved[0][0] if moved else next((resources.root / "scratch").iterdir())
            )
            saved = scratch.with_name(scratch.name + "_saved")
            scratch.rename(saved)
            scratch.mkdir()
            moved.append((scratch, saved))
        return rows

    monkeypatch.setattr(module, "_sample_rows", changed)
    with pytest.raises((ValueError, RuntimeError)):
        module.hourly_exposure(
            resources,
            qualified,
            USER,
            "BTC",
            at("2026-08-01"),
            at("2026-08-03"),
            max_price_age_seconds=7200,
        )
    if moved:
        scratch, saved = moved[0]
        scratch.rmdir()
        saved.rename(scratch)
        assert resources.audit()["reserved_bytes"] == module.SPILL_BYTES
    else:
        assert resources.audit()["reserved_bytes"] == 0


def test_seed_source_reverified_after_later_replay(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_hourly_exposure as module

    seed_paths = []
    original_seed, original_samples = module._seed_rows, module._sample_rows

    def seed(db, day, *args):
        seed_paths.extend(entry.path for entry in day.entries)
        return original_seed(db, day, *args)

    def samples(db, window, *args):
        result = original_samples(db, window, *args)
        assert seed_paths[0] not in [entry.path for entry in window.entries]
        with seed_paths[0].open("ab") as stream:
            stream.write(b"changed after seed")
        return result

    monkeypatch.setattr(module, "_seed_rows", seed)
    monkeypatch.setattr(module, "_sample_rows", samples)
    with pytest.raises(ValueError, match="changed"):
        module.hourly_exposure(
            resources,
            qualified,
            USER,
            "BTC",
            at("2026-08-02"),
            at("2026-08-03"),
            max_price_age_seconds=7200,
        )


def test_budget_exhaustion_precedes_sql(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_hourly_exposure as module

    resources.reserve("staging/" + "a" * 32, 6 * 1024**3, "payload")
    before = resources.audit()

    def unexpected(*args, **kwargs):
        raise AssertionError("unbudgeted SQL")

    monkeypatch.setattr(module.duckdb, "connect", unexpected)
    with pytest.raises(ValueError, match="budget"):
        module.hourly_exposure(
            resources,
            qualified,
            USER,
            "BTC",
            at("2026-08-01"),
            at("2026-08-03"),
            max_price_age_seconds=7200,
        )
    assert resources.audit() == before


def test_failed_spill_stays_charged(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_hourly_exposure as module

    def failed(*args):
        scratch = next((resources.root / "scratch").iterdir())
        (scratch / "unfinished").write_bytes(b"unfinished")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "_sample_rows", failed)
    with pytest.raises(RuntimeError, match="interrupted"):
        module.hourly_exposure(
            resources,
            qualified,
            USER,
            "BTC",
            at("2026-08-01"),
            at("2026-08-03"),
            max_price_age_seconds=7200,
        )
    assert resources.audit()["reserved_bytes"] == module.SPILL_BYTES


def test_many_native_fills_use_bounded_sample_chunks(tmp_path, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_hourly_exposure as module
    from .test_qualified_native_positions import custom_history

    fills = 100_002

    def change(key, block):
        first = block["events"][0][1]
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        if "20260801" in key:
            block["events"] = [
                [USER, dict(first, tid=i, startPosition=str(i * 2), dir="Open Long")]
                for i in range(hour * 4167, min((hour + 1) * 4167, fills))
            ]
        else:
            block["events"] = [[OTHER, first]]

    pin = custom_history(tmp_path, change, days=2)
    original = module._sample_rows
    sizes = []

    def counted(*args):
        rows = original(*args)
        sizes.append(len(rows))
        return rows

    monkeypatch.setattr(module, "_sample_rows", counted)
    result = module.hourly_exposure(
        resources,
        pin,
        USER,
        "BTC",
        at("2026-08-01"),
        at("2026-08-02T01:00"),
        max_price_age_seconds=7200,
    )
    assert sizes == [24, 1]
    assert result[0][1] is None
    assert result[-1][1] == fills * 2 * 100.0
    assert [value for _, value in result[1:24]] == [
        hour * 4167 * 2 * 100.0 for hour in range(1, 24)
    ]
    assert resources.audit()["reserved_bytes"] == 0


def test_engine_changed_during_last_source_check_cannot_return(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import qualified_hourly_exposure as module

    original = module._Source.verify

    def changed(source):
        original(source)
        engine = module._engine()
        monkeypatch.setattr(module, "_engine", lambda: dict(engine, changed=True))

    monkeypatch.setattr(module._Source, "verify", changed)
    with pytest.raises(ValueError, match="changed"):
        module.hourly_exposure(
            resources,
            qualified,
            USER,
            "BTC",
            at("2026-08-01"),
            at("2026-08-01T01:00"),
            max_price_age_seconds=7200,
        )


@pytest.mark.parametrize("query_kind", ["position", "market"])
def test_existing_native_queries_reject_recycled_scratch_inode(
    qualified, resources, monkeypatch, query_kind
):
    # Two replacements can release/reuse the original inode. Hold the directory
    # open across the whole query/cleanup rather than relying on inode numbers.
    if query_kind == "position":
        from arblab.hyperliquid_copy import qualified_native_positions as module

        hook = "_query_day"
        run = lambda: module.native_positions(
            resources, qualified, at("2026-08-02"), "BTC", [USER]
        )
    else:
        from arblab.hyperliquid_copy import qualified_market_queries as module

        hook = "_execute"
        run = lambda: module.market_volume(
            resources, qualified, "BTC", at("2026-08-01"), at("2026-08-02")
        )
    original = getattr(module, hook)
    saved_paths = []

    def replaced(*args):
        result = original(*args)
        scratch = next((resources.root / "scratch").iterdir())
        saved = scratch.with_name(scratch.name + "_saved")
        scratch.rename(saved)
        scratch.mkdir()
        scratch.rename(saved)
        scratch.mkdir()
        saved_paths.append(saved)
        return result

    monkeypatch.setattr(module, hook, replaced)
    try:
        with pytest.raises(ValueError, match="scratch.*changed"):
            run()
    finally:
        for saved in saved_paths:
            saved.rmdir()  # Empty test-created displaced directory only.
    assert resources.audit()["reserved_bytes"] == 2 * 1024**3


def test_day_and_window_metadata_are_released_and_source_view_is_lazy(
    qualified, resources, monkeypatch
):
    import weakref
    from arblab.hyperliquid_copy import qualified_hourly_exposure as module

    references, plans = [], []
    for name in ("QualifiedDay", "QualifiedWindow"):
        original = getattr(module, name)

        def tracked(*args, original=original):
            assert not any(reference() is not None for reference in references)
            source = original(*args)
            references.append(weakref.ref(source))
            return source

        monkeypatch.setattr(module, name, tracked)
    sample = module._sample_rows

    def inspected(db, *args):
        result = sample(db, *args)
        plans.append(db.execute("EXPLAIN SELECT * FROM native_source").fetchone()[1])
        return result

    monkeypatch.setattr(module, "_sample_rows", inspected)
    result = module.hourly_exposure(
        resources,
        qualified,
        UNKNOWN,
        "BTC",
        at("2026-08-02"),
        at("2026-08-04"),
        max_price_age_seconds=7200,
    )
    assert len(result) == 48 and all(value is None for _, value in result)
    assert len(references) == 3 and not any(
        reference() is not None for reference in references
    )
    assert len(plans) == 2 and all(
        "PARQUET" in plan and "COLUMN_DATA_SCAN" not in plan for plan in plans
    )


def test_closed_busy_or_naive_requests_rejected(qualified, resources):
    from arblab.hyperliquid_copy import qualified_hourly_exposure as module
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease, CacheBusyError

    with pytest.raises(CacheBusyError):
        with CacheLease(resources.root):
            pytest.fail("second query acquired busy cache")
    with pytest.raises(ValueError):
        module.hourly_exposure(
            resources,
            qualified,
            USER,
            "BTC",
            datetime(2026, 8, 1),
            at("2026-08-02"),
            max_price_age_seconds=7200,
        )
    resources.lease.__exit__(None, None, None)
    with pytest.raises(ValueError):
        module.hourly_exposure(
            resources,
            qualified,
            USER,
            "BTC",
            at("2026-08-01"),
            at("2026-08-02"),
            max_price_age_seconds=7200,
        )


def test_directory_pin_closes_descriptor_after_success(
    qualified, resources, monkeypatch
):
    import os
    from arblab.hyperliquid_copy.qualified_hourly_exposure import hourly_exposure

    original = os.open
    descriptors = []

    def opened(path, flags, *args, **kwargs):
        fd = original(path, flags, *args, **kwargs)
        if flags & os.O_DIRECTORY:
            descriptors.append(fd)
        return fd

    monkeypatch.setattr(os, "open", opened)
    hourly_exposure(
        resources,
        qualified,
        USER,
        "BTC",
        at("2026-08-01"),
        at("2026-08-02"),
        max_price_age_seconds=7200,
    )
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_dormant_seed_outside_90_days_and_fresh_other_wallet_price(tmp_path, resources):
    from arblab.hyperliquid_copy.qualified_hourly_exposure import hourly_exposure
    from .test_qualified_native_positions import custom_history

    def change(key, block):
        first = block["events"][0][1]
        user = USER if key.endswith("20250801/0.lz4") else OTHER
        block["events"] = [[user, first]]

    pin = custom_history(tmp_path, change, days=94, start=at("2025-08-01"))
    result = hourly_exposure(
        resources,
        pin,
        USER,
        "BTC",
        at("2025-11-02"),
        at("2025-11-03"),
        max_price_age_seconds=3600,
    )
    assert len(result) == 24 and all(value == 100.0 for _, value in result)
    assert resources.audit()["reserved_bytes"] == 0


def test_preorigin_spill_matches_clipped_reference(tmp_path, resources):
    from arblab.hyperliquid_copy.qualified_hourly_exposure import hourly_exposure
    from arblab.hyperliquid_copy.registration_partitions import publish_interior
    from .test_qualified_native_positions import custom_history

    def change(key, block):
        first = block["events"][0][1]
        block["events"] = [[OTHER, first]]
        if key.endswith("20260801/0.lz4"):
            moment = at("2026-07-31T23:59:59.999")
            block["block_time"] = moment.isoformat()
            block["events"] = [[USER, dict(first, time=int(moment.timestamp() * 1000))]]

    pin = custom_history(tmp_path, change, days=1)
    report = json.loads(Path(pin["path"]).read_text())
    stage = tmp_path / "interior"
    stage.mkdir()
    entries = publish_interior(
        report["files"], at("2026-08-01"), at("2026-08-02"), stage
    )
    with ProxyActivity(
        [stage / entry["name"] for entry in entries], temp_root=tmp_path
    ) as ref:
        expected = ref.hourly_exposure(
            USER, "BTC", at("2026-08-01"), at("2026-08-02"), max_price_age_seconds=7200
        )
    assert (
        hourly_exposure(
            resources,
            pin,
            USER,
            "BTC",
            at("2026-08-01"),
            at("2026-08-02"),
            max_price_age_seconds=7200,
        )
        == expected
    )
    assert all(value is None for _, value in expected)


@pytest.mark.parametrize("query_kind", ["position", "market"])
def test_caller_pin_changed_during_final_source_check_rejected(
    qualified, resources, monkeypatch, query_kind
):
    if query_kind == "position":
        from arblab.hyperliquid_copy import qualified_native_positions as module

        source_type = module._Source
        run = lambda: module.native_positions(
            resources, qualified, at("2026-08-02"), "BTC", [USER]
        )
    else:
        from arblab.hyperliquid_copy import qualified_market_queries as module

        source_type = module.QualifiedWindow
        run = lambda: module.market_volume(
            resources, qualified, "BTC", at("2026-08-01"), at("2026-08-02")
        )
    closed = False
    release, verify = module._release_empty_scratch, source_type.verify

    def cleanup(*args):
        nonlocal closed
        release(*args)
        closed = True

    def changed(source):
        verify(source)
        if closed:
            qualified["sha256"] = "0" * 64

    monkeypatch.setattr(module, "_release_empty_scratch", cleanup)
    monkeypatch.setattr(source_type, "verify", changed)
    with pytest.raises(ValueError, match="changed"):
        run()
