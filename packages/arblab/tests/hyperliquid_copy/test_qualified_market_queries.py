"""Causal market admission and volume over bounded qualified history."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from .test_candidate_day import resources
from .test_qualified_day import qualified


def at(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "decision", ["2026-08-01", "2026-08-01T00:00:00.001", "2026-08-03", "2026-08-04"]
)
def test_observed_matches_complete_reference(qualified, resources, tmp_path, decision):
    from arblab.hyperliquid_copy.qualified_market_queries import observed_markets

    report = json.loads(Path(qualified["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        expected = ref.observed(at(decision))
    assert observed_markets(resources, qualified, at(decision)) == expected
    assert resources.audit()["reserved_bytes"] == 0


@pytest.mark.parametrize(
    "start,decision",
    [
        ("2026-08-01", "2026-08-02"),
        ("2026-08-01T12:00", "2026-08-02T12:00"),
        ("2026-08-03", "2026-08-04"),
        ("2026-08-01", "2026-08-04"),
    ],
)
def test_volume_counts_counterparties_and_distant_duplicates_once(
    qualified, resources, tmp_path, start, decision
):
    from arblab.hyperliquid_copy.qualified_market_queries import market_volume

    report = json.loads(Path(qualified["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        expected = ref.volume("BTC", at(start), at(decision))
    assert (
        market_volume(resources, qualified, "BTC", at(start), at(decision)) == expected
    )
    assert resources.audit()["reserved_bytes"] == 0
    assert list((resources.root / "scratch").iterdir()) == []


@pytest.mark.parametrize(
    "start,end,coin",
    [
        ("2026-07-31", "2026-08-02", "BTC"),
        ("2026-08-01", "2026-08-05", "BTC"),
        ("2026-08-02", "2026-08-02", "BTC"),
        ("2026-08-03", "2026-08-02", "BTC"),
        ("2026-08-01", "2026-08-02", "ETH"),
    ],
)
def test_volume_invalid_coverage_or_coin_is_not_zero(
    qualified, resources, start, end, coin
):
    from arblab.hyperliquid_copy.qualified_market_queries import market_volume

    with pytest.raises(ValueError):
        market_volume(resources, qualified, coin, at(start), at(end))
    assert resources.audit()["reserved_bytes"] == 0


def test_observed_rejects_outside_coverage_and_naive_time(qualified, resources):
    from arblab.hyperliquid_copy.qualified_market_queries import observed_markets

    for decision in (at("2026-07-31"), at("2026-08-05"), datetime(2026, 8, 2)):
        with pytest.raises(ValueError):
            observed_markets(resources, qualified, decision)


def test_origin_empty_observation_still_requires_pinned_authority(qualified, resources):
    from arblab.hyperliquid_copy.qualified_market_queries import observed_markets

    with pytest.raises(ValueError):
        observed_markets(resources, dict(qualified, sha256="0" * 64), at("2026-08-01"))
    assert resources.audit()["reserved_bytes"] == 0


@pytest.fixture
def mixed(tmp_path):
    import lz4.frame
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history
    from .test_archive_job import inputs
    from .test_prefix_qualification import qualify

    _, source, _, _ = inputs(tmp_path, days=3)
    for key, body in list(source.bodies.items()):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        block = json.loads(lz4.frame.decompress(body))
        block["events"] = block["events"][:2]
        coin = ["BTC", "xyz:GOLD", "ETH"][hour // 8]
        for _, fill in block["events"]:
            fill.update(coin=coin, tid=hour, px=str((hour % 8 + 1) / 10))
        source.bodies[key] = lz4.frame.compress(json.dumps(block).encode() + b"\n")
    for key in list(source.bodies):
        if "20260803" in key:
            source.bodies[key] = source.bodies[key.replace("20260803", "20260801")]
    raw = download_archive(source, "2026-08-01", "2026-08-04", tmp_path / "downloaded")
    normalized = import_archive(
        raw,
        ["BTC", "ETH", "xyz:GOLD"],
        tmp_path / "normalized",
        retain_boundary_spill=True,
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    return qualify([compact], tmp_path)


@pytest.mark.parametrize(
    "decision", ["2026-08-01T08:00", "2026-08-01T08:00:00.001", "2026-08-04"]
)
def test_cross_class_observation_and_timestamp_disambiguated_volume(
    mixed, resources, tmp_path, decision
):
    from arblab.hyperliquid_copy.qualified_market_queries import (
        observed_markets,
        market_volume,
    )

    report = json.loads(Path(mixed["path"]).read_text())
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        assert observed_markets(resources, mixed, at(decision)) == ref.observed(
            at(decision)
        )
        for coin in report["coins"]:
            expected = ref.volume(coin, at("2026-08-01"), at(decision))
            assert market_volume(
                resources, mixed, coin, at("2026-08-01"), at(decision)
            ) == pytest.approx(expected, rel=1e-14, abs=1e-14)
    if decision == "2026-08-04":
        # Two distinct event days reuse tids; third physical day duplicates first.
        assert market_volume(
            resources, mixed, "BTC", at("2026-08-01"), at(decision)
        ) == pytest.approx(14.4)


@pytest.mark.parametrize(
    "kind", ["explicit", "general", "commodity", "threshold", "below", "above"]
)
def test_market_selection_matches_reference_including_thresholds(
    mixed, resources, tmp_path, kind
):
    from math import nextafter, inf
    from types import SimpleNamespace
    from arblab.hyperliquid_copy.qualified_market_queries import (
        observed_markets,
        market_volume,
    )
    from arblab.hyperliquid_copy.lab_config_v2 import (
        ExplicitUniverse,
        LiquidityUniverse,
    )
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
    from arblab.hyperliquid_copy.proxy_selection import select_proxy_markets
    from .test_proxy_selection import mapping

    report = json.loads(Path(mixed["path"]).read_text())
    mappings = ProxyMappings(
        [
            mapping(
                instrument_id="BTC",
                ticker="BTCUSDT",
                valid_from="2026-01-01",
                valid_to="2027-01-01",
            ),
            mapping(
                instrument_id="ETH",
                ticker="ETHUSDT",
                valid_from="2026-01-01",
                valid_to="2027-01-01",
            ),
            mapping(
                instrument_id="xyz:GOLD",
                ticker="GLD",
                provider="yahoo",
                asset_class="commodities",
                calendar="XNYS",
                valid_from="2026-01-01",
                valid_to="2027-01-01",
            ),
        ]
    )
    proxy = SimpleNamespace(
        observed=lambda decision: observed_markets(resources, mixed, decision),
        volume=lambda coin, start, decision: market_volume(
            resources, mixed, coin, start, decision
        ),
    )
    with ProxyActivity(
        [Path(f["path"]) for f in report["files"]], temp_root=tmp_path
    ) as ref:
        threshold = ref.volume("BTC", at("2026-08-01"), at("2026-08-02"))
        if kind == "explicit":
            config = ExplicitUniverse(
                classes=["crypto", "commodity"], instrument_ids=report["coins"]
            )
        elif kind == "commodity":
            config = LiquidityUniverse(
                general=False, classes=["commodity"], lookback_days=1
            )
        else:
            minimum = {
                "general": 0,
                "threshold": threshold,
                "below": nextafter(threshold, -inf),
                "above": nextafter(threshold, inf),
            }[kind]
            config = LiquidityUniverse(top_n=1, lookback_days=1, min_volume_usd=minimum)

        def select(activity):
            dataset = SimpleNamespace(
                activity=activity,
                mappings=mappings,
                coverage_start=at("2026-08-01"),
                coverage_end=at("2026-08-04"),
            )
            return select_proxy_markets(
                dataset, config, at("2026-08-03"), previous=["ETH"]
            )

        expected, expected_cohort = select(ref)
        actual, actual_cohort = select(proxy)
    assert actual_cohort == expected_cohort
    for row, original in zip(actual, expected, strict=True):
        volume, reference_volume = row.pop("volume_usd"), original.pop("volume_usd")
        assert row == original
        assert volume == (
            None
            if reference_volume is None
            else pytest.approx(reference_volume, rel=1e-14, abs=1e-14)
        )


def invoke(module, resources, pin, operation):
    if operation == "observed":
        return module.observed_markets(resources, pin, at("2026-08-02"))
    return module.market_volume(
        resources, pin, "BTC", at("2026-08-01"), at("2026-08-02")
    )


@pytest.mark.parametrize("operation", ["observed", "volume"])
@pytest.mark.parametrize(
    "fault", ["source", "report", "engine", "pin", "interrupt", "scratch"]
)
def test_query_changes_never_return_partial_result(
    qualified, resources, monkeypatch, operation, fault
):
    from arblab.hyperliquid_copy import qualified_market_queries as module

    original = module._execute
    moved = []

    def changed(db, *args):
        result = original(db, *args)
        if fault == "engine":
            engine = module._engine()
            monkeypatch.setattr(module, "_engine", lambda: dict(engine, changed=True))
        elif fault == "pin":
            qualified["sha256"] = "0" * 64
        elif fault in ("report", "source"):
            path = Path(qualified["path"])
            if fault == "source":
                path = Path(json.loads(path.read_text())["files"][0]["path"])
            with path.open("ab") as stream:
                stream.write(b"changed")
        elif fault == "interrupt":
            raise RuntimeError("interrupted")
        else:
            scratch = next((resources.root / "scratch").iterdir())
            saved = scratch.with_name(scratch.name + "_saved")
            scratch.rename(saved)
            scratch.mkdir()
            moved.append((scratch, saved))
        return result

    monkeypatch.setattr(module, "_execute", changed)
    with pytest.raises((ValueError, RuntimeError), match="changed|interrupted"):
        invoke(module, resources, qualified, operation)
    if moved:
        scratch, saved = moved[0]
        scratch.rmdir()
        saved.rename(scratch)
        assert resources.audit()["reserved_bytes"] == module.SPILL_BYTES
    else:
        assert resources.audit()["reserved_bytes"] == 0


@pytest.mark.parametrize("operation", ["observed", "volume"])
def test_budget_reserved_before_connection(
    qualified, resources, monkeypatch, operation
):
    from arblab.hyperliquid_copy import qualified_market_queries as module

    resources.reserve("staging/" + "a" * 32, 6 * 1024**3, "payload")
    before = resources.audit()

    def unexpected(*args, **kwargs):
        raise AssertionError("unbudgeted SQL")

    monkeypatch.setattr(module.duckdb, "connect", unexpected)
    with pytest.raises(ValueError, match="budget"):
        invoke(module, resources, qualified, operation)
    assert resources.audit() == before


def test_closed_or_busy_lease_cannot_start_query(qualified, resources):
    from arblab.hyperliquid_copy import qualified_market_queries as module
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease, CacheBusyError

    with pytest.raises(CacheBusyError):
        with CacheLease(resources.root):
            pytest.fail("second query obtained busy resources")
    resources.lease.__exit__(None, None, None)
    with pytest.raises(ValueError):
        invoke(module, resources, qualified, "observed")


def test_nonempty_orphan_spill_remains_charged(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_market_queries as module

    def failed(*args):
        scratch = next((resources.root / "scratch").iterdir())
        (scratch / "unfinished").write_bytes(b"unfinished")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "_execute", failed)
    with pytest.raises(RuntimeError, match="interrupted"):
        invoke(module, resources, qualified, "volume")
    assert resources.audit()["reserved_bytes"] == module.SPILL_BYTES


def test_final_verification_follows_cleanup(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_market_queries as module

    original = module._release_empty_scratch

    def changed(*args):
        original(*args)
        engine = module._engine()
        monkeypatch.setattr(module, "_engine", lambda: dict(engine, changed=True))

    monkeypatch.setattr(module, "_release_empty_scratch", changed)
    with pytest.raises(ValueError, match="changed"):
        invoke(module, resources, qualified, "observed")


def test_lazy_scan_and_volume_dedup_plan(qualified, resources, monkeypatch):
    from arblab.hyperliquid_copy import qualified_market_queries as module

    original = module._execute
    plans = []

    def explain(db, sql, parameters, maximum):
        plans.append(db.execute("EXPLAIN " + sql, parameters).fetchone()[1])
        return original(db, sql, parameters, maximum)

    monkeypatch.setattr(module, "_execute", explain)
    invoke(module, resources, qualified, "observed")
    invoke(module, resources, qualified, "volume")
    assert all("PARQUET" in plan and "COLUMN_DATA_SCAN" not in plan for plan in plans)
    assert "ROW_NUMBER" in plans[1] and "CTE_SCAN" in plans[1]


def test_day_span_rejected_before_metadata_allocation(
    qualified, resources, monkeypatch
):
    from arblab.hyperliquid_copy import qualified_market_queries as module
    from arblab.hyperliquid_copy.download import file_hash

    path = Path(qualified["path"])
    report = json.loads(path.read_text())
    report["source_start"] = "2020-01-01"
    path.write_text(json.dumps(report))
    qualified["sha256"] = file_hash(path)

    def unexpected(*args):
        raise AssertionError("metadata before day-bound check")

    monkeypatch.setattr(module, "QualifiedWindow", unexpected)
    monkeypatch.setattr(module, "QualifiedDay", unexpected)
    with pytest.raises(ValueError, match="day bound"):
        invoke(module, resources, qualified, "volume")


def test_preorigin_spill_does_not_extend_observation_coverage(tmp_path, resources):
    from arblab.hyperliquid_copy.qualified_market_queries import (
        observed_markets,
        market_volume,
    )
    from arblab.hyperliquid_copy.registration_partitions import publish_interior
    from .test_qualified_native_positions import custom_history

    def change(key, block):
        if key.endswith("20260801/0.lz4"):
            moment = at("2026-07-31T23:59:59.999")
            block["block_time"] = moment.isoformat()
            for _, fill in block["events"]:
                fill["time"] = int(moment.timestamp() * 1000)

    pin = custom_history(tmp_path, change, days=1)
    report = json.loads(Path(pin["path"]).read_text())
    stage = tmp_path / "interior"
    stage.mkdir()
    entries = publish_interior(
        report["files"], at("2026-08-01"), at("2026-08-02"), stage
    )
    with ProxyActivity([stage / e["name"] for e in entries], temp_root=tmp_path) as ref:
        for decision in (at("2026-08-01"), at("2026-08-01T01:00"), at("2026-08-02")):
            assert observed_markets(resources, pin, decision) == ref.observed(decision)
        assert market_volume(
            resources, pin, "BTC", at("2026-08-01"), at("2026-08-02")
        ) == ref.volume("BTC", at("2026-08-01"), at("2026-08-02"))


def test_large_native_volume_does_not_truncate_or_count_wallet_sides(
    tmp_path, resources
):
    from arblab.hyperliquid_copy.qualified_market_queries import market_volume
    from .test_qualified_native_positions import custom_history

    trades = 100_002

    def change(key, block):
        hour = int(key.rsplit("/", 1)[1].split(".")[0])
        first, second = block["events"][:2]
        block["events"] = []
        for tid in range(hour * 4167, min((hour + 1) * 4167, trades)):
            a = [first[0], dict(first[1], tid=tid)]
            b = [second[0], dict(second[1], tid=tid)]
            block["events"].extend([a, b, a])

    pin = custom_history(tmp_path, change, days=1)
    assert (
        market_volume(resources, pin, "BTC", at("2026-08-01"), at("2026-08-02"))
        == trades * 200.0
    )
    assert resources.audit()["reserved_bytes"] == 0
