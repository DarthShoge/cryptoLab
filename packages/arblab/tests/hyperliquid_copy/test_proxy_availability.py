from dataclasses import replace
from datetime import timedelta
import json

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.proxy_simulator import (
    simulate_proxy,
    ProxySimulationConfig,
)
from arblab.hyperliquid_copy.proxy_funding import FundingEvent
from .test_proxy_dataset import registered, scheduled_config
from .test_proxy_weekly import multiweek, weekly_config


def test_qualified_market_waits_for_full_lookback(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import ProxySelectionState

    data = multiweek(tmp_path)
    data.native_starts = {"BTC": day("2026-08-02")}
    c = weekly_config()
    state = ProxySelectionState(data, c)
    try:
        for date in ["2026-08-03", "2026-08-10"]:
            state.advance(day(date))
            assert state.active == []
            assert "insufficient_native_history" in state.market_rankings[-1]["reasons"]
        state.advance(day("2026-08-17"))
        assert state.active == ["BTC"]
    finally:
        data.activity.close()


def test_missing_preavailability_funding_is_not_fabricated():
    start, end = day("2026-08-03"), day("2026-08-05")
    available = start + timedelta(days=1)
    funding = [
        FundingEvent(
            "ETH",
            available + timedelta(hours=h),
            available + timedelta(hours=h),
            0.001,
            0,
        )
        for h in range(24)
    ]
    result = simulate_proxy(
        [],
        [],
        funding,
        start,
        end,
        ["ETH"],
        ProxySimulationConfig(),
        funding_starts={"ETH": available},
    )
    assert len(result.equity) == 49
    assert len(result.funding) == 24
    assert all(row["time"] >= available for row in result.funding)
    with pytest.raises(ValueError, match="funding coverage"):
        simulate_proxy(
            [],
            [],
            funding[1:],
            start,
            end,
            ["ETH"],
            ProxySimulationConfig(),
            funding_starts={"ETH": available},
        )
    with pytest.raises(ValueError, match="availability"):
        simulate_proxy(
            [dict(time=start, coin="ETH", value=1)],
            [],
            funding,
            start,
            end,
            ["ETH"],
            ProxySimulationConfig(),
            funding_starts={"ETH": available},
        )


def test_pipeline_routes_verified_funding_starts_to_strategy_and_benchmark(
    tmp_path, monkeypatch
):
    import arblab.hyperliquid_copy.lab_pipeline_proxy as module

    data = multiweek(tmp_path)
    c = weekly_config(end="2026-08-05")
    available = day(c.start)
    data.native_starts = {"BTC": data.coverage_start}
    data.funding_starts = {"BTC": available}
    calls = []
    original = module.simulate_proxy

    def capture(*args, **kwargs):
        calls.append(kwargs.get("funding_starts"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "simulate_proxy", capture)
    try:
        module.run_configured_proxy(data, c)
    finally:
        data.activity.close()

    assert calls[0] == {"BTC": available}
    assert calls[1] == {"BTC": available}
    assert calls[2] is None


@pytest.mark.parametrize(
    "fault", [None, "missing_coin", "bad_evidence", "unaligned", "contradiction"]
)
def test_registered_native_history_is_explicit_and_validated(tmp_path, fault):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    metadata = registered(tmp_path)
    record = dict(
        instrument_id="BTC",
        available_from="2026-08-01T00:00:00+00:00",
        evidence_sha256="c" * 64,
        description="Synthetic qualification fixture",
    )
    if fault == "bad_evidence":
        record["evidence_sha256"] = "invalid"
    if fault == "unaligned":
        record["available_from"] = "2026-08-01T00:01:00+00:00"
    if fault == "contradiction":
        record["available_from"] = "2026-08-03T00:00:00+00:00"
    metadata["native_history"] = [] if fault == "missing_coin" else [record]
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    if fault:
        with pytest.raises(ValueError, match="native history"):
            manifest = ProxyDatasetManifest(tmp_path)
            with manifest.load(
                temp_root=tmp_path,
                expected_hash=manifest.identity,
                config=scheduled_config(),
            ):
                pass
    else:
        manifest = ProxyDatasetManifest(tmp_path)
        with manifest.load(
            temp_root=tmp_path,
            expected_hash=manifest.identity,
            config=scheduled_config(),
        ) as loaded:
            assert loaded.native_starts == {"BTC": day("2026-08-01")}


def test_dynamic_universe_explains_future_unobserved_market(tmp_path):
    from arblab.hyperliquid_copy.proxy_selection import select_proxy_markets
    from arblab.hyperliquid_copy.lab_config_v2 import LiquidityUniverse

    data = multiweek(tmp_path)
    data.native_starts = {"BTC": data.coverage_start, "ETH": day("2026-08-05")}
    try:
        rows, cohort = select_proxy_markets(
            data,
            LiquidityUniverse(top_n=2, lookback_days=1),
            day("2026-08-03"),
            history_days=1,
        )
        row = next(row for row in rows if row["instrument_id"] == "ETH")
        assert "native_history_not_available" in row["reasons"]
        assert "not_yet_observed" in row["reasons"]
        assert not row["selected"]
        assert "ETH" not in cohort["members"]
    finally:
        data.activity.close()


@pytest.mark.parametrize("prices_after_warmup", [False, True])
def test_weekly_pipeline_admits_later_market_without_backdated_funding(
    tmp_path, prices_after_warmup
):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
    from .test_proxy_activity import fills, partition
    from .test_proxy_selection import mapping

    data = multiweek(tmp_path)
    c = weekly_config()
    c = replace(
        c,
        trader=replace(c.trader, lookback_days=1),
        market_universe=replace(c.market_universe, instrument_ids=["BTC", "ETH"]),
    )
    available = day("2026-08-04")
    rows = [
        replace(
            row,
            coin="ETH",
            tid=10000 + i,
            event_id=f"eth-{i}",
            exchange_time=day("2026-08-09") + timedelta(hours=i),
        )
        for i, row in enumerate(fills()[:2])
    ]
    rows.append(
        replace(
            rows[0],
            tid=20000,
            event_id="eth-reopen",
            exchange_time=day("2026-08-09") + timedelta(hours=23),
        )
    )
    data.activity.close()
    data.activity = ProxyActivity(
        [tmp_path / "fills.parquet", partition(tmp_path, rows, "eth")],
        temp_root=tmp_path,
    )
    data.native_starts = {"BTC": data.coverage_start, "ETH": available}
    data.mappings = ProxyMappings(
        [
            mapping(),
            mapping(instrument_id="ETH", ticker="ETHUSDT", valid_from="2025-01-01"),
        ]
    )
    price_start = available + timedelta(days=1) if prices_after_warmup else available
    data.bars += [
        replace(bar, instrument_id="ETH")
        for bar in data.bars
        if bar.start >= price_start
    ]
    data.funding += [
        replace(event, instrument_id="ETH")
        for event in data.funding
        if event.time >= available
    ]
    try:
        run = run_configured_proxy(data, c)
        assert run.market_cohorts[0]["members"] == ["BTC"]
        assert run.market_cohorts[1]["members"] == ["BTC", "ETH"]
        strategy = next(iter(run.result.strategies.values()))
        eth_fills = [row for row in strategy.fills if row["coin"] == "ETH"]
        assert eth_fills
        assert all(row["signal_time"] >= day("2026-08-10") for row in eth_fills)
        assert all(
            row["time"] >= available for row in strategy.funding if row["coin"] == "ETH"
        )
        assert len(strategy.equity) == 361
    finally:
        data.activity.close()


def test_run_ending_before_new_market_stays_valid(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import run_configured_proxy
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
    from .test_proxy_selection import mapping

    data = multiweek(tmp_path)
    c = weekly_config(end="2026-08-04")
    c = replace(
        c, market_universe=replace(c.market_universe, instrument_ids=["BTC", "ETH"])
    )
    data.native_starts = {"BTC": data.coverage_start, "ETH": day("2026-08-05")}
    data.mappings = ProxyMappings(
        [
            mapping(),
            mapping(instrument_id="ETH", ticker="ETHUSDT", valid_from="2025-01-01"),
        ]
    )
    try:
        run = run_configured_proxy(data, c)
        assert run.market_cohorts[0]["members"] == ["BTC"]
        row = next(row for row in run.market_rankings if row["instrument_id"] == "ETH")
        assert "native_history_not_available" in row["reasons"]
    finally:
        data.activity.close()


@pytest.mark.parametrize("cadence", ["weekly", "daily"])
def test_price_coverage_starts_after_proven_native_warmup(tmp_path, cadence):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import validate_proxy_run
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
    from .test_proxy_selection import mapping

    data = multiweek(tmp_path)
    c = weekly_config()
    c = replace(
        c,
        rebalance=cadence,
        trader=replace(c.trader, lookback_days=7, reselection=cadence),
        market_universe=replace(
            c.market_universe, instrument_ids=["BTC", "ETH"], reselection=cadence
        ),
    )
    available = day("2026-08-04")
    earliest = available + timedelta(days=7)
    data.native_starts = {"BTC": data.coverage_start, "ETH": available}
    data.mappings = ProxyMappings(
        [
            mapping(),
            mapping(instrument_id="ETH", ticker="ETHUSDT", valid_from="2025-01-01"),
        ]
    )
    data.bars += [
        replace(bar, instrument_id="ETH") for bar in data.bars if bar.start >= earliest
    ]
    try:
        validate_proxy_run(data, c)
        saved_bars = data.bars
        data.bars = [
            b
            for b in saved_bars
            if not (b.instrument_id == "BTC" and b.start == day(c.start))
        ]
        with pytest.raises(ValueError, match="Missing scheduled proxy bar: BTC"):
            validate_proxy_run(data, c)
        data.bars = saved_bars
        # Shorter lookbacks require the earlier prices rather than treating gaps as cash.
        with pytest.raises(ValueError, match="Missing scheduled proxy bar: ETH"):
            validate_proxy_run(
                data, replace(c, trader=replace(c.trader, lookback_days=1))
            )
        # Unproven history may not be used to waive coverage.
        data.native_starts.pop("ETH")
        with pytest.raises(ValueError, match="Missing scheduled proxy bar: ETH"):
            validate_proxy_run(data, c)
    finally:
        data.activity.close()


def test_first_xnys_admission_requires_prior_completed_close(tmp_path):
    from arblab.hyperliquid_copy.lab_pipeline_proxy import validate_proxy_run
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
    from arblab.hyperliquid_copy.proxy_bars import ProxyBar
    from arblab.hyperliquid_copy.proxy_sessions import hourly_windows
    from .test_proxy_selection import mapping

    data = multiweek(tmp_path)
    c = weekly_config()
    c = replace(
        c,
        trader=replace(c.trader, lookback_days=7),
        market_universe=replace(
            c.market_universe,
            instrument_ids=["BTC", "xyz:TSLA"],
            classes=["crypto", "equity"],
        ),
    )
    available, admission = day("2026-08-03"), day("2026-08-10")
    data.native_starts = {"BTC": data.coverage_start, "xyz:TSLA": available}
    data.mappings = ProxyMappings(
        [
            mapping(),
            mapping(
                instrument_id="xyz:TSLA",
                ticker="TSLA",
                provider="yahoo",
                asset_class="equities",
                calendar="XNYS",
                valid_from="2025-01-01",
            ),
        ]
    )
    prices = [
        ProxyBar("xyz:TSLA", a, b, 100, 110, 90, 105)
        for a, b in hourly_windows("XNYS", admission, day(c.end)).items()
    ]
    data.bars += prices
    funding = [replace(f, instrument_id="xyz:TSLA") for f in data.funding]
    signals = [dict(time=admission, coin="xyz:TSLA", value=0.5)]
    try:
        with pytest.raises(ValueError, match="completed.close seed"):
            validate_proxy_run(data, c)
        with pytest.raises(ValueError, match="No completed proxy bar"):
            simulate_proxy(
                signals,
                prices,
                funding,
                day(c.start),
                day(c.end),
                ["xyz:TSLA"],
                ProxySimulationConfig(),
            )
        a, b = list(
            hourly_windows("XNYS", day("2026-08-07"), day("2026-08-08")).items()
        )[-1]
        seed = ProxyBar("xyz:TSLA", a, b, 100, 110, 90, 105)
        data.bars.append(seed)
        validate_proxy_run(data, c)
        with pytest.raises(ValueError, match="completed.close seed"):
            validate_proxy_run(
                data, replace(c, proxy=replace(c.proxy, max_mark_age_seconds=86400))
            )
        result = simulate_proxy(
            signals,
            [seed, *prices],
            funding,
            day(c.start),
            day(c.end),
            ["xyz:TSLA"],
            ProxySimulationConfig(),
        )
        first_hour = next(
            f for f in result.funding if f["time"] == admission + timedelta(hours=14)
        )
        assert first_hour["qty"] > 0 and first_hour["mark"] == seed.close
    finally:
        data.activity.close()
