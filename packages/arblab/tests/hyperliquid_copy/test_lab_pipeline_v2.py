from dataclasses import replace
from datetime import timedelta
import pytest
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_config_v2 import LiquidityUniverse


def dataset():
    from arblab.hyperliquid_copy.lab_fixture_v2 import fixture_rows
    from arblab.hyperliquid_copy.lab_instruments import Catalogue
    from arblab.hyperliquid_copy.lab_volume import MarketVolume
    from arblab.hyperliquid_copy.market_data import MarketData
    from types import SimpleNamespace

    fills, books, funding, instruments, volume, config, metadata = fixture_rows()
    return SimpleNamespace(
        fills=fills,
        market=MarketData(books, funding),
        catalogue=Catalogue(instruments),
        volume=MarketVolume(volume),
        manifest=metadata,
    ), config


def test_rotating_market_replay_exit_targets_and_contributions():
    from arblab.hyperliquid_copy.lab_pipeline_v2 import run_configured_v2

    loaded, config = dataset()
    output = run_configured_v2(loaded, config)
    assert len(output.market_cohorts) == 2
    first, second = output.market_cohorts
    assert first["members"] != second["members"] and second["exits"]
    for coin in second["exits"]:
        assert (
            next(
                r
                for r in output.result.signals
                if r["time"] == day("2026-01-04") and r["coin"] == coin
            )["value"]
            == 0
        )
    for row in output.result.signals:
        pieces = [
            r
            for r in output.contributions
            if r["time"] == row["time"] and r["coin"] == row["coin"]
        ]
        assert sum(r["target_contribution"] for r in pieces) == pytest.approx(
            row["value"]
        )
    assert set(output.result.controls) == {("btc_buy_hold", 5), ("cash", None)}


def test_selection_schedules_forced_rerank_and_preview_parity():
    from arblab.hyperliquid_copy.lab_schedule import SelectionState, preview_selection

    loaded, config = dataset()
    config = replace(
        config, trader=replace(config.trader, scope="pooled", reselection="weekly")
    )
    state = SelectionState(loaded, config)
    state.advance(day(config.start))
    state.advance(day("2026-01-04"))
    assert state.trader_cohorts[-1]["decision_trigger"] == "market_change"
    rows, _, hypothetical = preview_selection(loaded, config, day("2026-01-04"), None)
    assert not hypothetical
    assert rows == [
        r for r in state.rankings if r["decision_time"] == day("2026-01-04")
    ]
    future = [
        replace(f, exchange_time=day("2026-01-06"), event_id=f.event_id + "future")
        for f in loaded.fills
    ]
    loaded.fills += future
    again, _, _ = preview_selection(loaded, config, day("2026-01-04"), None)
    assert again == rows


def test_nonminute_listing_cannot_hide_missing_hourly_funding():
    from arblab.hyperliquid_copy.lab_pipeline_v2 import validate_v2, participating_ids
    from arblab.hyperliquid_copy.lab_instruments import Catalogue

    loaded, config = dataset()
    loaded.catalogue = Catalogue(
        [
            replace(r, listed_at=r.listed_at + timedelta(seconds=30))
            if r.instrument_id == "demo:INDEX"
            else r
            for r in loaded.catalogue.records
        ]
    )
    loaded.fills = [
        r
        for r in loaded.fills
        if r.coin != "demo:INDEX"
        or r.exchange_time >= day("2026-01-01") + timedelta(seconds=30)
    ]
    loaded.market.funding = {
        k: v for k, v in loaded.market.funding.items() if k[0] != "demo:INDEX"
    }
    with pytest.raises(ValueError, match="Missing hourly funding"):
        validate_v2(loaded, config, participating_ids(loaded, config))


def test_execution_spec_changes_rejected_but_classification_changes_allowed():
    from arblab.hyperliquid_copy.lab_instruments import Catalogue

    loaded, _ = dataset()
    record = loaded.catalogue.records[0]
    before = replace(record, effective_to=day("2026-01-02"))
    after = replace(record, effective_from=day("2026-01-02"))
    assert Catalogue([before, replace(after, asset_class="commodity")])
    with pytest.raises(ValueError, match="execution specifications"):
        Catalogue([replace(before, model="unsupported"), after])
