from datetime import timedelta
import pytest
from arblab.hyperliquid_copy.lab_config import day


def instrument(identifier, asset_class="crypto", **changes):
    return (
        dict(
            instrument_id=identifier,
            display_name=identifier,
            venue=identifier.split(":")[0] if ":" in identifier else "core",
            asset_class=asset_class,
            base=identifier,
            quote="USD",
            settlement="USDC",
            multiplier=1.0,
            model="linear_usd_continuous_v1",
            known_at=day("2026-01-01"),
            effective_from=day("2026-01-01"),
            effective_to=None,
            listed_at=day("2026-01-01"),
            delisted_at=None,
        )
        | changes
    )


def volume(identifier, date, amount):
    start = day(date)
    return dict(
        instrument_id=identifier,
        interval_start=start,
        interval_end=start + timedelta(days=1),
        available_at=start + timedelta(days=1, seconds=5),
        notional_usd=amount,
    )


def test_market_selection_lag_ties_and_future_invariance():
    from arblab.hyperliquid_copy.lab_instruments import Catalogue
    from arblab.hyperliquid_copy.lab_volume import MarketVolume
    from arblab.hyperliquid_copy.lab_market_selection import select_markets
    from arblab.hyperliquid_copy.lab_config_v2 import LiquidityUniverse

    records = [
        instrument("demo:COIN"),
        instrument("demo:GOLD", "commodity"),
        instrument("demo:STOCK", "equity"),
    ]
    buckets = [
        volume(r["instrument_id"], d, a)
        for r, a in zip(records, [100, 300, 300])
        for d in ["2026-01-02", "2026-01-03"]
    ]
    settings = LiquidityUniverse(top_n=2, lookback_days=2)
    rows, cohort = select_markets(
        Catalogue(records), MarketVolume(buckets), settings, day("2026-01-05")
    )
    assert cohort["members"] == ["demo:GOLD", "demo:STOCK"]
    assert [r["volume_usd"] for r in rows if r["selected"]] == [600, 600]
    future = records + [
        instrument(
            "future:NEW",
            known_at=day("2026-01-06"),
            effective_from=day("2026-01-06"),
            listed_at=day("2026-01-06"),
        )
    ]
    assert (rows, cohort) == select_markets(
        Catalogue(future),
        MarketVolume(buckets + [volume("demo:COIN", "2026-01-04", 1e9)]),
        settings,
        day("2026-01-05"),
    )


def test_explicit_unknown_and_zero_volume_are_not_future_or_missing():
    from arblab.hyperliquid_copy.lab_instruments import Catalogue
    from arblab.hyperliquid_copy.lab_volume import MarketVolume
    from arblab.hyperliquid_copy.lab_market_selection import select_markets
    from arblab.hyperliquid_copy.lab_config_v2 import (
        ExplicitUniverse,
        LiquidityUniverse,
    )

    catalogue = Catalogue(
        [instrument("demo:ABC"), instrument("demo:NEW", known_at=day("2026-01-06"))]
    )
    rows, cohort = select_markets(
        catalogue,
        None,
        ExplicitUniverse(instrument_ids=["demo:NEW"]),
        day("2026-01-05"),
    )
    assert rows[0]["asset_class"] is None and rows[0]["reasons"] == ["not_yet_known"]
    rows, cohort = select_markets(
        catalogue,
        MarketVolume([volume("demo:ABC", "2026-01-03", 0)]),
        LiquidityUniverse(lookback_days=1),
        day("2026-01-05"),
    )
    assert cohort["members"] == ["demo:ABC"] and rows[0]["volume_usd"] == 0
    rows, _ = select_markets(
        catalogue,
        MarketVolume([]),
        LiquidityUniverse(lookback_days=1),
        day("2026-01-05"),
    )
    assert "missing_volume" in rows[0]["reasons"]


def test_invalid_catalogue_and_daily_volume_fail_closed():
    from arblab.hyperliquid_copy.lab_instruments import Catalogue
    from arblab.hyperliquid_copy.lab_volume import MarketVolume

    with pytest.raises(ValueError):
        Catalogue([instrument("BTC"), instrument("BTC")])
    row = volume("BTC", "2026-01-02", 10)
    for rows in [
        [row, row],
        [row | {"notional_usd": -1}],
        [row | {"available_at": day("2026-01-02")}],
    ]:
        with pytest.raises(ValueError):
            MarketVolume(rows)
