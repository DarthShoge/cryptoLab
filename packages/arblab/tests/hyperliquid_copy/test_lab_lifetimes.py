from datetime import timedelta
import pytest
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_instruments import Instrument
from arblab.hyperliquid_copy.market_data import MarketData
from arblab.hyperliquid_copy.simulator import simulate, SimulationConfig
from .test_lab_market_selection import instrument


def test_midrun_listing_needs_no_invented_earlier_mark():
    start = day("2026-01-03")
    listing = start + timedelta(minutes=1)
    end = start + timedelta(minutes=3)
    definition = Instrument(**instrument("demo:NEW", listed_at=listing))
    books = [
        dict(
            coin="demo:NEW",
            exch_time=listing + timedelta(minutes=i),
            bids=[dict(px=99.0, sz=1000.0)],
            asks=[dict(px=101.0, sz=1000.0)],
        )
        for i in range(3)
    ]
    market = MarketData(books, [])
    config = SimulationConfig(latency_seconds=0, deadband=0, min_trade_usd=0)
    result = simulate(
        [dict(time=listing, coin="demo:NEW", value=1)],
        market,
        start,
        end,
        ["demo:NEW"],
        config,
        lifetimes={"demo:NEW": definition},
    )
    assert result.equity[0]["equity"] == config.initial_equity
    assert result.fills and result.positions["demo:NEW"].qty > 0
    with pytest.raises(ValueError, match="lifetime"):
        simulate(
            [dict(time=start, coin="demo:NEW", value=1)],
            market,
            start,
            end,
            ["demo:NEW"],
            config,
            lifetimes={"demo:NEW": definition},
        )


def test_lifetime_execution_boundary_and_legacy_stability():
    start = day("2026-01-03")
    end = start + timedelta(minutes=2)
    definition = Instrument(**instrument("BTC", listed_at=start))
    books = [
        dict(
            coin="BTC",
            exch_time=start + timedelta(minutes=i),
            bids=[dict(px=99.0, sz=1000.0)],
            asks=[dict(px=101.0, sz=1000.0)],
        )
        for i in range(3)
    ]
    market = MarketData(books, [])
    config = SimulationConfig(latency_seconds=0)
    signals = [dict(time=start, coin="BTC", value=1)]
    assert simulate(signals, market, start, end, ["BTC"], config) == simulate(
        signals, market, start, end, ["BTC"], config, lifetimes={"BTC": definition}
    )
