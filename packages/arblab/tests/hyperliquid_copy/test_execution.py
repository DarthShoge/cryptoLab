from datetime import timedelta

import pytest

from .test_market_data import T, book_row


def test_walk_depth_fees_and_missing_book():
    from arblab.hyperliquid_copy.market_data import parse_book
    from arblab.hyperliquid_copy.execution import walk_book
    book = parse_book(book_row())
    buy = walk_book(book, 4, fee_bps=4.5)
    assert buy.filled_qty == 3 and buy.unfilled_qty == 1
    assert buy.vwap == pytest.approx(305/3)
    assert buy.fee == pytest.approx(305*.00045)
    assert buy.reason == "partial_depth"
    sell = walk_book(book, -1, fee_bps=4.5)
    assert sell.vwap == 99 and sell.filled_qty == -1
    assert sell.spread_cost == 1
    assert walk_book(None, 1).reason == "stale_book"


def test_perpetual_accounting_partial_close_flip_and_funding():
    from arblab.hyperliquid_copy.execution import Account
    account = Account(1000)
    account.fill("BTC", 2, 100, 1)
    assert account.cash == 999
    assert account.equity({"BTC":110}) == 1019
    account.fill("BTC", -1, 110, 1)
    assert account.cash == 1008
    assert account.realized_pnl == 10
    account.fill("BTC", -2, 120, 1)
    assert account.cash == 1027
    assert account.positions["BTC"].qty == -1
    assert account.positions["BTC"].entry == 120
    assert account.funding("BTC", 120, .001) == pytest.approx(.12)
    assert account.equity({"BTC":115}) == pytest.approx(1032.12)


def test_funding_before_delayed_fill_and_terminal_exposure():
    from arblab.hyperliquid_copy.market_data import MarketData
    from arblab.hyperliquid_copy.simulator import SimulationConfig, simulate
    rows = [book_row(T+timedelta(seconds=s)) for s in range(181)]
    market = MarketData(rows, [{"coin":"BTC", "timestamp":T+timedelta(minutes=1), "rate":.001}])
    signals = [{"time":T, "coin":"BTC", "value":1.0}]
    cfg = SimulationConfig(initial_equity=1000, asset_cap=.25, latency_seconds=60)
    result = simulate(signals, market, T, T+timedelta(minutes=2), ("BTC",), cfg)
    assert result.funding[0]["qty"] == 0
    assert result.fills[0]["book_time"] == T+timedelta(minutes=1)
    assert result.equity[-1]["equity"] < 1000
    assert result.positions["BTC"].qty > 0
    assert result.equity[-1]["equity"] == pytest.approx(result.cash + result.equity[-1]["unrealized_pnl"])


def test_stale_execution_notional_is_in_fill_ratio():
    from arblab.hyperliquid_copy.market_data import MarketData
    from arblab.hyperliquid_copy.simulator import SimulationConfig,simulate
    from arblab.hyperliquid_copy.report import summarize
    market = MarketData([book_row(T),book_row(T+timedelta(minutes=1))],[])
    result = simulate([dict(time=T,coin="BTC",value=1)],market,T,T+timedelta(minutes=1),("BTC",),SimulationConfig(latency_seconds=5))
    assert result.fills[0]["reason"] == "stale_book"
    assert result.fills[0]["requested_notional"] == 2500
    assert summarize(result)["fill_ratio"] == 0
