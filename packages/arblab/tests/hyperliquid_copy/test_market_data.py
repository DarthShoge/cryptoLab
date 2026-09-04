from datetime import datetime, timedelta, timezone

import pytest

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def book_row(at=T):
    return dict(exch_time=at, local_time=at, coin="BTC", best_bid=99., best_ask=101.,
                mid=100., spread_bps=200., bids=[dict(px=99., sz=1., n=1)],
                asks=[dict(px=101., sz=1., n=1), dict(px=102., sz=2., n=1)])


def test_books_lookup_dedup_and_conflict():
    from arblab.hyperliquid_copy.market_data import MarketData
    market = MarketData([book_row(), book_row()], [])
    assert market.mark("BTC", T + timedelta(seconds=60)) == 100
    with pytest.raises(ValueError, match="stale"):
        market.mark("BTC", T + timedelta(seconds=61))
    assert market.execution_book("BTC", T - timedelta(seconds=2)).mid == 100
    assert market.execution_book("BTC", T - timedelta(seconds=3)) is None
    with pytest.raises(ValueError, match="conflicting"):
        MarketData([book_row(), book_row() | {"asks": [dict(px=101., sz=2., n=1)]}], [])


def test_parquet_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from arblab.hyperliquid_copy.market_data import MarketData
    pq.write_table(pa.Table.from_pylist([book_row()]), tmp_path / "book.parquet")
    market = MarketData.from_parquet([tmp_path / "book.parquet"], [], T, T + timedelta(minutes=1))
    assert market.mark("BTC", T) == 100


def test_upstream_funding_and_paths(tmp_path):
    from arblab.hyperliquid_copy.market_data import MarketData
    from arblab.hyperliquid_copy.data_paths import market_partition
    from hyperliquid_data import coin_dir
    market = MarketData([book_row()], [{"timestamp":T,"rate":.001,"instrument":"BTC-PERP","source":"hyperliquid"}])
    assert market.funding["BTC",T] == .001
    assert market_partition(tmp_path,"BTC","2026-01-01").parts[-3] == coin_dir("BTC-PERP")
