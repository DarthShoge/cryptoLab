from dataclasses import replace
from datetime import timedelta

import pytest

from .test_archive import parse
from .test_market_data import T, book_row


def test_fill_quality_collects_errors_and_chain_discontinuity():
    from arblab.hyperliquid_copy.data_quality import validate_fills
    fill = parse().events[0]
    report = validate_fills([fill, fill, replace(fill, event_id="new", start_position=5)],
                            fill.exchange_time, fill.exchange_time+timedelta(seconds=1))
    codes = {i.code for i in report.issues}
    assert {"duplicate_event_id", "position_discontinuity", "invalid_position_delta"} <= codes
    assert not report.accepted
    with pytest.raises(ValueError, match="duplicate"):
        report.assert_accepted()


def test_coverage_per_coin_and_no_funding_imputation():
    from arblab.hyperliquid_copy.data_quality import validate_market
    from arblab.hyperliquid_copy.market_data import MarketData
    market = MarketData([book_row(T+timedelta(seconds=i)) for i in range(61)], [])
    report = validate_market(market, ("BTC", "ETH"), T, T+timedelta(minutes=1), research=True)
    assert not report.accepted
    assert any(i.code == "mark_coverage" and "ETH" in i.detail for i in report.issues)
    report = validate_market(market, ("BTC",), T, T+timedelta(hours=1), research=True)
    assert any(i.code == "funding_coverage" for i in report.issues)


def test_manifest_hash_ignores_locations_and_order():
    from arblab.hyperliquid_copy.data_quality import dataset_hash
    one = [{"source_key":"a", "sha256":"abc", "size":3, "local_path":"/a"}]
    two = [{"size":3, "sha256":"abc", "source_key":"a", "local_path":"/b"}]
    assert dataset_hash(one) == dataset_hash(two)


def test_naive_and_nonfinite_fills_are_quality_issues_not_sort_errors():
    from arblab.hyperliquid_copy.data_quality import validate_fills
    fill = parse().events[0]
    broken = replace(fill,event_id="bad",exchange_time=fill.exchange_time.replace(tzinfo=None))
    report = validate_fills([fill,broken,replace(fill,event_id="nan",px=float("nan"))],
                            fill.exchange_time,fill.exchange_time+timedelta(seconds=1))
    assert not report.accepted
    assert next(i for i in report.issues if i.code == "invalid_data").count == 2
