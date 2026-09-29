from datetime import timedelta
from dataclasses import replace
from types import SimpleNamespace

import pytest

from arblab.hyperliquid_copy.lab_config_v2 import ExplicitUniverse, LiquidityUniverse
from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings
from .test_proxy_mapping import mapping as base_mapping
from .test_proxy_activity import fills, partition


def mapping(**changes):
    return base_mapping(
        **(
            dict(
                instrument_id="BTC",
                provider="binance",
                ticker="BTCUSDT",
                asset_class="crypto",
                calendar="24/7",
            )
            | changes
        )
    )


def test_proxy_general_ranks_observed_mapped_native_volume_only(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from arblab.hyperliquid_copy.proxy_selection import select_proxy_markets

    first = fills()[0]
    at = first.exchange_time.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=3)
    rows = [
        replace(
            first,
            coin=c,
            tid=i,
            event_id=str(i),
            sz=size,
            exchange_time=at - timedelta(days=2, hours=-1),
        )
        for i, (c, size) in enumerate([("BTC", 1), ("ETH", 5), ("UNMAPPED", 10)])
    ]
    rows.append(replace(first, coin="FUTURE", exchange_time=at + timedelta(hours=1)))
    mappings = ProxyMappings(
        [
            mapping(
                instrument_id=c,
                ticker=c + "USDT",
                valid_from="2025-01-01",
                valid_to="2027-01-01",
            )
            for c in ["BTC", "ETH", "FUTURE"]
        ]
    )
    with ProxyActivity([partition(tmp_path, rows)], temp_root=tmp_path) as activity:
        dataset = SimpleNamespace(
            activity=activity,
            mappings=mappings,
            coverage_start=at - timedelta(days=3),
            coverage_end=at + timedelta(days=1),
        )
        ranked, cohort = select_proxy_markets(
            dataset, LiquidityUniverse(top_n=1, lookback_days=1), at
        )
    assert cohort["members"] == ["ETH"]
    assert {r["instrument_id"] for r in ranked} == {"BTC", "ETH", "UNMAPPED"}
    unknown = next(r for r in ranked if r["instrument_id"] == "UNMAPPED")
    assert unknown["reasons"] == ["missing_proxy_mapping"]
    assert ranked[0]["volume_usd"] == 5 * first.px


def test_explicit_proxy_classes_and_unobserved_are_visible(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from arblab.hyperliquid_copy.proxy_selection import select_proxy_markets

    first = fills()[0]
    at = first.exchange_time + timedelta(hours=1)
    mappings = ProxyMappings([mapping(valid_from="2025-01-01", valid_to="2027-01-01")])
    with ProxyActivity([partition(tmp_path, [first])], temp_root=tmp_path) as activity:
        dataset = SimpleNamespace(
            activity=activity,
            mappings=mappings,
            coverage_start=at - timedelta(days=3),
            coverage_end=at + timedelta(days=1),
        )
        rows, cohort = select_proxy_markets(
            dataset, ExplicitUniverse(instrument_ids=["BTC", "ETH"]), at
        )
        assert cohort["members"] == ["BTC"]
        assert (
            "not_yet_observed"
            in next(r for r in rows if r["instrument_id"] == "ETH")["reasons"]
        )
        rows, cohort = select_proxy_markets(
            dataset, ExplicitUniverse(classes=["equity"], instrument_ids=["BTC"]), at
        )
        assert not cohort["members"]
        assert rows[0]["reasons"] == ["class_not_selected"]


def test_missing_volume_warmup_is_not_zero(tmp_path):
    from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
    from arblab.hyperliquid_copy.proxy_selection import select_proxy_markets

    first = fills()[0]
    at = first.exchange_time + timedelta(hours=1)
    with ProxyActivity([partition(tmp_path, [first])], temp_root=tmp_path) as activity:
        dataset = SimpleNamespace(
            activity=activity,
            mappings=ProxyMappings(
                [mapping(valid_from="2025-01-01", valid_to="2027-01-01")]
            ),
            coverage_start=first.exchange_time,
            coverage_end=at + timedelta(days=1),
        )
        rows, cohort = select_proxy_markets(
            dataset, LiquidityUniverse(lookback_days=1), at
        )
        assert rows[0]["volume_usd"] is None
        assert "missing_volume_coverage" in rows[0]["reasons"]
        assert not cohort["members"]
