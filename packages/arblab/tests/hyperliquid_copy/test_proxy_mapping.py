from dataclasses import replace
from datetime import datetime, timezone

import pytest


def mapping(**changes):
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMapping

    fields = dict(
        instrument_id="xyz:GOLD",
        provider="yahoo",
        ticker="GLD",
        asset_class="commodities",
        quote_currency="USD",
        calendar="XNYS",
        unit="ETF share",
        adjustment="raw",
        valid_from="2026-08-01",
        valid_to="2026-09-01",
        description="Gold ETF return proxy; not native gold units",
        provenance="researcher mapping; https://www.spdrgoldshares.com/usa/",
    )
    return ProxyMapping(**(fields | changes))


def test_mapping_explicitly_preserves_units_and_effective_window():
    m = mapping()
    assert m.unit == "ETF share"
    assert m.active(datetime(2026, 8, 3, tzinfo=timezone.utc))
    assert not m.active(datetime(2026, 9, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    "changes",
    [
        {"quote_currency": "GBP"},
        {"calendar": "CMES"},
        {"adjustment": "automatic"},
        {"ticker": "../../secret"},
        {"provider": "unknown"},
        {"description": ""},
        {"provenance": ""},
        {"valid_to": "2026-07-31"},
        {"asset_class": "other"},
        {"provider": "binance", "calendar": "XNYS"},
    ],
)
def test_rejects_ambiguous_or_unsupported_mapping(changes):
    with pytest.raises(ValueError):
        mapping(**changes)


def test_mapping_set_rejects_overlaps_and_lists_unmapped_ids():
    from arblab.hyperliquid_copy.proxy_mapping import ProxyMappings

    m = mapping()
    with pytest.raises(ValueError, match="overlap"):
        ProxyMappings([m, replace(m, ticker="IAU")])
    mappings = ProxyMappings([m])
    at = datetime(2026, 8, 3, tzinfo=timezone.utc)
    assert mappings.at("xyz:GOLD", at) == m
    assert mappings.at("other:GOLD", at) is None
    assert mappings.exclusions(["xyz:GOLD", "other:GOLD"], at) == {
        "other:GOLD": "missing_proxy_mapping"
    }
