import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from api import providers


def test_invalid_wallet_is_rejected_before_network_access():
    with pytest.raises(ValueError):
        providers.validate_wallet("not-an-address")
    assert providers.validate_wallet("11111111111111111111111111111111")


def test_native_and_wrapped_sol_are_added_once(monkeypatch):
    def rpc(method, params):
        if method == "getBalance":
            return {"value": 2_000_000_000}
        return (
            {
                "value": [
                    {
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "mint": providers.SOL_MINT,
                                        "tokenAmount": {
                                            "amount": "3000000000",
                                            "decimals": 9,
                                        },
                                    }
                                }
                            }
                        }
                    }
                ]
            }
            if params[1]["programId"] == providers.TOKEN_PROGRAMS[0]
            else {"value": []}
        )

    monkeypatch.setattr(providers, "rpc", rpc)
    tokens = providers.wallet_tokens("wallet", {"SOL": 100})
    assert len(tokens) == 1
    assert tokens[0]["amount"] == 5


def test_provider_errors_do_not_disclose_credentials():
    assert "secret" not in providers.safe_error(
        RuntimeError("https://rpc/?api-key=secret")
    )


def test_chart_rejects_unsupported_assets_before_request():
    with pytest.raises(ValueError):
        providers.market_candles("FAKE")


def test_imported_chart_is_anchored_to_historical_time(monkeypatch):
    queries = []

    def fake_request(url):
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(url).query)
        queries.append(query)
        start = int(query["start"][0])
        return [[start, 90, 110, 100, 105, 200]]

    monkeypatch.setattr(providers, "json_request", fake_request)
    providers.market_candles("SOL", "1d", 30, end_time=1700086400)
    assert int(queries[0]["end"][0]) <= 1700086400


def test_inception_candles_paginate_beyond_old_600_bar_cap(monkeypatch):
    from urllib.parse import parse_qs, urlparse

    end = 1700006400 // 86400 * 86400
    start = end - 750 * 86400
    queries = []

    def fake_request(url):
        query = parse_qs(urlparse(url).query)
        a, b = int(query["start"][0]), int(query["end"][0])
        queries.append((a, b))
        return [[t, 90, 110, 100, 105, 200] for t in range(a, b + 86400, 86400)]

    monkeypatch.setattr(providers, "json_request", fake_request)
    candles = providers.market_candles("SOL", "1d", 0, end_time=end, start_time=start)
    assert len(candles) == 750
    assert candles[0]["time"] == start
    assert candles[-1]["time"] == end - 86400
    assert len(queries) == 3


def test_hourly_requested_range_is_not_silently_capped(monkeypatch):
    from urllib.parse import parse_qs, urlparse

    def fake_request(url):
        q = parse_qs(urlparse(url).query)
        return [
            [t, 1, 2, 1, 2, 1]
            for t in range(int(q["start"][0]), int(q["end"][0]), 3600)
        ]

    monkeypatch.setattr(providers, "json_request", fake_request)
    assert len(providers.market_candles("SOL", "1h", 30, end_time=1700006400)) == 720


def test_kamino_indexed_legs_replace_generic_rpc_record_without_losing_swap():
    records = [
        {"id": "same", "signature": "same", "type": "kamino", "time": 1},
        {"id": "dex", "signature": "dex", "type": "buy", "time": 2},
    ]
    protocol = [
        {"id": "borrow", "signature": "same", "type": "borrow", "time": 1},
        {"id": "deposit", "signature": "same", "type": "deposit", "time": 1},
    ]
    result = providers.merge_activity(records, protocol)
    assert {t["id"] for t in result} == {"dex", "borrow", "deposit"}
