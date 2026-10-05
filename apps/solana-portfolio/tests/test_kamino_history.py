import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from api import kamino_history as kh


def metric(day, equity, deposits=None, borrows=None):
    return {
        "timestamp": f"2025-01-{day:02d}T00:00:00.000Z",
        "refreshedStats": {"netAccountValue": str(equity)},
        "deposits": [{"amount": "100", "mintAddress": "mint"}]
        if deposits is None
        else deposits,
        "borrows": [] if borrows is None else borrows,
        "obligationSolValues": {"solPrice": "100"},
    }


def event(day, action="initObligation", signature="signature", amount="0"):
    return {
        "createdOn": f"2025-01-{day:02d}T12:00:00.000Z",
        "timestamp": str(1735689600000 + (day - 1) * 86400000 + 43200000),
        "transactionSignature": signature,
        "transactionName": action,
        "liquidityToken": "SOL",
        "liquidityTokenMint": "mint",
        "liquidityTokenAmount": amount,
        "liquidityTokenPrice": "100",
        "liquidityUsdValue": str(float(amount) * 100),
    }


def test_missing_active_bucket_is_omitted_without_carry_forward_or_zero():
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [event(1)], "b": [event(2)]}}],
        {"a": [metric(1, 10), metric(2, 11), metric(3, 12)], "b": [metric(2, 20)]},
    )
    assert [p["equity"] for p in data["history"]] == [10, 31]
    assert data["coverage"]["omittedBuckets"] == 1
    assert len(data["series"][0]["history"]) == 3
    assert data["coverage"]["pointInTime"] is False
    assert data["returnsComplete"] is False
    assert all(p["externalFlow"] is None for p in data["history"])


def test_withdrawal_does_not_prove_closed_obligation():
    data = kh.build_history(
        [
            {
                "market": "market",
                "obligations": {
                    "a": [event(1)],
                    "b": [
                        event(1),
                        event(
                            2,
                            "withdrawObligationCollateralAndRedeemReserveCollateral",
                            amount="1",
                        ),
                    ],
                },
            }
        ],
        {"a": [metric(1, 10), metric(3, 12)], "b": [metric(1, 20)]},
    )
    assert [p["equity"] for p in data["history"]] == [30]


def test_zero_snapshot_confirms_inactive_until_next_action():
    data = kh.build_history(
        [
            {
                "market": "market",
                "obligations": {"a": [event(1)], "b": [event(1), event(4)]},
            }
        ],
        {
            "a": [metric(1, 10), metric(2, 11), metric(3, 12), metric(4, 13)],
            "b": [metric(1, 20), metric(2, 0, deposits=[])],
        },
    )
    assert [p["equity"] for p in data["history"]] == [30, 11, 12]


def test_composite_events_keep_distinct_ids_and_liquidation_identity():
    actions = [
        event(1, "borrowObligationLiquidityV2", amount="2"),
        event(1, "depositReserveLiquidityAndObligationCollateralV2", amount="2"),
        event(1, "liquidateObligationAndRedeemReserveCollateralV2", amount="1"),
    ]
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": actions}}], {"a": []}
    )
    trades = data["trades"]
    assert len({t["id"] for t in trades}) == 3
    assert {t["type"] for t in trades} == {"borrow", "deposit", "liquidation"}
    assert trades[0]["time"] == 1735732800
    assert trades[0]["amount"] == 2
    assert trades[0]["price"] == 100
    assert trades[0]["quote"] == "USD"


def test_unpriced_and_nonfinite_metrics_never_become_zero_equity():
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [event(1)]}}],
        {"a": [metric(1, "NaN"), metric(2, -12)]},
    )
    assert data["series"][0]["history"][0]["equity"] is None
    assert [p["equity"] for p in data["history"]] == [-12]


def test_loader_caches_by_wallet_and_hides_provider_exception_text(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(kh, "CACHE_DIRECTORY", tmp_path)
    calls = []

    def fetch(url):
        calls.append(url)
        if "/users/" in url:
            return [{"market": "market", "obligations": {"a": [event(1)]}}]
        raise RuntimeError("https://rpc/?api-key=secret")

    monkeypatch.setattr(kh, "api_request", fetch)
    wallet = "11111111111111111111111111111111"
    result = kh.load_kamino_history(wallet)
    assert result["history"] == []
    assert result["coverage"]["failedObligations"] == 1
    assert "secret" not in str(result["warnings"])
    kh.load_kamino_history(wallet)
    assert len(calls) == 2


def test_invalid_wallet_rejected_before_network(monkeypatch):
    monkeypatch.setattr(kh, "api_request", lambda url: pytest.fail("network invoked"))
    with pytest.raises(ValueError):
        kh.load_kamino_history("invalid")


def test_gaps_in_every_obligation_are_disclosed_and_not_interpolated():
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [event(1)]}}],
        {"a": [metric(1, 10), metric(4, 14)]},
    )
    assert len(data["history"]) == 2
    assert data["coverage"]["missingDailyBuckets"] == 2
    assert any("daily buckets" in warning for warning in data["warnings"])


def test_explicit_close_allows_zero_only_after_the_close_bucket():
    data = kh.build_history(
        [
            {
                "market": "market",
                "obligations": {
                    "a": [event(1)],
                    "b": [event(1), event(2, "closeObligation")],
                },
            }
        ],
        {"a": [metric(1, 10), metric(2, 11), metric(3, 12)], "b": [metric(1, 20)]},
    )
    assert [p["equity"] for p in data["history"]] == [30, 12]


def risk_metric(day, adjusted_debt="100", limit="110", raw_debt="50"):
    point = metric(day, 10)
    point["refreshedStats"].update(
        userTotalBorrowBorrowFactorAdjusted=adjusted_debt,
        borrowLiquidationLimit=limit,
        userTotalBorrow=raw_debt,
    )
    return point


def test_historical_health_uses_adjusted_debt_not_raw_debt():
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [event(1)]}}],
        {"a": [risk_metric(1)]},
    )
    point = data["series"][0]["healthHistory"][0]
    assert point["health"] == pytest.approx(1.1)
    assert point["status"] == "available"
    assert point["adjustedDebt"] == 100
    assert point["liquidationLimit"] == 110
    assert data["healthHistory"][0]["health"] == pytest.approx(1.1)


@pytest.mark.parametrize(
    "debt,limit",
    [
        (None, "100"),
        ("NaN", "100"),
        ("-1", "100"),
        ("100", None),
        ("100", "-1"),
        ("100", "Infinity"),
        ("0", "-1"),
        (False, "100"),
    ],
)
def test_invalid_protocol_risk_is_unavailable_without_raw_debt_fallback(debt, limit):
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [event(1)]}}],
        {"a": [risk_metric(1, debt, limit)]},
    )
    assert data["series"][0]["healthHistory"][0]["status"] == "unavailable"
    assert data["healthHistory"][0]["health"] is None
    assert data["healthHistory"][0]["status"] == "unavailable"


def test_no_debt_is_distinct_from_missing_health_and_zero_limit_is_liquidation():
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [event(1)]}}],
        {"a": [risk_metric(1, "0", "100"), risk_metric(2, "100", "0")]},
    )
    assert data["healthHistory"][0]["status"] == "no-debt"
    assert data["healthHistory"][0]["health"] is None
    assert data["healthHistory"][1]["status"] == "available"
    assert data["healthHistory"][1]["health"] == 0


def test_aggregate_health_is_worst_individual_and_missing_active_day_is_explicit():
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [event(1)], "b": [event(2)]}}],
        {
            "a": [risk_metric(1, "0"), risk_metric(2), risk_metric(3)],
            "b": [risk_metric(2, "100", "105")],
        },
    )
    assert [p["status"] for p in data["healthHistory"]] == [
        "no-debt",
        "available",
        "unavailable",
    ]
    assert data["healthHistory"][1]["health"] == pytest.approx(1.05)
    assert len(data["healthHistory"]) == 3
    assert len(data["history"]) == 2


def test_confirmed_empty_obligation_does_not_block_later_health():
    empty = risk_metric(2, "0", "0")
    empty.update(deposits=[], borrows=[])
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [event(1)], "b": [event(1)]}}],
        {
            "a": [risk_metric(1), risk_metric(2), risk_metric(3)],
            "b": [risk_metric(1), empty],
        },
    )
    assert data["healthHistory"][-1]["status"] == "available"
    assert data["healthHistory"][-1]["health"] == pytest.approx(1.1)


def test_old_disk_cache_is_invalidated_for_health_schema(tmp_path, monkeypatch):
    import json
    import time

    monkeypatch.setattr(kh, "CACHE_DIRECTORY", tmp_path)
    wallet = "11111111111111111111111111111111"
    cache = tmp_path / f"mainnet-{wallet}-kamino-history.json"
    cache.write_text(json.dumps({"cachedAt": time.time(), "data": {"old": True}}))
    monkeypatch.setattr(kh, "api_request", lambda url: [])
    result = kh.load_kamino_history(wallet)
    assert "healthHistory" in result
    assert json.loads(cache.read_text())["cacheVersion"] == kh.CACHE_VERSION


def test_liquidation_legs_preserve_collateral_and_debt_roles():
    collateral = {
        **event(1, "liquidateObligationAndRedeemReserveCollateralV2", amount="3"),
        "isLiquidationWithdrawal": True,
    }
    debt = {
        **event(1, "liquidateObligationAndRedeemReserveCollateralV2", amount="2"),
        "isLiquidationWithdrawal": False,
    }
    unknown = event(1, "liquidateObligationAndRedeemReserveCollateral", amount="1")
    data = kh.build_history(
        [{"market": "market", "obligations": {"a": [collateral, debt, unknown]}}],
        {"a": []},
    )
    assert all(t["type"] == "liquidation" for t in data["trades"])
    assert {t["liquidationRole"] for t in data["trades"]} == {
        "collateral-seized",
        "debt-repaid",
        "unknown",
    }
