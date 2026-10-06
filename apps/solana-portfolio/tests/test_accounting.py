import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.accounting import account_summary, history_metrics, validate_history
from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition


def snapshot(collateral=100, debt=40, factor=1.25):
    return AccountSnapshot(
        [CollateralPosition("SOL", collateral, 100, 0.65, 0.75)],
        [DebtPosition("ETH", debt, 100, factor)],
    )


def test_equity_includes_wallet_and_subtracts_both_types_of_debt():
    account = snapshot()
    account.debt.append(DebtPosition("USDC", 1000, 1))
    result = account_summary(
        [("market-a", account)], [{"symbol": "SOL", "amount": 5, "price": 100}]
    )
    assert result["netEquity"] == 5500
    assert result["supplied"] == 10000
    assert result["debt"] == 5000
    assert result["exposure"]["SOL"] == 105
    assert result["exposure"]["ETH"] == -40
    assert result["loans"][0]["ltv"] == 0.6
    assert result["loans"][0]["health"] == 1.25


def test_health_uses_risk_adjusted_debt_and_keeps_obligations_separate():
    result = account_summary(
        [("healthy", snapshot(debt=1)), ("at-risk", snapshot(debt=65))], []
    )
    assert result["health"] == pytest.approx(7500 / 8125)
    assert len(result["loans"]) == 2
    assert result["loans"][1]["liquidationBuffer"] == -625


def test_unpriced_wallet_token_never_becomes_zero_equity():
    result = account_summary([], [{"symbol": "UNKNOWN", "amount": 10, "price": None}])
    assert result["netEquity"] is None
    assert result["knownEquity"] == 0
    assert result["unpricedCount"] == 1
    assert result["health"] is None


def test_deposits_are_not_trading_profit():
    history = [
        {"time": 100, "equity": 1000, "externalFlow": 0, "solPrice": 100},
        {"time": 200, "equity": 1600, "externalFlow": 500, "solPrice": 100},
        {"time": 300, "equity": 1680, "externalFlow": 0, "solPrice": 100},
    ]
    result = history_metrics(history, complete=True)
    assert result["pnl"] == 180
    assert result["returnPct"] == pytest.approx(15.5)
    assert result["solPnl"] == pytest.approx(1.8)
    assert history_metrics(history, complete=False)["returnPct"] is None


def test_drawdown_is_on_flow_adjusted_performance_not_account_value():
    history = [
        {"time": 1, "equity": 1000, "externalFlow": 0, "solPrice": 100},
        {"time": 2, "equity": 1800, "externalFlow": 1000, "solPrice": 100},
    ]
    assert history_metrics(history, complete=True)["drawdownPct"] == pytest.approx(-20)


def test_import_requires_ordered_finite_equity_and_explicit_coverage():
    with pytest.raises(ValueError):
        validate_history(
            {"history": [{"time": 2, "equity": float("nan")}], "complete": True}
        )
    with pytest.raises(ValueError):
        validate_history({"history": [{"time": 2, "equity": 100}], "complete": True})
    data = {
        "source": {
            "name": "Test export",
            "account": "Test wallet",
            "equityScope": "wallet-and-kamino",
            "cashFlowCoverage": "complete",
        },
        "complete": True,
        "flowTiming": "period-end",
        "history": [
            {"time": 1, "equity": 100, "externalFlow": 0, "solPrice": 10},
            {"time": 2, "equity": 110, "externalFlow": 0, "solPrice": 10},
        ],
        "trades": [],
    }
    assert validate_history(data)["history"] == data["history"]


def valid_import():
    return {
        "source": {
            "name": "Wallet export",
            "account": "Test account",
            "equityScope": "wallet-and-kamino",
            "cashFlowCoverage": "complete",
        },
        "complete": True,
        "flowTiming": "period-end",
        "history": [
            {"time": 100, "equity": 100, "externalFlow": 0, "solPrice": 10},
            {"time": 200, "equity": 120, "externalFlow": 0, "solPrice": 10},
        ],
        "trades": [
            {
                "id": "t1",
                "time": 150,
                "type": "buy",
                "asset": "SOL",
                "amount": 1,
                "price": 10,
            }
        ],
    }


def test_import_rejects_object_fields_that_would_crash_react():
    for key in ("protocol", "funding", "quote", "signature"):
        data = valid_import()
        data["trades"][0][key] = {}
        with pytest.raises(ValueError):
            validate_history(data)


def test_import_requires_cash_flow_provenance_for_complete_history():
    data = valid_import()
    data["source"]["cashFlowCoverage"] = "partial"
    with pytest.raises(ValueError):
        validate_history(data)


def test_total_loss_has_negative_100_percent_return_and_drawdown():
    history = [
        {"time": 1, "equity": 100, "externalFlow": 0, "solPrice": 10},
        {"time": 2, "equity": 0, "externalFlow": 0, "solPrice": 10},
    ]
    result = history_metrics(history, complete=True)
    assert result["returnPct"] == -100
    assert result["drawdownPct"] == -100
