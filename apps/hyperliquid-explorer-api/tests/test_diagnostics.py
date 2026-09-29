import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hyperliquid_explorer_api.repository import Repository, ReportError


@pytest.fixture
def diagnostic_fixture(tmp_path):
    run = tmp_path / "hyperliquid_trader_ensemble_diagnostic"
    run.mkdir()
    start = datetime(2025, 9, 1, tzinfo=timezone.utc)
    rows = [dict(time=start + timedelta(hours=i), equity=100 + i / 100,
                 cash=100.0, unrealized_pnl=i / 100, gross_exposure=20.0,
                 net_exposure=-10.0, signal_name="direction_equal", latency_seconds=5)
            for i in range(24 * 15 + 1)]
    pq.write_table(pa.Table.from_pylist(rows), run / "equity_curve.parquet")
    control = [{**r, "control_name": "btc_buy_hold", "equity": 100.0,
                "cash": 100.0, "unrealized_pnl": 0.0} for r in rows]
    pq.write_table(pa.Table.from_pylist(control), run / "control_equity_curve.parquet")
    scenario = dict(scenario_type="strategy", signal_name="direction_equal",
                    latency_seconds=5, sampling_interval_seconds=3600,
                    final_equity=103.6, final_collateral=100, final_unrealized_pnl=3.6,
                    total_return=.036, fee_drag=.01, funding_drag=.02,
                    residual_positions={"BTC": {"qty": .01, "entry": 1000}})
    summary = dict(research_eligible=False, warnings=["approximate_proxy_priced"],
                   scenarios=[scenario, dict(scenario_type="control", control_name="btc_buy_hold",
                                            latency_seconds=5, sampling_interval_seconds=3600)])
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "reconciliation.json").write_text(json.dumps(dict(accepted=False, issues=["smoke_only_unreconciled"])))
    pq.write_table(pa.Table.from_pylist([dict(fee=1.0, time=start, coin="BTC", signal_name="direction_equal", latency_seconds=5)]), run / "simulated_fills.parquet")
    pq.write_table(pa.Table.from_pylist([dict(cash_delta=-2.0, time=start, coin="BTC", signal_name="direction_equal", latency_seconds=5)]), run / "funding_ledger.parquet")
    item = dict(id="a" * 32, name="Weekly fixture", status="completed", kind="backtest",
                run_id=run.name, dataset_id="fixture", config_hash="fixed", provenance={},
                config=dict(start="2025-09-01", end="2025-09-16", rebalance="weekly",
                            follower=dict(aggregation="direction_equal", latency_seconds=5, initial_equity=100)))
    return Repository(tmp_path), item, run


def test_diagnostics_preserves_flags_and_reconciles(diagnostic_fixture):
    from hyperliquid_explorer_api.diagnostic_service import build_diagnostics
    repo, item, _ = diagnostic_fixture
    data = build_diagnostics(repo, item).model_dump(mode="json")
    assert data["research_eligible"] is False
    assert data["reconciliation"]["accepted"] is False
    assert "smoke_only_unreconciled" in data["warnings"]
    assert data["series"]["strategy"]["samples"] == 361
    assert data["statistics"]["count"]["value"] == 2
    assert data["statistics"]["mean_ci_low"]["value"] is None
    assert data["accounting"]["fees_usd"]["value"] == 1
    assert data["accounting"]["funding_usd"]["value"] == 2
    assert all(c["passed"] for c in data["checks"])
    assert data["worst_weeks"][0]["max_gross_usd"] == 20
    assert data["worst_weeks"][0]["mean_net_usd"] == -10
    assert data["cohort_reason"]
    json.dumps(data, allow_nan=False)


def test_missing_or_invalid_benchmark_does_not_hide_strategy(diagnostic_fixture):
    from hyperliquid_explorer_api.diagnostic_service import build_diagnostics
    repo, item, run = diagnostic_fixture
    (run / "control_equity_curve.parquet").unlink()
    data = build_diagnostics(repo, item)
    assert data.statistics["count"].value == 2
    assert data.statistics["correlation"].reason
    assert data.series["btc"].reason


def test_invalid_equity_retains_card_and_warnings(diagnostic_fixture):
    from hyperliquid_explorer_api.diagnostic_service import build_diagnostics
    repo, item, run = diagnostic_fixture
    rows = pq.read_table(run / "equity_curve.parquet").to_pylist()
    rows.pop(12)
    pq.write_table(pa.Table.from_pylist(rows), run / "equity_curve.parquet")
    data = build_diagnostics(repo, item)
    assert data.config == item["config"]
    assert data.series["strategy"].reason
    assert not data.series["strategy"].weeks
    assert data.statistics["mean"].value is None


def test_wrong_latency_is_not_used(diagnostic_fixture):
    from hyperliquid_explorer_api.diagnostic_service import build_diagnostics
    repo, item, _ = diagnostic_fixture
    item["config"]["follower"]["latency_seconds"] = 999
    data = build_diagnostics(repo, item)
    assert data.series["strategy"].reason
    assert data.stored_metrics == {}


def test_accounting_mismatch_is_explicit(diagnostic_fixture):
    from hyperliquid_explorer_api.diagnostic_service import build_diagnostics
    repo, item, run = diagnostic_fixture
    summary = json.loads((run / "summary.json").read_text())
    summary["scenarios"][0]["final_collateral"] = 99
    (run / "summary.json").write_text(json.dumps(summary))
    data = build_diagnostics(repo, item)
    assert any(c.passed is False for c in data.checks)
    assert data.research_eligible is False


def test_unknown_and_incomplete_are_actionable(diagnostic_fixture):
    from hyperliquid_explorer_api.diagnostic_service import get_diagnostics
    repo, item, _ = diagnostic_fixture
    def missing(_):
        raise ValueError("Unknown experiment")
    with pytest.raises(ReportError, match="Unknown experiment"):
        get_diagnostics(SimpleNamespace(store=SimpleNamespace(get=missing)), repo, "bad")
    item["status"] = "running"
    with pytest.raises(ReportError, match="completed"):
        get_diagnostics(SimpleNamespace(store=SimpleNamespace(get=lambda _: item)), repo, item["id"])


def test_read_only_http_contract(diagnostic_fixture, tmp_path):
    import asyncio
    import httpx
    from hyperliquid_explorer_api.app import create_app
    repo, item, _ = diagnostic_fixture
    app = create_app(repo.root, lab_root=tmp_path / "isolated_lab")
    app.state.lab = SimpleNamespace(store=SimpleNamespace(get=lambda _: item))
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get(f"/api/lab/experiments/{item['id']}/diagnostics")
            assert response.status_code == 200, response.text
            assert response.json()["experiment_id"] == item["id"]
            assert response.headers["Cache-Control"] == "no-store"
    asyncio.run(request())


@pytest.mark.parametrize("filename,field", [("equity_curve.parquet", "time"), ("simulated_fills.parquet", "time")])
def test_null_timestamp_only_disables_affected_section(diagnostic_fixture, filename, field):
    from hyperliquid_explorer_api.diagnostic_service import build_diagnostics
    repo, item, run = diagnostic_fixture
    table = pq.read_table(run / filename)
    rows = table.to_pylist()
    rows[0][field] = None
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), run / filename)
    data = build_diagnostics(repo, item)
    assert data.config == item["config"]
    if filename.startswith("equity"):
        assert data.series["strategy"].reason
    else:
        assert data.statistics["count"].value == 2
        assert data.accounting["fees_usd"].value is None


@pytest.mark.parametrize("timestamp", [None, datetime(2025, 9, 1, tzinfo=timezone.utc)])
def test_malformed_cohort_is_explicit_and_finite(diagnostic_fixture, timestamp):
    from hyperliquid_explorer_api.diagnostic_service import build_diagnostics
    repo, item, run = diagnostic_fixture
    pq.write_table(pa.Table.from_pylist([dict(decision_time=timestamp, coin="BTC", selected_count=5,
                                             membership_turnover=float("nan"))]), run / "cohort_history.parquet")
    data = build_diagnostics(repo, item)
    assert data.cohort_reason
    assert data.statistics["count"].value == 2
    assert not data.cohorts or data.cohorts[0].membership_turnover is None
    json.dumps(data.model_dump(mode="json"), allow_nan=False)
