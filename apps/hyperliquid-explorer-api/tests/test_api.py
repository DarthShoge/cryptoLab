import hashlib
import json
import shutil

import pytest


def test_inventory_detail_and_read_only(client, report_root):
    root, run = report_root
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / run).iterdir()
    }
    assert client.get("/api/health").json()["status"] == "ok"
    listed = client.get("/api/runs").json()
    assert listed[0]["id"] == run and listed[0]["synthetic"]
    detail = client.get(f"/api/runs/{run}").json()
    assert len(detail["scenarios"]) == 29
    assert detail["initial_equity"] == 10000
    assert all(s["metrics"]["sharpe"] is None for s in detail["scenarios"])
    assert detail["scenarios"][0]["metrics"]["terminal_open_drawdown"] == 1
    assert client.post("/api/runs").status_code == 405
    assert client.get("/api/unknown").status_code == 404
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / run).iterdir()
    }


def test_queries_filters_cash_and_empty_ledger(client, report_root):
    base = f"/api/runs/{report_root[1]}"
    eq = client.get(base + "/equity").json()
    assert eq["total"] == 4 and len(eq["rows"]) == 4
    assert eq["rows"][-1]["drawdown"] > 0
    cash = client.get(
        base + "/equity", params={"scenario_type": "control", "name": "cash"}
    ).json()
    assert all(r["equity"] == 10000 for r in cash["rows"])
    fills = client.get(base + "/fills", params={"coin": "BTC", "page_size": 1}).json()
    assert len(fills["rows"]) == 1 and fills["total"] == 3
    assert client.get(base + "/fills", params={"coin": "ETH"}).json()["total"] == 0
    assert client.get(base + "/fills", params={"page_size": 201}).status_code == 422
    assert client.get(base + "/equity", params={"name": "not-real"}).status_code == 404
    assert (
        client.get(base + "/traders", params={"scope": "global"}).json()["total"] == 6
    )
    assert (
        client.get(base + "/traders", params={"wallet": "00005"}).json()["total"] == 1
    )
    assert client.get(base + "/cohorts").json()["total"] == 1
    assert client.get(base + "/funding").json()["available"]
    assert not client.get(
        base + "/funding", params={"scenario_type": "control", "name": "cash"}
    ).json()["available"]


def test_analytics_short_history_and_raw_download(client, report_root):
    base = f"/api/runs/{report_root[1]}"
    response = client.get(
        base + "/analytics",
        params={"benchmark_type": "control", "benchmark_name": "cash"},
    )
    assert response.status_code == 200, response.text
    metrics = response.json()["metrics"]
    assert (
        metrics["sharpe"]["value"] is None
        and metrics["sharpe"]["reason"] == "synthetic demo"
    )
    assert metrics["max_drawdown"]["value"] > 0
    assert metrics["fees_usd"]["value"] == pytest.approx(0.41175)
    assert metrics["benchmark_excess_return"]["value"] < 0
    assert client.get(base + "/artifacts/report.md").status_code == 200
    assert client.get(base + "/artifacts/.env").status_code == 404


def test_symlinks_malformed_and_missing_artifacts(client, report_root, tmp_path):
    root, run = report_root
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (root / "hyperliquid_trader_ensemble_escape").symlink_to(
        outside, target_is_directory=True
    )
    assert client.get("/api/runs/hyperliquid_trader_ensemble_escape").status_code == 404
    bad = root / "hyperliquid_trader_ensemble_bad"
    bad.mkdir()
    (bad / "summary.json").write_text("broken")
    rows = client.get("/api/runs").json()
    assert any(r["id"] == bad.name and not r["available"] for r in rows)
    assert str(root) not in client.get(f"/api/runs/{bad.name}").text
    (root / run / "trader_scores.parquet").unlink()
    assert not client.get(f"/api/runs/{run}/traders").json()["available"]
    secret = outside / "secret.md"
    secret.write_text("secret")
    (root / run / "report.md").unlink()
    (root / run / "report.md").symlink_to(secret)
    assert client.get(f"/api/runs/{run}/artifacts/report.md").status_code == 404


def test_corrupt_curve_does_not_hide_valid_reports(client, report_root):
    root, run = report_root
    bad = root / "hyperliquid_trader_ensemble_corrupt"
    shutil.copytree(root / run, bad)
    (bad / "equity_curve.parquet").write_bytes(b"not parquet")
    response = client.get("/api/runs")
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()}
    assert rows[run]["available"]
    assert not rows[bad.name]["available"]


def test_empty_inventory_and_static_containment(tmp_path):
    from fastapi.testclient import TestClient
    from hyperliquid_explorer_api.app import create_app

    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<h1>Explorer</h1>")
    secret = tmp_path / "secret.txt"
    secret.write_text("outside build")
    (web / "escape.txt").symlink_to(secret)
    with TestClient(create_app(tmp_path / "absent", web)) as api:
        assert api.get("/api/runs").json() == []
        assert api.get("/").text == "<h1>Explorer</h1>"
        assert api.get("/deep/link").status_code == 200
        assert api.get("/escape.txt").status_code == 404
        assert api.get("/%2e%2e/secret.txt").status_code == 404
        assert api.get("/api/unknown").json() == {"detail": "Not found"}
        assert api.get("/assets/missing.js").status_code == 404


def test_metadata_redaction_and_filter_validation(client, report_root):
    root, run = report_root
    path = root / run / "config.json"
    config = json.loads(path.read_text())
    config.update(
        api_key="do-not-expose", nested={"source_path": "/private/thing", "coin": "BTC"}
    )
    path.write_text(json.dumps(config))
    detail = client.get(f"/api/runs/{run}").json()
    assert "api_key" not in detail["config"]
    assert detail["config"]["nested"] == {"coin": "BTC"}
    for suffix in (
        "/traders?decision_date=invalid",
        "/fills?page=0",
        "/equity?max_points=2001",
    ):
        assert client.get(f"/api/runs/{run}" + suffix).status_code == 422
    assert (
        client.get(
            f"/api/runs/{run}/traders", params={"scope": "BTC' OR 1=1 --"}
        ).json()["total"]
        == 0
    )
