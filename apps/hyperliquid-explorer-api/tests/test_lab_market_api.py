import pytest


@pytest.fixture
def cross_client(tmp_path):
    from fastapi.testclient import TestClient
    from hyperliquid_explorer_api.app import create_app
    from arblab.hyperliquid_copy.lab_fixture_v2 import write_fixture

    write_fixture(tmp_path / "lab" / "datasets" / "cross")
    with TestClient(
        create_app(tmp_path / "reports", lab_root=tmp_path / "lab")
    ) as client:
        yield client


def test_cross_class_run_history_clone_and_comparison(cross_client):
    from test_lab_api import completed

    client = cross_client
    headers = {"X-Lab-Token": client.get("/api/lab/bootstrap").json()["token"]}
    dataset = client.get("/api/lab/datasets").json()[0]
    assert dataset["liquidity_available"]
    instruments = client.get(
        "/api/lab/datasets/cross/instruments",
        params={"asset_class": "equity", "page_size": 1},
    ).json()
    assert instruments["total"] == 2 and len(instruments["rows"]) == 1
    body = dict(
        name="Cross-class hypothesis",
        dataset_id="cross",
        config=dataset["default_config"],
    )
    submitted = client.post("/api/lab/experiments", json=body, headers=headers)
    assert submitted.status_code == 202, submitted.text
    first = completed(client, submitted.json()["id"])
    path = f"/api/lab/experiments/{first['id']}/market-universe"
    cohorts = client.get(path).json()
    assert cohorts["total"] == 2 and any(r["exits"] for r in cohorts["rows"])
    assert client.get(path, params={"page_size": 201}).status_code == 422
    rows = client.get(
        path,
        params={
            "table": "rankings",
            "instrument_id": "demo:GOLD",
            "decision_date": "2026-01-03",
        },
    ).json()
    assert rows["total"] == 1 and rows["rows"][0]["volume_usd"] > 0
    clone = client.post(
        f"/api/lab/experiments/{first['id']}/clone", json={}, headers=headers
    ).json()
    clone["config"]["market_universe"]["top_n"] = 2
    second = client.post("/api/lab/experiments", json=clone, headers=headers)
    assert second.status_code == 202, second.text
    second = completed(client, second.json()["id"])
    comparison = client.get(
        "/api/lab/compare", params={"ids": first["id"] + "," + second["id"]}
    ).json()
    assert "market_universe" in comparison["differences"]
    assert comparison["series"][0]["market_membership_turnover"] is not None


def test_empty_market_selection_is_saved_as_cash_with_typed_evidence(cross_client):
    from test_lab_api import completed

    client = cross_client
    headers = {"X-Lab-Token": client.get("/api/lab/bootstrap").json()["token"]}
    config = client.get("/api/lab/datasets").json()[0]["default_config"]
    config["market_universe"]["min_volume_usd"] = 1e15
    response = client.post(
        "/api/lab/experiments",
        json=dict(dataset_id="cross", config=config),
        headers=headers,
    )
    assert response.status_code == 202, response.text
    saved = completed(client, response.json()["id"])
    rows = client.get(
        f"/api/lab/experiments/{saved['id']}/universe", params={"table": "rankings"}
    ).json()
    assert rows["total"] == 0


def test_unscheduled_preview_is_explicitly_hypothetical(cross_client):
    from test_lab_api import completed

    client = cross_client
    headers = {"X-Lab-Token": client.get("/api/lab/bootstrap").json()["token"]}
    config = client.get("/api/lab/datasets").json()[0]["default_config"]
    config["market_universe"]["reselection"] = "weekly"
    config["trader"].update(scope="pooled", reselection="weekly")
    response = client.post(
        "/api/lab/previews",
        json=dict(
            dataset_id="cross", config=config, decision_date="2026-01-04", scope=None
        ),
        headers=headers,
    )
    assert response.status_code == 202, response.text
    saved = completed(client, response.json()["id"])
    info = client.get(f"/api/lab/experiments/{saved['id']}/preview-info")
    assert info.status_code == 200 and info.json()["hypothetical"] is True


def test_inactive_v2_fields_are_safe_validation_errors(cross_client):
    client = cross_client
    headers = {"X-Lab-Token": client.get("/api/lab/bootstrap").json()["token"]}
    config = client.get("/api/lab/datasets").json()[0]["default_config"]
    config["market_universe"]["weights"] = {"BTC": 1}
    response = client.post(
        "/api/lab/preflight",
        json=dict(dataset_id="cross", config=config),
        headers=headers,
    )
    assert response.status_code == 422, response.text
    assert "Traceback" not in response.text
