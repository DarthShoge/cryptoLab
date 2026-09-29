import time

import pytest


@pytest.fixture
def lab_client(tmp_path):
    from fastapi.testclient import TestClient
    from hyperliquid_explorer_api.app import create_app
    from arblab.hyperliquid_copy.lab_fixture import write_fixture

    write_fixture(tmp_path / "lab" / "datasets" / "demo")
    with TestClient(
        create_app(tmp_path / "reports", lab_root=tmp_path / "lab")
    ) as client:
        yield client


def submit(client, **changes):
    bootstrap = client.get("/api/lab/bootstrap").json()
    dataset = client.get("/api/lab/datasets").json()[0]
    body = (
        dict(
            name="SOL and ETH test", dataset_id="demo", config=dataset["default_config"]
        )
        | changes
    )
    response = client.post(
        "/api/lab/experiments", json=body, headers={"X-Lab-Token": bootstrap["token"]}
    )
    assert response.status_code == 202, response.text
    return response.json(), bootstrap["token"]


def completed(client, identifier, *, timeout_seconds=25):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        item = client.get("/api/lab/experiments/" + identifier).json()
        if item["status"] not in {"queued", "running"}:
            assert item["status"] == "completed", item
            return item
        time.sleep(0.05)
    pytest.fail("Local worker did not finish")


def test_explicit_clone_upgrade_preserves_original(lab_client):
    item, token = submit(lab_client)
    completed(lab_client, item["id"])
    clone = lab_client.post(
        f"/api/lab/experiments/{item['id']}/clone?upgrade=true",
        json={},
        headers={"X-Lab-Token": token},
    )
    assert clone.status_code == 200
    assert clone.json()["config"]["schema_version"] == "hyperliquid_copy_lab_v2"
    assert (
        lab_client.get(f"/api/lab/experiments/{item['id']}").json()["config"]
        == item["config"]
    )


def test_mutations_require_local_token_and_safe_config(lab_client):
    client = lab_client
    assert client.post("/api/lab/experiments", json={}).status_code == 403
    token = client.get("/api/lab/bootstrap").json()["token"]
    headers = {"X-Lab-Token": token, "Origin": "https://untrusted.example"}
    assert (
        client.post("/api/lab/experiments", json={}, headers=headers).status_code == 403
    )
    assert (
        client.get(
            "/api/lab/bootstrap", headers={"Host": "untrusted.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/lab/experiments", content="{}", headers={"X-Lab-Token": token}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/api/lab/experiments",
            json={"dataset_id": "demo", "name": "x", "config": {"ignored": True}},
            headers={"X-Lab-Token": token},
        ).status_code
        == 422
    )


def test_run_save_clone_compare_and_historical_universe(lab_client):
    client = lab_client
    item, token = submit(client)
    result = completed(client, item["id"])
    assert result["run_id"] and result["artifact_hashes"]
    original_hash = result["config_hash"]
    base = "/api/lab/experiments/" + item["id"]
    saved = client.patch(
        base + "/metadata",
        json={"name": "Saved hypothesis", "notes": "Top two, separate asset cohorts"},
        headers={"X-Lab-Token": token},
    ).json()
    assert saved["config_hash"] == original_hash and saved["name"] == "Saved hypothesis"
    cohorts = client.get(base + "/universe", params={"table": "cohorts"}).json()
    assert cohorts["total"] == 4
    assert any(r["exits"] for r in cohorts["rows"])
    rankings = client.get(
        base + "/universe",
        params={
            "table": "rankings",
            "decision_date": "2026-01-04",
            "scope": "SOL",
            "page_size": 2,
        },
    ).json()
    assert len(rankings["rows"]) == 2 and rankings["total"] == 6
    assert rankings["rows"][0]["percentiles"]
    wallet = rankings["rows"][0]["user"]
    assert (
        client.get(
            base + "/universe", params={"table": "rankings", "wallet": wallet}
        ).json()["total"]
        >= 2
    )
    draft = client.post(base + "/clone", json={}, headers={"X-Lab-Token": token}).json()
    draft["config"]["top_n"] = 3
    second, _ = submit(client, **draft)
    completed(client, second["id"])
    assert second["id"] != item["id"]
    comparison = client.get(
        "/api/lab/compare",
        params={"ids": item["id"] + "," + second["id"], "units": "growth"},
    ).json()
    assert len(comparison["series"]) == 2
    assert "top_n" in comparison["differences"]
    assert comparison["series"][0]["curve"]["rows"][0]["equity"] == 1


def test_preview_matches_saved_decision_and_cancellation(lab_client):
    client = lab_client
    item, token = submit(client)
    result = completed(client, item["id"])
    response = client.post(
        "/api/lab/previews",
        json={
            "dataset_id": "demo",
            "config": item["config"],
            "decision_date": "2026-01-03",
            "scope": "SOL",
        },
        headers={"X-Lab-Token": token},
    )
    assert response.status_code == 202, response.text
    preview = completed(client, response.json()["id"])
    assert preview["kind"] == "cohort_preview" and preview["run_id"] is None
    actual = client.get(
        f"/api/lab/experiments/{item['id']}/universe",
        params={"table": "rankings", "decision_date": "2026-01-03", "scope": "SOL"},
    ).json()
    expected = client.get(
        f"/api/lab/experiments/{preview['id']}/universe", params={"table": "rankings"}
    ).json()
    assert actual["rows"] == expected["rows"]
    assert all(
        e["kind"] == "backtest" for e in client.get("/api/lab/experiments").json()
    )
    queued, _ = submit(client)
    cancelled = client.post(
        f"/api/lab/experiments/{queued['id']}/cancel",
        json={},
        headers={"X-Lab-Token": token},
    )
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
