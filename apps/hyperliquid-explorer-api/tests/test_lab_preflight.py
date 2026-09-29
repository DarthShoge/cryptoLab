import pytest


@pytest.fixture
def preflight_client(tmp_path):
    from fastapi.testclient import TestClient
    from arblab.hyperliquid_copy.lab_fixture import write_fixture
    from hyperliquid_explorer_api.app import create_app

    write_fixture(tmp_path / "lab" / "datasets" / "demo")
    with TestClient(
        create_app(tmp_path / "reports", lab_root=tmp_path / "lab")
    ) as client:
        yield client


def request_parts(client):
    token = client.get("/api/lab/bootstrap").json()["token"]
    config = client.get("/api/lab/datasets").json()[0]["default_config"]
    return {"name": "Preflight regression", "dataset_id": "demo", "config": config}, {
        "X-Lab-Token": token
    }


def test_preflight_explains_coverage_and_never_saves(preflight_client):
    client = preflight_client
    body, headers = request_parts(client)
    body["config"].update(start="2025-01-05", end="2026-01-08", lookback_days=90)
    response = client.post("/api/lab/preflight", json=body, headers=headers)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ready"] is False
    issues = {i["code"]: i for i in result["issues"]}
    assert {"insufficient_warmup", "end_after_coverage"} <= issues.keys()
    assert issues["end_after_coverage"]["available"] == "2026-01-05"
    assert "90" in issues["insufficient_warmup"]["message"]
    rejected = client.post("/api/lab/experiments", json=body, headers=headers)
    assert rejected.status_code == 422
    assert rejected.json()["issues"] == result["issues"]
    assert client.get("/api/lab/experiments").json() == []


def test_preflight_is_advisory_but_submission_checks_bytes(
    preflight_client, monkeypatch
):
    client = preflight_client
    body, headers = request_parts(client)
    import hyperliquid_explorer_api.lab_datasets as datasets

    def inaccessible_hash(path):
        raise ValueError("private parser failure /operator/private/data")

    monkeypatch.setattr(datasets, "file_hash", inaccessible_hash)
    response = client.post("/api/lab/preflight", json=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["ready"] is True
    assert response.json()["config_hash"]
    assert response.json()["estimates"]["input_rows"] > 0
    rejected = client.post("/api/lab/experiments", json=body, headers=headers)
    assert rejected.status_code == 422
    assert "private" not in rejected.text
    assert client.get("/api/lab/experiments").json() == []


def test_preflight_guard_and_parser_privacy(preflight_client, monkeypatch):
    client = preflight_client
    body, headers = request_parts(client)
    assert client.post("/api/lab/preflight", json=body).status_code == 403
    assert (
        client.post(
            "/api/lab/preflight",
            json=body,
            headers=headers | {"Origin": "https://untrusted.example"},
        ).status_code
        == 403
    )
    assert (
        client.post("/api/lab/preflight", content="{}", headers=headers).status_code
        == 415
    )

    def malformed_manifest(identifier):
        raise ValueError("secret filename without a slash")

    monkeypatch.setattr(client.app.state.lab.catalog, "manifest", malformed_manifest)
    response = client.post("/api/lab/preflight", json=body, headers=headers)
    assert response.status_code == 422
    assert "secret" not in response.text
