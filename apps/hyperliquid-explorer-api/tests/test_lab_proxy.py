from dataclasses import asdict
from datetime import timedelta
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def write_proxy_fixture(directory):
    from arblab.hyperliquid_copy.lab_fixture import fixture_rows
    from arblab.hyperliquid_copy.lab_config import day
    from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxy
    from arblab.hyperliquid_copy.lab_config_codec import migrate_v1_to_v2
    from arblab.hyperliquid_copy.proxy_bars import ProxyBar
    from arblab.hyperliquid_copy.proxy_funding import FundingEvent
    from arblab.hyperliquid_copy.download import file_hash

    directory.mkdir(parents=True)
    fills, _, rates, original, _ = fixture_rows()
    coins = ["BTC", "ETH", "SOL"]
    migrated = migrate_v1_to_v2(original).to_dict()
    migrated.pop("schema_version")
    config = LabConfigProxy(**migrated)
    start = day("2026-01-01")
    bars = [
        ProxyBar(
            c,
            start + timedelta(hours=h),
            start + timedelta(hours=h + 1),
            100 + h,
            101 + h,
            100 + h,
            101 + h,
        )
        for c in coins
        for h in range(96)
    ]
    funding = [
        FundingEvent(r["coin"], r["timestamp"], r["timestamp"], r["rate"], 0)
        for r in rates
    ]
    for name, rows in [("fills-0000", fills), ("bars", bars), ("funding", funding)]:
        pq.write_table(
            pa.Table.from_pylist([asdict(r) for r in rows]),
            directory / f"{name}.parquet",
        )
    mappings = [
        dict(
            instrument_id=c,
            provider="binance",
            ticker=c + "USDT",
            asset_class="crypto",
            quote_currency="USD",
            calendar="24/7",
            unit="synthetic native unit",
            adjustment="raw",
            valid_from="2026-01-01",
            valid_to="2026-01-06",
            description="Synthetic integration mapping",
            provenance="synthetic test fixture",
        )
        for c in coins
    ]
    metadata = dict(
        schema="hyperliquid_lab_proxy_dataset_v1",
        name="Synthetic hourly proxy fixture",
        synthetic=True,
        coverage_start="2026-01-01",
        coverage_end="2026-01-05",
        coins=coins,
        fee_semantics="gross_excludes_fee",
        coverage_note="Fabricated integration data; not research evidence",
        mappings=mappings,
        default_config=config.to_dict(),
        activity_provenance=dict(scope="synthetic", complete=False, source_keys=[]),
        price_source_manifest_hash="a" * 64,
        funding_source_manifest_hash="b" * 64,
        price_policy=dict(
            adjustment="raw", corporate_actions="none_detected", rolls="not_applicable"
        ),
        files=[
            dict(
                name=p.name,
                bytes=p.stat().st_size,
                rows=pq.ParquetFile(p).metadata.num_rows,
                sha256=file_hash(p),
            )
            for p in sorted(directory.glob("*.parquet"))
        ],
    )
    (directory / "manifest.json").write_text(json.dumps(metadata))
    return config


def write_scheduled_fixture(directory):
    from arblab.hyperliquid_copy.lab_config import day
    from arblab.hyperliquid_copy.lab_config_proxy import LabConfigProxyScheduled
    from arblab.hyperliquid_copy.proxy_bars import ProxyBar
    from arblab.hyperliquid_copy.proxy_funding import FundingEvent
    from arblab.hyperliquid_copy.download import file_hash

    original = write_proxy_fixture(directory)
    value = original.to_dict()
    value.pop("schema_version")
    value.update(start="2026-01-05", end="2026-01-20")
    value["trader"]["reselection"] = "weekly"
    value["market_universe"]["reselection"] = "weekly"
    c = LabConfigProxyScheduled(**value)
    start = day("2026-01-01")
    coins = ["BTC", "ETH", "SOL"]
    bars = [
        ProxyBar(
            coin,
            start + timedelta(hours=h),
            start + timedelta(hours=h + 1),
            100 + h * 0.01,
            101 + h * 0.01,
            100 + h * 0.01,
            101 + h * 0.01,
        )
        for coin in coins
        for h in range(480)
    ]
    rates = [
        FundingEvent(
            coin, start + timedelta(hours=h), start + timedelta(hours=h), 0.00001, 0
        )
        for coin in coins
        for h in range(480)
    ]
    for name, rows in [("bars", bars), ("funding", rates)]:
        pq.write_table(
            pa.Table.from_pylist([asdict(r) for r in rows]),
            directory / f"{name}.parquet",
        )
    meta = json.loads((directory / "manifest.json").read_text())
    meta.update(
        name="Synthetic multiweek scheduled proxy",
        coverage_end="2026-01-21",
        default_config=c.to_dict(),
    )
    for m in meta["mappings"]:
        m["valid_to"] = "2026-01-22"
    meta["files"] = [
        dict(
            name=p.name,
            bytes=p.stat().st_size,
            rows=pq.ParquetFile(p).metadata.num_rows,
            sha256=file_hash(p),
        )
        for p in sorted(directory.glob("*.parquet"))
    ]
    (directory / "manifest.json").write_text(json.dumps(meta))
    return c


def write_annual_fixture(directory):
    from tempfile import TemporaryDirectory
    from pathlib import Path
    import shutil
    from arblab.hyperliquid_copy.download import file_hash
    from packages.arblab.tests.hyperliquid_copy.test_proxy_annual import (
        annual_dataset,
        annual_config,
    )

    write_proxy_fixture(directory)
    with TemporaryDirectory() as scratch:
        data = annual_dataset(Path(scratch))
        try:
            shutil.copyfile(
                Path(scratch) / "annual.parquet", directory / "fills-0000.parquet"
            )
            for name, rows in [("bars", data.bars), ("funding", data.funding)]:
                pq.write_table(
                    pa.Table.from_pylist([asdict(r) for r in rows]),
                    directory / f"{name}.parquet",
                )
            config = annual_config()
            meta = json.loads((directory / "manifest.json").read_text())
            meta.update(
                name="Synthetic annual weekly copy strategy",
                coins=["BTC"],
                coverage_start=str(data.coverage_start.date()),
                coverage_end=str(data.coverage_end.date()),
                mappings=[asdict(m) for m in data.mappings.records],
                default_config=config.to_dict(),
            )
            meta["files"] = [
                dict(
                    name=p.name,
                    bytes=p.stat().st_size,
                    rows=pq.ParquetFile(p).metadata.num_rows,
                    sha256=file_hash(p),
                )
                for p in sorted(directory.glob("*.parquet"))
            ]
            (directory / "manifest.json").write_text(json.dumps(meta))
        finally:
            data.activity.close()
    return config


@pytest.mark.parametrize(
    "annual,validation_mode",
    [(False, "full_v1"), (True, "full_v1"), (False, "sharded_v1")],
)
def test_scheduled_proxy_http_save_clone_compare(tmp_path, annual, validation_mode):
    from fastapi.testclient import TestClient
    from hyperliquid_explorer_api.app import create_app
    from test_lab_api import completed

    config = (write_annual_fixture if annual else write_scheduled_fixture)(
        tmp_path / "datasets" / "scheduled"
    )
    path = tmp_path / "datasets" / "scheduled" / "manifest.json"
    metadata = json.loads(path.read_text()) | {"validation_mode": validation_mode}
    path.write_text(json.dumps(metadata))
    from hyperliquid_explorer_api.lab_datasets import DatasetCatalog
    from arblab.hyperliquid_copy.scheduled_activity import ScheduledActivity
    import sqlite3

    catalog = DatasetCatalog(tmp_path)
    frozen = catalog.preflight("scheduled", config)
    with catalog.load("scheduled", config, frozen) as loaded:
        assert isinstance(loaded.activity, ScheduledActivity)
        checkpoint_db = loaded.activity.store.path
    with TestClient(
        create_app(tmp_path / "external_reports", lab_root=tmp_path)
    ) as client:
        headers = {"X-Lab-Token": client.get("/api/lab/bootstrap").json()["token"]}
        body = dict(
            name="Weekly scheduled", dataset_id="scheduled", config=config.to_dict()
        )
        response = client.post("/api/lab/experiments", json=body, headers=headers)
        assert response.status_code == 202, response.text
        first = completed(
            client, response.json()["id"], timeout_seconds=90 if annual else 25
        )
        assert first["provenance"]["engine"] == "copy_lab_proxy_v2"
        assert first["config"]["rebalance"] == "weekly"
        clone = client.post(
            f"/api/lab/experiments/{first['id']}/clone", json={}, headers=headers
        ).json()
        assert clone["config"]["rebalance"] == "weekly"
        clone["config"]["rebalance"] = "daily"
        clone["config"]["trader"]["reselection"] = "daily"
        clone["config"]["market_universe"]["reselection"] = "daily"
        response = client.post("/api/lab/experiments", json=clone, headers=headers)
        assert response.status_code == 202, response.text
        second = completed(
            client, response.json()["id"], timeout_seconds=90 if annual else 25
        )
        compare = client.get(f"/api/lab/compare?ids={first['id']},{second['id']}")
        assert compare.status_code == 200, compare.text
        assert compare.json()["series"][0]["config"]["rebalance"] == "weekly"
        assert compare.json()["series"][1]["config"]["rebalance"] == "daily"
        for item, decisions in [
            (first, 53 if annual else 3),
            (second, 365 if annual else 15),
        ]:
            report = tmp_path / "reports" / item["run_id"]
            assert pq.read_table(report / "equity_curve.parquet").num_rows == (
                8761 if annual else 361
            )
            assert pq.read_table(
                report / "control_funding_ledger.parquet"
            ).num_rows == (8760 if annual else 360)
            assert (report / "proxy_requests.parquet").is_file()
            scores = tmp_path / "results" / item["id"] / "rankings.parquet"
            assert (
                scores.read_bytes() == (report / "trader_scores.parquet").read_bytes()
            )
            if annual:
                assert pq.read_table(scores).num_rows == decisions
                assert pq.read_table(report / "simulated_fills.parquet").num_rows > 0
        assert all(s["curve"]["rows"] for s in compare.json()["series"])
        with sqlite3.connect(checkpoint_db) as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM checkpoints WHERE json_extract(metadata, '$.parent') IS NULL"
                ).fetchone()[0]
                == 1
            )
            assert db.execute("SELECT count(*) FROM checkpoints").fetchone()[0] > 1


def test_sharded_preflight_rejects_hourly_configuration(tmp_path):
    from hyperliquid_explorer_api.lab_proxy_datasets import inspect

    directory = tmp_path / "datasets" / "sharded"
    config = write_proxy_fixture(directory)
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(json.loads(path.read_text()) | {"validation_mode": "sharded_v1"})
    )
    _, result = inspect(directory, config)
    assert not result["ready"]
    assert "scheduled_config_required" in {i["code"] for i in result["issues"]}


def test_proxy_catalogue_inspection_and_load_do_not_require_books_or_listing_dates(
    tmp_path,
):
    from hyperliquid_explorer_api.lab_datasets import DatasetCatalog
    from arblab.hyperliquid_copy.lab_config_codec import parse_lab_config

    config = write_proxy_fixture(tmp_path / "datasets" / "proxy")
    catalog = DatasetCatalog(tmp_path)
    listed = catalog.list()[0]
    assert listed["available"]
    assert listed["pricing_mode"] == "hourly_proxy"
    assert listed["liquidity_available"]
    assert parse_lab_config(listed["default_config"]) == config
    assert catalog.instruments("proxy")["rows"][0]["listed_at"] is None
    assert catalog.inspect("proxy", config)["ready"]
    frozen = catalog.preflight("proxy", config)
    with catalog.load("proxy", config, frozen) as loaded:
        assert loaded.activity.count > 0
        assert not hasattr(loaded, "market")


def test_proxy_dataset_requires_proxy_config_not_legacy_dispatch(tmp_path):
    from hyperliquid_explorer_api.lab_datasets import DatasetCatalog
    from arblab.hyperliquid_copy.lab_config import LabConfig
    from arblab.hyperliquid_copy.lab_validation import LabValidationError

    write_proxy_fixture(tmp_path / "datasets" / "proxy")
    with pytest.raises(LabValidationError, match="hourly proxy"):
        DatasetCatalog(tmp_path).preflight("proxy", LabConfig())


@pytest.mark.parametrize("kind", ["backtest", "cohort_preview"])
def test_proxy_typed_submission_worker_and_publication(tmp_path, kind):
    from hyperliquid_explorer_api.lab_models import Submission, Preview
    from hyperliquid_explorer_api.lab_jobs import LabJobs
    from hyperliquid_explorer_api.lab_worker import execute

    config = write_proxy_fixture(tmp_path / "datasets" / "proxy")
    body = dict(
        name="Hourly saved scenario", dataset_id="proxy", config=config.to_dict()
    )
    request = (
        Preview(**body, decision_date=config.start, scope="ETH")
        if kind == "cohort_preview"
        else Submission(**body)
    )
    jobs = LabJobs(tmp_path)
    item = jobs.submit(request, kind=kind)
    assert item["provenance"]["engine"] == "copy_lab_proxy_v1"
    assert "exchange-calendars" in item["provenance"]["dependencies"]
    jobs.store.transition(item["id"], "queued", "running")
    execute(tmp_path, item["id"])
    run_id, hashes = jobs.publish(jobs.store.get(item["id"]))
    jobs.store.transition(
        item["id"], "running", "completed", run_id=run_id, artifact_hashes=hashes
    )
    saved = jobs.store.get(item["id"])
    assert saved["status"] == "completed"
    assert (tmp_path / "results" / saved["id"] / "market_cohorts.parquet").is_file()
    assert not list((tmp_path / "datasets").glob("proxy_activity_*"))
    if kind == "backtest":
        summary = json.loads(
            (tmp_path / "reports" / saved["run_id"] / "summary.json").read_text()
        )
        assert "approximate_proxy_priced" in summary["warnings"]
        assert summary["scenarios"][0]["execution_model"] == "hourly_proxy_open"


@pytest.mark.parametrize("date", ["2026-01-05", "2026-01-07"])
def test_scheduled_preview_publishes_disk_evidence_and_paginates(
    tmp_path, monkeypatch, date
):
    from arblab.hyperliquid_copy.lab_config import day
    from arblab.hyperliquid_copy.lab_pipeline_proxy import ProxySelectionState
    from arblab.hyperliquid_copy.lab_schedule import preview_selection
    from hyperliquid_explorer_api.lab_datasets import DatasetCatalog
    from hyperliquid_explorer_api.lab_models import Preview
    from hyperliquid_explorer_api.lab_jobs import LabJobs
    from hyperliquid_explorer_api.lab_queries import universe
    from hyperliquid_explorer_api import lab_worker

    config = write_scheduled_fixture(tmp_path / "datasets" / "scheduled")
    catalog = DatasetCatalog(tmp_path)
    provenance = catalog.preflight("scheduled", config)
    with catalog.load("scheduled", config, provenance) as loaded:
        expected, _, hypothetical = preview_selection(
            loaded, config, day(date), "ETH", state_type=ProxySelectionState
        )
    jobs = LabJobs(tmp_path)
    item = jobs.submit(
        Preview(
            name="Disk preview",
            dataset_id="scheduled",
            config=config.to_dict(),
            decision_date=date,
            scope="ETH",
        ),
        kind="cohort_preview",
    )
    jobs.store.transition(item["id"], "queued", "running")
    monkeypatch.setattr(
        lab_worker,
        "preview_selection",
        lambda *a, **k: pytest.fail("scheduled worker used list preview"),
    )
    lab_worker.execute(tmp_path, item["id"])
    run_id, hashes = jobs.publish(jobs.store.get(item["id"]))
    jobs.store.transition(
        item["id"], "running", "completed", run_id=run_id, artifact_hashes=hashes
    )
    directory = tmp_path / "results" / item["id"]
    assert json.loads((directory / "preview.json").read_text()) == {
        "hypothetical": hypothetical
    }
    assert pq.read_metadata(directory / "rankings.parquet").num_rows == len(expected)
    from arblab.hyperliquid_copy.ranking_artifact import RANKING_SCHEMA

    normalized = [
        {k: None if v == {} else v for k, v in row.items()} for row in expected
    ]
    assert pq.read_table(directory / "rankings.parquet").equals(
        pa.Table.from_pylist(normalized, schema=RANKING_SCHEMA)
    )
    page = universe(jobs, item["id"], "rankings", page_size=1)
    assert page.total == len(expected) and len(page.rows) == min(1, len(expected))
    excluded = universe(jobs, item["id"], "rankings", page_size=1, selection="excluded")
    assert excluded.total == sum(not row["eligible"] for row in expected)
    if len(expected) > 1:
        second = universe(jobs, item["id"], "rankings", page=2, page_size=1)
        assert len(second.rows) == 1 and second.rows != page.rows


def test_proxy_http_saved_runs_analytics_and_comparison(tmp_path):
    from fastapi.testclient import TestClient
    from hyperliquid_explorer_api.app import create_app
    from test_lab_api import completed

    config = write_proxy_fixture(tmp_path / "datasets" / "proxy")
    with TestClient(
        create_app(tmp_path / "external_reports", lab_root=tmp_path)
    ) as client:
        token = client.get("/api/lab/bootstrap").json()["token"]
        headers = {"X-Lab-Token": token}
        listed = client.get("/api/lab/datasets")
        assert listed.status_code == 200, listed.text
        assert listed.json()[0]["pricing_mode"] == "hourly_proxy"
        instruments = client.get("/api/lab/datasets/proxy/instruments").json()
        assert instruments["rows"][0]["listed_at"] is None
        body = dict(
            name="Hourly HTTP baseline", dataset_id="proxy", config=config.to_dict()
        )
        check = client.post("/api/lab/preflight", json=body, headers=headers)
        assert check.status_code == 200 and check.json()["ready"], check.text
        response = client.post("/api/lab/experiments", json=body, headers=headers)
        assert response.status_code == 202, response.text
        first = completed(client, response.json()["id"])
        analytics = client.get(
            f"/api/runs/{first['run_id']}/analytics?benchmark_name=btc_buy_hold"
        )
        assert analytics.status_code == 200, analytics.text
        assert analytics.json()["metrics"]["initial_equity"]["value"] == 10000
        assert "hourly" in analytics.json()["conventions"]
        assert "365 × 24" in analytics.json()["conventions"]
        funding = client.get(
            f"/api/runs/{first['run_id']}/funding?scenario_type=control&name=btc_buy_hold"
        )
        assert funding.status_code == 200, funding.text
        assert funding.json()["available"] and funding.json()["total"] > 0
        from hyperliquid_explorer_api.repository import Repository

        repo = Repository(tmp_path / "external_reports", lab_root=tmp_path)
        assert repo.artifact(first["run_id"], "proxy_requests.parquet").is_file()
        market = client.get(
            f"/api/lab/experiments/{first['id']}/market-universe?table=rankings"
        )
        assert market.status_code == 200, market.text
        assert market.json()["rows"][0]["proxy_ticker"]
        assert market.json()["rows"][0]["availability_basis"]
        clone = client.post(
            f"/api/lab/experiments/{first['id']}/clone", json={}, headers=headers
        ).json()
        clone["config"]["proxy"]["slippage_bps"] = 25
        response = client.post("/api/lab/experiments", json=clone, headers=headers)
        assert response.status_code == 202, response.text
        second = completed(client, response.json()["id"])
        comparison = client.get(f"/api/lab/compare?ids={first['id']},{second['id']}")
        assert comparison.status_code == 200, comparison.text
        assert len(comparison.json()["series"]) == 2
        assert all(s["curve"]["rows"] for s in comparison.json()["series"])
        assert (
            "Different proxy execution/mark assumptions"
            in comparison.json()["warnings"]
        )


@pytest.mark.parametrize("action", ["cancel", "close", "tick"])
def test_coordinator_removes_owned_scratch_after_worker_exit(tmp_path, action):
    import subprocess
    import sys
    from hyperliquid_explorer_api.lab_jobs import LabJobs

    jobs = LabJobs(tmp_path)
    item = jobs.store.create("Interrupted proxy", "proxy", {}, {})
    jobs.store.transition(item["id"], "queued", "running")
    scratch = tmp_path / "scratch" / item["id"]
    scratch.mkdir(parents=True)
    (scratch / "spill.tmp").write_bytes(b"owned temporary data")
    unrelated = tmp_path / "datasets" / "keep.parquet"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"user data")
    jobs.active = item["id"]
    jobs.process = subprocess.Popen(
        [sys.executable, "-c", "import signal; signal.pause()"]
    )
    try:
        if action == "cancel":
            jobs.cancel(item["id"])
        elif action == "close":
            from threading import Thread

            jobs.thread = Thread(target=lambda: None)
            jobs.thread.start()
            jobs.close()
        else:
            jobs.process.terminate()
            jobs.process.wait(timeout=5)
            jobs.tick()
        assert not scratch.exists()
        assert unrelated.read_bytes() == b"user data"
    finally:
        if jobs.process:
            jobs.process.kill()
            jobs.process.wait(timeout=5)


def test_coordinator_persists_worker_stderr_and_exit_code(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from hyperliquid_explorer_api import lab_jobs as module

    jobs = module.LabJobs(tmp_path)
    item = jobs.store.create(
        "Failing worker",
        "proxy",
        {},
        {"source_hashes": module.source_fingerprint()},
    )
    original = subprocess.Popen

    def failing_worker(command, **kwargs):
        return original(
            [
                sys.executable,
                "-c",
                "import sys; print('diagnostic marker', file=sys.stderr); sys.exit(7)",
            ],
            **kwargs,
        )

    monkeypatch.setattr(module.subprocess, "Popen", failing_worker)
    jobs.tick()
    deadline = time.monotonic() + 5
    while jobs.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    jobs.tick()

    failed = jobs.store.get(item["id"])
    assert failed["status"] == "failed"
    assert "exit code 7" in failed["error"]
    assert "worker_logs" in failed["error"]
    log = tmp_path / "worker_logs" / f"{item['id']}.stderr.log"
    assert log.read_text().strip() == "diagnostic marker"
