import sys
from dataclasses import asdict
from pathlib import Path

import pytest


@pytest.fixture
def report_root(tmp_path):
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "packages/arblab/tests"))
    from hyperliquid_copy.test_pipeline import fixture
    from arblab.hyperliquid_copy.pipeline import run_offline
    from arblab.hyperliquid_copy.ranking import RankingConfig
    from arblab.hyperliquid_copy.report import write_report

    fills, market, start, end, warmup = fixture()
    config = dict(
        mode="smoke_only",
        research_eligible=False,
        coins=["BTC"],
        closed_pnl_fee_semantics="unknown",
        ranking=asdict(
            RankingConfig(
                min_active_days=1, min_episodes=1, min_notional=0, min_minutes=0
            )
        ),
    )
    result = run_offline(
        fills, market, start, end, warmup, config, dataset_sha256="fixture"
    )
    run = tmp_path / "hyperliquid_trader_ensemble_demo_20260905"
    write_report(run, result, config, dict(synthetic=True, start=start, end=end), {})
    return tmp_path, run.name


@pytest.fixture
def client(report_root):
    from fastapi.testclient import TestClient
    from hyperliquid_explorer_api.app import create_app

    with TestClient(create_app(report_root[0])) as client:
        yield client
