from dataclasses import asdict, replace
from datetime import timedelta

import pytest

from .test_ranking import history
from .test_market_data import book_row


def fixture():
    from arblab.hyperliquid_copy.market_data import MarketData
    fills = history()
    start = max(f.exchange_time for f in fills).replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1)
    # Prior round trip qualifies each wallet; reopen before the simulation.
    for f in list(fills):
        if f.side == "B":
            fills.append(replace(f, exchange_time=start-timedelta(minutes=2),event_id=f.event_id+":reopen"))
    warmup = start-timedelta(minutes=3)
    end = start+timedelta(minutes=3)
    books = [book_row(warmup+timedelta(seconds=s)) for s in range(361)]
    return fills, MarketData(books,[]), start, end, warmup


def test_pipeline_signals_causal_and_twelve_scenarios(tmp_path):
    from arblab.hyperliquid_copy.pipeline import run_offline
    from arblab.hyperliquid_copy.ranking import RankingConfig
    from arblab.hyperliquid_copy.report import write_report
    fills, market, start,end,warmup = fixture()
    config = dict(mode="smoke_only", research_eligible=False, coins=["BTC"],
                  closed_pnl_fee_semantics="unknown", scale_lookback_days=2, scale_quantile=.95,
                  ranking=asdict(RankingConfig(min_active_days=1,min_episodes=1,min_notional=0,min_minutes=0)))
    result = run_offline(fills,market,start,end,warmup,config, dataset_sha256="fixture")
    assert len(result.strategies) == 12
    assert len(result.controls) == 17
    assert len(result.signals) == 9
    assert all("latency_seconds" not in r for r in result.signals)
    assert all(r["value"] > 0 for r in result.signals)
    destination = tmp_path/"report"
    write_report(destination,result,config,{"dataset_hash":"fixture"},{"role":"smoke"})
    assert (destination/"summary.json").exists()
    assert "INTEGRATION ONLY" in (destination/"report.md").read_text()
    assert (destination/"simulated_fills.parquet").exists()
    with pytest.raises(FileExistsError):
        write_report(destination,result,config,{}, {})
    again = run_offline(list(reversed(fills)),market,start,end,warmup,config,dataset_sha256="fixture")
    assert result.signals == again.signals
    assert result.strategies == again.strategies


def test_report_preserves_evidence_bytes_before_hashing(tmp_path):
    from arblab.hyperliquid_copy.pipeline import run_offline
    from arblab.hyperliquid_copy.ranking import RankingConfig
    from arblab.hyperliquid_copy.report import write_report
    from arblab.hyperliquid_copy.download import file_hash
    fills,market,start,end,warmup = fixture()
    config = dict(mode="smoke_only",research_eligible=False,coins=["BTC"],closed_pnl_fee_semantics="unknown",
                  ranking=asdict(RankingConfig(min_active_days=1,min_episodes=1,min_notional=0,min_minutes=0)))
    result = run_offline(fills,market,start,end,warmup,config,dataset_sha256="x")
    evidence = b'{ "accepted": false, "issues": ["fixture"] }\n'
    summary = write_report(tmp_path/"out",result,config,{}, {},reconciliation_bytes=evidence)
    assert (tmp_path/"out/reconciliation.json").read_bytes() == evidence
    assert summary["artifact_hashes"]["reconciliation.json"] == file_hash(tmp_path/"out/reconciliation.json")
