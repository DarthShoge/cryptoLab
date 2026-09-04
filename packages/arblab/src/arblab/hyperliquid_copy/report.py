"""Risk-first offline artifacts. Never overwrite a prior experiment."""
from dataclasses import asdict
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import canonical_json, semantic_hash
from .download import file_hash
from .metrics import compute_performance


def summarize(result):
    metrics = compute_performance(result.equity)
    initial = result.equity[0]["equity"]
    notional = sum(abs(f["filled_qty"])*(f["vwap"] or 0) for f in result.fills)
    # Fill ratio is quantity-weighted only inside each asset, then notional-weighted
    # across requests; raw BTC and ETH units must not be added as one denominator.
    requested_usd = sum(f["requested_notional"] for f in result.fills)
    filled_usd = sum(abs(f["filled_qty"])*f["signal_mid"] for f in result.fills)
    metrics.update(turnover=notional/initial,fee_drag=sum(f["fee"] for f in result.fills)/initial,
                   funding_drag=-sum(f["cash_delta"] for f in result.funding)/initial,
                   fill_ratio=filled_usd/requested_usd if requested_usd else None,
                   stale_requests=sum(f["reason"] == "stale_book" for f in result.fills),
                   residual_gross_exposure=result.equity[-1]["gross_exposure"],
                   residual_net_exposure=result.equity[-1]["net_exposure"],
                   final_equity=result.equity[-1]["equity"],final_collateral=result.cash,
                   final_unrealized_pnl=result.equity[-1]["unrealized_pnl"],
                   residual_positions={c:asdict(p) for c,p in sorted(result.positions.items()) if p.qty},
                   max_gross_leverage=max(r["gross_exposure"]/r["equity"] for r in result.equity),
                   time_net_long=sum(r["net_exposure"] > 0 for r in result.equity)/len(result.equity),
                   time_net_short=sum(r["net_exposure"] < 0 for r in result.equity)/len(result.equity),
                   time_net_flat=sum(r["net_exposure"] == 0 for r in result.equity)/len(result.equity),
                   liquidation_count=0)
    if abs(metrics["final_equity"]-metrics["final_collateral"]-metrics["final_unrealized_pnl"]) > .01:
        raise ValueError("terminal accounting reconciliation failed")
    return metrics


def _parquet(path,rows,empty_fields):
    # Nested empty dictionaries cannot be serialized as a Parquet struct.
    table = pa.Table.from_pylist(rows) if rows else pa.table({k:pa.array([],type=t) for k,t in empty_fields.items()})
    pq.write_table(table,path,compression="zstd")


def write_report(destination,result,config,manifest,trial,reconciliation=None, *, reconciliation_bytes=None):
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    summaries, equity, fills, funding, control_equity, control_fills = [], [], [], [], [], []
    for kind,scenarios in [("strategy",result.strategies),("control",result.controls)]:
        for (name,latency),scenario in scenarios.items():
            key = {"signal_name" if kind == "strategy" else "control_name":name,"latency_seconds":latency}
            summaries.append(dict(scenario_type=kind,**key,**summarize(scenario)))
            (equity if kind == "strategy" else control_equity).extend(r | key for r in scenario.equity)
            (fills if kind == "strategy" else control_fills).extend(r | key for r in scenario.fills)
            if kind == "strategy":
                funding.extend(r | key for r in scenario.funding)
    inputs = {"config.json":config,"data_manifest.json":manifest,"trial.json":trial,
              "reconciliation.json":reconciliation or {"accepted":False,"issues":["smoke_only_unreconciled"]}}
    for name,value in inputs.items():
        (destination/name).write_bytes(canonical_json(value))
    if reconciliation_bytes is not None:
        json.loads(reconciliation_bytes)  # Reject malformed evidence before hashing.
        (destination/"reconciliation.json").write_bytes(reconciliation_bytes)
    scenario_schema = {"signal_name":pa.string(),"latency_seconds":pa.int64()}
    artifacts = [("trader_scores",result.scores,{"user":pa.string(),"score":pa.float64()}),
                 ("cohort_history",result.cohorts,{"members":pa.list_(pa.string())}),
                 ("signals",result.signals,{"signal_name":pa.string(),"value":pa.float64()}),
                 ("simulated_fills",fills,scenario_schema | {"filled_qty":pa.float64()}),
                 ("equity_curve",equity,scenario_schema | {"equity":pa.float64()}),
                 ("funding_ledger",funding,scenario_schema | {"cash_delta":pa.float64()}),
                 ("control_fills",control_fills,{"control_name":pa.string(),"latency_seconds":pa.int64(),"filled_qty":pa.float64()}),
                 ("control_equity_curve",control_equity,{"control_name":pa.string(),"equity":pa.float64()})]
    for name,rows,schema in artifacts:
        _parquet(destination/f"{name}.parquet",rows,schema)
    summary = dict(schema="hyperliquid_copy_report_v1",research_eligible=config["research_eligible"],
                   warnings=result.warnings,scenarios=summaries,signal_checksum=semantic_hash(result.signals))
    summary["result_checksum"] = semantic_hash(summary)
    summary["artifact_hashes"] = {p.name:file_hash(p) for p in sorted(destination.iterdir()) if p.is_file()}
    (destination/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines = ["# Hyperliquid trader ensemble", "", "## Data and reconciliation warnings", ""]
    lines.extend(f"- {warning}" for warning in result.warnings)
    lines.extend(["", "## Outcome versus benchmarks", "", "| Scenario | Latency | Return | Max DD | Fees / capital | Residual gross |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in summaries:
        name = row.get("signal_name") or row["control_name"]
        lines.append(f"| {name} | {row['latency_seconds']} | {row['total_return']:.4%} | {row['max_drawdown']:.4%} | {row['fee_drag']:.4%} | ${row['residual_gross_exposure']:,.2f} |")
    lines.extend(["", "## Drawdown, costs and latency", "",
                  "All four latencies use identical signal rows. Unfilled depth is canceled; no synthetic liquidity is added. Funding uses settlement mid as an oracle-price approximation.",
                  "", "## Cohort turnover and concentration", "",
                  "See cohort_history.parquet for every daily membership and cutoff; unknown wallets remain in the weight denominator.",
                  "", "## Position reconciliation", "",
                  "Follower equity reconciles to remaining collateral plus unrealized PnL. Open positions are not force-closed; native quantities and average entries are retained in summary.json.",
                  "", "## Forensic observations", "",
                  "The summary records stale requests, depth fill ratios, funding drag and residual exposure. Inspect simulated_fills.parquet around the largest equity changes before interpreting performance.",
                  "", "## Conclusions and next hypotheses", "",
                  "This run is integration evidence only." if not config["research_eligible"] else "These are backtest observations, not a claim of future profitability. Apply the pre-registered validation gate before opening the locked test.",
                  "No parameters were optimized by this report. Copy-trading performance remains sensitive to cohort selection, latency, external hedges and missing account state.",
                  "", "## Artifacts", "", "See summary.json for file checksums and the shared signal checksum."])
    (destination/"report.md").write_text("\n".join(lines)+"\n")
    return summary
