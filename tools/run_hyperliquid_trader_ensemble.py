"""Causal, local-only historical runs. This tool cannot place exchange orders."""
import argparse
import fcntl
from datetime import datetime, timedelta
import json
from pathlib import Path

from arblab.hyperliquid_copy.contracts import UTC, canonical_json, semantic_hash
from arblab.hyperliquid_copy.configuration import validate_configuration
from arblab.hyperliquid_copy.pipeline import run_offline
from arblab.hyperliquid_copy.reconciliation import validate_reconciliation
from arblab.hyperliquid_copy.report import summarize, write_report
from arblab.hyperliquid_copy.storage import load_dataset, midnight
from arblab.hyperliquid_copy.trial_registry import TrialRegistry, chronological_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    run = sub.add_parser("run")
    run.add_argument("--config",type=Path,required=True)
    run.add_argument("--cache-root",type=Path,required=True)
    run.add_argument("--registry",type=Path)
    run.add_argument("--study-id",required=True)
    run.add_argument("--split",choices=["development","validation","locked-test"],required=True)
    run.add_argument("--start",required=True,help="Dataset first day, including warmup")
    run.add_argument("--end",required=True,help="Dataset end day, exclusive")
    run.add_argument("--reconciliation",type=Path)
    run.add_argument("--unlock-test")
    run.add_argument("--output-root",type=Path,default=Path("reports"))
    run.add_argument("--dry-run",action="store_true")
    run.add_argument("--max-rows",type=int,default=1000000)
    select = sub.add_parser("select-development")
    select.add_argument("--registry",type=Path,required=True)
    select.add_argument("--study-id",required=True)
    select.add_argument("--dataset-hash",required=True)
    args = parser.parse_args(argv)
    trial_id,registry,execution_lock = None,None,None
    try:
        if args.command == "select-development":
            print(TrialRegistry(args.registry).select(args.study_id,args.dataset_hash))
            return 0
        config = json.loads(args.config.read_text())
        validate_configuration(config)
        config_hash = semantic_hash(config)
        smoke = config["mode"] == "smoke_only"
        if smoke and args.split != "development":
            raise ValueError("smoke cannot enter validation or locked-test")
        if (args.unlock_test is not None and args.split != "locked-test") or (args.split == "locked-test" and args.unlock_test != config_hash):
            raise ValueError("matching --unlock-test required only for locked-test")
        if config["research_eligible"] and (not args.reconciliation or config["closed_pnl_fee_semantics"] == "unknown"):
            raise ValueError("research requires resolved fee semantics and dated --reconciliation evidence")
        start,end = midnight(args.start),midnight(args.end)
        eligible_start = start+timedelta(days=max(config["ranking"]["lookback_days"],config["scale_lookback_days"]))
        if end <= eligible_start:
            raise ValueError("insufficient warmup/history")
        sim_start,sim_end = (eligible_start,end) if smoke else chronological_splits(eligible_start,end)[args.split]
        warmup_start = sim_start-timedelta(days=config["scale_lookback_days"])
        if args.dry_run:
            print(canonical_json(dict(config_hash=config_hash,simulation_start=sim_start,simulation_end=sim_end,
                                      market_warmup=warmup_start,network_calls=0,research_eligible=config["research_eligible"])).decode())
            return 0
        fills,market,manifest = load_dataset(args.cache_root,start,end,config["coins"],
                                            market_start=eligible_start-timedelta(days=config["scale_lookback_days"]),max_rows=args.max_rows)
        evidence = json.loads(args.reconciliation.read_text()) if args.reconciliation else None
        if config["research_eligible"]:
            validate_reconciliation(evidence,manifest["dataset_hash"],end)
            if evidence["closed_pnl_fee_semantics"] != config["closed_pnl_fee_semantics"]:
                raise ValueError("config/reconciliation semantic mismatch")
        registry = TrialRegistry(args.registry or args.cache_root/"hyperliquid_copy/trials/trial_registry.jsonl")
        candidate_id = registry.begin(study_id=args.study_id,dataset_hash=manifest["dataset_hash"],config_hash=config_hash,
                                  split=args.split,smoke=smoke,unlock=args.unlock_test)
        execution_lock = registry.path.with_name(candidate_id+".run.lock").open("a")
        try:
            fcntl.flock(execution_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("trial is already running in another process") from None
        previous = registry.status(candidate_id)
        if previous["event"] == "completed":
            print(previous["artifact_directory"])
            return 0
        trial_id = candidate_id
        # Restrict source rows before passing to quality and ranking; no future split leakage.
        result = run_offline([f for f in fills if f.exchange_time < sim_end],market,sim_start,sim_end,warmup_start,
                             config,dataset_sha256=manifest["dataset_hash"])
        destination = args.output_root/("hyperliquid_trader_ensemble_"+datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        write_report(destination,result,config,manifest,{"trial_id":trial_id,"split":args.split},evidence,
                     reconciliation_bytes=args.reconciliation.read_bytes() if args.reconciliation else None)
        registry.complete(trial_id,summarize(result.strategies["conviction_trimmed",5]),artifact_directory=str(destination))
        trial_id = None
        print(destination)
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        if trial_id and registry:
            registry.complete(trial_id,{},failed=True)
        parser.error(str(exc))
    finally:
        if execution_lock is not None:
            execution_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
