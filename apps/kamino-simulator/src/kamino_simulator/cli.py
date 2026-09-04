import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from arblab.kamino_risk import (
    AccountSnapshot,
    CollateralPosition,
    DebtPosition,
    apply_actions,
    liquidation_prices,
    scenario_report,
)
from arblab.kamino_onchain import load_onchain_snapshot
from arblab.paths import fixture_path


def _parse_snapshot(payload: Dict[str, Any]) -> AccountSnapshot:
    collateral = [
        CollateralPosition(
            symbol=item["symbol"],
            amount=float(item["amount"]),
            price=float(item["price"]),
            ltv=float(item["ltv"]),
            liquidation_threshold=float(item["liquidation_threshold"]),
        )
        for item in payload.get("collateral", [])
    ]
    debt = [
        DebtPosition(
            symbol=item["symbol"],
            amount=float(item["amount"]),
            price=float(item["price"]),
        )
        for item in payload.get("debt", [])
    ]
    return AccountSnapshot(collateral=collateral, debt=debt)


def _format_report(title: str, report: Dict[str, float]) -> str:
    lines = [title]
    for key, value in report.items():
        lines.append(f"  {key}: {value:.6f}")
    return "\n".join(lines)


def run_simulation(payload: Dict[str, Any]) -> str:
    snapshot = _parse_snapshot(payload)
    baseline_report = scenario_report(snapshot)
    baseline_prices = liquidation_prices(snapshot)

    actions: List[Dict[str, float]] = payload.get("actions", [])
    if actions:
        snapshot = apply_actions(snapshot, actions)
    scenario = scenario_report(snapshot)
    scenario_prices = liquidation_prices(snapshot)

    output = [
        _format_report("Baseline:", baseline_report),
        json.dumps({"baseline_liquidation_prices": baseline_prices}, indent=2),
    ]
    if actions:
        output.extend(
            [
                _format_report("After actions:", scenario),
                json.dumps({"scenario_liquidation_prices": scenario_prices}, indent=2),
            ]
        )
    return "\n".join(output)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Kamino liquidation risk simulator (defaults to the bundled offline sample)."
    )
    parser.add_argument(
        "--input",
        help="Path to JSON collateral/debt data (default: bundled offline sample).",
    )
    parser.add_argument("--obligation", help="Kamino obligation account address.")
    parser.add_argument(
        "--program-id",
        default=os.getenv("KAMINO_PROGRAM_ID", "KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD"),
        help="Kamino lending program id (default: mainnet KLend).",
    )
    parser.add_argument("--rpc-url", default=os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"))
    parser.add_argument("--idl", help="Path to Kamino Anchor IDL JSON for decoding accounts.")
    parser.add_argument("--obligation-account-name", default="Obligation")
    parser.add_argument("--reserve-account-name", default="Reserve")
    args = parser.parse_args()

    payload: Optional[Dict[str, Any]] = None
    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif args.obligation and args.program_id and args.idl:
        snapshot = load_onchain_snapshot(
            obligation_address=args.obligation,
            program_id=args.program_id,
            rpc_url=args.rpc_url,
            idl_path=args.idl,
            obligation_account_name=args.obligation_account_name,
            reserve_account_name=args.reserve_account_name,
        )
        payload = {
            "collateral": [
                {
                    "symbol": position.symbol,
                    "amount": position.amount,
                    "price": position.price,
                    "ltv": position.ltv,
                    "liquidation_threshold": position.liquidation_threshold,
                }
                for position in snapshot.collateral
            ],
            "debt": [
                {
                    "symbol": position.symbol,
                    "amount": position.amount,
                    "price": position.price,
                }
                for position in snapshot.debt
            ],
            "actions": [],
        }
    elif args.obligation:
        parser.error("Provide --input or all of --obligation, --program-id, and --idl.")
    else:
        sample_path: Path = fixture_path("kamino_sample.json")
        with sample_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

    print(run_simulation(payload))


if __name__ == "__main__":
    main()
