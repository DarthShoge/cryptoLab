"""Verify the persisted observed-history evidence bound to a registered dataset."""

from datetime import datetime
import json
import re

from .contracts import utc
from .download import file_hash
from .lab_config import day

POLICY_V1 = "complete_source_observed_history_v1"
POLICY_V2 = "complete_source_observed_history_v2"


def verify_native_history_evidence(directory, metadata, starts):
    if (
        "native_history_policy" not in metadata
        and "native_history_evidence" not in metadata
    ):
        return starts  # Legacy metadata uses native starts for funding too.
    pin = metadata.get("native_history_evidence")
    policy = metadata.get("native_history_policy")
    if (
        policy not in (POLICY_V1, POLICY_V2)
        or not isinstance(pin, dict)
        or set(pin) != {"name", "sha256"}
        or pin["name"] != "native_history_evidence.json"
        or not isinstance(pin["sha256"], str)
        or not re.fullmatch("[a-f0-9]{64}", pin["sha256"])
    ):
        raise ValueError("Invalid native-history evidence sidecar pin")
    path = directory / pin["name"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("Missing/unsafe native-history evidence sidecar")
    if file_hash(path) != pin["sha256"]:
        raise ValueError("Native-history evidence sidecar identity changed")
    evidence = json.loads(path.read_bytes())
    if (
        evidence.get("schema")
        != (
            "hyperliquid_native_history_evidence_v2"
            if policy == POLICY_V2
            else "hyperliquid_native_history_evidence_v1"
        )
        or evidence.get("policy") != policy
        or evidence.get("listing_dates_verified") is not False
        or evidence.get("native_availability_qualified") is not True
        or evidence.get("research_eligible") is not False
    ):
        raise ValueError("Invalid native-history evidence policy")
    try:
        declared = {
            coin: utc(datetime.fromisoformat(at))
            for coin, at in evidence["starts"].items()
        }
        funding_starts = (
            {
                coin: utc(datetime.fromisoformat(at))
                for coin, at in evidence["funding_starts"].items()
            }
            if policy == POLICY_V2
            else declared
        )
        funding_ends = (
            {
                coin: utc(datetime.fromisoformat(at))
                for coin, at in evidence["funding_ends"].items()
            }
            if policy == POLICY_V2
            else None
        )
        funding_rows = metadata.get("funding_history")
        funding_matches = policy == POLICY_V1 or (
            isinstance(funding_rows, list)
            and len(funding_rows) == len(declared)
            and {
                row["instrument_id"]: (
                    utc(datetime.fromisoformat(row["available_from"])),
                    utc(datetime.fromisoformat(row["available_until"])),
                    row["evidence_sha256"],
                )
                for row in funding_rows
            }
            == {
                coin: (funding_starts[coin], funding_ends[coin], pin["sha256"])
                for coin in declared
            }
            and all(
                set(row)
                == {
                    "instrument_id",
                    "available_from",
                    "available_until",
                    "evidence_sha256",
                    "description",
                }
                and isinstance(row["description"], str)
                and 0 < len(row["description"].strip()) <= 1000
                for row in funding_rows
            )
            and set(funding_starts) == set(funding_ends) == set(declared)
            and all(
                at.minute == at.second == at.microsecond == 0
                    and funding_ends[coin] >= day(evidence["coverage_end"])
                and at < funding_ends[coin]
                for coin, at in funding_starts.items()
            )
        )
        matches = (
            declared == starts
            and set(declared)
            == set(metadata["coins"])
            == set(evidence["inputs"]["coins"])
            and all(
                evidence[k] == metadata[k] for k in ("coverage_start", "coverage_end")
            )
            and all(
                row["evidence_sha256"] == pin["sha256"]
                for row in metadata["native_history"]
            )
            and evidence["inputs"]["qualification"]
            == evidence["observed"]["qualification"]
            and evidence["inputs"]["qualification"]["sha256"]
            == metadata["activity_provenance"]["source_manifest_hash"]
            and evidence["funding_bundle"]["sha256"]
            == metadata["funding_source_manifest_hash"]
            and funding_matches
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("Invalid native-history evidence bindings") from error
    if not matches or file_hash(path) != pin["sha256"]:
        raise ValueError("Native-history evidence does not match dataset bindings")
    return funding_starts
