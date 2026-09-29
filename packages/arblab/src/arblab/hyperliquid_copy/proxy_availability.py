"""Qualified native-history starts, distinct from proxy mapping/listing dates."""

from datetime import datetime
import re

from .contracts import utc


def native_history_starts(data, coins, start, end):
    if "native_history" not in data:
        return {}  # Preserve legacy registered dataset semantics.
    rows = data["native_history"]
    if not isinstance(rows, list) or len(rows) != len(coins):
        raise ValueError("Invalid native history market coverage")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "instrument_id",
            "available_from",
            "evidence_sha256",
            "description",
        }:
            raise ValueError("Invalid native history record")
        coin = row["instrument_id"]
        if not isinstance(coin, str) or coin not in coins or coin in result:
            raise ValueError("Invalid native history instrument")
        try:
            at = utc(datetime.fromisoformat(row["available_from"]))
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid native history start") from error
        if not start <= at < end or at.minute or at.second or at.microsecond:
            raise ValueError("Invalid native history start hour")
        if not isinstance(row["evidence_sha256"], str) or not re.fullmatch(
            "[a-f0-9]{64}", row["evidence_sha256"]
        ):
            raise ValueError("Invalid native history evidence")
        if (
            not isinstance(row["description"], str)
            or not 0 < len(row["description"].strip()) <= 1000
        ):
            raise ValueError("Invalid native history description")
        result[coin] = at
    return result


def validate_native_fills(activity, starts):
    for coin, at in starts.items():
        if activity.db.execute(
            "SELECT 1 FROM fills WHERE coin=? AND exchange_time<? LIMIT 1", [coin, at]
        ).fetchone():
            raise ValueError(f"Fill contradicts qualified native history: {coin}")
