"""Stage scope/content checks; none of these qualify global event coverage."""

from datetime import timedelta
import json

from .compact_catalog import _validated
from .download import file_hash
from .lab_config import day
from .proxy_archive_download import _validate_resume
from .proxy_compact import _inputs, _sync


def read(path):
    if path.stat().st_size > 1_000_000:
        raise ValueError("Archive stage manifest byte limit exceeded")
    return json.loads(path.read_text())


def raw(path, batch):
    data = _validate_resume(path)
    actual = [{k: obj[k] for k in ("key", "bytes", "etag")} for obj in data["objects"]]
    if actual != batch["objects"] or (
        data["start"],
        data["end"],
        data["expected_bytes"],
    ) != (batch["start"], batch["end"], batch["bytes"]):
        raise ValueError("Raw archive frozen batch identity mismatch")
    return data


def _scope(data, batch, coins, source):
    if (
        (data.get("start"), data.get("end"), data.get("coins"))
        != (batch["start"], batch["end"], coins)
        or data.get("boundary_spill_retained") is not True
        or data.get("scope") != "all_wallets_for_declared_markets"
        or data.get("source_manifest") != str(source["path"])
        or data.get("source_manifest_sha256") != source["sha256"]
        or data.get("source_keys") != sorted(o["key"] for o in batch["objects"])
    ):
        raise ValueError("Normalized archive frozen scope/source mismatch")


def normalized(path, batch, coins, source):
    identity = file_hash(path)
    data = read(path)
    _scope(data, batch, coins, source)
    paths, _, _ = _inputs(path, data)
    for partition in paths:
        _sync(partition)
    _sync(path)
    _sync(path.parent)
    _sync(path.parent.parent)
    if file_hash(path) != identity:
        raise ValueError("Normalized archive identity changed")
    return data


def compact(path, batch, coins, raw_source, normalized_source):
    data = read(path)
    _scope(data["source_evidence"], batch, coins, raw_source)
    expected_days = [
        (day(batch["start"]) + timedelta(days=i)).date().isoformat()
        for i in range(batch["days"])
    ]
    if (
        data.get("partitioning") != "source_day"
        or data.get("validation") != "exact_canonical_projection_only"
        or data.get("source_manifest") != str(normalized_source["path"])
        or data.get("source_manifest_sha256") != normalized_source["sha256"]
        or [e.get("source_day") for e in data["files"]] != expected_days
    ):
        raise ValueError("Compact archive frozen source/layout mismatch")
    _validated(path)
    return data
