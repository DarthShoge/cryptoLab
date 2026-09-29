"""Verify registered canonical copies against their durable qualified archive.

This is a binding check, not independent archive-completion/listing qualification.
Publication must already have passed the complete-job and native-history gates.
"""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import re

from .archive import archive_keys
from .archive_cache import _safe
from .derived_publication import _encode, _inputs
from .derived_cache_resources import _regular, MAX_BYTES as CACHE_BYTES, METADATA_BYTES
from .derived_cache_policy import (
    EXPANDED_BYTES,
    _PinnedFile,
    _engine as expansion_engine,
)
from .derived_cache_policy_64 import (
    EXPANDED_64_BYTES,
    _engine_64 as expansion_64_engine,
)
from .download import file_hash
from .lab_config import day
from .feature_history_policy import checked_feature_policy
from .annual_execution_policy import checked_execution_policy
from .ranking_staging_policy import validate_policy as checked_staging_policy
from .native_history_evidence import verify_native_history_evidence
from .prefix_qualification import _previous, _engine
from .proxy_availability import native_history_starts
from .qualified_day import QualifiedFile
from .registration_provenance import verify_registration_provenance, MAX_BYTES

POLICY = "qualified_source_seed_v1"


def cache_reference(value):
    if (
        type(value) is not dict
        or set(value)
        not in (
            {"path", "identity"},
            {"path", "identity", "expansion_receipt"},
            {
                "path",
                "identity",
                "expansion_receipt",
                "expansion_64gib_receipt",
            },
        )
        or type(value["path"]) is not str
        or not 1 <= len(value["path"]) <= 4096
        or type(value["identity"]) is not str
        or not 1 <= len(value["identity"]) <= 256
    ):
        raise ValueError("Expected explicit existing derived-cache reference")
    frozen = _inputs("registered_cache_reference", value)
    path = Path(frozen["path"])
    if not path.is_absolute() or ".." in path.parts or not path.is_dir():
        raise ValueError("Expected existing absolute derived-cache directory")
    _safe(path)
    marker_path = path / ".initialized"
    if (
        not 0 < _regular(marker_path).st_size <= 4096
        or not 0 < _regular(path / "resources.sqlite3").st_size <= METADATA_BYTES // 2
        or _regular(path / ".resource.lock").st_size != 0
    ):
        raise ValueError("Missing/oversized existing cache initialization")
    try:
        with _PinnedFile(marker_path, 4096) as pin:
            marker = json.loads(pin.read())
            expected_limit = (
                EXPANDED_64_BYTES
                if "expansion_64gib_receipt" in frozen
                else EXPANDED_BYTES
                if "expansion_receipt" in frozen
                else CACHE_BYTES
            )
            if (
                type(marker) is not dict
                or set(marker)
                != {"version", "identity", "limit_bytes", "nonce", "root"}
                or type(marker["version"]) is not int
                or marker["version"] != 2
                or marker["identity"] != frozen["identity"]
                or type(marker["limit_bytes"]) is not int
                or marker["limit_bytes"] != expected_limit
                or not isinstance(marker["nonce"], str)
                or not re.fullmatch("[a-f0-9]{32}", marker["nonce"])
                or _encode(marker["root"])
                != _encode([path.stat().st_dev, path.stat().st_ino])
            ):
                raise ValueError("Existing cache identity/budget mismatch")
            if "expansion_receipt" in frozen:
                _expanded_receipt(marker, frozen["expansion_receipt"])
            if "expansion_64gib_receipt" in frozen:
                _expanded_64_receipt(
                    marker,
                    frozen["expansion_receipt"],
                    frozen["expansion_64gib_receipt"],
                )
            pin.check()
            if _encode(value) != _encode(frozen):
                raise ValueError("Derived-cache reference changed during verification")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid cache initialization marker") from exc
    # Full catalog/accounting verification remains mandatory under CacheLease.
    # This precheck does no writes and prevents lock creation in arbitrary dirs.
    return frozen


def _expanded_receipt(marker, receipt):
    # Structural binding only. The owned loader must still verify the retained
    # receipt publication and full resource accounting under its CacheLease.
    if (
        type(receipt) is not dict
        or set(receipt) != {"schema", "old_marker", "new_limit", "approval", "engine"}
        or type(receipt["schema"]) is not int
        or receipt["schema"] != 1
        or type(receipt["new_limit"]) is not int
        or receipt["new_limit"] != EXPANDED_BYTES
        or _encode(receipt["old_marker"])
        != _encode(dict(marker, limit_bytes=CACHE_BYTES))
        or type(receipt["approval"]) is not str
        or not 1 <= len(receipt["approval"].strip()) <= 1024
        or _encode(receipt["engine"]) != _encode(expansion_engine())
    ):
        raise ValueError("Invalid explicit cache expansion receipt")


def _expanded_64_receipt(marker, predecessor, receipt):
    old_marker = dict(marker, limit_bytes=EXPANDED_BYTES)
    if (
        type(receipt) is not dict
        or set(receipt)
        != {
            "schema",
            "predecessor_receipt",
            "old_marker",
            "new_limit",
            "approval",
            "engine",
        }
        or type(receipt["schema"]) is not int
        or receipt["schema"] != 1
        or _encode(receipt["predecessor_receipt"]) != _encode(predecessor)
        or _encode(receipt["old_marker"]) != _encode(old_marker)
        or type(receipt["new_limit"]) is not int
        or receipt["new_limit"] != EXPANDED_64_BYTES
        or type(receipt["approval"]) is not str
        or not 1 <= len(receipt["approval"].strip()) <= 1024
        or _encode(receipt["engine"]) != _encode(expansion_64_engine())
    ):
        raise ValueError("Invalid explicit 64-GiB cache expansion receipt")


def _source_inputs(directory):
    path = directory / "activity_source.json"
    _safe(path)
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("Registered source descriptor byte limit exceeded")
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("Expected registered source descriptor")
    return value


def _keys(start, end):
    return sorted(
        key
        for i in range((end - start).days)
        for key in archive_keys((start + timedelta(days=i)).date().isoformat())
    )


def _verify(manifest):
    metadata, directory = manifest.metadata, manifest.directory
    frozen = _encode(metadata)
    feature_policy = checked_feature_policy(metadata.get("feature_history_policy"))
    staging_policy = checked_staging_policy(metadata.get("ranking_staging_policy"))
    if (
        metadata.get("validation_mode") != "qualified_v1"
        or metadata.get("history_policy") != POLICY
    ):
        raise ValueError("Explicit qualified registration/source-seed policy required")
    cache = cache_reference(metadata.get("derived_cache"))
    checked_execution_policy(
        metadata.get("execution_policy"),
        feature_history_policy=feature_policy,
        ranking_staging_policy=staging_policy,
        registration_policy=metadata.get("history_policy"),
        cache_reference=cache,
    )
    begin, finish = day(metadata["coverage_start"]), day(metadata["coverage_end"])
    verify_registration_provenance(directory, metadata)
    starts = native_history_starts(metadata, metadata["coins"], begin, finish)
    verify_native_history_evidence(directory, metadata, starts)
    if not starts or "registration_provenance" not in metadata:
        raise ValueError(
            "Qualified registration requires copied native/source evidence"
        )
    inputs = _source_inputs(directory)
    pin = inputs["qualification"]
    report = _previous(pin, _engine())
    origin, end = day(report["source_start"]), day(report["source_end"])
    if (
        not 0 < (end - origin).days <= 732
        or not origin + timedelta(days=1) <= begin < finish <= end - timedelta(days=1)
        or (manifest.coverage_start, manifest.coverage_end) != (begin, finish)
        or inputs["coverage_start"] != metadata["coverage_start"]
        or inputs["coverage_end"] != metadata["coverage_end"]
        or inputs["source_start"] != report["source_start"]
        or inputs["source_end"] != report["source_end"]
        or inputs["coins"] != report["coins"]
        or inputs["coins"] != metadata["coins"]
        or pin["sha256"] != metadata["activity_provenance"]["source_manifest_hash"]
        or file_hash(directory / "activity_qualification.json") != pin["sha256"]
    ):
        raise ValueError("Registered coverage/qualified source binding mismatch")
    retained_keys, all_keys = _keys(begin, finish), _keys(origin, end)
    padding_keys = sorted(set(all_keys) - set(retained_keys))
    if (
        inputs["source_keys"] != retained_keys
        or inputs["padding_source_keys"] != padding_keys
        or metadata["activity_provenance"]["source_keys"] != retained_keys
        or metadata["activity_provenance"]["padding_source_keys"] != padding_keys
        or sorted(o["key"] for raw in report["raw_sources"] for o in raw["objects"])
        != all_keys
    ):
        raise ValueError("Registered archive source-key binding mismatch")
    entries = report["files"]
    if type(entries) is not list or not 1 <= len(entries) <= 4998:
        raise ValueError("Registered canonical file bound exceeded")
    if inputs["files"] != [
        {k: e[k] for k in ("path", "sha256", "bytes", "rows")} for e in entries
    ]:
        raise ValueError("Registered source inventory mismatch")
    originals = tuple(QualifiedFile.parse(entry) for entry in entries)
    if (
        len({e.path for e in originals}) != len(originals)
        or sum(e.bytes for e in originals) > 64 * 1024**3
    ):
        raise ValueError("Duplicate/oversized registered canonical source")
    retained = [e for e in metadata["files"] if e["name"].startswith("fills-")]
    expected = [
        dict(name=f"fills-{i:04d}.parquet", bytes=e.bytes, rows=e.rows, sha256=e.sha256)
        for i, e in enumerate(originals)
    ]
    if retained != expected or len({e.schema_sha256 for e in originals}) != 1:
        raise ValueError("Registered canonical copy inventory mismatch")
    for original, entry in zip(originals, expected):
        original.verify()
        destination = directory / entry["name"]
        if manifest.paths[entry["name"]] != destination:
            raise ValueError("Registered canonical destination mismatch")
        replace(original, path=destination).verify()
    for original in originals:
        original.verify()
    if _previous(pin, _engine()) != report:
        raise ValueError("Qualified registration source changed")
    verify_registration_provenance(directory, metadata)
    verify_native_history_evidence(directory, metadata, starts)
    if (
        _encode(metadata) != frozen
        or cache_reference(metadata["derived_cache"]) != cache
    ):
        raise ValueError("Qualified registration metadata/cache changed")
    return dict(pin), cache


def verify_qualified_registration(manifest):
    """Read-only validation; never acquire, create, recover or refund a cache."""
    try:
        return _verify(manifest)
    except (KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError("Malformed qualified registration binding") from exc
