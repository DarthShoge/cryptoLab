"""Read-only complete-key inventory planning; never permission to acquire data."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

from .archive import archive_keys
from .archive_budget import MAX_JOB_BYTES, MAX_OBJECTS
from .download import file_hash
from .lab_config import day
from .proxy_archive_download import MAX_BYTES, MAX_OBJECT_BYTES


def object_index(objects):
    """Validate exact frozen identities without creating budget state."""
    if not isinstance(objects, list) or not 0 < len(objects) <= MAX_OBJECTS:
        raise ValueError("Invalid archive object scope")
    result = {}
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {"key", "bytes", "etag"}:
            raise ValueError("Invalid archive object fields")
        key, size, etag = obj["key"], obj["bytes"], obj["etag"]
        match = (
            re.fullmatch(
                r"node_fills(?:_by_block)?/hourly/(\d{8})/(?:[0-9]|1[0-9]|2[0-3])\.lz4",
                key,
            )
            if isinstance(key, str)
            else None
        )
        if (
            not match
            or type(size) is not int
            or not 0 < size <= MAX_BYTES
            or not isinstance(etag, str)
            or not 0 < len(etag) <= 200
            or key in result
        ):
            raise ValueError("Invalid or duplicate frozen archive object")
        date = datetime.strptime(match[1], "%Y%m%d").date().isoformat()
        if key not in archive_keys(date):
            raise ValueError("Archive format does not match source day")
        result[key] = (size, etag)
    if sum(size for size, _ in result.values()) > MAX_JOB_BYTES:
        raise ValueError("Archive exceeds job byte ceiling")
    return result


def plan_archive(inventory_path, *, max_batch_bytes=MAX_BYTES):
    if type(max_batch_bytes) is not int or not 0 < max_batch_bytes <= MAX_BYTES:
        raise ValueError("Invalid batch byte budget")
    path = Path(inventory_path).resolve(strict=True)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Inventory metadata byte limit exceeded")
    identity = file_hash(path)
    data = json.loads(path.read_text())
    if data.get("schema") != "hyperliquid_annual_metadata_audit_v1":
        raise ValueError("Unsupported archive inventory")
    begin, finish = day(data["start"]), day(data["end"])
    if not 0 < (finish - begin).days <= 732 or finish > datetime.now(timezone.utc):
        raise ValueError("Inventory requires 1–732 completed source days")
    index = object_index(data["objects"])
    expected, batches, current = set(), [], None
    for offset in range((finish - begin).days):
        date = begin + timedelta(days=offset)
        keys = archive_keys(date.date().isoformat())
        if not set(keys).issubset(index):
            raise ValueError("Missing expected archive source keys")
        expected.update(keys)
        objects = [
            dict(key=key, bytes=index[key][0], etag=index[key][1]) for key in keys
        ]
        size = sum(o["bytes"] for o in objects)
        if size > max_batch_bytes:
            raise ValueError("Whole source day exceeds batch byte limit")
        if (
            current is None
            or current["days"] == 7
            or current["bytes"] + size > max_batch_bytes
        ):
            current = dict(
                start=date.date().isoformat(), end=None, days=0, bytes=0, objects=[]
            )
            batches.append(current)
        current["end"] = (date + timedelta(days=1)).date().isoformat()
        current["days"] += 1
        current["bytes"] += size
        current["objects"].extend(objects)
    total = sum(b["bytes"] for b in batches)
    if set(index) != expected or data.get("missing") != []:
        raise ValueError("Inventory source-key scope mismatch")
    for field, value in (
        ("total_bytes", total),
        ("expected_objects", len(expected)),
        ("listed_expected_objects", len(expected)),
    ):
        if type(data.get(field)) is not int or data[field] != value:
            raise ValueError("Inventory declared totals mismatch")
    if file_hash(path) != identity:
        raise ValueError("Inventory changed during planning")
    unsupported = [
        obj
        for batch in batches
        for obj in batch["objects"]
        if obj["bytes"] > MAX_OBJECT_BYTES
    ]
    return dict(
        schema="hyperliquid_archive_plan_v1",
        inventory=str(path),
        inventory_sha256=identity,
        start=data["start"],
        end=data["end"],
        max_batch_bytes=max_batch_bytes,
        total_bytes=total,
        object_count=len(expected),
        batches=batches,
        unsupported_objects=unsupported,
        transfer_ready=not unsupported,
        authorization="not_granted_by_plan",
        coverage="source_keys_only_not_event_time_qualification",
    )
