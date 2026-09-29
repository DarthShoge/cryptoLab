"""Explicitly approved, size-frozen native fill archive acquisition.

No automatic retries: failed transfers retain their full reservation in the audit.
This bounds requested object bytes, not AWS billing or network protocol overhead.
"""

from contextlib import closing
from datetime import timedelta, datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import tempfile

from .archive import archive_keys
from .lab_config import day
from .download import BUCKET, file_hash
from .proxy_compact import _sync

MAX_BYTES = 6 * 1024**3
LEGACY_MAX_OBJECT_BYTES = 128 * 1024**2
MAX_OBJECT_BYTES = 384 * 1024**2


def download_archive(
    s3, start, end, output_root, *, max_bytes=MAX_BYTES, progress=None
):
    begin, finish = day(start), day(end)
    if not 0 < (finish - begin).days <= 7 or finish > datetime.now(timezone.utc):
        raise ValueError("Archive acquisition requires 1–7 completed UTC days")
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("Invalid approved byte budget")
    objects = []
    for offset in range((finish - begin).days):
        for key in archive_keys(str((begin + timedelta(days=offset)).date())):
            meta = s3.head_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
            size, etag = meta["ContentLength"], meta["ETag"]
            if type(size) is not int or not 0 < size <= MAX_OBJECT_BYTES or not etag:
                raise ValueError("Invalid or oversized archive metadata")
            objects.append(
                dict(
                    key=key, bytes=size, etag=etag, file=f"fills_{len(objects):04d}.lz4"
                )
            )
    total = sum(o["bytes"] for o in objects)
    if total > max_bytes:
        raise ValueError("Archive exceeds approved byte budget; no GET issued")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="proxy_archive_", dir=output_root))
    manifest = dict(
        schema="hyperliquid_proxy_archive_v1",
        bucket=BUCKET,
        start=start,
        end=end,
        objects=objects,
        expected_bytes=total,
        max_bytes=max_bytes,
        max_object_bytes=MAX_OBJECT_BYTES,
        reserved_bytes=0,
        complete=False,
        captured_at=datetime.now(timezone.utc).isoformat(),
    )
    path = root / "manifest.json"

    _checkpoint(path, manifest)
    return resume_archive(s3, path, progress=progress)


def _checkpoint(path, manifest):
    pending = path.parent / "manifest.pending"
    if pending.is_symlink():
        raise ValueError("Symlink archive manifest staging file")
    with pending.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    _sync(path.parent)


def _validate_resume(path):
    if path.stat().st_size > 1_000_000:
        raise ValueError("Archive manifest byte limit exceeded")
    data = json.loads(path.read_text())
    if (
        data.get("schema") != "hyperliquid_proxy_archive_v1"
        or data.get("bucket") != BUCKET
    ):
        raise ValueError("Unsupported archive manifest")
    begin, finish = day(data["start"]), day(data["end"])
    if not 0 < (finish - begin).days <= 7 or finish > datetime.now(timezone.utc):
        raise ValueError("Invalid archive dates")
    expected = [
        key
        for offset in range((finish - begin).days)
        for key in archive_keys(str((begin + timedelta(days=offset)).date()))
    ]
    objects = data["objects"]
    if not isinstance(objects, list) or [obj["key"] for obj in objects] != expected:
        raise ValueError("Archive source keys changed")
    object_limit = data.get("max_object_bytes", LEGACY_MAX_OBJECT_BYTES)
    if type(object_limit) is not int or not 0 < object_limit <= MAX_OBJECT_BYTES:
        raise ValueError("Invalid original archive object limit")
    total = reserved = 0
    pending_seen = False
    for index, obj in enumerate(objects):
        if obj["file"] != f"fills_{index:04d}.lz4":
            raise ValueError("Invalid archive filename")
        if (
            type(obj["bytes"]) is not int
            or not 0 < obj["bytes"] <= object_limit
            or not isinstance(obj["etag"], str)
            or not obj["etag"]
        ):
            raise ValueError("Invalid archive object metadata")
        total += obj["bytes"]
        target = path.parent / obj["file"]
        partial = target.with_suffix(".partial")
        if target.is_symlink() or partial.exists() or partial.is_symlink():
            raise ValueError("Unsafe or interrupted archive object")
        status = obj.get("status")
        if status == "requested":
            raise ValueError(
                "Interrupted request retains its reservation; separately approved recovery required"
            )
        if status == "downloaded":
            if (
                pending_seen
                or not target.is_file()
                or target.stat().st_size != obj["bytes"]
                or file_hash(target) != obj.get("sha256")
            ):
                raise ValueError("Completed archive object identity changed")
            reserved += obj["bytes"]
        elif status is None and "status" not in obj and not target.exists():
            pending_seen = True
        else:
            raise ValueError("Invalid archive object state")
    for field, expected_value in (
        ("expected_bytes", total),
        ("reserved_bytes", reserved),
    ):
        if type(data.get(field)) is not int or data[field] != expected_value:
            raise ValueError("Archive byte accounting mismatch")
    if (
        type(data.get("max_bytes")) is not int
        or not total <= data["max_bytes"] <= MAX_BYTES
    ):
        raise ValueError("Invalid original archive budget")
    if type(data.get("complete")) is not bool or (data["complete"] and pending_seen):
        raise ValueError("Invalid archive completion state")
    return data


def resume_archive(s3, manifest_path, *, progress=None):
    """Continue unrequested objects only, preserving the original batch budget.

    A requested-but-uncommitted object needs separately approved recovery; never
    retry it here. Completed content is hash-verified even for a completed batch.
    This is a batch primitive, not an annual job's lifetime budget authority.
    """
    path = Path(manifest_path).absolute()
    if path.name != "manifest.json" or any(
        p.is_symlink() for p in (path, *path.parents)
    ):
        raise ValueError("Unsafe archive manifest path")
    lock = path.parent / ".acquisition.lock"
    if lock.is_symlink():
        raise ValueError("Symlink archive acquisition lock")
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Archive acquisition is already running") from exc
        manifest = _validate_resume(path)
        if manifest["complete"]:
            return path
        return _transfer(s3, path, manifest, progress)


def _transfer(s3, path, manifest, progress):
    root, objects = path.parent, manifest["objects"]

    def checkpoint():
        _checkpoint(path, manifest)

    if progress:
        progress(
            dict(
                manifest=str(path),
                objects=len(objects),
                expected_bytes=manifest["expected_bytes"],
            )
        )
    for index, obj in enumerate(objects):
        if obj.get("status") == "downloaded":
            continue
        manifest["reserved_bytes"] += obj["bytes"]
        obj["status"] = "requested"
        checkpoint()  # Account for the entire object even if transfer is interrupted.
        response = s3.get_object(
            Bucket=BUCKET, Key=obj["key"], RequestPayer="requester", IfMatch=obj["etag"]
        )
        with closing(response["Body"]) as body:
            if (
                response.get("ContentLength") != obj["bytes"]
                or response.get("ETag") != obj["etag"]
            ):
                raise ValueError(
                    "Archive identity/length changed after metadata freeze"
                )
            target = root / obj["file"]
            temporary = target.with_suffix(".partial")
            digest, count = hashlib.sha256(), 0
            with temporary.open("xb") as stream:
                while chunk := body.read(min(1024**2, obj["bytes"] - count + 1)):
                    count += len(chunk)
                    if count > obj["bytes"]:
                        raise ValueError("Archive transfer length exceeded frozen size")
                    stream.write(chunk)
                    digest.update(chunk)
                if count != obj["bytes"]:
                    raise ValueError("Archive transfer length shorter than frozen size")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
            _sync(root)
        obj.update(status="downloaded", sha256=digest.hexdigest())
        checkpoint()
        if progress:
            progress(
                dict(
                    downloaded=index + 1,
                    objects=len(objects),
                    reserved_bytes=manifest["reserved_bytes"],
                )
            )
    manifest["complete"] = True
    checkpoint()
    return path
