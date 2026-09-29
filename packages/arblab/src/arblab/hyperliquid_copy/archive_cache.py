"""Hash-verified local archive reuse in front of the lifetime-budget adapter."""

import hashlib
from pathlib import Path
from types import MappingProxyType

from .archive_budget import BudgetedArchiveSource
from .archive_plan import object_index
from .download import BUCKET, file_hash
from .proxy_archive_download import _validate_resume


def _safe(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Unsafe symlink cache path")


class VerifiedArchiveCache:
    def __init__(self, manifests, objects):
        self.scope = MappingProxyType(object_index(objects))
        if not isinstance(manifests, (list, tuple)) or len(manifests) > 1000:
            raise ValueError("Invalid cache manifest count")
        entries, evidence, seen = {}, [], set()
        for value in manifests:
            path = Path(value).absolute()
            _safe(path)
            if path.name != "manifest.json" or path in seen:
                raise ValueError("Invalid or duplicate cache manifest")
            seen.add(path)
            identity = file_hash(path)
            data = _validate_resume(path)
            if not data["complete"]:
                raise ValueError("Cache acquisition is incomplete")
            for obj in data["objects"]:
                key = obj["key"]
                if key in entries or self.scope.get(key) != (obj["bytes"], obj["etag"]):
                    raise ValueError("Cache duplicate or frozen identity mismatch")
                entries[key] = MappingProxyType(
                    dict(
                        path=str(path.parent / obj["file"]),
                        bytes=obj["bytes"],
                        etag=obj["etag"],
                        sha256=obj["sha256"],
                        manifest=str(path),
                        manifest_sha256=identity,
                    )
                )
            if file_hash(path) != identity:
                raise ValueError("Cache manifest identity changed")
            evidence.append(
                dict(
                    manifest=str(path),
                    sha256=identity,
                    objects=len(data["objects"]),
                    bytes=data["expected_bytes"],
                )
            )
        self.entries = MappingProxyType(entries)
        self.evidence = evidence
        self.total_bytes = sum(e["bytes"] for e in entries.values())

    def verify(self, key):
        entry = self.entries[key]
        path, manifest = Path(entry["path"]), Path(entry["manifest"])
        _safe(path)
        _safe(manifest)
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_hash(path) != entry["sha256"]
            or file_hash(manifest) != entry["manifest_sha256"]
        ):
            raise ValueError("Cached archive identity changed")
        return entry


class _VerifiedBody:
    def __init__(self, entry):
        self.stream = Path(entry["path"]).open("rb", buffering=0)
        self.expected = (entry["bytes"], entry["sha256"])
        self.count, self.digest = 0, hashlib.sha256()

    def read(self, size):
        if type(size) is not int or not 0 < size <= 1024**2:
            raise ValueError("Invalid bounded cache body read")
        value = self.stream.read(size)
        self.count += len(value)
        self.digest.update(value)
        if self.count > self.expected[0] or (
            not value and (self.count, self.digest.hexdigest()) != self.expected
        ):
            raise ValueError("Streamed cache identity changed")
        return value

    def close(self):
        self.stream.close()


class CachedArchiveSource:
    def __init__(self, cache, remote=None):
        if remote is not None and (
            not isinstance(remote, BudgetedArchiveSource)
            or remote.budget.objects != cache.scope
        ):
            raise ValueError("Cached source requires matching lifetime-budget scope")
        self.cache, self.remote = cache, remote

    def _request(self, args, *, get):
        fields = {"Bucket", "Key", "RequestPayer"} | ({"IfMatch"} if get else set())
        if (
            set(args) != fields
            or args["Bucket"] != BUCKET
            or args["RequestPayer"] != "requester"
            or args["Key"] not in self.cache.scope
            or (get and args["IfMatch"] != self.cache.scope[args["Key"]][1])
        ):
            raise ValueError("Unscoped or changed cache request")
        if args["Key"] in self.cache.entries:
            return self.cache.verify(args["Key"])
        if self.remote is None:
            raise ValueError(
                "Archive object not cached; network acquisition is disabled"
            )
        return None

    def head_object(self, **args):
        entry = self._request(args, get=False)
        if entry is not None:
            return dict(ContentLength=entry["bytes"], ETag=entry["etag"])
        return self.remote.head_object(**args)

    def get_object(self, **args):
        entry = self._request(args, get=True)
        if entry is not None:
            return dict(
                ContentLength=entry["bytes"],
                ETag=entry["etag"],
                Body=_VerifiedBody(entry),
            )
        return self.remote.get_object(**args)
