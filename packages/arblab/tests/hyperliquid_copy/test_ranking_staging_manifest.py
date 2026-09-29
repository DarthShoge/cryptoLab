from copy import deepcopy
import json

import pytest


def manifest():
    roles = {
        "manifest": ("staging", "", 64 * 1024, "payload"),
        "scratch": ("scratch", "", 3 * 1024**3, "scratch"),
        "metrics": ("staging", ".parquet", 512 * 1024**2, "payload"),
        "scores": ("staging", ".parquet", 512 * 1024**2, "payload"),
        "ranking": ("artifacts", ".parquet", 512 * 1024**2, "payload"),
    }
    return dict(
        schema=1,
        context={"query": {"decision": "2026-08-03"}, "engine": "a" * 64},
        allocations={
            role: dict(
                token=f"{i:032x}",
                path=f"{directory}/{i:032x}{suffix}",
                maximum=maximum,
                purpose=purpose,
            )
            for i, (role, (directory, suffix, maximum, purpose)) in enumerate(
                roles.items(), 1
            )
        },
    )


def test_manifest_canonical_roundtrip_is_detached():
    from arblab.hyperliquid_copy.ranking_staging_manifest import (
        encode_manifest,
        decode_manifest,
    )

    source = manifest()
    encoded = encode_manifest(source)
    decoded = decode_manifest(encoded)
    assert decoded == source
    source["context"]["query"]["decision"] = "changed"
    assert decode_manifest(encoded) == decoded
    assert encode_manifest(decoded) == encoded


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "extra",
        "missing_role",
        "extra_role",
        "duplicate_token",
        "duplicate_path",
        "path",
        "namespace",
        "purpose",
        "bound",
        "boolean_bound",
        "token",
        "context",
        "oversized",
    ],
)
def test_manifest_rejects_invalid_ownership(fault):
    from arblab.hyperliquid_copy.ranking_staging_manifest import encode_manifest

    data = manifest()
    row = data["allocations"]["metrics"]
    if fault == "schema":
        data["schema"] = True
    elif fault == "extra":
        data["extra"] = 1
    elif fault == "missing_role":
        del data["allocations"]["scores"]
    elif fault == "extra_role":
        data["allocations"]["other"] = deepcopy(row)
    elif fault == "duplicate_token":
        row["token"] = data["allocations"]["scores"]["token"]
    elif fault == "duplicate_path":
        row["path"] = data["allocations"]["scores"]["path"]
    elif fault == "path":
        row["path"] = "staging/../secrets"
    elif fault == "namespace":
        row["path"] = "artifacts/" + "b" * 32 + ".parquet"
    elif fault == "purpose":
        row["purpose"] = "scratch"
    elif fault == "bound":
        row["maximum"] += 1
    elif fault == "boolean_bound":
        row["maximum"] = True
    elif fault == "token":
        row["token"] = "not-a-token"
    elif fault == "context":
        data["context"] = {}
    else:
        data["context"]["large"] = "x" * 65536
    with pytest.raises(ValueError):
        encode_manifest(data)


@pytest.mark.parametrize(
    "raw",
    [b"{}", b"[1]", b"x" * 65537, b"\xff", b'{"schema":1,"schema":2}', b"[" * 1000],
    ids=["empty", "array", "oversized", "utf8", "duplicate", "depth"],
)
def test_manifest_decode_rejects_malformed_bytes(raw):
    from arblab.hyperliquid_copy.ranking_staging_manifest import decode_manifest

    with pytest.raises(ValueError):
        decode_manifest(raw)


def test_manifest_decode_rejects_noncanonical_encoding():
    from arblab.hyperliquid_copy.ranking_staging_manifest import decode_manifest

    with pytest.raises(ValueError):
        decode_manifest(json.dumps(manifest(), indent=2).encode())
