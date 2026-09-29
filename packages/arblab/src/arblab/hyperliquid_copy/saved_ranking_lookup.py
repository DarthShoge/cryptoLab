"""Bounded exact-query receipt discovery; no payload reads or catalogue writes."""

import hashlib
import json
import re

from .annual_execution_policy import POLICY as ANNUAL_EXECUTION_POLICY
from .derived_publication import (
    MAX_ARTIFACTS,
    MAX_DESCRIPTOR_BYTES,
    MAX_PUBLICATIONS,
    _encode,
    _inputs,
    publication_key,
)

KIND = "saved_feature_ranking"


def _descriptor(resources, key, raw, digest):
    if type(raw) is not str or hashlib.sha256(raw.encode()).hexdigest() != digest:
        raise ValueError("Missing, oversized or changed ranking receipt descriptor")
    try:
        data = json.loads(raw)
        if (
            type(data) is not dict
            or set(data) != {"schema", "kind", "inputs", "artifacts"}
            or type(data["schema"]) is not int
            or data["schema"] != 1
            or _encode(data).decode() != raw
            or type(data["artifacts"]) is not list
            or not 1 <= len(data["artifacts"]) <= MAX_ARTIFACTS
        ):
            raise ValueError("Invalid saved ranking descriptor")
        inputs = _inputs(data["kind"], data["inputs"])
        if publication_key(data["kind"], inputs) != key:
            raise ValueError("Changed publication kind/inputs/key")
        # Classify only authenticated bounded metadata: a corrupt kind must not
        # hide a saved receipt and turn it into a replay-triggering cache miss.
        if data["kind"] != KIND:
            return None
        if len(data["artifacts"]) != 1:
            raise ValueError("Expected one saved ranking artifact")
        artifact = data["artifacts"][0]
        if (
            type(artifact) is not dict
            or set(artifact) != {"token", "path", "bytes", "sha256"}
            or type(artifact["bytes"]) is not int
            or not 0 <= artifact["bytes"] <= resources.limit
            or type(artifact["token"]) is not str
            or not re.fullmatch("[a-f0-9]{32}", artifact["token"])
            or type(artifact["sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", artifact["sha256"])
        ):
            raise ValueError("Invalid saved ranking artifact metadata")
        resources._path(artifact["path"])
        if not artifact["path"].startswith("artifacts/"):
            raise ValueError("Invalid saved ranking artifact namespace")
        query_keys = set(inputs["query"]) if type(inputs["query"]) is dict else set()
        if (
            set(inputs) != {"query", "ranking"}
            or type(inputs["query"]) is not dict
            or query_keys
            not in (
                {"source", "metrics", "selection", "engine"},
                {
                    "source",
                    "metrics",
                    "selection",
                    "engine",
                    "execution_policy",
                },
            )
            or "execution_policy" in inputs["query"]
            and inputs["query"]["execution_policy"] != ANNUAL_EXECUTION_POLICY
            or type(inputs["ranking"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", inputs["ranking"])
        ):
            raise ValueError("Invalid saved ranking receipt inputs/key")
        return inputs
    except (KeyError, TypeError, RecursionError) as exc:
        raise ValueError("Invalid saved ranking receipt structure") from exc


def find_receipt_inputs(resources, query):
    """Return one exact candidate or None; caller must load and finally recheck."""
    resources.lease.check()
    expected, found = _encode(query), None
    with resources._connect() as db:
        if (
            db.execute("SELECT count(*) FROM publications").fetchone()[0]
            > MAX_PUBLICATIONS
        ):
            raise ValueError("Ranking receipt catalogue row limit exceeded")
        for key, raw, digest in db.execute(
            "SELECT key, CASE WHEN length(CAST(descriptor AS BLOB)) BETWEEN 1 AND ? "
            "THEN descriptor END, sha256 FROM publications",
            (MAX_DESCRIPTOR_BYTES,),
        ):
            inputs = _descriptor(resources, key, raw, digest)
            if inputs is not None and _encode(inputs["query"]) == expected:
                if found is not None:
                    raise ValueError("Ambiguous saved ranking receipts for query")
                found = inputs
    resources.lease.check()
    return found
