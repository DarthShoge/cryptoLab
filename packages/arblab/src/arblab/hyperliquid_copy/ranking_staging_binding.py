"""Standalone bounded provenance for a complete staged ranking artifact."""

from pathlib import Path
from datetime import datetime
import re

import pyarrow.parquet as pq

from . import ranking_staging_sources as sources
from . import ranking_staging_score as scoring
from .bound_scoring_context import _read_publication
from .contracts import semantic_hash, utc
from .derived_publication import _encode, _inputs
from .disk_metric_rows import MAX_BYTES, MAX_ROWS, MAX_GROUPS, METRIC_ROWS_SCHEMA
from .disk_score_query import SCORED_SCHEMA
from .download import file_hash
from .ranking_artifact import RANKING_SCHEMA
from .ranking_staging_policy import POLICY
from .ranking_staging_provenance import validate_source

KIND = "staged_cohort_rankings"


def _engine():
    return semantic_hash(
        dict(
            source=sources._engine(),
            scoring=scoring._engine().decode(),
            code={
                name: file_hash(Path(__file__).with_name(name))
                for name in (
                    "ranking_staging_binding.py",
                    "ranking_staging_capture.py",
                    "ranking_staging_cleanup.py",
                    "ranking_staging_policy.py",
                    "ranking_staging_provenance.py",
                    "ranking_staging_read.py",
                )
            },
        )
    )


def selected_digest(rows):
    return semantic_hash(
        [
            {
                key: value.isoformat() if key == "decision_time" else value
                for key, value in row.items()
                if key not in ("market_decision_time", "decision_trigger")
            }
            for row in rows
        ]
    )


def summary(result):
    return dict(
        candidate_count=result.candidate_count,
        eligible_count=result.eligible_count,
        requested_count=result.requested_count,
        selected_count=len(result.selected),
        selected_sha256=selected_digest(result.selected),
    )


def _descriptor(query, source, candidate, consumed, ranking, counts):
    return _inputs(
        KIND,
        dict(
            schema=1,
            policy=POLICY,
            query=query,
            source=source,
            candidate=candidate,
            consumed=consumed,
            ranking=ranking,
            summary=counts,
            engine=_engine(),
        ),
    )


def preflight(query, prepared, session):
    source = prepared.verify()
    actual, candidate = _read_publication(
        prepared.resources, prepared.candidate.key, "candidate_history"
    )
    if actual != prepared.candidate:
        raise ValueError("Staged preflight candidate changed")
    validate_source(source, candidate, query, session)
    prepared.verify_identity()
    session.verify_identity()
    maximum = dict(bytes=MAX_BYTES, rows=MAX_ROWS, sha256="f" * 64)
    counts = dict(
        candidate_count=MAX_ROWS,
        eligible_count=MAX_ROWS,
        requested_count=MAX_ROWS,
        selected_count=MAX_ROWS,
        selected_sha256="f" * 64,
    )
    _descriptor(
        query, source, candidate, dict(metrics=maximum, scores=maximum), maximum, counts
    )
    return dict(source=source, candidate=candidate)


def _file(pin, schema):
    pin.verify()
    with pq.ParquetFile(pin.path) as reader:
        if (
            reader.schema_arrow != schema
            or reader.metadata.num_rows > MAX_ROWS
            or reader.metadata.num_row_groups > MAX_GROUPS
        ):
            raise ValueError("Invalid staged binding schema/counts")
        rows = reader.metadata.num_rows
    return dict(bytes=pin.bytes, sha256=pin.sha256, rows=rows)


def bind_pending(owner, prepared, pending, query, expected_source):
    owner.verify()
    if (
        owner._context.get("query") != query
        or owner._context.get("producer") != prepared.key
    ):
        raise ValueError("Staged owner/query binding mismatch")
    consumed = dict(
        metrics=_file(pending.metrics, METRIC_ROWS_SCHEMA),
        scores=_file(pending.scores, SCORED_SCHEMA),
    )
    ranking = _file(pending.ranking, RANKING_SCHEMA)
    for role in ("metrics", "scores", "ranking"):
        if getattr(pending, role).path != owner.path(role):
            raise ValueError("Staged binding artifact is not owned")
        owner.verify_file(role)
    candidate_path = prepared.resources.root / prepared.candidate.artifacts[0].path
    count = pq.read_metadata(candidate_path).num_rows
    info = pending.summary
    if (
        any(item["rows"] != count for item in (*consumed.values(), ranking))
        or info["candidate_count"] != count
    ):
        raise ValueError("Staged candidate/metric/score/ranking counts differ")
    counts = {
        key: info[key]
        for key in (
            "candidate_count",
            "eligible_count",
            "requested_count",
            "selected_count",
        )
    }
    counts["selected_sha256"] = selected_digest(info["selected"])
    source = prepared.verify()
    if _encode(source) != _encode(expected_source["source"]):
        raise ValueError("Staged producer provenance changed")
    return _descriptor(
        query, source, expected_source["candidate"], consumed, ranking, counts
    )


def verify_saved_binding(resources, inputs, receipt, query, result, session):
    """No temporary files, candidate builders or feature reconstruction on read."""
    publication, body = _read_publication(resources, inputs["ranking"], KIND)
    if (
        set(body)
        != {
            "schema",
            "policy",
            "query",
            "source",
            "candidate",
            "consumed",
            "ranking",
            "summary",
            "engine",
        }
        or type(body["schema"]) is not int
        or body["schema"] != 1
        or body["policy"] != POLICY
        or body["engine"] != _engine()
        or _encode(body["query"]) != _encode(query)
        or publication.artifacts != receipt.artifacts
        or body["summary"] != summary(result)
    ):
        raise ValueError("Saved staged provenance/query/selection mismatch")
    validate_source(body["source"], body["candidate"], query, session)
    if any(
        type(body["summary"][key]) is not int
        or not 0 <= body["summary"][key] <= MAX_ROWS
        for key in (
            "candidate_count",
            "eligible_count",
            "requested_count",
            "selected_count",
        )
    ):
        raise ValueError("Invalid staged summary counts")
    if set(body["consumed"]) != {"metrics", "scores"}:
        raise ValueError("Missing consumed staging evidence")
    for item in (*body["consumed"].values(), body["ranking"]):
        if (
            set(item) != {"bytes", "rows", "sha256"}
            or type(item["bytes"]) is not int
            or not 0 < item["bytes"] <= MAX_BYTES
            or type(item["rows"]) is not int
            or item["rows"] != result.candidate_count
            or type(item["sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", item["sha256"])
        ):
            raise ValueError("Invalid consumed staging evidence")
    pin = receipt.artifacts[0]
    if body["ranking"] != dict(
        bytes=pin.bytes, sha256=pin.sha256, rows=result.candidate_count
    ):
        raise ValueError("Saved staged ranking pin mismatch")
    for batch in result.iter_batches():
        if any(
            row["decision_time"]
            != utc(datetime.fromisoformat(query["metrics"]["decision_time"]))
            or row["coin"] != query["metrics"]["scope"]
            for row in batch
        ):
            raise ValueError("Saved staged ranking row context mismatch")
    return body
