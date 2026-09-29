"""Read-time causal metadata checks without reopening retired feature files."""

from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace

from . import candidate_history, candidate_metric_producer, feature_metric_producer
from . import qualified_window, feature_window, ranking_staging_sources
from .annual_execution_policy import execution_policy
from .contracts import semantic_hash
from .derived_publication import _encode, publication_key
from .disk_cohort_scoring import metric_context, _selection
from .disk_metric_rows import MAX_BYTES
from .download import file_hash
from .qualified_day import micros


def _digest(value):
    return type(value) is str and re.fullmatch("[a-f0-9]{64}", value) is not None


def _window(session, start, end):
    # Same conservative membership rule as QualifiedWindow, using the already
    # fully verified session's bounded immutable metadata (no event replay).
    entries = [
        e
        for e in session._files
        if e.rows and e.max_time >= micros(start) and e.min_time < micros(end)
    ]
    return dict(
        schema=1,
        report_sha256=session.inputs()["pin"]["sha256"],
        qualification_engine=semantic_hash(qualified_window._engine()),
        start=start.isoformat(),
        end=end.isoformat(),
        coins=list(session.coins),
        schema_sha256=session._files[0].schema_sha256,
        file_count=len(entries),
        file_membership_sha256=hashlib.sha256(
            _encode(
                [
                    (e.sha256, e.bytes, e.rows, e.min_time, e.max_time, e.schema_sha256)
                    for e in entries
                ]
            )
        ).hexdigest(),
        derivation_code=dict(qualified_window._code()),
    )


def validate_source(source, candidate, query, session):
    try:
        _validate(source, candidate, query, session)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Malformed staged causal provenance") from exc


def _validate(source, candidate, query, session):
    if (
        type(source) is not dict
        or set(source)
        != {"key", "route", "report", "origin", "inputs", "config", "engine"}
        or source["key"]
        != semantic_hash({k: v for k, v in source.items() if k != "key"})
        or source["route"] not in ("raw", "features")
        or source["report"] != query["source"]["pin"]
        or source["origin"] != query["source"]["origin"]
        or source["engine"] != ranking_staging_sources._engine()
        or set(source["inputs"]) != {"context", "provenance"}
        or _encode(source["inputs"]["context"]) != _encode(query["metrics"])
    ):
        raise ValueError("Saved staged source binding mismatch")
    metrics = query["metrics"]
    decision = datetime.fromisoformat(metrics["decision_time"])
    config = SimpleNamespace(**source["config"])
    # JSON object order is not scoring order; reconstruct only from the saved
    # ordered term names while still checking the original weights/directions.
    names = [term[0] for term in metrics["terms"]]
    if set(config.metric_weights) != set(names):
        raise ValueError("Saved staged scoring terms mismatch")
    config.metric_weights = {name: config.metric_weights[name] for name in names}
    if (
        _encode(
            metric_context(config, decision, metrics["scope"], metrics["semantics"])
        )
        != _encode(metrics)
        or _selection(Path("."), config, decision, metrics["scope"])
        != query["selection"]
    ):
        raise ValueError("Saved staged effective configuration mismatch")
    provenance = source["inputs"]["provenance"]
    policy = execution_policy(provenance.get("execution_policy"))
    route = source["route"]
    field = "source" if route == "raw" else "features"
    producer = candidate_metric_producer if route == "raw" else feature_metric_producer
    if (
        set(provenance)
        != {field, "candidates", "engine", "max_bytes", "max_partition_rows"}
        | ({"execution_policy"} if policy is not None else set())
        or provenance["engine"] != producer._engine()
        or type(provenance["max_bytes"]) is not int
        or provenance["max_bytes"] != MAX_BYTES
        or type(provenance["max_partition_rows"]) is not int
        or not 1 <= provenance["max_partition_rows"] <= 250000
        or publication_key("candidate_history", candidate) != provenance["candidates"]
    ):
        raise ValueError("Saved staged producer/candidate provenance mismatch")
    origin = datetime.fromisoformat(source["origin"]).replace(tzinfo=decision.tzinfo)
    span = decision - origin
    candidate_fields = {
        "schema",
        "report_sha256",
        "source_start",
        "decision",
        "coins",
        "scope",
        "day_count",
        "daily_chain_sha256",
        "engine",
        "max_bytes",
    }
    if policy is not None:
        candidate_fields |= {"execution_policy", "candidate_day_bytes"}
    if (
        set(candidate) != candidate_fields
        or type(candidate["schema"]) is not int
        or candidate["schema"] != 1
        or candidate["report_sha256"] != source["report"]["sha256"]
        or candidate["source_start"] != source["origin"]
        or candidate["decision"] != decision.isoformat()
        or candidate["coins"] != sorted(metrics["coins"])
        or candidate["scope"] != metrics["scope"]
        or type(candidate["day_count"]) is not int
        or candidate["day_count"] != span.days + bool(span.seconds or span.microseconds)
        or not _digest(candidate["daily_chain_sha256"])
        or candidate["engine"] != candidate_history._engine()
        or type(candidate["max_bytes"]) is not int
        or candidate["max_bytes"]
        != (MAX_BYTES if policy is None else policy.candidate_history_bytes)
        or policy is not None
        and (
            candidate["execution_policy"] != policy.name
            or candidate["candidate_day_bytes"] != policy.candidate_day_bytes
        )
    ):
        raise ValueError("Saved staged candidate causal context mismatch")
    start = decision - timedelta(days=config.lookback_days)
    if route == "raw":
        if _encode(provenance["source"]) != _encode(_window(session, start, decision)):
            raise ValueError("Saved staged raw lookback mismatch")
    else:
        first = start.replace(hour=0, minute=0, second=0, microsecond=0)
        last = decision.replace(hour=0, minute=0, second=0, microsecond=0)
        if last != decision:
            last += timedelta(days=1)
        window = provenance["features"]
        if (
            set(window)
            != {
                "schema",
                "source",
                "start",
                "end",
                "coins",
                "semantics",
                "days",
                "code_sha256",
            }
            or type(window["schema"]) is not int
            or window["schema"] != 1
            or window["start"] != start.isoformat()
            or window["end"] != decision.isoformat()
            or window["coins"]
            != ([metrics["scope"]] if metrics["scope"] else sorted(metrics["coins"]))
            or window["semantics"] != metrics["semantics"]
            or window["code_sha256"] != file_hash(Path(feature_window.__file__))
            or type(window["days"]) is not list
            or len(window["days"]) != (last - first).days
            or len(set(window["days"])) != len(window["days"])
            or not all(_digest(key) for key in window["days"])
            or _encode(window["source"]) != _encode(_window(session, first, last))
        ):
            raise ValueError("Saved staged feature lookback mismatch")
