"""Budgeted score/rank publication over causally qualified metric publications.

Upstream metric producers must establish complete candidate membership and exact
metrics. Their required verifier returns the current verified provenance mapping;
this wrapper checks it before/after work and reuse, never infers it from fills.
"""

from pathlib import Path
import uuid

import duckdb
import pyarrow

from .contracts import utc, symbol
from .derived_day_builder import _release_empty_scratch
from .derived_publication import PublishedArtifacts, publication_key, _inputs, _encode
from .disk_metric_rows import MAX_BYTES
from .disk_score_query import ScoreQuery, scoring_terms
from .disk_cohort_query import CohortQuery
from .disk_score_result import read_cohort
from .download import file_hash
from .lab_config import bounded

SPILL_BYTES = 2 * 1024**3


def metric_context(config, decision, scope, semantics):
    if semantics not in ("gross_excludes_fee", "net_includes_fee"):
        raise ValueError("Unresolved metric fee semantics")
    coins = config.coins
    if (
        type(coins) not in (list, tuple)
        or not 1 <= len(coins) <= 50
        or len(set(coins)) != len(coins)
    ):
        raise ValueError("Invalid metric market context")
    coins = [symbol(c) for c in coins]
    if scope is not None and scope not in coins:
        raise ValueError("Metric scope outside configured markets")
    values = {}
    for name, lo, hi, integer in (
        ("lookback_days", 1, 732, True),
        ("min_active_days", 0, 3650, True),
        ("min_episodes", 0, 100000, True),
        ("min_notional", 0, 1e15, False),
        ("min_volume", 0, 1e15, False),
        ("min_minutes", 0, 1e7, False),
    ):
        value = getattr(config, name)
        bounded(value, name, lo, hi, integer=integer)
        values[name] = value
    return dict(
        schema=1,
        decision_time=utc(decision).isoformat(),
        scope=scope,
        coins=coins,
        semantics=semantics,
        metric_config=values,
        terms=scoring_terms(config),
    )


def _selection(root, config, decision, scope):
    # Constructor only validates/freezes parameters; it performs no I/O/query.
    query = CohortQuery(root / "unused", root, config, decision, scope)
    return dict(
        selection=config.selection,
        top_n=query.top_n,
        top_fraction=query.fraction,
        minimum=query.minimum,
        maximum=query.maximum,
        aggregation=query.aggregation,
    )


def _engine():
    root = Path(__file__).parent
    names = (
        "disk_metric_rows.py",
        "disk_score_query.py",
        "disk_cohort_query.py",
        "disk_score_result.py",
        "disk_cohort_scoring.py",
        "derived_publication.py",
        "derived_cache_resources.py",
        "derived_cache_lease.py",
        "derived_day_builder.py",
        "checkpoint_io.py",
        "ranking_artifact.py",
        "lab_market_evidence.py",
        "lab_config.py",
        "contracts.py",
    )
    return dict(
        schema=1,
        duckdb=duckdb.__version__,
        pyarrow=pyarrow.__version__,
        code={n: file_hash(root / n) for n in names},
    )


def score_metric_artifact(
    resources,
    metric_inputs,
    config,
    decision,
    scope,
    semantics,
    *,
    verify_source,
    max_bytes=MAX_BYTES,
):
    resources.lease.check()
    if (
        not callable(verify_source)
        or type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_BYTES
    ):
        raise ValueError("Invalid scoring verifier/resource bounds")
    inputs = _inputs("candidate_metrics", metric_inputs)
    context = metric_context(config, decision, scope, semantics)
    selection = _selection(resources.root, config, decision, scope)
    if (
        set(inputs) != {"context", "provenance"}
        or type(inputs["provenance"]) is not dict
        or not inputs["provenance"]
        or _encode(inputs["context"]) != _encode(context)
    ):
        raise ValueError("Metric input context mismatch")
    publications = PublishedArtifacts(resources)
    engine = _engine()

    def verify():
        resources.lease.check()
        provenance = _inputs("metric_provenance", verify_source())
        if _encode(provenance) != _encode(inputs["provenance"]):
            raise ValueError("Metric upstream provenance changed")
        if (
            _engine() != engine
            or _encode(metric_context(config, decision, scope, semantics))
            != _encode(context)
            or _encode(_selection(resources.root, config, decision, scope))
            != _encode(selection)
        ):
            raise ValueError("Scoring engine/configuration changed")
        source = publications.lookup("candidate_metrics", inputs)
        if source is None or len(source.artifacts) != 1:
            raise ValueError("Missing single-artifact metric publication")
        return source

    source = verify()
    score_inputs = dict(
        source=source.key, engine=engine, context=context, max_bytes=max_bytes
    )
    score_key = publication_key("candidate_scores", score_inputs)
    final_inputs = dict(
        scores=score_key, selection=selection, engine=engine, max_bytes=max_bytes
    )
    scored = publications.lookup("candidate_scores", score_inputs)
    final = publications.lookup("cohort_rankings", final_inputs)
    if final is not None:
        if scored is None:
            raise ValueError("Missing score publication dependency")
        result = read_cohort(resources.root, final, selection)
        verify()
        return result
    relative = f"scratch/{uuid.uuid4().hex}"
    scratch_token = resources.reserve(relative, SPILL_BYTES, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    info = scratch.stat()
    identity = info.st_dev, info.st_ino

    def output():
        relative = f"artifacts/{uuid.uuid4().hex}.parquet"
        return resources.reserve(
            relative, max_bytes, "payload"
        ), resources.root / relative

    try:
        if scored is None:
            with ScoreQuery(
                resources.root / source.artifacts[0].path, scratch, config
            ) as query:
                token, path = output()
                query.write_scores(path, max_bytes=max_bytes)
            resources.settle(token)
            verify()
            scored = publications.publish("candidate_scores", score_inputs, [token])
        if len(scored.artifacts) != 1:
            raise ValueError("Expected one score artifact")
        with CohortQuery(
            resources.root / scored.artifacts[0].path, scratch, config, decision, scope
        ) as query:
            token, path = output()
            query.write_rankings(path, max_bytes=max_bytes)
        resources.settle(token)
        verify()
        # Reverify intermediate pins as well as upstream before final publication.
        if publications.lookup("candidate_scores", score_inputs) != scored:
            raise ValueError("Score publication changed")
        final = publications.publish("cohort_rankings", final_inputs, [token])
    finally:
        _release_empty_scratch(resources, scratch_token, scratch, identity)
    result = read_cohort(resources.root, final, selection)
    verify()
    return result
