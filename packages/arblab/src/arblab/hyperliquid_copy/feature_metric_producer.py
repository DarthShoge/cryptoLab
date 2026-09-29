"""Complete feature-backed metrics and rankings under the existing cache budget."""

from contextlib import closing
from datetime import timedelta
import hashlib
from pathlib import Path
import uuid

import duckdb
import pyarrow

from .annual_execution_policy import execution_policy
from .candidate_history import build_candidate_history
from .candidate_metric_producer import _candidate_rows, _engine as legacy_engine
from . import feature_publication, qualified_window
from .contracts import semantic_hash, utc
from .derived_cache_resources import _regular
from .derived_day_builder import _release_empty_scratch
from .derived_publication import PublishedArtifacts, _inputs, _encode
from .disk_cohort_scoring import metric_context, score_metric_artifact
from .disk_metric_rows import MAX_BYTES, write_metric_rows
from .download import file_hash
from .merged_feature_stream import merge_feature_metric_rows_from_days
from .feature_metric_summary import (
    discard_metric_summary,
    stage_metric_summaries,
    summary_metric_rows,
)
from .feature_window import FeatureWindow
from .prefix_qualification import _previous
from .qualified_day import _day
from .query_directory_pin import pin_directory
from .feature_writer_compatibility import compatible_code

SCRATCH_BYTES = 3 * 1024**3


def _engine():
    root = Path(__file__).parent
    return dict(
        schema=1,
        legacy=legacy_engine(),
        duckdb=duckdb.__version__,
        pyarrow=pyarrow.__version__,
        feature_engine=feature_publication.feature_engine(),
        qualification_engine=semantic_hash(qualified_window._engine()),
        code=compatible_code(
            {
                name: file_hash(root / name)
                for name in (
                    "feature_metric_producer.py",
                    "feature_candidate_merge.py",
                    "feature_wallet_metrics.py",
                    "merged_feature_stream.py",
                    "feature_metric_summary.py",
                    "feature_metric_result.py",
                    "feature_window.py",
                    "feature_publication.py",
                    "feature_records.py",
                    "feature_order.py",
                    "query_directory_pin.py",
                    "lab_config.py",
                )
            }
        ),
    )


def _prepare(
    resources,
    report_pin,
    source_start,
    days,
    decision,
    config,
    scope,
    semantics,
    max_bytes,
    max_partition_rows,
    execution_policy_name=None,
):
    resources.lease.check()
    if (
        type(max_bytes) is not int
        or not 0 < max_bytes <= MAX_BYTES
        or type(max_partition_rows) is not int
        or not 1 <= max_partition_rows <= 250000
    ):
        raise ValueError("Invalid feature metric resource bounds")
    decision = utc(decision)
    policy = execution_policy(execution_policy_name)
    context = metric_context(config, decision, scope, semantics)
    pin = dict(report_pin)
    window = FeatureWindow(
        resources,
        pin,
        days,
        decision - timedelta(days=config.lookback_days),
        decision,
        [scope] if scope else sorted(config.coins),
        semantics,
    )
    origin = _day(source_start).date().isoformat()
    if origin != window.days[0].inputs["origin"]:
        raise ValueError("Candidate source origin must match complete feature history")
    candidate = build_candidate_history(
        resources,
        pin,
        origin,
        decision,
        config.coins,
        scope,
        max_bytes=(MAX_BYTES if policy is None else policy.candidate_history_bytes),
        execution_policy_name=execution_policy_name,
    )
    source_files = tuple(
        Path(entry["path"])
        for entry in _previous(pin, qualified_window._engine())["files"]
    )
    provenance = dict(
        features=window.inputs(),
        candidates=candidate.key,
        engine=_engine(),
        max_bytes=max_bytes,
        max_partition_rows=max_partition_rows,
    )
    if policy is not None:
        provenance["execution_policy"] = policy.name
    inputs = _inputs(
        "candidate_metrics",
        dict(
            context=context,
            provenance=provenance,
        ),
    )
    snapshot = _encode(inputs)

    def candidate_stats():
        result = []
        for artifact in candidate.artifacts:
            info = _regular(resources.root / artifact.path)
            result.append(
                (
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            )
        return tuple(result)

    def source_stats():
        digest = hashlib.sha256()
        try:
            for path in source_files:
                info = path.lstat()
                digest.update(
                    _encode(
                        (
                            str(path),
                            info.st_dev,
                            info.st_ino,
                            info.st_mode,
                            info.st_nlink,
                            info.st_size,
                            info.st_mtime_ns,
                            info.st_ctime_ns,
                        )
                    )
                )
        except OSError as exc:
            raise ValueError("Candidate source file unavailable") from exc
        return digest.digest()

    def verify():
        resources.lease.check()
        before = window._stats(), candidate_stats(), source_stats()
        current = build_candidate_history(
            resources,
            pin,
            origin,
            decision,
            config.coins,
            scope,
            max_bytes=(MAX_BYTES if policy is None else policy.candidate_history_bytes),
            execution_policy_name=execution_policy_name,
        )
        window.verify()
        # Candidate rows are checked on full stream consumption as well. Recheck
        # their immutable file pins after window verification on reuse/scoring.
        for artifact in candidate.artifacts:
            path = resources.root / artifact.path
            if (
                _regular(path).st_size != artifact.bytes
                or file_hash(path) != artifact.sha256
            ):
                raise ValueError("Feature candidate artifact changed")
        actual = dict(
            context=metric_context(config, decision, scope, semantics),
            provenance=dict(
                features=window.inputs(),
                candidates=current.key,
                engine=_engine(),
                max_bytes=max_bytes,
                max_partition_rows=max_partition_rows,
            ),
        )
        if policy is not None:
            actual["provenance"]["execution_policy"] = policy.name
        if (
            current != candidate
            or _encode(actual) != snapshot
            or _encode(inputs) != snapshot
            or (window._stats(), candidate_stats(), source_stats()) != before
        ):
            raise ValueError("Feature metric inputs/configuration/engine changed")
        return actual["provenance"]

    return window, candidate, inputs, verify


def _produce(
    resources,
    window,
    candidate,
    config,
    scope,
    semantics,
    max_bytes,
    max_partition_rows,
):
    relative = f"scratch/{uuid.uuid4().hex}"
    token = resources.reserve(relative, SCRATCH_BYTES, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    identity = scratch.stat().st_dev, scratch.stat().st_ino
    spill = scratch / "spill"
    spill.mkdir()
    spill_identity = spill.stat().st_dev, spill.stat().st_ino
    with pin_directory(scratch, identity), pin_directory(spill, spill_identity):
        try:
            with (
                closing(_candidate_rows(resources, candidate)) as candidates,
                closing(
                    _staged_metric_rows(window, candidates, config, scratch)
                ) as rows,
            ):
                payload = write_metric_rows(
                    resources, rows, list(config.metric_weights), max_bytes=max_bytes
                )
        finally:
            resources.lease.check()
            info = spill.lstat()
            if (
                spill.is_symlink()
                or not spill.is_dir()
                or (info.st_dev, info.st_ino) != spill_identity
            ):
                raise ValueError(
                    "Feature metric spill identity changed; retained scratch"
                )
            if not any(spill.iterdir()):
                spill.rmdir()
            _release_empty_scratch(resources, token, scratch, identity)
    return payload


def _staged_metric_rows(window, candidates, config, scratch):
    summary = stage_metric_summaries(window, scratch)
    complete = False
    try:
        with closing(
            summary_metric_rows(summary, candidates, config, temp_root=scratch)
        ) as rows:
            yield from rows
        complete = True
    finally:
        if complete:
            discard_metric_summary(summary)


def build_feature_candidate_metrics(
    resources,
    report_pin,
    source_start,
    days,
    decision,
    config,
    scope,
    semantics,
    *,
    max_bytes=MAX_BYTES,
    max_partition_rows=250000,
    execution_policy_name=None,
):
    window, candidate, inputs, verify = _prepare(
        resources,
        report_pin,
        source_start,
        days,
        decision,
        config,
        scope,
        semantics,
        max_bytes,
        max_partition_rows,
        execution_policy_name,
    )
    publications = PublishedArtifacts(resources)
    if publications.lookup("candidate_metrics", inputs) is not None:
        verify()
        return inputs
    payload = _produce(
        resources,
        window,
        candidate,
        config,
        scope,
        semantics,
        max_bytes,
        max_partition_rows,
    )
    # _produce must finish ownership-checked cleanup before anything is visible.
    verify()
    publications.publish("candidate_metrics", inputs, [payload])
    return inputs


def build_and_score_features(
    resources,
    report_pin,
    source_start,
    days,
    decision,
    config,
    scope,
    semantics,
    *,
    max_bytes=MAX_BYTES,
    max_partition_rows=250000,
    execution_policy_name=None,
):
    inputs = build_feature_candidate_metrics(
        resources,
        report_pin,
        source_start,
        days,
        decision,
        config,
        scope,
        semantics,
        max_bytes=max_bytes,
        max_partition_rows=max_partition_rows,
        execution_policy_name=execution_policy_name,
    )
    _, _, expected, verify = _prepare(
        resources,
        report_pin,
        source_start,
        days,
        decision,
        config,
        scope,
        semantics,
        max_bytes,
        max_partition_rows,
        execution_policy_name,
    )
    if _encode(expected) != _encode(inputs):
        raise ValueError("Feature metric inputs changed before scoring")
    return score_metric_artifact(
        resources,
        inputs,
        config,
        decision,
        scope,
        semantics,
        verify_source=verify,
        max_bytes=max_bytes,
    )
