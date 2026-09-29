"""Complete candidate/ordered-fill merge for the budgeted metric producer.

The caller must supply verified, fully ordered inputs and reserve shared scratch.
Rows are unpublished staging: only complete exhaustion plus final source checks
can establish a complete candidate-metric publication.
"""

from itertools import groupby
from math import isfinite
from contextlib import closing
from datetime import timedelta
from pathlib import Path
import re
import uuid

import pyarrow.parquet as pq

from .annual_execution_policy import execution_policy
from .candidate_history import build_candidate_history, SCHEMA as CANDIDATE_SCHEMA
from .contracts import utc
from .derived_cache_resources import _regular
from .derived_day_builder import _release_empty_scratch
from .derived_publication import PublishedArtifacts, _encode
from .disk_cohort_scoring import metric_context, score_metric_artifact
from .disk_metric_rows import write_metric_rows, MAX_BYTES, MAX_ROWS, MAX_GROUPS
from .download import file_hash
from .lab_config import METRICS
from .ordered_wallet_partitions import OrderedWalletPartitions
from .qualified_window import QualifiedWindow
from .ranking import RankingConfig
from .streaming_wallet_metrics import stream_wallet_metrics


def merge_metric_rows(candidates, fills, decision, config, semantics, *, temp_root):
    """Use one wallet group/lookahead, preserving dormant candidates and arithmetic.

    Ownership of both input iterators stays with the caller, which must close
    them on interruption. No SQL-backed iterator may remain open during metrics.
    """
    legacy = RankingConfig(
        lookback_days=config.lookback_days,
        min_active_days=config.min_active_days,
        min_episodes=config.min_episodes,
        min_notional=config.min_notional,
        min_minutes=config.min_minutes,
    )
    groups = groupby(fills, key=lambda fill: fill.user)
    current = next(groups, None)
    previous = None
    for user in candidates:
        if (
            type(user) is not str
            or not re.fullmatch(r"0x[0-9a-f]{40}", user)
            or previous is not None
            and user <= previous
        ):
            raise ValueError("Invalid or unordered candidate identity")
        previous = user
        if current is not None and current[0] < user:
            raise ValueError("History wallet absent from complete candidate index")
        active = current is not None and current[0] == user
        result = stream_wallet_metrics(
            current[1] if active else iter(()),
            decision,
            legacy,
            semantics,
            temp_root=temp_root,
        )
        metrics = {key: result.metrics.get(key) for key in METRICS}
        if not result.closing_notional:
            metrics["pnl_efficiency"] = None
        metrics["gross_volume"] = result.gross_volume
        reasons = list(result.exclusions)
        if result.gross_volume < config.min_volume:
            reasons.append("insufficient_gross_volume")
        if any(
            metrics[k] is None or not isfinite(metrics[k])
            for k in config.metric_weights
        ):
            reasons.append("missing_ranking_metric")
        yield dict(user=user, metrics=metrics, exclusions=reasons)
        if active:
            current = next(groups, None)
    if current is not None:
        raise ValueError("History wallet absent from complete candidate index")


SCRATCH_BYTES = 3 * 1024**3


def _engine():
    root = Path(__file__).parent
    names = (
        "candidate_metric_producer.py",
        "candidate_history.py",
        "candidate_day.py",
        "qualified_window.py",
        "ordered_wallet_partitions.py",
        "wallet_partition_plan.py",
        "wallet_replay_groups.py",
        "streaming_wallet_metrics.py",
        "wallet_metric_spool.py",
        "ranking.py",
        "episodes.py",
        "contracts.py",
        "disk_metric_rows.py",
        "disk_cohort_scoring.py",
        "derived_publication.py",
        "derived_cache_resources.py",
        "derived_cache_lease.py",
        "derived_day_builder.py",
        "checkpoint_io.py",
    )
    return dict(schema=1, code={name: file_hash(root / name) for name in names})


def _prepare(
    resources,
    report_pin,
    source_start,
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
        or not 0 < max_partition_rows <= 250_000
    ):
        raise ValueError("Invalid candidate metric output/partition bound")
    decision = utc(decision)
    policy = execution_policy(execution_policy_name)
    context = metric_context(config, decision, scope, semantics)
    pin = dict(report_pin)
    engine = _engine()
    source = QualifiedWindow(
        pin, decision - timedelta(days=config.lookback_days), decision
    )
    candidate = build_candidate_history(
        resources,
        pin,
        source_start,
        decision,
        config.coins,
        scope,
        max_bytes=(MAX_BYTES if policy is None else policy.candidate_history_bytes),
        execution_policy_name=execution_policy_name,
    )
    provenance = dict(
        source=source.inputs(),
        candidates=candidate.key,
        engine=engine,
        max_bytes=max_bytes,
        max_partition_rows=max_partition_rows,
    )
    if policy is not None:
        provenance["execution_policy"] = policy.name
    inputs = dict(context=context, provenance=provenance)
    snapshot = _encode(inputs)

    def verify():
        resources.lease.check()
        source.verify()
        current = build_candidate_history(
            resources,
            pin,
            source_start,
            decision,
            config.coins,
            scope,
            max_bytes=(MAX_BYTES if policy is None else policy.candidate_history_bytes),
            execution_policy_name=execution_policy_name,
        )
        actual = dict(
            context=metric_context(config, decision, scope, semantics),
            provenance=dict(
                source=source.inputs(),
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
        ):
            raise ValueError("Candidate metric source/configuration/engine changed")
        return actual["provenance"]

    return source, candidate, inputs, verify


def _candidate_rows(resources, publication):
    if len(publication.artifacts) != 1:
        raise ValueError("Expected single complete candidate artifact")
    pin = publication.artifacts[0]
    path = resources.root / pin.path
    if _regular(path).st_size != pin.bytes or file_hash(path) != pin.sha256:
        raise ValueError("Candidate artifact identity changed")
    count = 0
    with pq.ParquetFile(path) as reader:
        expected = reader.metadata.num_rows
        if (
            expected > MAX_ROWS
            or reader.metadata.num_row_groups > MAX_GROUPS
            or reader.schema_arrow != CANDIDATE_SCHEMA
        ):
            raise ValueError("Candidate artifact schema/metadata limit")
        for batch in reader.iter_batches(batch_size=4096, use_threads=False):
            if batch.nbytes > 64 * 1024**2:
                raise ValueError("Candidate decoded batch limit")
            for user in batch.column(0).to_pylist():
                count += 1
                yield user
    if (
        count != expected
        or _regular(path).st_size != pin.bytes
        or file_hash(path) != pin.sha256
    ):
        raise ValueError("Candidate artifact changed or incompletely consumed")


def _ordered_fills(reader, scratch, max_partition_rows):
    with reader.verified_batch():
        for part in reader.plan(max_rows=max_partition_rows):
            if not part.physical_rows:
                continue
            path = scratch / f"{uuid.uuid4().hex}.parquet"
            artifact = reader.write(part, path)
            info = _regular(path)
            identity = info.st_dev, info.st_ino
            with closing(reader.read(artifact)) as fills:
                yield from fills
            reader._verify_artifact(artifact)
            info = _regular(path)
            if (info.st_dev, info.st_ino) != identity:
                raise ValueError(
                    "Owned ordered partition identity changed; retained scratch"
                )
            path.unlink()  # This invocation's fully consumed, verified temporary file only.


def build_candidate_metrics(
    resources,
    report_pin,
    source_start,
    decision,
    config,
    scope,
    semantics,
    *,
    max_bytes=MAX_BYTES,
    max_partition_rows=250_000,
    execution_policy_name=None,
):
    """Return inputs for a complete, verified candidate_metrics publication.

    One pending 3GiB scratch reservation covers sequential query spill, one
    ordered partition and one numeric spool. The final metric writer separately
    reserves its output before this lazy stream performs any sort/query work.
    """
    source, candidate, inputs, verify = _prepare(
        resources,
        report_pin,
        source_start,
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
    relative = f"scratch/{uuid.uuid4().hex}"
    token = resources.reserve(relative, SCRATCH_BYTES, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    info = scratch.stat()
    identity = info.st_dev, info.st_ino
    spill = scratch / "spill"
    spill.mkdir()
    info = spill.stat()
    spill_identity = info.st_dev, info.st_ino
    try:
        reader = OrderedWalletPartitions(source, config.coins, scratch, scope)
        with (
            closing(_candidate_rows(resources, candidate)) as candidates,
            closing(_ordered_fills(reader, scratch, max_partition_rows)) as fills,
            closing(
                merge_metric_rows(
                    candidates, fills, decision, config, semantics, temp_root=scratch
                )
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
            raise ValueError("Owned metric spill identity changed; retained scratch")
        if not any(spill.iterdir()):
            spill.rmdir()
        _release_empty_scratch(resources, token, scratch, identity)
    verify()
    publications.publish("candidate_metrics", inputs, [payload])
    return inputs


def build_and_score_candidates(
    resources,
    report_pin,
    source_start,
    decision,
    config,
    scope,
    semantics,
    *,
    max_bytes=MAX_BYTES,
    max_partition_rows=250_000,
    execution_policy_name=None,
):
    inputs = build_candidate_metrics(
        resources,
        report_pin,
        source_start,
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
        decision,
        config,
        scope,
        semantics,
        max_bytes,
        max_partition_rows,
        execution_policy_name,
    )
    if _encode(expected) != _encode(inputs):
        raise ValueError("Metric inputs changed before scoring")
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
