"""Prepare candidate inputs before admission; subsequent verification is read-only."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import candidate_metric_producer as raw
from . import feature_metric_producer as features
from .annual_execution_policy import execution_policy
from .bound_scoring_context import _read_publication
from .candidate_history import CandidateHistory
from .contracts import utc, semantic_hash
from .derived_publication import PublishedArtifacts, _encode
from .derived_cache_policy import _PinnedFile
from .derived_cache_resources import METADATA_BYTES
from .disk_metric_rows import MAX_BYTES
from .download import file_hash
from .qualified_day import _day
from .qualified_window import QualifiedWindow


def _engine():
    return dict(
        raw=raw._engine(),
        features=features._engine(),
        code={
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "ranking_staging_sources.py",
                "ranking_staging_producer.py",
                "ranking_staging_rows.py",
                "ranking_staging_artifact.py",
                "bound_scoring_context.py",
                "derived_cache_policy.py",
            )
        },
    )


@dataclass(frozen=True)
class PreparedStagingSource:
    resources: object
    window: object
    candidate: object
    config: object
    decision: object
    scope: str | None
    semantics: str
    route: str
    max_partition_rows: int
    execution_policy: str | None
    key: str
    _verify: object
    _identity_guard: object

    def verify(self):
        return self._verify(self)

    def verify_identity(self):
        """Cheap final guard after full verification and external callbacks."""
        self._identity_guard(self)


def prepare_staging_source(
    resources,
    report_pin,
    source_start,
    decision,
    config,
    scope,
    semantics,
    *,
    days=None,
    max_partition_rows=250000,
    execution_policy_name=None,
):
    """Build only reusable candidate inputs; no temporary ranking allocations."""
    decision = utc(decision)
    policy = execution_policy(execution_policy_name)
    route = "raw" if days is None else "features"
    frozen_config, frozen_pin = _encode(vars(config)), _encode(report_pin)
    engine_functions = (_engine, raw._engine, features._engine)
    engine = _engine()
    args = (resources, report_pin, source_start)
    tail = (
        decision,
        config,
        scope,
        semantics,
        MAX_BYTES,
        max_partition_rows,
        execution_policy_name,
    )
    if days is None:
        window, candidate, inputs, _ = raw._prepare(*args, *tail)
    else:
        window, candidate, inputs, _ = features._prepare(*args, days, *tail)
    # Capture the exact causal daily chain while builders are still allowed.
    # Later verification never calls either legacy producer's rebuilding verifier.
    with CandidateHistory(
        resources,
        report_pin,
        source_start,
        decision,
        config.coins,
        scope,
        execution_policy_name=execution_policy_name,
    ) as history:
        dependencies = [
            ("candidate_day", publication) for publication in history.publications
        ]
        expected_history = dict(
            **history.inputs(),
            max_bytes=(MAX_BYTES if policy is None else policy.candidate_history_bytes),
        )
    dependencies.append(("candidate_history", candidate))
    bindings = []
    for kind, publication in dependencies:
        actual, bound = _read_publication(resources, publication.key, kind)
        if (
            actual != publication
            or kind == "candidate_history"
            and bound != expected_history
        ):
            raise ValueError("Prepared candidate chain mismatch")
        bindings.append((kind, publication, bound))
    candidate_source = QualifiedWindow(report_pin, _day(source_start), decision)
    paths = {
        candidate_source._anchor.report_path,
        candidate_source._anchor.witness.path,
        candidate_source.witness.path,
    }
    paths.update(entry.path for entry in candidate_source.entries)
    paths.update(
        resources.root / pin.path
        for _, publication, _ in bindings
        for pin in publication.artifacts
    )
    if days is not None:
        paths.update(
            resources.root / pin.path
            for day in window.days
            for pin in day.publication.artifacts
        )
    # Pin physical identities of the exact named code dependencies as well as
    # data, so the final guard does not need to invoke another engine hasher.
    stack = [engine]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for name, value in item.items():
                if name.endswith(".py") and Path(name).name == name:
                    paths.add(Path(__file__).with_name(name))
                stack.append(value)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    paths = tuple(sorted(paths))

    def physical():
        digest = hashlib.sha256()
        for path in paths:
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
        return digest.digest()

    identity = physical()

    def catalogue(observer):
        """Authenticate exact bindings without re-hashing immutable payloads.

        Payload content is fully verified once below.  Subsequent guards bind
        the same catalogue rows to the same physical file identities; an
        in-place write changes ctime and is rejected by ``physical()``.
        """
        rows = []
        for kind, publication, bound in bindings:
            row = observer.execute(
                "SELECT descriptor,sha256 FROM publications WHERE key=?",
                (publication.key,),
            ).fetchone()
            if row is None:
                raise ValueError("Prepared candidate publication changed or missing")
            try:
                descriptor = json.loads(row[0])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Prepared candidate publication changed") from exc
            expected = dict(
                schema=1,
                kind=kind,
                inputs=bound,
                artifacts=[
                    dict(
                        token=pin.token,
                        path=pin.path,
                        bytes=pin.bytes,
                        sha256=pin.sha256,
                    )
                    for pin in publication.artifacts
                ],
            )
            raw = _encode(expected)
            if _encode(descriptor) != raw or row[1] != hashlib.sha256(raw).hexdigest():
                raise ValueError("Prepared candidate publication changed")
            for pin in publication.artifacts:
                allocation = observer.execute(
                    "SELECT path,maximum,purpose,state,bytes,sha256 "
                    "FROM allocations WHERE token=?",
                    (pin.token,),
                ).fetchone()
                if (
                    allocation is None
                    or allocation[0] != pin.path
                    or allocation[2:]
                    != (
                        "payload",
                        "retained",
                        pin.bytes,
                        pin.sha256,
                    )
                    or allocation[1] < pin.bytes
                ):
                    raise ValueError("Prepared candidate allocation changed")
            rows.append((publication.key, row[1]))
        return tuple(rows)

    with resources._connect() as observer:
        catalogue_identity = catalogue(observer)
    context = dict(
        route=route,
        report=report_pin,
        origin=source_start,
        inputs=inputs,
        config=vars(config),
        engine=engine,
    )
    frozen = _encode(context)
    key = semantic_hash(context)

    def verify_identity(current):
        resources.lease.check()
        if (
            current.resources is not resources
            or current.window is not window
            or current.candidate != candidate
            or current.config is not config
            or (
                current.decision,
                current.scope,
                current.semantics,
                current.route,
                current.max_partition_rows,
                current.execution_policy,
                current.key,
            )
            != (
                decision,
                scope,
                semantics,
                route,
                max_partition_rows,
                execution_policy_name,
                key,
            )
        ):
            raise ValueError("Prepared staging binding changed")
        if _encode(vars(config)) != frozen_config or _encode(report_pin) != frozen_pin:
            raise ValueError("Prepared staging configuration/source changed")
        if (
            (_engine, raw._engine, features._engine) != engine_functions
            or _encode(context) != frozen
            or physical() != identity
        ):
            raise ValueError("Prepared staging physical/context identity changed")

    fully_verified = False

    def verify(current):
        nonlocal fully_verified
        verify_identity(current)
        with (
            _PinnedFile(resources.path, METADATA_BYTES // 2) as database,
            _PinnedFile(resources.marker, 4096) as marker,
            resources._connect() as observer,
        ):
            version = observer.execute("PRAGMA data_version").fetchone()
            if not fully_verified:
                candidate_source.verify()
                window.verify()
                publications = PublishedArtifacts(resources)
                for kind, publication, bound in bindings:
                    if publications.lookup(kind, bound) != publication:
                        raise ValueError(
                            "Prepared candidate publication changed or missing"
                        )
            if (
                catalogue(observer) != catalogue_identity
                or physical() != identity
                or _engine() != engine
                or _encode(context) != frozen
            ):
                raise ValueError("Prepared staging source/context/engine changed")
            if observer.execute("PRAGMA data_version").fetchone() != version:
                raise ValueError("Prepared source catalogue changed")
            database.check()
            marker.check()
            verify_identity(current)
            fully_verified = True
        return dict(key=key, **context)

    prepared = PreparedStagingSource(
        resources,
        window,
        candidate,
        config,
        decision,
        scope,
        semantics,
        route,
        max_partition_rows,
        execution_policy_name,
        key,
        verify,
        verify_identity,
    )
    prepared.verify()
    return prepared
