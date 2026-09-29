"""Explicit verified ranking receipts, independent of intermediate cache lifetime."""

from dataclasses import dataclass, fields
from datetime import timedelta
from pathlib import Path
import re

from . import feature_metric_producer as producer
from . import candidate_metric_producer as raw_producer
from .annual_execution_policy import execution_policy
from .saved_ranking_provenance import verify_raw_provenance
from .bound_scoring_context import (
    BoundScoredCohort,
    bind_scored_context,
    _read_publication,
)
from .contracts import semantic_hash, utc
from .derived_publication import PublishedArtifacts, _encode, _inputs
from .disk_cohort_scoring import metric_context, _selection, _engine as scoring_engine
from .disk_score_result import ScoredCohort, read_cohort
from .download import file_hash
from .qualified_source_session import QualifiedSourceSession, _identity
from .saved_ranking_lookup import KIND, find_receipt_inputs
from .ranking_staging_policy import validate_policy


def _engine():
    root = Path(__file__).parent
    return semantic_hash(
        dict(
            producer=producer._engine(),
            raw_producer=raw_producer._engine(),
            scoring=scoring_engine(),
            code={
                name: file_hash(root / name)
                for name in (
                    "saved_feature_rankings.py",
                    "saved_ranking_provenance.py",
                    "saved_ranking_lookup.py",
                    "bound_scoring_context.py",
                    "qualified_source_session.py",
                )
            },
        )
    )


@dataclass(frozen=True, init=False)
class SavedFeatureRankings:
    _resources: object
    _source: QualifiedSourceSession
    _engine_digest: str
    _staging_policy: str | None
    _execution_policy: str | None

    def __init__(
        self,
        resources,
        source_session,
        *,
        staging_policy=None,
        execution_policy_name=None,
    ):
        validate_policy(staging_policy)
        policy = execution_policy(execution_policy_name)
        if policy is not None and staging_policy is None:
            raise ValueError("Annual execution requires staged ranking policy")
        if type(source_session) is not QualifiedSourceSession:
            raise ValueError("Verified qualified source session required")
        object.__setattr__(self, "_staging_policy", staging_policy)
        object.__setattr__(
            self, "_execution_policy", None if policy is None else policy.name
        )
        engine = self._current_engine()
        resources.lease.check()
        source_session.verify()
        object.__setattr__(self, "_resources", resources)
        object.__setattr__(self, "_source", source_session)
        object.__setattr__(self, "_engine_digest", engine)
        self._verify()

    def _current_engine(self):
        base = _engine()
        if self._staging_policy is None:
            return base
        from .ranking_staging_binding import _engine as staged_engine

        value = dict(base=base, policy=self._staging_policy, staging=staged_engine())
        if self._execution_policy is not None:
            value["execution_policy"] = self._execution_policy
        return semantic_hash(value)

    def _verify(self):
        self._resources.lease.check()
        self._source.verify()
        if self._current_engine() != self._engine_digest:
            raise ValueError("Saved ranking engine changed")
        self._source.verify_identity()
        self._resources.lease.check()

    def _query(self, decision, config, scope, semantics):
        decision = utc(decision)
        context = metric_context(config, decision, scope, semantics)
        if not self._source.origin <= decision - timedelta(
            days=config.lookback_days
        ) < decision <= self._source.finish or not set(config.coins) <= set(
            self._source.coins
        ):
            raise ValueError("Saved ranking query outside qualified source")
        result = dict(
            source=self._source.inputs(),
            metrics=context,
            selection=_selection(self._resources.root, config, decision, scope),
            engine=self._engine_digest,
        )
        if self._execution_policy is not None:
            result["execution_policy"] = self._execution_policy
        return result

    def capture(self, days, decision, config, scope, semantics):
        return self._capture(days, decision, config, scope, semantics, raw=False)

    def capture_raw(self, decision, config, scope, semantics):
        return self._capture(None, decision, config, scope, semantics, raw=True)

    def capture_staged(
        self, decision, config, scope, semantics, *, days=None, validation=None
    ):
        from .ranking_staging_capture import capture_staged

        return capture_staged(
            self, decision, config, scope, semantics, days=days, validation=validation
        )

    def _capture(self, days, decision, config, scope, semantics, *, raw):
        if self._staging_policy is not None:
            raise ValueError("Staging policy requires staged capture")
        self._verify()
        query = self._query(decision, config, scope, semantics)
        frozen = _encode(query)
        existing = self.find(decision, config, scope, semantics)
        if existing is not None:
            publication, inputs = _read_publication(
                self._resources, existing.publication.key, KIND
            )
            # Discovery owns the final catalogue/artifact guard. Do not run an
            # additional verifier after that guard has closed.
            confirmed = self.find(decision, config, scope, semantics)
            if confirmed is None or confirmed.publication != publication:
                raise ValueError("Existing saved ranking changed during capture")
            if _encode(self._query(decision, config, scope, semantics)) != frozen:
                raise ValueError("Saved ranking capture context changed")
            return inputs
        if raw:
            result = raw_producer.build_and_score_candidates(
                self._resources,
                query["source"]["pin"],
                query["source"]["origin"],
                decision,
                config,
                scope,
                semantics,
            )
        else:
            result = producer.build_and_score_features(
                self._resources,
                query["source"]["pin"],
                query["source"]["origin"],
                days,
                decision,
                config,
                scope,
                semantics,
            )
        # Existing helper verifies publication links, artifact binding, complete
        # metric/selection context and current scoring engine, including empties.
        bound = bind_scored_context(
            self._resources, result, config, decision, scope, semantics
        )
        if raw:
            verify_raw_provenance(
                self._resources, bound, query, config, decision, scope
            )
        else:
            self._check_source_link(bound, query)
        self._verify()
        if _encode(self._query(decision, config, scope, semantics)) != frozen:
            raise ValueError("Saved ranking capture context changed")
        inputs = _inputs(KIND, dict(query=query, ranking=bound.publication.key))
        PublishedArtifacts(self._resources).publish(
            KIND, inputs, [p.token for p in bound.publication.artifacts]
        )
        loaded = self.load(inputs, decision, config, scope, semantics)
        if (
            loaded.artifact != bound.artifact
            or loaded.candidate_count != bound.candidate_count
            or loaded.eligible_count != bound.eligible_count
            or loaded._selected_json != bound._selected_json
        ):
            raise ValueError("Saved ranking capture result mismatch")
        return inputs

    def find(self, decision, config, scope, semantics):
        self._verify()
        query = self._query(decision, config, scope, semantics)
        frozen = _encode(query)
        catalogue = _identity(self._resources.path)
        # Observe committed writes using one connection, independently of file
        # timestamp granularity. No transaction spans the scan or payload reads.
        with self._resources._connect() as observer:
            version = observer.execute("PRAGMA data_version").fetchone()[0]
            inputs = find_receipt_inputs(self._resources, query)
            result = (
                None
                if inputs is None
                else self.load(inputs, decision, config, scope, semantics)
            )
            if result is not None:
                identity = _identity(result.artifact.path)
                result.artifact.verify()
            self._verify()
            if (
                _identity(self._resources.path) != catalogue
                or _encode(self._query(decision, config, scope, semantics)) != frozen
                or result is not None
                and _identity(result.artifact.path) != identity
                or observer.execute("PRAGMA data_version").fetchone()[0] != version
            ):
                raise ValueError("Saved ranking discovery catalogue/query changed")
        return result

    def _check_source_link(self, bound, query):
        try:
            _, final = _read_publication(
                self._resources, bound.publication.key, "cohort_rankings"
            )
            _, scores = _read_publication(
                self._resources, final["scores"], "candidate_scores"
            )
            _, metrics = _read_publication(
                self._resources, scores["source"], "candidate_metrics"
            )
            features = metrics["provenance"]["features"]
            if (
                features["source"]["report_sha256"] != query["source"]["pin"]["sha256"]
                or features["semantics"] != query["metrics"]["semantics"]
                or metrics["provenance"]["engine"] != producer._engine()
            ):
                raise ValueError("Saved ranking feature source mismatch")
        except (KeyError, TypeError) as exc:
            raise ValueError("Missing verified feature ranking provenance") from exc

    def load(self, inputs, decision, config, scope, semantics):
        if self._staging_policy is None:
            return self._load(inputs, decision, config, scope, semantics)
        from .ranking_staging_read import catalogue_guard

        with catalogue_guard(self._resources):
            return self._load(inputs, decision, config, scope, semantics)

    def _load(self, inputs, decision, config, scope, semantics):
        normalized = _inputs(KIND, inputs)
        self._verify()
        query = self._query(decision, config, scope, semantics)
        if (
            set(normalized) != {"query", "ranking"}
            or type(normalized["ranking"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", normalized["ranking"])
            or _encode(normalized["query"]) != _encode(query)
        ):
            raise ValueError("Saved ranking receipt/query mismatch")
        publication = PublishedArtifacts(self._resources).lookup(KIND, normalized)
        if publication is None:
            raise ValueError("Saved ranking receipt is not published")
        if len(publication.artifacts) != 1:
            raise ValueError("Expected one saved ranking artifact")
        path = self._resources.root / publication.artifacts[0].path
        identity = _identity(path)
        result = read_cohort(self._resources.root, publication, query["selection"])
        if self._staging_policy is not None:
            from .ranking_staging_binding import verify_saved_binding

            verify_saved_binding(
                self._resources, normalized, publication, query, result, self._source
            )
        if PublishedArtifacts(self._resources).lookup(KIND, normalized) != publication:
            raise ValueError("Saved ranking publication changed")
        self._verify()
        if _identity(path) != identity:
            raise ValueError("Saved ranking artifact identity changed")
        if _encode(_inputs(KIND, inputs)) != _encode(normalized) or _encode(
            self._query(decision, config, scope, semantics)
        ) != _encode(query):
            raise ValueError("Saved ranking inputs/publication changed")
        return BoundScoredCohort(
            **{
                field.name: getattr(result, field.name)
                for field in fields(ScoredCohort)
            },
            bound_decision=utc(decision),
            bound_scope=scope,
        )
