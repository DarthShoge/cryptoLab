"""Explicit staged capture; independent saved reload precedes owned disposal."""

from . import ranking_staging_sources as sources
from . import ranking_staging_binding as binding
from .annual_execution_policy import execution_policy, require_descriptor_bound
from .bound_scoring_context import _read_publication
from .derived_publication import PublishedArtifacts, _encode, _inputs
from .derived_cache_policy import _PinnedFile
from .disk_metric_rows import MAX_BYTES
from .ranking_staging_cleanup import finish_staging
from .ranking_staging_owner import RankingStagingOwner
from .ranking_staging_producer import produce_staged_metrics
from .ranking_staging_score import score_pending_metrics
from .ranking_staging_policy import POLICY
from .ranking_staging_read import catalogue_guard
from .saved_ranking_lookup import KIND


def capture_staged(
    receipts, decision, config, scope, semantics, *, days=None, validation=None
):
    if receipts._staging_policy != POLICY:
        raise ValueError("Explicit ranking staging policy required")
    if validation is not None and not callable(validation):
        raise ValueError("Invalid staged validation callback")
    receipts._verify()
    query = receipts._query(decision, config, scope, semantics)
    frozen = _encode(query)
    existing = receipts.find(decision, config, scope, semantics)
    if existing is not None:
        publication, inputs = _read_publication(
            receipts._resources, existing.publication.key, KIND
        )
        if validation is not None:
            validation()
        confirmed = receipts.find(decision, config, scope, semantics)
        if (
            confirmed is None
            or confirmed.publication != publication
            or _encode(receipts._query(decision, config, scope, semantics)) != frozen
        ):
            raise ValueError("Saved staged result changed")
        return inputs
    resources = receipts._resources
    with resources._connect() as db:
        if db.execute(
            "SELECT 1 FROM allocations WHERE state='pending' LIMIT 1"
        ).fetchone():
            raise ValueError("Pending obligations block staged production")
    prepared = sources.prepare_staging_source(
        resources,
        query["source"]["pin"],
        query["source"]["origin"],
        decision,
        config,
        scope,
        semantics,
        days=days,
        execution_policy_name=receipts._execution_policy,
    )
    source = binding.preflight(query, prepared, receipts._source)
    owner = RankingStagingOwner.create(
        resources, dict(query=query, producer=prepared.key)
    )
    try:
        metrics = produce_staged_metrics(owner, prepared)
        pending = score_pending_metrics(
            owner,
            metrics,
            config,
            decision,
            scope,
            semantics,
            verify_source=prepared.verify,
        )
        provenance = binding.bind_pending(owner, prepared, pending, query, source)

        def context():
            receipts._verify()
            prepared.verify()
            if validation is not None:
                validation()
            if _encode(receipts._query(decision, config, scope, semantics)) != frozen:
                raise ValueError("Staged capture query changed")
            prepared.verify_identity()
            receipts._source.verify_identity()

        context()
        pending.ranking.verify()
        token = owner._allocations["ranking"]["token"]
        resources.settle(token)
        annual = execution_policy(receipts._execution_policy)
        if annual is not None:
            require_descriptor_bound(
                resources,
                binding.KIND,
                provenance,
                [token],
                annual.active_ranking_descriptor_bytes,
            )
        final = PublishedArtifacts(resources).publish(binding.KIND, provenance, [token])
        inputs = _inputs(KIND, dict(query=query, ranking=final.key))
        context()
        if annual is not None:
            require_descriptor_bound(
                resources,
                KIND,
                inputs,
                [token],
                annual.active_ranking_descriptor_bytes,
            )
        receipt = PublishedArtifacts(resources).publish(KIND, inputs, [token])

        def verify_receipt():
            with (
                catalogue_guard(resources),
                _PinnedFile(pending.ranking.path, MAX_BYTES) as artifact,
            ):
                loaded = receipts.load(inputs, decision, config, scope, semantics)
                if (
                    loaded.publication != receipt
                    or binding.summary(loaded) != provenance["summary"]
                ):
                    raise ValueError("Independent staged receipt reload mismatch")
                context()
                artifact.check()
                resources.lease.check()
                return loaded.publication

        verify_receipt()
        finish_staging(owner, pending, receipt, verify_receipt=verify_receipt)
        verify_receipt()
        return inputs
    finally:
        owner.close()
