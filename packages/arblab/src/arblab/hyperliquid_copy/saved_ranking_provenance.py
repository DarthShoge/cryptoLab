"""Authenticate raw ranking provenance before publishing a standalone receipt."""

from datetime import timedelta

from .bound_scoring_context import _read_publication
from .candidate_history import build_candidate_history
from .candidate_metric_producer import _engine
from .contracts import utc
from .disk_metric_rows import MAX_BYTES
from .qualified_window import QualifiedWindow


def verify_raw_provenance(resources, bound, query, config, decision, scope):
    try:
        _, final = _read_publication(
            resources, bound.publication.key, "cohort_rankings"
        )
        _, scores = _read_publication(resources, final["scores"], "candidate_scores")
        _, metrics = _read_publication(resources, scores["source"], "candidate_metrics")
        provenance = metrics["provenance"]
        if (
            set(provenance)
            != {"source", "candidates", "engine", "max_bytes", "max_partition_rows"}
            or provenance["engine"] != _engine()
            or type(provenance["max_bytes"]) is not int
            or not 0 < provenance["max_bytes"] <= MAX_BYTES
            or type(provenance["max_partition_rows"]) is not int
            or not 0 < provenance["max_partition_rows"] <= 250_000
        ):
            raise ValueError("Invalid raw ranking producer provenance")
        pin = query["source"]["pin"]
        decision = utc(decision)
        source = QualifiedWindow(
            pin, decision - timedelta(days=config.lookback_days), decision
        )
        if provenance["source"] != source.inputs():
            raise ValueError("Raw ranking lookback source mismatch")
        # This verifies origin, causal day membership, decision, coins and scope,
        # not merely the declared report hash or number of candidate wallets.
        candidate = build_candidate_history(
            resources, pin, query["source"]["origin"], decision, config.coins, scope
        )
        actual, _ = _read_publication(
            resources, provenance["candidates"], "candidate_history"
        )
        if actual != candidate:
            raise ValueError("Raw ranking candidate-history source mismatch")
        source.verify()
        if provenance["engine"] != _engine():
            raise ValueError("Raw ranking producer engine changed")
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing verified raw ranking provenance") from exc
