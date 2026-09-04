from dataclasses import replace
from datetime import timedelta

from .test_ranking import history


def test_fee_semantics_requires_three_consistent_episodes():
    from arblab.hyperliquid_copy.reconciliation import resolve_fee_semantics
    fills = history(3)
    assert resolve_fee_semantics(fills) == "gross_excludes_fee"
    assert resolve_fee_semantics(fills[:2]) == "unknown"
    net = [replace(f, closed_pnl=f.closed_pnl-.02) if f.side == "A" else f for f in fills]
    assert resolve_fee_semantics(net) == "net_includes_fee"
    assert resolve_fee_semantics(net[:2]+fills[2:]) == "unknown"
    assert resolve_fee_semantics([replace(f, fee_token="HYPE") for f in fills]) == "unknown"


def test_reconciliation_checks_cutoff_positions_identity_and_hash():
    from arblab.hyperliquid_copy.reconciliation import reconcile, validate_reconciliation
    fills = history(3)
    cutoff = max(f.exchange_time for f in fills)+timedelta(seconds=1)
    artifact = reconcile(fills, fills, {(f.user,f.coin):0 for f in fills}, cutoff,
                         dataset_sha256="dataset", provider="fixture", evidence_hashes=["a", "b"])
    assert artifact["accepted"]
    validate_reconciliation(artifact, "dataset", cutoff)
    artifact["closed_pnl_fee_semantics"] = "net_includes_fee"
    import pytest
    with pytest.raises(ValueError, match="checksum"):
        validate_reconciliation(artifact, "dataset", cutoff)
