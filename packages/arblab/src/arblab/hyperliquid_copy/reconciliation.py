"""Read-only source comparisons; no fee-semantic inference from a leaderboard."""
from collections import defaultdict
from math import isclose

from .contracts import semantic_hash, utc
from .positions import PositionReplay


def resolve_fee_semantics(fills, *, tolerance=.000001, minimum_episodes=3):
    grouped = defaultdict(list)
    for f in sorted(fills, key=lambda f:f.order_key):
        grouped[f.user,f.coin].append(f)
    verdicts = []
    for rows in grouped.values():
        opening = None
        for f in rows:
            if abs(f.start_position) < 1e-9 and abs(f.post_position) > 1e-9:
                opening = f
            elif opening is not None:
                if abs(f.post_position) < 1e-9 and isclose(opening.sz, f.sz) and all(x.fee_token == "USDC" for x in (opening,f)):
                    gross = opening.post_position*(f.px-opening.px)
                    source = opening.closed_pnl+f.closed_pnl
                    net = gross-opening.fee-f.fee
                    matches = {name for name,value in [("gross_excludes_fee",gross),("net_includes_fee",net)]
                               if isclose(source,value,abs_tol=tolerance,rel_tol=0)}
                    verdicts.append(matches)
                opening = None
    common = set.intersection(*verdicts) if len(verdicts) >= minimum_episodes else set()
    return next(iter(common)) if len(common) == 1 else "unknown"


def fill_identity(f):
    return f.user, f.coin, f.tid, f.oid, f.exchange_time


def reconcile(archive, provider_fills, snapshot, cutoff, *, dataset_sha256, provider, evidence_hashes):
    cutoff = utc(cutoff)
    wallets = {user for user,_ in snapshot}
    archive = [f for f in archive if f.user in wallets and f.exchange_time < cutoff]
    external = [f for f in provider_fills if f.user in wallets and f.exchange_time < cutoff]
    archive_ids = {fill_identity(f):f for f in archive}
    provider_ids = {fill_identity(f):f for f in external}
    issues = []
    # Compare the provider's bounded observation window, not absent old Info history.
    for key, f in provider_ids.items():
        other = archive_ids.get(key)
        if other is None or any(getattr(f,k) != getattr(other,k) for k in ("px","sz","side","start_position","post_position","closed_pnl","fee","fee_token")):
            issues.append("fill_mismatch")
    if not provider_ids or not evidence_hashes:
        issues.append("missing_provider_evidence")
    positions = PositionReplay(archive).snapshot(cutoff, inclusive=False)
    deltas = []
    for key in sorted(set(positions)|set(snapshot)):
        known = key in positions
        delta = positions.get(key, 0)-snapshot.get(key, 0)
        deltas.append(dict(user=key[0], coin=key[1], delta=delta, known=known))
        if not known or abs(delta) > 1e-8:
            issues.append("position_mismatch")
    semantics = resolve_fee_semantics(external)
    if semantics == "unknown":
        issues.append("unresolved_fee_semantics")
    result = dict(schema="reconciliation_v1", dataset_sha256=dataset_sha256, provider=provider,
                  cutoff=cutoff, wallets=sorted(wallets), matches=len(provider_ids)-issues.count("fill_mismatch"),
                  position_deltas=deltas, closed_pnl_fee_semantics=semantics,
                  evidence_hashes=sorted(evidence_hashes), issues=sorted(set(issues)), accepted=not issues)
    return result | {"semantic_checksum":semantic_hash(result)}


def validate_reconciliation(artifact, dataset_sha256, cutoff):
    checksum = artifact.get("semantic_checksum")
    if checksum != semantic_hash({k:v for k,v in artifact.items() if k != "semantic_checksum"}):
        raise ValueError("reconciliation checksum mismatch")
    if not artifact["accepted"] or artifact["dataset_sha256"] != dataset_sha256 or not artifact["evidence_hashes"]:
        raise ValueError("unaccepted/out-of-scope reconciliation")
    if artifact["closed_pnl_fee_semantics"] == "unknown":
        raise ValueError("unresolved fee semantics")
    from datetime import datetime
    observed = artifact["cutoff"]
    if isinstance(observed,str):
        observed = datetime.fromisoformat(observed.replace("Z","+00:00"))
    if utc(observed) != utc(cutoff):
        raise ValueError("reconciliation cutoff mismatch")
