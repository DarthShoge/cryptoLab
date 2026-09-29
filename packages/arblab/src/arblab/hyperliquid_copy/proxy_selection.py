"""Proxy universe admission uses past observations, never fictitious listing metadata."""

from datetime import timedelta

from .contracts import utc
from .lab_config_v2 import ExplicitUniverse

CLASSES = {
    "crypto": "crypto",
    "commodities": "commodity",
    "equities": "equity",
    "indices": "index",
}


def select_proxy_markets(dataset, universe, decision, previous=None, *, history_days=0):
    decision = utc(decision)
    observed = set(dataset.activity.observed(decision))
    explicit = isinstance(universe, ExplicitUniverse)
    native_starts = getattr(dataset, "native_starts", {})
    ids = universe.instrument_ids if explicit else sorted(observed | set(native_starts))
    begin = (
        None
        if explicit
        else decision
        - timedelta(days=universe.lookback_days + universe.publication_lag_days)
    )
    end = None if explicit else decision - timedelta(days=universe.publication_lag_days)
    rows = []
    for coin in ids:
        mapping = dataset.mappings.at(coin, decision)
        reasons = []
        volume = None
        if coin in native_starts:
            if decision < native_starts[coin]:
                reasons.append("native_history_not_available")
            elif decision - timedelta(days=history_days) < native_starts[coin]:
                reasons.append("insufficient_native_history")
        if coin not in observed:
            reasons.append("not_yet_observed")
        if mapping is None:
            reasons.append("missing_proxy_mapping")
        else:
            asset_class = CLASSES[mapping.asset_class]
            if not universe.general and asset_class not in universe.classes:
                reasons.append("class_not_selected")
            if not explicit:
                if begin < utc(dataset.coverage_start) or end > utc(
                    dataset.coverage_end
                ):
                    reasons.append("missing_volume_coverage")
                else:
                    volume = dataset.activity.volume(coin, begin, end)
                    if volume < universe.min_volume_usd:
                        reasons.append("below_minimum_volume")
        rows.append(
            dict(
                decision_time=decision,
                effective_at=decision,
                instrument_id=coin,
                display_name=coin,
                venue=coin.split(":")[0] if ":" in coin else "hyperliquid",
                asset_class=CLASSES[mapping.asset_class] if mapping else None,
                volume_usd=volume,
                rank=None,
                eligible=not reasons,
                selected=False,
                reasons=reasons,
                budget=0.0,
                window_start=begin,
                window_end=end,
                availability_basis="historically_observed_not_listing_time",
                proxy_ticker=mapping.ticker if mapping else None,
            )
        )
    eligible = sorted(
        (r for r in rows if r["eligible"]),
        key=lambda r: (-(r["volume_usd"] or 0), r["instrument_id"]),
    )
    chosen = eligible if explicit else eligible[: universe.top_n]
    for rank, row in enumerate(eligible, 1):
        row["rank"] = None if explicit else rank
        row["selected"] = row in chosen
        if row["selected"]:
            row["budget"] = (
                universe.weights[row["instrument_id"]]
                if explicit and universe.allocation == "custom"
                else 1 / len(chosen)
            )
        else:
            row["reasons"] = ["rank_below_cutoff"]
    members = sorted(r["instrument_id"] for r in chosen)
    old, new = set(previous or []), set(members)
    entries, exits = sorted(new - old), sorted(old - new)
    cohort = dict(
        decision_time=decision,
        effective_at=decision,
        members=members,
        entries=entries,
        exits=exits,
        candidate_count=len(rows),
        eligible_count=len(eligible),
        selected_count=len(members),
        requested_count=len(ids) if explicit else universe.top_n,
        retention=len(old & new) / len(old) if old else None,
        membership_turnover=(len(entries) + len(exits)) / (len(old) + len(new))
        if previous is not None and old | new
        else (0.0 if previous is not None else None),
    )
    return sorted(
        rows, key=lambda r: (not r["selected"], r["rank"] or 0, r["instrument_id"])
    ), cohort
