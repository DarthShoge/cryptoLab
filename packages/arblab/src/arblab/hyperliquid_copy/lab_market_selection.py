"""Pure causal market admission before ranking traders."""

from datetime import timedelta
from .lab_config_v2 import ExplicitUniverse, CLASSES


def select_markets(
    catalogue,
    volume,
    universe,
    decision,
    previous=None,
    normalization_days=0,
    effective_at=None,
):
    explicit = isinstance(universe, ExplicitUniverse)
    ids = universe.instrument_ids if explicit else catalogue.known_ids(decision)
    rows = []
    for identifier in ids:
        instrument = catalogue.at(identifier, decision)
        reasons = []
        amount = None
        if instrument is None:
            reasons.append(
                "not_yet_known"
                if identifier not in catalogue.known_ids(decision)
                else "no_effective_catalogue_record"
            )
        else:
            if instrument.asset_class not in (
                CLASSES if universe.general else universe.classes
            ):
                reasons.append("class_not_selected")
            if not instrument.active(decision):
                reasons.append("not_listed")
            if not instrument.supported:
                reasons.append("unsupported_contract_model")
            if normalization_days and instrument.listed_at > decision - timedelta(
                days=normalization_days
            ):
                reasons.append("insufficient_normalization_history")
            if not explicit:
                amount = (
                    volume.trailing(
                        identifier,
                        decision,
                        universe.lookback_days,
                        universe.publication_lag_days,
                    )
                    if volume is not None
                    else None
                )
                if amount is None:
                    reasons.append("missing_volume")
                elif amount < universe.min_volume_usd:
                    reasons.append("below_minimum_volume")
        rows.append(
            dict(
                decision_time=decision,
                effective_at=effective_at or decision,
                instrument_id=identifier,
                display_name=instrument.display_name if instrument else None,
                venue=instrument.venue if instrument else None,
                asset_class=instrument.asset_class if instrument else None,
                volume_usd=amount,
                rank=None,
                eligible=not reasons,
                selected=False,
                reasons=reasons,
                budget=0.0,
                window_start=None
                if explicit
                else decision
                - timedelta(
                    days=universe.lookback_days + universe.publication_lag_days
                ),
                window_end=None
                if explicit
                else decision - timedelta(days=universe.publication_lag_days),
            )
        )
    eligible = sorted(
        [r for r in rows if r["eligible"]],
        key=lambda r: (-(r["volume_usd"] or 0), r["instrument_id"]),
    )
    chosen = eligible if explicit else eligible[: universe.top_n]
    for rank, row in enumerate(eligible, 1):
        row["rank"] = None if explicit else rank
        row["selected"] = row in chosen
        if not row["selected"]:
            row["reasons"] = ["rank_below_cutoff"]
        else:
            row["budget"] = (
                universe.weights[row["instrument_id"]]
                if explicit and universe.allocation == "custom"
                else 1 / len(chosen)
            )
    members = sorted(r["instrument_id"] for r in chosen)
    old, new = set(previous or []), set(members)
    entries, exits = sorted(new - old), sorted(old - new)
    cohort = dict(
        decision_time=decision,
        effective_at=effective_at or decision,
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
