"""Conservative observed-history qualification, explicitly not listing dates."""

from datetime import datetime
from pathlib import Path

from .annual_registration import completed_job_inputs
from .download import file_hash
from .lab_config import day
from .native_history_scan import scan_qualified_history
from .registration_market_inputs import verified_market_bundle

POLICY = "complete_source_observed_history_v2"


def qualify_observed_history(job_root, funding_pin, *, start, end, temp_root):
    policy_hash = file_hash(Path(__file__))
    inputs = completed_job_inputs(job_root, start=start, end=end)
    observed = scan_qualified_history(inputs["qualification"])
    funding = verified_market_bundle(funding_pin, kind="funding", temp_root=temp_root)[
        "manifest"
    ]
    if set(funding["intervals"]) != set(inputs["coins"]):
        raise ValueError("Funding/native history scope mismatch")
    begin, finish = day(start), day(end)
    starts, funding_starts, funding_ends = {}, {}, {}
    for coin in inputs["coins"]:
        row = observed["markets"][coin]
        if not row["rows"] or row["first_event"] is None:
            raise ValueError(f"Absent/unobserved native market: {coin}")
        first = datetime.fromisoformat(row["first_event"])
        if first >= finish:
            raise ValueError(
                f"Native market first observed after retained interval: {coin}"
            )
        if coin == "BTC" and first > begin:
            raise ValueError("BTC benchmark lacks full-window observed history")
        at = max(begin, first.replace(minute=0, second=0, microsecond=0))
        funding_begin, funding_end = map(
            datetime.fromisoformat, funding["intervals"][coin]
        )
        if (
            funding_begin >= finish
            or funding_end < finish
            or any(
                value.minute or value.second or value.microsecond
                for value in (funding_begin, funding_end)
            )
            or coin == "BTC"
            and funding_begin > begin
        ):
            raise ValueError(
                f"Funding coverage does not span required exposure: {coin}; native={at.isoformat()}, funding={funding_begin.isoformat()}..{funding_end.isoformat()}"
            )
        starts[coin] = at.isoformat()
        funding_starts[coin] = funding_begin.isoformat()
        funding_ends[coin] = funding_end.isoformat()
    if completed_job_inputs(job_root, start=start, end=end) != inputs:
        raise ValueError("Native-history source identity changed")
    if file_hash(Path(__file__)) != policy_hash:
        raise ValueError("Native-history qualification policy changed")
    return dict(
        schema="hyperliquid_native_history_evidence_v2",
        policy=POLICY,
        listing_dates_verified=False,
        native_availability_qualified=True,
        research_eligible=False,
        starts=starts,
        funding_starts=funding_starts,
        funding_ends=funding_ends,
        observed=observed,
        inputs=inputs,
        funding_bundle=dict(funding_pin),
        coverage_start=start,
        coverage_end=end,
        policy_sha256=policy_hash,
        limitation="Complete scoped source coverage and observed events; not listing dates, pre-observation history or complete wallet account equity.",
    )
