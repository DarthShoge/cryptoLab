"""Pure, conservative admission model for bounded annual execution.

Every term is an enforced maximum from the producer modules or an explicit
caller measurement.  Compression and expected encoded sizes are never credits.
"""

from dataclasses import dataclass
from datetime import datetime
from math import ceil

from .annual_execution_policy import ANNUAL_BOUNDED_ROLLING
from .cache_retirement_journal import MAX_BYTES as LEGACY_RETIREMENT_BYTES
from .candidate_day import SPILL_BYTES as CANDIDATE_SPILL_BYTES
from .derived_cache_policy_64 import EXPANDED_64_BYTES
from .derived_cache_resources import FREE_RESERVE, METADATA_BYTES, MAX_RECORDS
from .derived_publication import MAX_PUBLICATIONS
from .feature_resume_anchor import MAX_BYTES as LEGACY_ANCHOR_BYTES
from .ranking_staging_manifest import ROLES

SQLITE_PAGE_BYTES = 4096
SQLITE_MAX_PAGES = 4000
ALLOCATION_ROW_BYTES = 1024
CANDIDATE_DAY_DESCRIPTOR_BYTES = 2 * 1024
FEATURE_DAY_DESCRIPTOR_BYTES = 8 * 1024
ANCHOR_DESCRIPTOR_BYTES = 4 * 1024
ACTIVE_RANKING_DESCRIPTOR_BYTES = ANNUAL_BOUNDED_ROLLING.active_ranking_descriptor_bytes

if (
    ANNUAL_BOUNDED_ROLLING.feature_anchor_bytes > LEGACY_ANCHOR_BYTES
    or ANNUAL_BOUNDED_ROLLING.retirement_journal_bytes > LEGACY_RETIREMENT_BYTES
):
    raise ValueError("Annual policy cap exceeds a legacy producer cap")


@dataclass(frozen=True)
class CacheState:
    retained_bytes: int
    reserved_bytes: int
    allocation_count: int
    publication_count: int
    database_pages: int
    unknown_objects: bool
    pending_allocations: int


@dataclass(frozen=True)
class ExecutionRun:
    name: str
    ticks: tuple[datetime, ...]
    scope_count: int


@dataclass(frozen=True)
class ExecutionEnvelope:
    terms: dict
    formula: dict
    assumptions: tuple[str, ...]
    admitted: bool
    blocking_bounds: tuple[str, ...]


def bounded_metadata_growth(
    *,
    new_allocations,
    source_days,
    feature_peak_days,
    anchor_count,
    ranking_retirements,
    feature_retirements,
):
    """Policy-enforced encoded row allowance, including conservative index space."""
    values = (
        new_allocations,
        source_days,
        feature_peak_days,
        anchor_count,
        ranking_retirements,
        feature_retirements,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Invalid bounded metadata terms")
    return (
        new_allocations * ALLOCATION_ROW_BYTES
        + source_days * CANDIDATE_DAY_DESCRIPTOR_BYTES
        + feature_peak_days * FEATURE_DAY_DESCRIPTOR_BYTES
        + anchor_count * ANCHOR_DESCRIPTOR_BYTES
        + (ranking_retirements + 1) * ANNUAL_BOUNDED_ROLLING.operation_descriptor_bytes
        + (feature_retirements + 1) * ANNUAL_BOUNDED_ROLLING.retirement_descriptor_bytes
        + 3 * ACTIVE_RANKING_DESCRIPTOR_BYTES
    )


def _checked_run(run):
    if (
        not isinstance(run, ExecutionRun)
        or type(run.name) is not str
        or not run.name
        or type(run.ticks) is not tuple
        or not run.ticks
        or type(run.scope_count) is not int
        or run.scope_count <= 0
        or any(
            not isinstance(tick, datetime) or tick.tzinfo is None for tick in run.ticks
        )
        or tuple(sorted(run.ticks)) != run.ticks
        or len(set(run.ticks)) != len(run.ticks)
    ):
        raise ValueError("Invalid annual execution run")
    return run


def _gap_days(ticks):
    if len(ticks) == 1:
        return 1
    return max(
        1,
        max(
            ceil((right - left).total_seconds() / 86400)
            for left, right in zip(ticks, ticks[1:], strict=False)
        ),
    )


def _run_built_days(run, lookback_days):
    elapsed = ceil((run.ticks[-1] - run.ticks[0]).total_seconds() / 86400)
    return lookback_days + max(0, elapsed)


def calculate_envelope(
    *,
    cache,
    runs,
    source_days,
    lookback_days,
    free_disk_bytes,
    projected_metadata_bytes,
    cache_limit_bytes=EXPANDED_64_BYTES,
):
    """Return named peak terms and blockers without mutating cache or inputs."""
    if (
        not isinstance(cache, CacheState)
        or type(runs) is not tuple
        or not runs
        or any(_checked_run(run) is not run for run in runs)
        or any(
            type(value) is not int or value < 0
            for value in (
                cache.retained_bytes,
                cache.reserved_bytes,
                cache.allocation_count,
                cache.publication_count,
                cache.database_pages,
                source_days,
                free_disk_bytes,
            )
        )
        or type(lookback_days) is not int
        or lookback_days <= 0
        or projected_metadata_bytes is not None
        and (type(projected_metadata_bytes) is not int or projected_metadata_bytes < 0)
        or type(cache_limit_bytes) is not int
        or cache_limit_bytes <= METADATA_BYTES
    ):
        raise ValueError("Invalid annual capacity inputs")

    run_peaks = {
        f"{run.name}_feature_peak_days": lookback_days + _gap_days(run.ticks)
        for run in runs
    }
    feature_peak_days = max(run_peaks.values())
    feature_retirements = sum(
        max(0, _run_built_days(run, lookback_days) - lookback_days) for run in runs
    )
    terminal_window_retirements = (len(runs) - 1) * lookback_days
    feature_retirements += terminal_window_retirements
    anchor_count = sum(len(run.ticks) for run in runs)
    ranking_events = sum(len(run.ticks) * run.scope_count for run in runs)
    ranking_retirements = ranking_events
    retirement_count = ranking_retirements + feature_retirements

    staging_reservation = sum(row[2] for row in ROLES.values())
    staging_retained = sum(
        row[2] for name, row in ROLES.items() if name not in ("scratch",)
    )
    transient_reservation = max(staging_reservation, CANDIDATE_SPILL_BYTES)
    candidate_day_bytes = source_days * ANNUAL_BOUNDED_ROLLING.candidate_day_bytes
    feature_bytes = feature_peak_days * ANNUAL_BOUNDED_ROLLING.feature_day_bytes
    anchor_bytes = anchor_count * ANNUAL_BOUNDED_ROLLING.feature_anchor_bytes
    receipt_bytes = retirement_count * ANNUAL_BOUNDED_ROLLING.retirement_journal_bytes
    live_ranking_bytes = (
        ANNUAL_BOUNDED_ROLLING.candidate_history_bytes + staging_retained
    )
    cache_peak = (
        cache.retained_bytes
        + cache.reserved_bytes
        + candidate_day_bytes
        + feature_bytes
        + anchor_bytes
        + receipt_bytes
        + live_ranking_bytes
        + transient_reservation
        + METADATA_BYTES
    )

    # One completed receipt row per operation, plus one transient intent row.
    retirement_publications = retirement_count + 1
    projected_publications = (
        cache.publication_count
        + source_days
        + anchor_count
        + feature_peak_days
        + retirement_publications
        + 3  # active candidate history, saved ranking and staged ranking
    )
    projected_allocations = (
        cache.allocation_count
        + source_days
        + anchor_count
        + feature_peak_days * ANNUAL_BOUNDED_ROLLING.feature_day_artifacts
        + retirement_count
        + len(ROLES)
        + 1
    )
    projected_pages = (
        None
        if projected_metadata_bytes is None
        else cache.database_pages + ceil(projected_metadata_bytes / SQLITE_PAGE_BYTES)
    )
    existing_cache_bytes = cache.retained_bytes + cache.reserved_bytes + METADATA_BYTES
    incremental_cache_bytes = max(0, cache_peak - existing_cache_bytes)
    disk_peak = incremental_cache_bytes + FREE_RESERVE

    terms = dict(
        **run_peaks,
        cache_limit_bytes=cache_limit_bytes,
        cache_peak_bytes=cache_peak,
        free_disk_bytes=free_disk_bytes,
        free_disk_required_bytes=disk_peak,
        incremental_cache_bytes=incremental_cache_bytes,
        feature_peak_days=feature_peak_days,
        feature_peak_bytes=feature_bytes,
        candidate_day_bytes=candidate_day_bytes,
        anchor_count=anchor_count,
        anchor_bytes=anchor_bytes,
        ranking_event_count=ranking_events,
        ranking_retirement_receipt_count=ranking_retirements,
        feature_retirement_receipt_count=feature_retirements,
        terminal_window_retirement_count=terminal_window_retirements,
        retirement_receipt_count=retirement_count,
        retirement_receipt_bytes=receipt_bytes,
        retirement_publication_rows=retirement_publications,
        transient_reservation_bytes=transient_reservation,
        live_ranking_bytes=live_ranking_bytes,
        projected_publication_count=projected_publications,
        publication_limit=MAX_PUBLICATIONS,
        projected_allocation_count=projected_allocations,
        feature_day_artifact_limit=ANNUAL_BOUNDED_ROLLING.feature_day_artifacts,
        allocation_limit=MAX_RECORDS,
        projected_database_pages=projected_pages,
        sqlite_max_pages=SQLITE_MAX_PAGES,
        projected_metadata_bytes=projected_metadata_bytes,
    )
    blockers = []
    if cache.pending_allocations or cache.reserved_bytes:
        blockers.append("pending_cache_allocations")
    if cache.unknown_objects:
        blockers.append("unknown_cache_objects")
    if cache_peak > cache_limit_bytes:
        blockers.append("cache_byte_limit")
    if disk_peak > free_disk_bytes:
        blockers.append("free_disk_peak")
    if projected_publications > MAX_PUBLICATIONS:
        blockers.append("publication_count_limit")
    if projected_allocations > MAX_RECORDS:
        blockers.append("allocation_count_limit")
    if projected_pages is None:
        blockers.append("unknown_encoded_metadata_growth")
    elif projected_pages > SQLITE_MAX_PAGES:
        blockers.append("sqlite_page_limit")

    return ExecutionEnvelope(
        terms=terms,
        formula=dict(
            cache_peak="existing + candidate_days + feature_peak + anchors + retirement_receipts + live_ranking + transient + metadata",
            publications="existing + candidate_days + anchors + peak_features + one_row_per_retirement + one_active_intent + active_ranking_rows",
            feature_peak="max(lookback + schedule_gap, two_lookbacks_on_backward_restart)",
        ),
        assumptions=(
            "Every configured output cap is charged without a compression credit.",
            "Anchors and committed retirement journals remain retained.",
            "A completed terminal feature window is retired before a cadence restarts backward.",
            "Each ranking event uses one ordered multi-target retirement journal.",
        ),
        admitted=not blockers,
        blocking_bounds=tuple(blockers),
    )
