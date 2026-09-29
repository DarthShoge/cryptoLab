from datetime import datetime, timedelta, timezone

from arblab.hyperliquid_copy.annual_execution_capacity import (
    CacheState,
    ExecutionRun,
    calculate_envelope,
)
from arblab.hyperliquid_copy.derived_cache_policy_64 import EXPANDED_64_BYTES

UTC = timezone.utc
DAY = 24 * 60 * 60


def _ticks(days, step):
    origin = datetime(2025, 9, 1, tzinfo=UTC)
    return tuple(origin + timedelta(days=n) for n in range(0, days, step))


def _state(**changes):
    values = dict(
        retained_bytes=5_000_000_000,
        reserved_bytes=0,
        allocation_count=496,
        publication_count=203,
        database_pages=180,
        unknown_objects=False,
        pending_allocations=0,
    )
    values.update(changes)
    return CacheState(**values)


def _model(**changes):
    values = dict(
        cache=_state(),
        runs=(
            ExecutionRun("weekly", _ticks(365, 7), scope_count=4),
            ExecutionRun("daily", _ticks(365, 1), scope_count=4),
        ),
        source_days=457,
        lookback_days=90,
        free_disk_bytes=800 * 1024**3,
        projected_metadata_bytes=8 * 1024**2,
    )
    values.update(changes)
    return calculate_envelope(**values)


def test_weekly_creation_before_retirement_needs_97_feature_days():
    result = calculate_envelope(
        cache=_state(),
        runs=(ExecutionRun("weekly", _ticks(365, 7), scope_count=4),),
        source_days=457,
        lookback_days=90,
        free_disk_bytes=800 * 1024**3,
        projected_metadata_bytes=1024,
    )

    assert result.terms["weekly_feature_peak_days"] >= 97


def test_backward_second_run_requires_terminal_window_retirement():
    one = _model(runs=(ExecutionRun("weekly", _ticks(365, 7), 4),))
    two = _model()

    assert two.terms["feature_peak_days"] == 97
    assert two.terms["terminal_window_retirement_count"] == 90
    assert two.terms["anchor_count"] > one.terms["anchor_count"]
    assert two.terms["retirement_receipt_count"] > one.terms["retirement_receipt_count"]


def test_anchors_and_one_ordered_retirement_per_ranking_remain_charged():
    result = _model()
    ranking_events = result.terms["ranking_event_count"]

    assert result.terms["anchor_bytes"] == result.terms["anchor_count"] * 8 * 1024**2
    assert result.terms["ranking_retirement_receipt_count"] == ranking_events
    assert result.terms["retirement_receipt_bytes"] == (
        result.terms["retirement_receipt_count"] * 64 * 1024
    )
    assert result.terms["retirement_publication_rows"] == (
        result.terms["retirement_receipt_count"] + 1
    )


def test_pending_or_unknown_cache_state_blocks_execution():
    result = _model(
        cache=_state(pending_allocations=1, reserved_bytes=4096, unknown_objects=True)
    )

    assert not result.admitted
    assert "pending_cache_allocations" in result.blocking_bounds
    assert "unknown_cache_objects" in result.blocking_bounds


def test_free_disk_and_cache_limit_are_independent_bounds():
    result = _model(free_disk_bytes=1, cache_limit_bytes=32 * 1024**3)

    assert result.terms["cache_limit_bytes"] < EXPANDED_64_BYTES
    assert "cache_byte_limit" in result.blocking_bounds
    assert "free_disk_peak" in result.blocking_bounds


def test_metadata_must_be_resolved_and_fit_sqlite_page_limit():
    unresolved = _model(projected_metadata_bytes=None)
    oversized = _model(projected_metadata_bytes=4001 * 4096)

    assert "unknown_encoded_metadata_growth" in unresolved.blocking_bounds
    assert "sqlite_page_limit" in oversized.blocking_bounds
    assert oversized.terms["sqlite_max_pages"] == 4000


def test_batched_retirement_fits_publication_limit():
    result = _model()

    assert result.terms["projected_publication_count"] <= 10_000
    assert "publication_count_limit" not in result.blocking_bounds
