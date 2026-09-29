"""Merge immutable day-ordered feature streams without an external sort.

Each feature day is already ordered by wallet and native event key.  A heap merge
therefore produces the same complete window order as the disk sorter while only
holding one decoded row per day.  Source iterators remain owned here so early
consumer closure still runs every day's final verification.
"""

import heapq

from .feature_publication import FeatureDay
from .feature_candidate_merge import merge_feature_metric_rows
from .feature_window import FeatureWindow
from .wallet_day_features import EpisodeObservation


def _key(row):
    return (
        row.user,
        *row.order_key,
        0 if isinstance(row, EpisodeObservation) else 1,
    )


def _selected(rows, window):
    for row in rows:
        if window.start <= row.order_key[0] < window.end and row.coin in window.coins:
            yield row


def merge_feature_days(window):
    """Yield a verified feature window in wallet/native-key order.

    Memory is O(number of calendar days) rather than O(window rows).  Closing the
    returned generator closes and re-verifies every opened day before rechecking
    the complete window binding.
    """

    if not isinstance(window, FeatureWindow):
        raise ValueError("Qualified feature window required")
    window.verify()
    sources = []
    selected = []
    error = None
    try:
        for day in window.days:
            if not isinstance(day, FeatureDay):
                raise ValueError("Qualified feature day required")
            source = day.observations()
            sources.append(source)
            selected.append(_selected(source, window))
        yield from heapq.merge(*selected, key=_key)
    finally:
        for source in reversed(sources):
            try:
                source.close()
            except BaseException as exc:
                error = error or exc
        try:
            window.verify()
        except BaseException as exc:
            error = error or exc
        if error is not None:
            raise error


def merge_feature_metric_rows_from_days(
    window, candidates, config, semantics, *, temp_root
):
    """Reduce a feature window without materialising an externally sorted copy."""

    observations = merge_feature_days(window)
    try:
        yield from merge_feature_metric_rows(
            candidates,
            observations,
            window.end,
            config,
            semantics,
            temp_root=temp_root,
        )
    finally:
        observations.close()
