"""Open a bounded, windowed compact-history reader with causal prefix seeds.

Seeds are rebuilt, not persisted, in this version. Frozen-prefix validation still
scans old input and keeps existing byte/file/spill ceilings. This is not yet an
annual streaming ingest or a coverage-qualified registered dataset adapter.
"""

from datetime import datetime, timezone
from pathlib import Path

from .contracts import utc
from .download import file_hash
from .proxy_activity import ProxyActivity


def open_rolling_activity(
    catalog,
    manifest_ids,
    start,
    end,
    *,
    temp_root,
    max_input_bytes=8 * 1024**3,
    **limits,
):
    start, end = utc(start), utc(end)
    if start >= end:
        raise ValueError("Increasing query window required")
    if type(max_input_bytes) is not int or not 0 < max_input_bytes <= 8 * 1024**3:
        raise ValueError("Invalid bounded prefix input limit")
    entries = catalog.partitions(
        manifest_ids, datetime.min.replace(tzinfo=timezone.utc), end
    )
    if not entries:
        raise ValueError("No retained prefix; cannot infer complete empty history")
    if sum(e["bytes"] for e in entries) > max_input_bytes:
        raise ValueError("Rolling prefix input byte limit exceeded")
    paths = [Path(e["path"]) for e in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate physical prefix paths")

    def verify():
        for path, entry in zip(paths, entries):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != entry["bytes"]
                or file_hash(path) != entry["sha256"]
            ):
                raise ValueError("Compact partition identity changed")

    verify()
    activity = ProxyActivity(
        paths,
        temp_root=temp_root,
        query_window=(start, end),
        max_input_bytes=max_input_bytes,
        **limits,
    )
    try:
        verify()
    except BaseException:
        activity.close()
        raise
    return activity
