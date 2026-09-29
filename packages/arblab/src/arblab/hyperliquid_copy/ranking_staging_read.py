"""Protect standalone staged receipt reads across verification callbacks."""

from contextlib import contextmanager

from .derived_cache_policy import _PinnedFile
from .derived_cache_resources import METADATA_BYTES


@contextmanager
def catalogue_guard(resources):
    with (
        _PinnedFile(resources.path, METADATA_BYTES // 2) as database,
        _PinnedFile(resources.marker, 4096) as marker,
        resources._connect() as observer,
    ):
        version = observer.execute("PRAGMA data_version").fetchone()
        yield
        if observer.execute("PRAGMA data_version").fetchone() != version:
            raise ValueError("Staged receipt catalogue changed during validation")
        database.check()
        marker.check()
        resources.lease.check()
