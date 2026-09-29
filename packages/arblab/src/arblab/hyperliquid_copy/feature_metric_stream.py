"""Finish each wallet-range reduction before ordering the next feature partition."""

from contextlib import closing, contextmanager
import os
import re
import uuid

from .derived_cache_resources import _regular
from .feature_candidate_merge import merge_feature_metric_rows

_END = object()


class _Candidates:
    def __init__(self, rows):
        self.rows, self.previous = iter(rows), None
        self.current = self._next()

    def _next(self):
        value = next(self.rows, _END)
        if value is not _END:
            if (
                type(value) is not str
                or not re.fullmatch(r"0x[0-9a-f]{40}", value)
                or self.previous is not None
                and value <= self.previous
            ):
                raise ValueError("Invalid or unordered candidate identity")
            self.previous = value
        return value

    def within(self, part):
        while self.current is not _END and self.current < part.upper:
            if self.current < part.lower:
                raise ValueError("Candidate outside complete partition order")
            yield self.current
            self.current = self._next()


@contextmanager
def _observations(reader, part):
    if not part.physical_rows:
        yield iter(())
        return
    artifact = reader.write(part, reader.scratch / f"{uuid.uuid4().hex}.parquet")
    with artifact.path.open("rb") as handle:
        info = os.fstat(handle.fileno())
        identity = info.st_dev, info.st_ino
        with closing(reader.read(artifact)) as rows:
            yield rows
        reader._verify_artifact(artifact)
        info = _regular(artifact.path)
        if (info.st_dev, info.st_ino) != identity:
            raise ValueError("Ordered feature scratch identity changed")
        artifact.path.unlink()  # Fully consumed exact-owned FD-pinned output only.


def partition_metric_rows(reader, candidates, config, semantics, max_partition_rows):
    cursor = _Candidates(candidates)
    with reader.verified_batch():
        for part in reader.plan(max_rows=max_partition_rows):
            # Even empty physical intervals can contain dormant candidates.
            with _observations(reader, part) as observations:
                with (
                    closing(cursor.within(part)) as users,
                    closing(
                        merge_feature_metric_rows(
                            users,
                            observations,
                            reader.window.end,
                            config,
                            semantics,
                            temp_root=reader.scratch,
                        )
                    ) as metrics,
                ):
                    yield from metrics
                if next(observations, None) is not None:
                    raise ValueError("Incomplete feature partition consumption")
        if cursor.current is not _END:
            raise ValueError("Candidate outside complete feature partition domain")
