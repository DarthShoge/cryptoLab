"""Full raw/feature candidate metrics in one pre-reserved invocation."""

from contextlib import closing, contextmanager
import os
import stat

from .candidate_metric_producer import (
    _candidate_rows,
    _ordered_fills,
    merge_metric_rows,
)
from .feature_metric_producer import _staged_metric_rows as staged_metric_rows
from .ordered_wallet_partitions import OrderedWalletPartitions
from .derived_publication import _encode
from .ranking_staging_artifact import _identity
from .ranking_staging_manifest import MAX_BYTES, encode_manifest
from .ranking_staging_owner import RankingStagingOwner
from .ranking_staging_rows import write_pending_metrics
from .ranking_staging_sources import PreparedStagingSource


def _owned(owner, artifact):
    """No hashing callbacks after final source verification."""
    owner._resources.lease.check()
    if (
        _encode(owner._context) != owner._frozen
        or encode_manifest(
            dict(schema=1, context=owner._context, allocations=owner._allocations)
        )
        != owner._encoded
        or os.pread(owner._fds["manifest"], MAX_BYTES + 1, 0) != owner._encoded
    ):
        raise ValueError("Staging production manifest/context changed")
    for path, fd in owner._directories.items():
        actual, held = path.lstat(), os.fstat(fd)
        if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
            held.st_dev,
            held.st_ino,
        ):
            raise ValueError("Staging production namespace changed")
    for role, fd in owner._fds.items():
        owner._check_fd(role, fd)
    if _identity(artifact.path.lstat()) != artifact.identity:
        raise ValueError("Staged metric artifact changed")


def _write_complete(owner, rows, config):
    complete = False

    def tracked():
        nonlocal complete
        yield from rows
        complete = True

    with closing(tracked()) as stream:
        artifact = write_pending_metrics(owner, stream, list(config.metric_weights))
    if not complete:
        raise ValueError("Incomplete staged metric stream")
    return artifact


@contextmanager
def _complete_rows(rows):
    complete = False

    def tracked():
        nonlocal complete
        yield from rows
        complete = True

    with closing(rows), closing(tracked()) as stream:
        yield stream
        if not complete:
            raise ValueError("Incomplete staged input consumption")


def produce_staged_metrics(owner, prepared):
    if (
        type(owner) is not RankingStagingOwner
        or type(prepared) is not PreparedStagingSource
        or owner._resources is not prepared.resources
        or owner._context.get("producer") != prepared.key
    ):
        raise ValueError("Expected matching prepared staging source/owner")
    owner.verify()
    prepared.verify()
    scratch = owner.path("scratch")
    spill = scratch / "spill"
    spill.mkdir()
    fd = os.open(spill, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if prepared.route == "raw":
            reader = OrderedWalletPartitions(
                prepared.window, prepared.config.coins, scratch, prepared.scope
            )
            with (
                _complete_rows(
                    _candidate_rows(prepared.resources, prepared.candidate)
                ) as candidates,
                _complete_rows(
                    _ordered_fills(reader, scratch, prepared.max_partition_rows)
                ) as fills,
                closing(
                    merge_metric_rows(
                        candidates,
                        fills,
                        prepared.decision,
                        prepared.config,
                        prepared.semantics,
                        temp_root=scratch,
                    )
                ) as rows,
            ):
                artifact = _write_complete(owner, rows, prepared.config)
        elif prepared.route == "features":
            with (
                _complete_rows(
                    _candidate_rows(prepared.resources, prepared.candidate)
                ) as candidates,
                closing(
                    staged_metric_rows(
                        prepared.window,
                        candidates,
                        prepared.config,
                        scratch,
                    )
                ) as rows,
            ):
                artifact = _write_complete(owner, rows, prepared.config)
        else:
            raise ValueError("Unknown prepared staging route")
        owner.verify()
        artifact.verify()
        prepared.verify()
        _owned(owner, artifact)
        actual, held = spill.lstat(), os.fstat(fd)
        if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
            held.st_dev,
            held.st_ino,
        ):
            raise ValueError("Staged metric spill identity changed")
        with os.scandir(fd) as entries:
            if next(entries, None) is not None:
                raise ValueError("Staged metric spill is not empty")
        os.rmdir("spill", dir_fd=owner._fds["scratch"])
        os.fsync(owner._fds["scratch"])
        owner.verify()
        artifact.verify()
        prepared.verify()
        _owned(owner, artifact)
        return artifact
    finally:
        os.close(fd)  # Failed files/allocations are never refunded or adopted.
