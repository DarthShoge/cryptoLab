"""Transfer unused pending reservations between the two feature-day streams.

The day and cache caps never increase. Actual written bytes stay charged, and
each write (including a footer) consults the current reservation. A transfer is
atomic in the existing ledger; no settled artifact or publication is modified.
"""

import shutil

from .checkpoint_io import _CappedOutput
from .derived_cache_resources import FREE_RESERVE, METADATA_BYTES


def rebalance(owner, receiver, required):
    if owner.max_day_bytes is None:
        return
    current = receiver.maximum if receiver is not None else 0
    if required > owner.max_shard_bytes:
        return
    donors = []
    for stream in owner.streams:
        if stream is receiver or stream.token in stream.tokens:
            continue
        stream._check()
        used = max(1, stream.handle.tell() if stream.handle is not None else 0)
        available = stream.maximum - used
        if available > 0:
            donors.append((stream, available))
    free = (
        0
        if receiver is None
        else owner.max_day_bytes - owner.day_settled_bytes - owner.day_reserved_bytes
    )
    available = free + sum(size for _, size in donors)
    if available < required - current:
        return
    wanted = min(owner.max_shard_bytes, max(required, current + 8 * 1024**2))
    amount = min(available, wanted - current)
    if amount <= 0:
        return
    extra = min(free, amount)
    changes, remaining = [], amount - extra
    for stream, available in donors:
        take = min(available, remaining)
        if take:
            changes.append((stream, stream.maximum - take))
            remaining -= take
    if receiver is not None:
        receiver._check()
        changes.append((receiver, current + amount))
    with owner.resources._connect() as db, db:
        db.execute("BEGIN IMMEDIATE")
        if extra:
            totals, unwritten = owner.resources._audit(db)
            if totals["total_bytes"] + extra > owner.resources.limit:
                raise ValueError("Cache budget exceeded")
            if (
                shutil.disk_usage(owner.resources.root).free
                < unwritten + extra + METADATA_BYTES + FREE_RESERVE
            ):
                raise ValueError("Insufficient cache free space")
        for stream, maximum in changes:
            row = owner.resources._pending(db, stream.token)
            if row != (
                stream.path.relative_to(owner.resources.root).as_posix(),
                stream.maximum,
                "payload",
                "pending",
            ):
                raise ValueError("Feature reservation changed during rebalance")
            stream._check()
            db.execute(
                "UPDATE allocations SET maximum=? WHERE token=?",
                (maximum, stream.token),
            )
    for stream, maximum in changes:
        stream.maximum = maximum
    if receiver is None:
        owner.day_reserved_bytes -= amount
    else:
        owner.day_reserved_bytes += extra


class FeatureOutput(_CappedOutput):
    def __init__(self, stream):
        super().__init__(stream.handle, stream.maximum)
        self.stream = stream

    def write(self, value):
        stream = self.stream
        required = self.tell() + len(value)
        if required > stream.maximum:
            rebalance(stream.owner, stream, required)
        # Another stream may have borrowed from us since our preceding write.
        self.limit = stream.maximum
        return super().write(value)
