"""Accounted bounded feature sinks. Finalized tokens are NOT a publication.

The caller reserves source scratch separately within this same cache's ledger,
exhausts verified source/prior state and publishes all returned tokens atomically.
Exceptions retain partial file obligations; this module never deletes payloads.
"""

from datetime import timedelta
import stat
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from .feature_writer_budget import FeatureOutput, rebalance
from .derived_cache_resources import _regular
from .episode_state import _timestamp, encode_episode_state
from .feature_order import ObservationOrder
from .feature_records import (
    CHECKPOINT_SCHEMA,
    OBSERVATION_SCHEMA,
    encode_checkpoint,
    encode_observation,
)


class _ShardStream:
    def __init__(self, owner, schema):
        self.owner, self.schema = owner, schema
        self.tokens, self.buffer = [], []
        self.writer = self.handle = None
        self.identity = None
        self.buffer_bytes = self.rows = self.groups = 0
        self._reserve()

    def _reserve(self):
        owner = self.owner
        if owner.artifacts >= owner.max_artifacts:
            raise ValueError("Combined feature artifact limit exceeded")
        relative = f"artifacts/{uuid.uuid4().hex}.parquet"
        self.maximum = owner._reserve_maximum()
        self.token = owner.resources.reserve(relative, self.maximum, "payload")
        owner.artifacts += 1
        self.path = owner.resources.root / relative
        info = self.path.parent.lstat()
        self.parent_identity = (info.st_dev, info.st_ino)
        self.rows = self.groups = 0

    def _check(self):
        self.owner.resources.lease.check()
        parent = self.path.parent.lstat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or (parent.st_dev, parent.st_ino) != self.parent_identity
        ):
            raise ValueError("Feature artifact namespace identity changed")
        if self.identity is not None:
            info = _regular(self.path)
            if (info.st_dev, info.st_ino) != self.identity:
                raise ValueError("Feature writer payload identity changed")

    def _open(self):
        if self.writer is not None:
            return
        self._check()
        self.handle = self.path.open("xb")
        info = _regular(self.path)
        self.identity = (info.st_dev, info.st_ino)
        self.writer = pq.ParquetWriter(
            FeatureOutput(self),
            self.schema,
            compression="zstd",
        )

    def _flush(self):
        if not self.buffer:
            return
        self._check()
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        if table.nbytes > 16 * 1024**2:
            raise ValueError("Feature decoded writer batch byte limit exceeded")
        self._open()
        self.writer.write_table(table)
        self._check()
        self.rows += len(self.buffer)
        self.groups += 1
        self.buffer.clear()
        self.buffer_bytes = 0

    def _close(self, settle):
        if settle:
            self._flush()
            self._open()  # Empty streams still have explicit-schema files.
        try:
            if self.writer is not None:
                writer, self.writer = self.writer, None
                writer.close()
            if self.handle is not None:
                self.handle.flush()
            # Pin the original inode until both path verification and settlement
            # finish. Closing first would permit inode reuse to defeat the check.
            self._check()
            if settle:
                size = _regular(self.path).st_size
                self.owner.resources.settle(self.token)
                self._check()
                self.owner._settled(self.maximum, size)
                self.tokens.append(self.token)
        finally:
            if self.handle is not None:
                self.handle.close()
                self.handle = None
        self.identity = None

    def add(self, row):
        size = sum(
            len(v.encode()) if type(v) is str else len(v) if type(v) is bytes else 8
            for v in row.values()
        )
        if size > 80 * 1024 or size > self.owner.buffer_bytes:
            raise ValueError("Feature encoded row/buffer byte limit exceeded")
        if self.rows + len(self.buffer) == self.owner.shard_rows or self.groups == 4096:
            self._close(True)
            self._reserve()
        if self.buffer and (
            len(self.buffer) == self.owner.buffer_rows
            or self.buffer_bytes + size > self.owner.buffer_bytes
        ):
            self._flush()
            if self.groups == 4096:
                self._close(True)
                self._reserve()
        self.buffer.append(row)
        self.buffer_bytes += size
        if len(self.buffer) == self.owner.buffer_rows:
            self._flush()


class FeatureWriter:
    def __init__(
        self,
        resources,
        *,
        day,
        semantics,
        max_shard_bytes=128 * 1024**2,
        shard_rows=250000,
        buffer_rows=256,
        buffer_bytes=8 * 1024**2,
        max_artifacts=5000,
        max_day_bytes=None,
    ):
        for value, maximum, minimum in (
            (max_shard_bytes, 128 * 1024**2, 1),
            (shard_rows, 250000, 1),
            (buffer_rows, 256, 1),
            (buffer_bytes, 8 * 1024**2, 1),
            (max_artifacts, 5000, 2),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("Invalid feature writer resource bound")
        if max_day_bytes is not None and (
            type(max_day_bytes) is not int
            or not 2 <= max_day_bytes <= 256 * 1024**2
            or max_day_bytes < 2 * max_shard_bytes
        ):
            raise ValueError("Invalid aggregate feature-day byte bound")
        self.day = _timestamp(day)
        encode_episode_state(
            {}, user="0x" + "0" * 40, cutoff=self.day, semantics=semantics
        )
        try:
            self.cutoff = self.day + timedelta(days=1)
        except OverflowError as exc:
            raise ValueError("Feature writer day outside timestamp range") from exc
        resources.lease.check()
        self.resources, self.semantics = resources, semantics
        self.max_shard_bytes, self.shard_rows = max_shard_bytes, shard_rows
        self.buffer_rows, self.buffer_bytes = buffer_rows, buffer_bytes
        self.max_artifacts, self.artifacts = max_artifacts, 0
        self.max_day_bytes = max_day_bytes
        self.day_settled_bytes = self.day_reserved_bytes = 0
        self.order, self.previous_user = ObservationOrder(), None
        self.state, self.streams = "new", []

    def _reserve_maximum(self):
        if self.max_day_bytes is None:
            return self.max_shard_bytes
        remaining = (
            self.max_day_bytes - self.day_settled_bytes - self.day_reserved_bytes
        )
        if remaining <= 0:
            rebalance(self, None, 1)
            remaining = (
                self.max_day_bytes - self.day_settled_bytes - self.day_reserved_bytes
            )
        maximum = min(self.max_shard_bytes, remaining)
        if maximum <= 0:
            raise ValueError("Aggregate feature-day byte limit exceeded")
        self.day_reserved_bytes += maximum
        return maximum

    def _settled(self, maximum, size):
        if self.max_day_bytes is None:
            return
        self.day_reserved_bytes -= maximum
        self.day_settled_bytes += size
        if (
            self.day_reserved_bytes < 0
            or self.day_settled_bytes + self.day_reserved_bytes > self.max_day_bytes
        ):
            raise ValueError("Aggregate feature-day byte limit exceeded")

    def __enter__(self):
        if self.state != "new":
            raise ValueError("Feature writer cannot be reopened")
        self.state = "open"
        try:
            for schema in (OBSERVATION_SCHEMA, CHECKPOINT_SCHEMA):
                self.streams.append(_ShardStream(self, schema))
        except BaseException:
            self.state = "failed"
            raise  # No payload handles opened; reservations remain charged.
        return self

    def _check(self):
        if self.state != "open":
            raise ValueError("Feature writer is not open")
        self.resources.lease.check()

    def add_observation(self, observation):
        self._check()
        try:
            row = encode_observation(observation)
            if not self.day <= row["exchange_time"] < self.cutoff:
                raise ValueError("Feature observation outside day")
            self.order.add(row)
            self.streams[0].add(row)
        except BaseException:
            self.state = "failed"
            raise

    def add_checkpoint(self, payload, *, user):
        self._check()
        try:
            row = encode_checkpoint(
                payload, user=user, cutoff=self.cutoff, semantics=self.semantics
            )
            if self.previous_user is not None and user <= self.previous_user:
                raise ValueError("Duplicate or unordered checkpoint wallet")
            self.streams[1].add(row)
            self.previous_user = user
        except BaseException:
            self.state = "failed"
            raise

    def finish(self):
        self._check()
        try:
            self.order.finish()
            for stream in self.streams:
                stream._close(True)
            self.state = "finished"
            return tuple(token for stream in self.streams for token in stream.tokens)
        except BaseException:
            self.state = "failed"
            raise

    def __exit__(self, kind, *_):
        error = None
        for stream in self.streams:
            try:
                if stream.writer is not None or stream.handle is not None:
                    stream._close(False)
            except BaseException as exc:
                error = error or exc
            finally:
                stream.buffer.clear()
        self.state = "closed"
        if kind is None and error is not None:
            raise error
