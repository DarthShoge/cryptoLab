"""Explicit retained feature intervals; no history advancement or retirement."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from .annual_execution_policy import execution_policy
from .checkpoint_io import _CappedOutput
from .derived_cache_resources import _regular
from .derived_publication import PublishedArtifacts, _encode, _inputs
from .download import file_hash
from .feature_publication import FeatureDay, feature_engine
from .feature_window import FeatureWindow
from .qualified_window import _engine as qualification_engine
from .query_directory_pin import pin_directory

KIND = "qualified_feature_anchor"
MAX_BYTES = 64 * 1024**2
SCHEMA = pa.schema([pa.field("day", pa.string()), pa.field("inputs", pa.binary())])


def _digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _engine():
    return _digest(
        dict(
            code=file_hash(Path(__file__)),
            window=file_hash(Path(__file__).with_name("feature_window.py")),
            features=feature_engine(),
            qualification=qualification_engine(),
        )
    )


def _window(resources, pin, days, coins, semantics):
    if type(days) not in (tuple, list) or not 1 <= len(days) <= 732:
        raise ValueError("Expected bounded retained feature interval")
    if any(not isinstance(d, FeatureDay) for d in days):
        raise ValueError("Expected published feature days")
    result = FeatureWindow(
        resources, pin, days, days[0].day, days[-1].cutoff, coins, semantics
    )
    if list(result.coins) != result.days[0].inputs["coins"]:
        raise ValueError("Anchor requires full feature scope")
    return result


def _binding(window):
    policy_names = {day.inputs.get("execution_policy") for day in window.days}
    if len(policy_names) != 1:
        raise ValueError("Anchor feature-day policy mismatch")
    policy = execution_policy(next(iter(policy_names)))
    value = dict(
        schema=1,
        report_sha256=window.source.inputs()["report_sha256"],
        origin=window.days[0].inputs["origin"],
        first=window.start.isoformat(),
        cutoff=window.end.isoformat(),
        coins=list(window.coins),
        semantics=window.semantics,
        day_count=len(window.days),
        window_sha256=_digest(window.inputs()),
        engine=_engine(),
    )
    if policy is not None:
        value.update(
            execution_policy=policy.name,
            max_anchor_bytes=policy.feature_anchor_bytes,
        )
    return _inputs(
        KIND,
        value,
    )


def _write(resources, window, maximum=MAX_BYTES):
    relative = f"artifacts/{uuid.uuid4().hex}.parquet"
    token = resources.reserve(relative, maximum, "payload")
    path = resources.root / relative
    info = path.parent.lstat()
    with (
        pin_directory(path.parent, (info.st_dev, info.st_ino)),
        path.open("xb") as handle,
    ):
        info = os.fstat(handle.fileno())
        identity = info.st_dev, info.st_ino

        def check():
            resources.lease.check()
            info = _regular(path)
            if (info.st_dev, info.st_ino) != identity:
                raise ValueError("Anchor output identity changed")

        with pq.ParquetWriter(
            _CappedOutput(handle, maximum), SCHEMA, compression="zstd"
        ) as writer:
            for offset in range(0, len(window.days), 8):
                rows = [
                    dict(
                        day=d.day.isoformat(),
                        inputs=_encode(_inputs("qualified_features_day", d.inputs)),
                    )
                    for d in window.days[offset : offset + 8]
                ]
                batch = pa.Table.from_pylist(rows, schema=SCHEMA)
                if batch.nbytes > 1024**2:
                    raise ValueError("Anchor writer batch limit exceeded")
                check()
                writer.write_table(batch)
                check()
        handle.flush()
        check()
        resources.settle(token)
        check()
    return token


def _read(resources, publication):
    if len(publication.artifacts) != 1:
        raise ValueError("Expected one anchor artifact")
    pin = publication.artifacts[0]
    if pin.bytes > MAX_BYTES:
        raise ValueError("Anchor artifact byte limit exceeded")
    result = []
    with pq.ParquetFile(resources.root / pin.path) as reader:
        if (
            reader.schema_arrow != SCHEMA
            or not 1 <= reader.metadata.num_rows <= 733
            or reader.metadata.num_row_groups > 92
        ):
            raise ValueError("Invalid anchor schema/metadata")
        # Check before Arrow decompresses binary cells. Limiting both encoded
        # row-group size and row count also bounds dictionary expansion.
        for index in range(reader.metadata.num_row_groups):
            group = reader.metadata.row_group(index)
            if not 1 <= group.num_rows <= 8 or not 0 < group.total_byte_size <= 1024**2:
                raise ValueError("Anchor row group resource limit exceeded")
        for batch in reader.iter_batches(batch_size=8, use_threads=False):
            if batch.nbytes > 1024**2:
                raise ValueError("Anchor reader batch limit exceeded")
            for row in batch.to_pylist():
                raw = row["inputs"]
                if type(raw) is not bytes or not 0 < len(raw) <= 64 * 1024:
                    raise ValueError("Invalid anchor descriptor size")
                try:
                    inputs = _inputs("qualified_features_day", json.loads(raw))
                except (
                    TypeError,
                    RecursionError,
                    UnicodeError,
                    json.JSONDecodeError,
                ) as exc:
                    raise ValueError("Invalid anchor descriptor") from exc
                if _encode(inputs) != raw:
                    raise ValueError("Noncanonical anchor descriptor")
                day = FeatureDay(resources, inputs)
                if row["day"] != day.day.isoformat():
                    raise ValueError("Anchor day mismatch")
                result.append(day)
        if len(result) != reader.metadata.num_rows:
            raise ValueError("Incomplete anchor consumption")
    return tuple(result)


@dataclass(frozen=True, init=False)
class FeatureAnchor:
    resources: object
    publication: object
    days: tuple
    first: object
    cutoff: object
    _encoded: bytes
    _window: object
    _caller_pin: object
    _pin_encoded: bytes

    def __init__(self, resources, report_pin, inputs):
        resources.lease.check()
        data = _inputs(KIND, inputs)
        policy = execution_policy(data.get("execution_policy"))
        maximum = MAX_BYTES if policy is None else policy.feature_anchor_bytes
        expected_keys = {
            "schema",
            "report_sha256",
            "origin",
            "first",
            "cutoff",
            "coins",
            "semantics",
            "day_count",
            "window_sha256",
            "engine",
        }
        if policy is not None:
            expected_keys |= {"execution_policy", "max_anchor_bytes"}
        if (
            set(data) != expected_keys
            or data.get("max_anchor_bytes", maximum) != maximum
        ):
            raise ValueError("Invalid feature anchor policy context")
        pin_encoded = _encode(report_pin)
        if data.get("engine") != _engine():
            raise ValueError("Anchor engine changed")
        publication = PublishedArtifacts(resources).lookup(KIND, data)
        if publication is None:
            raise ValueError("Feature anchor is not published")
        if sum(pin.bytes for pin in publication.artifacts) > maximum:
            raise ValueError("Feature anchor byte limit exceeded")
        days = _read(resources, publication)
        window = _window(
            resources, report_pin, days, data.get("coins"), data.get("semantics")
        )
        for name, value in dict(
            resources=resources,
            publication=publication,
            days=days,
            first=window.start,
            cutoff=window.end,
            _encoded=_encode(data),
            _window=window,
            _caller_pin=report_pin,
            _pin_encoded=pin_encoded,
        ).items():
            object.__setattr__(self, name, value)
        self.verify()

    @property
    def inputs(self):
        return json.loads(self._encoded)

    def _stats(self):
        info = _regular(self.resources.root / self.publication.artifacts[0].path)
        return (
            self._window._stats(),
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def verify(self):
        self.resources.lease.check()
        before = self._stats()
        if _encode(self._caller_pin) != self._pin_encoded:
            raise ValueError("Anchor source context changed")
        self._window.verify()
        if (
            PublishedArtifacts(self.resources).lookup(KIND, self.inputs)
            != self.publication
        ):
            raise ValueError("Anchor publication changed")
        if (
            _encode(_binding(self._window)) != self._encoded
            or self._stats() != before
            or _encode(self._caller_pin) != self._pin_encoded
        ):
            raise ValueError("Anchor context changed")
        self.resources.lease.check()


def publish_feature_anchor(resources, report_pin, days, coins, semantics):
    if type(days) not in (tuple, list) or type(coins) not in (tuple, list):
        raise ValueError("Expected bounded feature day and market sequences")
    pin_encoded, original_days, original_coins = (
        _encode(report_pin),
        tuple(days),
        tuple(coins),
    )
    window = _window(resources, report_pin, days, coins, semantics)
    inputs = _binding(window)
    frozen = _encode(inputs)

    def verify():
        before = window._stats()
        if (
            _encode(report_pin) != pin_encoded
            or tuple(days) != original_days
            or tuple(coins) != original_coins
        ):
            raise ValueError("Anchor caller context changed")
        window.verify()
        if (
            _encode(_binding(window)) != frozen
            or _encode(report_pin) != pin_encoded
            or tuple(days) != original_days
            or tuple(coins) != original_coins
            or window._stats() != before
        ):
            raise ValueError("Anchor context changed")
        resources.lease.check()

    verify()
    publications = PublishedArtifacts(resources)
    if publications.lookup(KIND, inputs) is None:
        maximum = inputs.get("max_anchor_bytes", MAX_BYTES)
        token = _write(resources, window, maximum)
        verify()
        policy = execution_policy(inputs.get("execution_policy"))
        if policy is not None:
            with resources._connect() as db:
                pins = publications._records(db, [token])
            descriptor = _encode(
                dict(
                    schema=1,
                    kind=KIND,
                    inputs=inputs,
                    artifacts=[asdict(pin) for pin in pins],
                )
            )
            if len(descriptor) > policy.feature_anchor_descriptor_bytes:
                raise ValueError("Feature anchor publication descriptor limit exceeded")
        publications.publish(KIND, inputs, [token])
    result = FeatureAnchor(resources, report_pin, inputs)
    verify()
    return result
