"""Verified immutable feature-file opening, not independent source qualification.

The qualified builder supplies expected inputs and proves contiguous consumption.
Readers recheck catalog/payload identity and enforce day, scope, order and pairing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .candidate_metric_producer import _engine as metric_engine
from .annual_execution_policy import execution_policy
from .contracts import symbol
from .derived_publication import Publication, PublishedArtifacts, _inputs, _encode
from .download import file_hash
from .feature_writer_compatibility import compatible_code
from .episode_state import encode_episode_state
from .feature_order import ObservationOrder
from .feature_records import (
    OBSERVATION_SCHEMA,
    CHECKPOINT_SCHEMA,
    decode_observation,
    decode_checkpoint,
)
from .qualified_day import _day, _digest

KIND = "qualified_features_day"


def feature_engine():
    root = Path(__file__).parent
    return dict(
        metric=metric_engine(),
        pyarrow=pa.__version__,
        code=compatible_code(
            {
                name: file_hash(root / name)
                for name in (
                    "feature_publication.py",
                    "feature_day_builder.py",
                    "query_directory_pin.py",
                    "feature_records.py",
                    "feature_order.py",
                    "feature_writer.py",
                    "wallet_day_features.py",
                    "episode_state.py",
                    "qualified_day.py",
                    "qualified_window.py",
                )
            }
        ),
    )


def _context(inputs):
    data = _inputs(KIND, inputs)
    base = {
        "schema",
        "source",
        "origin",
        "day",
        "coins",
        "semantics",
        "previous",
        "engine",
    }
    bounded = base | {
        "execution_policy",
        "max_day_bytes",
        "max_day_artifacts",
    }
    if (
        set(data) not in (base, bounded)
        or type(data["schema"]) is not int
        or data["schema"] != 1
    ):
        raise ValueError("Invalid feature publication context")
    policy = execution_policy(data.get("execution_policy"))
    if policy is None:
        if "max_day_bytes" in data:
            raise ValueError("Unbound feature-day byte limit")
    elif (
        data.get("max_day_bytes") != policy.feature_day_bytes
        or data.get("max_day_artifacts") != policy.feature_day_artifacts
    ):
        raise ValueError("Feature-day policy limit mismatch")
    day, origin = _day(data["day"]), _day(data["origin"])
    try:
        cutoff = day + timedelta(days=1)
    except OverflowError as exc:
        raise ValueError("Feature day outside timestamp range") from exc
    if day < origin or (data["previous"] is None) != (day == origin):
        raise ValueError("Feature previous-day/origin context mismatch")
    if data["previous"] is not None:
        _digest(data["previous"])
    coins = data["coins"]
    if (
        type(coins) is not list
        or not 1 <= len(coins) <= 50
        or coins != sorted({symbol(c) for c in coins})
    ):
        raise ValueError("Invalid feature market scope")
    source = data["source"]
    if (
        type(source) is not dict
        or source.get("start") != day.isoformat()
        or source.get("end") != cutoff.isoformat()
        or type(source.get("coins")) is not list
        or not set(coins) <= set(source["coins"])
    ):
        raise ValueError("Feature source interval/scope mismatch")
    _digest(source.get("report_sha256"))
    encode_episode_state(
        {}, user="0x" + "0" * 40, cutoff=cutoff, semantics=data["semantics"]
    )
    if data["engine"] != feature_engine():
        raise ValueError("Feature engine changed")
    return data, day, cutoff


@dataclass(frozen=True, init=False)
class FeatureDay:
    resources: object
    day: datetime
    cutoff: datetime
    _encoded: bytes
    publication: Publication
    _observations: tuple
    _checkpoints: tuple

    def __init__(self, resources, inputs):
        data, day, cutoff = _context(inputs)
        publication = PublishedArtifacts(resources).lookup(KIND, data)
        if publication is None:
            raise ValueError("Feature day is not published")
        observations, checkpoints = [], []
        for pin in publication.artifacts:
            path = resources.root / pin.path
            with pq.ParquetFile(path) as reader:
                if (
                    pin.bytes > 128 * 1024**2
                    or reader.metadata.num_rows > 250000
                    or reader.metadata.num_row_groups > 4096
                ):
                    raise ValueError("Feature shard metadata limit exceeded")
                schema = reader.schema_arrow
                if schema == OBSERVATION_SCHEMA and not checkpoints:
                    observations.append(pin)
                elif schema == CHECKPOINT_SCHEMA:
                    checkpoints.append(pin)
                else:
                    raise ValueError("Feature shard schema/order mismatch")
        maximum = data.get("max_day_bytes")
        if (
            maximum is not None
            and sum(pin.bytes for pin in publication.artifacts) > maximum
        ):
            raise ValueError("Aggregate feature-day byte limit exceeded")
        if not observations or not checkpoints:
            raise ValueError("Both feature streams require explicit shards")
        for name, value in dict(
            resources=resources,
            day=day,
            cutoff=cutoff,
            _encoded=_encode(data),
            publication=publication,
            _observations=tuple(observations),
            _checkpoints=tuple(checkpoints),
        ).items():
            object.__setattr__(self, name, value)
        self.verify()

    @property
    def inputs(self):
        return json.loads(self._encoded)

    def verify(self):
        inputs, day, cutoff = _context(self.inputs)
        if (
            day != self.day
            or cutoff != self.cutoff
            or PublishedArtifacts(self.resources).lookup(KIND, inputs)
            != self.publication
        ):
            raise ValueError("Feature publication identity changed")
        # Payload verification can be lengthy. Check code/context again afterward
        # so a change during the final lookup cannot validate a stale derivation.
        _context(self.inputs)

    def _rows(self, pins, schema):
        for pin in pins:
            with pq.ParquetFile(self.resources.root / pin.path) as reader:
                if (
                    reader.schema_arrow != schema
                    or reader.metadata.num_rows > 250000
                    or reader.metadata.num_row_groups > 4096
                ):
                    raise ValueError("Feature shard schema/metadata changed")
                count = 0
                for batch in reader.iter_batches(batch_size=256, use_threads=False):
                    if batch.nbytes > 32 * 1024**2:
                        raise ValueError(
                            "Feature decoded reader batch byte limit exceeded"
                        )
                    for row in batch.to_pylist():
                        count += 1
                        yield row
                if count != reader.metadata.num_rows:
                    raise ValueError("Incomplete feature shard consumption")

    def observations(self):
        self.verify()
        order = ObservationOrder()
        coins = self.inputs["coins"]
        try:
            for row in self._rows(self._observations, OBSERVATION_SCHEMA):
                observation = decode_observation(row)
                if (
                    observation.coin not in coins
                    or not self.day <= observation.order_key[0] < self.cutoff
                ):
                    raise ValueError("Feature observation outside day/scope")
                order.add(row)
                yield observation
            order.finish()
        finally:
            self.verify()

    def checkpoints(self):
        self.verify()
        inputs, previous = self.inputs, None
        try:
            for row in self._rows(self._checkpoints, CHECKPOINT_SCHEMA):
                payload = decode_checkpoint(
                    row,
                    user=row["user"],
                    cutoff=self.cutoff,
                    semantics=inputs["semantics"],
                )
                if (
                    previous is not None
                    and row["user"] <= previous
                    or any(
                        e["coin"] not in inputs["coins"] for e in payload["episodes"]
                    )
                ):
                    raise ValueError("Checkpoint wallet order/market scope mismatch")
                previous = row["user"]
                yield payload
        finally:
            self.verify()
