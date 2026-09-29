"""Complete qualified event-day derivation under one shared resource lease."""

from contextlib import closing
from dataclasses import asdict
from datetime import timedelta
from itertools import groupby
import os
import uuid

from .contracts import symbol
from .annual_execution_policy import execution_policy
from .derived_cache_resources import _regular
from .derived_day_builder import _release_empty_scratch
from .derived_publication import PublishedArtifacts, _encode
from .episode_state import _timestamp, encode_episode_state
from .feature_publication import FeatureDay, KIND, feature_engine
from .feature_writer import FeatureWriter
from .ordered_wallet_partitions import OrderedWalletPartitions
from .prefix_qualification import _previous, _engine as qualification_engine
from .qualified_day import _day
from .qualified_window import QualifiedWindow
from .query_directory_pin import pin_directory
from .wallet_day_features import derive_wallet_day

SCRATCH_BYTES = 3 * 1024**3


def _prepare(
    resources, report_pin, day, coins, semantics, previous, execution_policy_name=None
):
    resources.lease.check()
    day = _day(day) if type(day) is str else _timestamp(day)
    encode_episode_state({}, user="0x" + "0" * 40, cutoff=day, semantics=semantics)
    try:
        cutoff = day + timedelta(days=1)
    except OverflowError as exc:
        raise ValueError("Feature day outside timestamp range") from exc
    source = QualifiedWindow(report_pin, day, cutoff)
    report = _previous(dict(report_pin), qualification_engine())
    origin = _day(report["source_start"])
    if (
        type(coins) not in (list, tuple)
        or not 1 <= len(coins) <= 50
        or list(coins) != sorted({symbol(c) for c in coins})
        or not set(coins) <= set(source.coins)
    ):
        raise ValueError("Invalid qualified feature market subset")
    engine = feature_engine()
    if previous is None:
        if day != origin:
            raise ValueError("Missing previous feature day")
    else:
        if not isinstance(previous, FeatureDay) or previous.resources is not resources:
            raise ValueError("Previous feature day must use the current resource lease")
        previous.verify()
        old = previous.inputs
        if (
            day == origin
            or previous.cutoff != day
            or old["origin"] != origin.date().isoformat()
            or old["coins"] != list(coins)
            or old["semantics"] != semantics
            or old["engine"] != engine
            or old["source"]["report_sha256"] != source.inputs()["report_sha256"]
            or old.get("execution_policy") != execution_policy_name
        ):
            raise ValueError(
                "Previous feature day is not contiguous or context matched"
            )
    inputs = dict(
        schema=1,
        source=source.inputs(),
        origin=origin.date().isoformat(),
        day=day.date().isoformat(),
        coins=list(coins),
        semantics=semantics,
        previous=previous.publication.key if previous else None,
        engine=engine,
    )
    policy = execution_policy(execution_policy_name)
    if policy is not None:
        inputs.update(
            execution_policy=policy.name,
            max_day_bytes=policy.feature_day_bytes,
            max_day_artifacts=policy.feature_day_artifacts,
        )
    return source, inputs


def _verify(source, previous, inputs):
    source.verify()
    if previous is not None:
        previous.verify()
    if source.inputs() != inputs["source"] or feature_engine() != inputs["engine"]:
        raise ValueError("Feature source/engine changed during derivation")


def _fills(reader, scratch, counts):
    with reader.verified_batch():
        for part in reader.plan():
            if not part.physical_rows:
                continue
            path = scratch / f"{uuid.uuid4().hex}.parquet"
            artifact = reader.write(part, path)
            with path.open("rb") as pin:
                info = os.fstat(pin.fileno())
                identity = (info.st_dev, info.st_ino)
                with closing(reader.read(artifact)) as rows:
                    for fill in rows:
                        counts["source"] += 1
                        yield fill
                reader._verify_artifact(artifact)
                info = _regular(path)
                if (info.st_dev, info.st_ino) != identity:
                    raise ValueError("Feature ordered scratch payload identity changed")
                path.unlink()  # Only this fully consumed, verified, FD-pinned file.


def _states(previous):
    if previous is not None:
        yield from previous.checkpoints()


def _merge(fills, states, writer, inputs, counts):
    groups = groupby(fills, key=lambda fill: fill.user)
    group = next(groups, None)
    state = next(states, None)

    def on_fill(observation):
        writer.add_observation(observation)
        counts["output"] += 1

    while group is not None or state is not None:
        user = min(
            [
                value
                for value in (
                    group[0] if group else None,
                    state["user"] if state else None,
                )
                if value is not None
            ]
        )
        rows = group[1] if group is not None and group[0] == user else iter(())
        checkpoint = (
            state
            if state is not None and state["user"] == user
            else encode_episode_state(
                {}, user=user, cutoff=writer.day, semantics=inputs["semantics"]
            )
        )
        result = derive_wallet_day(
            rows,
            user=user,
            day=writer.day,
            semantics=inputs["semantics"],
            checkpoint=checkpoint,
            on_fill=on_fill,
            on_episode=writer.add_observation,
        )
        if next(rows, None) is not None:
            raise ValueError("Incomplete wallet feature consumption")
        if result.checkpoint["episodes"]:
            writer.add_checkpoint(result.checkpoint, user=user)
        if group is not None and group[0] == user:
            group = next(groups, None)
        if state is not None and state["user"] == user:
            state = next(states, None)
    if counts["source"] != counts["output"]:
        raise ValueError("Feature fill observation/source count mismatch")


def _derive(resources, source, inputs, previous):
    relative = f"scratch/{uuid.uuid4().hex}"
    token = resources.reserve(relative, SCRATCH_BYTES, "scratch")
    scratch = resources.root / relative
    scratch.mkdir()
    info = scratch.stat()
    identity = (info.st_dev, info.st_ino)
    spill = scratch / "spill"
    spill.mkdir()
    info = spill.stat()
    spill_identity = (info.st_dev, info.st_ino)
    with pin_directory(scratch, identity), pin_directory(spill, spill_identity):
        try:
            counts = dict(source=0, output=0)
            reader = OrderedWalletPartitions(source, inputs["coins"], scratch)
            with FeatureWriter(
                resources,
                day=source.start,
                semantics=inputs["semantics"],
                max_day_bytes=inputs.get("max_day_bytes"),
                max_artifacts=inputs.get("max_day_artifacts", 5000),
            ) as writer:
                with (
                    closing(_fills(reader, scratch, counts)) as fills,
                    closing(_states(previous)) as states,
                ):
                    _merge(fills, states, writer, inputs, counts)
                return writer.finish()
        finally:
            resources.lease.check()
            info = spill.lstat()
            if (
                spill.is_symlink()
                or not spill.is_dir()
                or (info.st_dev, info.st_ino) != spill_identity
            ):
                raise ValueError("Feature spill identity changed; retained scratch")
            if not any(spill.iterdir()):
                spill.rmdir()
            _release_empty_scratch(resources, token, scratch, identity)


def build_feature_day(
    resources,
    report_pin,
    day,
    coins,
    semantics,
    previous=None,
    *,
    execution_policy_name=None,
):
    source, inputs = _prepare(
        resources,
        report_pin,
        day,
        coins,
        semantics,
        previous,
        execution_policy_name,
    )
    publications = PublishedArtifacts(resources)
    if publications.lookup(KIND, inputs) is not None:
        _verify(source, previous, inputs)
        result = FeatureDay(resources, inputs)
        _verify(source, previous, inputs)
        return result
    tokens = _derive(resources, source, inputs, previous)
    _verify(source, previous, inputs)
    if policy := execution_policy(execution_policy_name):
        with resources._connect() as db:
            pins = publications._records(db, tokens)
        descriptor = _encode(
            dict(
                schema=1,
                kind=KIND,
                inputs=inputs,
                artifacts=[asdict(pin) for pin in pins],
            )
        )
        if len(descriptor) > policy.feature_day_descriptor_bytes:
            raise ValueError("Feature day publication descriptor limit exceeded")
    publications.publish(KIND, inputs, tokens)
    result = FeatureDay(resources, inputs)
    _verify(source, previous, inputs)
    return result
