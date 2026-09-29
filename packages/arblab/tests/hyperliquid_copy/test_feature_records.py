from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.wallet_day_features import (
    FillObservation,
    EpisodeObservation,
)
from arblab.hyperliquid_copy.episode_state import encode_episode_state
from .test_wallet_day_features import BASE, DAY, SEMANTICS, run


def codec():
    from arblab.hyperliquid_copy import feature_records

    return feature_records


def example(kind="fill"):
    key = (DAY, -1, "native/hour", 1, 2, "event")
    if kind == "episode":
        return EpisodeObservation(BASE.user, "BTC", key, -0.0, 1.5, 3)
    return FillObservation(
        BASE.user,
        "BTC",
        key,
        2**53 + 1,
        1,
        "B",
        -1,
        0,
        -0.0,
        True,
        -0.0,
        2**53 + 1,
        2**53 + 1,
    )


@pytest.mark.parametrize("kind", ["fill", "episode"])
def test_typed_parquet_roundtrip(tmp_path, kind):
    module = codec()
    original = example(kind)
    row = module.encode_observation(original)
    path = tmp_path / "observations.parquet"
    pq.write_table(pa.Table.from_pylist([row], schema=module.OBSERVATION_SCHEMA), path)
    actual = module.decode_observation(pq.read_table(path).to_pylist()[0])
    assert asdict(actual) == asdict(original)
    for name, expected in asdict(original).items():
        value = getattr(actual, name)
        assert type(value) is type(expected)
        if isinstance(value, float):
            assert value.hex() == expected.hex()


def test_actual_transducer_observations_roundtrip(tmp_path):
    module, observations = codec(), []
    rows = [
        replace(BASE, exchange_time=DAY),
        replace(
            BASE,
            exchange_time=DAY + timedelta(minutes=1),
            start_position=1,
            post_position=0,
            side="A",
            closed_pnl=-0.0,
        ),
    ]
    run(rows, on_fill=observations.append, on_episode=observations.append)
    assert [type(o).__name__ for o in observations] == [
        "FillObservation",
        "EpisodeObservation",
        "FillObservation",
    ]
    path = tmp_path / "observations.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [module.encode_observation(o) for o in observations],
            schema=module.OBSERVATION_SCHEMA,
        ),
        path,
    )
    assert [
        module.decode_observation(row) for row in pq.read_table(path).to_pylist()
    ] == observations


def test_checkpoint_payload_is_explicit_and_context_bound(tmp_path):
    module = codec()
    args = dict(user=BASE.user, cutoff=DAY, semantics=SEMANTICS)
    payload = encode_episode_state({}, **args)
    row = module.encode_checkpoint(payload, **args)
    path = tmp_path / "state.parquet"
    pq.write_table(pa.Table.from_pylist([row], schema=module.CHECKPOINT_SCHEMA), path)
    stored = pq.read_table(path).to_pylist()[0]
    assert module.decode_checkpoint(stored, **args) == payload
    with pytest.raises(ValueError):
        module.decode_checkpoint(stored, **(args | {"cutoff": DAY + timedelta(days=1)}))
    with pytest.raises(ValueError):
        module.decode_checkpoint(dict(stored, user="0x" + "f" * 40), **args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("px", True),
        ("net_pnl", float("nan")),
        ("gross_volume", 1 << 4096),
        ("crossed", 1),
        ("side", "X"),
        ("coin", "bad!"),
        ("user", BASE.user.upper()),
        ("px", []),
        ("order_key", (DAY, 0, "key", 1, 2)),
        ("order_key", (DAY, 2**63, "key", 1, 2, "event")),
        ("order_key", (DAY, 0, "x" * 16385, 1, 2, "event")),
        ("order_key", (DAY.replace(tzinfo=None), 0, "key", 1, 2, "event")),
        (
            "order_key",
            (
                datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
                0,
                "key",
                1,
                2,
                "event",
            ),
        ),
    ],
    ids=[f"invalid_{i}" for i in range(13)],
)
def test_invalid_observation_rejected(field, value):
    with pytest.raises(ValueError):
        codec().encode_observation(replace(example(), **{field: value}))


@pytest.mark.parametrize(
    "payload",
    [
        b"[]",
        b"{",
        b"x" * 65537,
        b'{"px":NaN}',
        b'{"px":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
    ],
    ids=["array", "syntax", "oversized", "nan", "nested"],
)
def test_malformed_payload_rejected(payload):
    module = codec()
    row = module.encode_observation(example())
    with pytest.raises(ValueError):
        module.decode_observation(dict(row, payload=payload))


@pytest.mark.parametrize(
    "field,value", [("minutes", -1), ("fragments", 0), ("fragments", 1.0)]
)
def test_invalid_episode_counters_rejected(field, value):
    with pytest.raises(ValueError):
        codec().encode_observation(replace(example("episode"), **{field: value}))
