from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from arblab.hyperliquid_copy.episodes import PositionEpisode, leader_net_pnl
from arblab.hyperliquid_copy.streaming_wallet_metrics import _episode
from .test_ranking import history

USER = "0x" + "a" * 40
CUTOFF = datetime(2026, 8, 2, tzinfo=timezone.utc)
SEMANTICS = "gross_excludes_fee"


def encode(active, **kwargs):
    from arblab.hyperliquid_copy.episode_state import encode_episode_state

    return encode_episode_state(
        active, **(dict(user=USER, cutoff=CUTOFF, semantics=SEMANTICS) | kwargs)
    )


def decode(payload, **kwargs):
    from arblab.hyperliquid_copy.episode_state import decode_episode_state

    return decode_episode_state(
        payload, **(dict(user=USER, cutoff=CUTOFF, semantics=SEMANTICS) | kwargs)
    )


def state():
    return {
        coin: PositionEpisode(
            USER,
            coin,
            CUTOFF - timedelta(days=400),
            pnl=-0.0,
            fill_count=4,
            taker_count=2,
            fees=-0.1,
            peak_notional=2**53 + 1,
            left_censored=True,
        )
        for coin in ("ETH", "BTC")
    }


def test_exact_json_roundtrip_and_no_aliases():
    active = state()
    payload = encode(active)
    assert [r["coin"] for r in payload["episodes"]] == ["BTC", "ETH"]
    restored = decode(json.loads(json.dumps(payload)))
    assert {coin: asdict(e) for coin, e in restored.items()} == {
        coin: asdict(e) for coin, e in active.items()
    }
    for e in restored.values():
        assert e.pnl.hex() == (-0.0).hex()
        assert type(e.peak_notional) is int and e.peak_notional == 2**53 + 1
    active["BTC"].pnl = 7.0
    restored["ETH"].fill_count = 9
    payload["episodes"][0]["fees"] = 9.0
    assert payload["episodes"][0]["pnl"].hex() == (-0.0).hex()
    assert payload["episodes"][1]["fill_count"] == 4
    assert restored["BTC"].fees == -0.1


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("user", "0x" + "b" * 40),
        ("cutoff", CUTOFF + timedelta(days=1)),
        ("semantics", "net_includes_fee"),
    ],
)
def test_even_empty_state_cannot_be_rebound(empty, field, value):
    payload = encode({} if empty else state())
    assert decode(payload) == ({} if empty else state())
    with pytest.raises(ValueError):
        decode(payload, **{field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"user": USER.upper()},
        {"user": []},
        {"cutoff": CUTOFF.replace(tzinfo=None)},
        {"cutoff": CUTOFF + timedelta(seconds=1)},
        {"cutoff": "2026-08-02"},
        {"cutoff": datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))},
        {"semantics": "unknown"},
        {"semantics": []},
    ],
)
def test_invalid_context_rejected(kwargs):
    with pytest.raises(ValueError):
        encode({}, **kwargs)


@pytest.mark.parametrize(
    "fault", ["closed", "mixed_wallet", "coin_key", "too_many", "wrong_type"]
)
def test_invalid_active_state_rejected(fault):
    active = state()
    if fault == "closed":
        active["BTC"].closed_at = CUTOFF - timedelta(hours=1)
    elif fault == "mixed_wallet":
        active["BTC"].user = "0x" + "b" * 40
    elif fault == "coin_key":
        active["BTC"].coin = "SOL"
    elif fault == "too_many":
        active = {f"C{i}": replace(active["BTC"], coin=f"C{i}") for i in range(51)}
    else:
        active = []
    with pytest.raises(ValueError):
        encode(active)


@pytest.mark.parametrize(
    "fault",
    [
        "header_extra",
        "header_missing",
        "schema",
        "row_extra",
        "row_missing",
        "order",
        "duplicate",
        "rows_type",
        "too_many",
        "coin",
        "opened_naive",
        "opened_at_cutoff",
        "opened_long",
        "opened_overflow",
        "pnl_nan",
        "pnl_bool",
        "fees_string",
        "peak_negative",
        "huge_int",
        "count_bool",
        "count_zero",
        "count_overflow",
        "takers_negative",
        "takers_excess",
        "censor_int",
        "nested",
        "payload_bytes",
    ],
)
def test_malformed_payload_rejected_before_restore(fault):
    data = encode(state())
    row = data["episodes"][0]
    if fault == "header_extra":
        data["extra"] = 1
    elif fault == "header_missing":
        del data["cutoff"]
    elif fault == "schema":
        data["schema"] = "future"
    elif fault == "row_extra":
        row["extra"] = 1
    elif fault == "row_missing":
        del row["pnl"]
    elif fault == "order":
        data["episodes"].reverse()
    elif fault == "duplicate":
        data["episodes"].append(deepcopy(row))
    elif fault == "rows_type":
        data["episodes"] = {}
    elif fault == "too_many":
        data["episodes"] = [row] * 51
    elif fault == "coin":
        row["coin"] = "x" * 81
    elif fault == "opened_naive":
        row["opened_at"] = "2026-08-01T00:00:00"
    elif fault == "opened_at_cutoff":
        row["opened_at"] = CUTOFF.isoformat()
    elif fault == "opened_long":
        row["opened_at"] = "x" * 100000
    elif fault == "opened_overflow":
        row["opened_at"] = "0001-01-01T00:00:00+01:00"
    elif fault == "pnl_nan":
        row["pnl"] = float("nan")
    elif fault == "pnl_bool":
        row["pnl"] = True
    elif fault == "fees_string":
        row["fees"] = "1.0"
    elif fault == "peak_negative":
        row["peak_notional"] = -1
    elif fault == "huge_int":
        row["peak_notional"] = 1 << 4096
    elif fault == "count_bool":
        row["fill_count"] = True
    elif fault == "count_zero":
        row["fill_count"] = 0
    elif fault == "count_overflow":
        row["fill_count"] = 1 << 63
    elif fault == "takers_negative":
        row["taker_count"] = -1
    elif fault == "takers_excess":
        row["taker_count"] = row["fill_count"] + 1
    elif fault == "censor_int":
        row["left_censored"] = 1
    elif fault == "nested":
        row["pnl"] = data
    else:
        data["episodes"] = [
            dict(
                row,
                coin=f"C{i:02d}",
                pnl=1 << 4000,
                fees=1 << 4000,
                peak_notional=1 << 4000,
            )
            for i in range(50)
        ]
    with pytest.raises(ValueError):
        decode(data)


@pytest.mark.parametrize("semantics", ["gross_excludes_fee", "net_includes_fee"])
def test_midnight_resume_matches_uninterrupted_episode_arithmetic(semantics):
    base = history(1)[0]
    specs = [
        ("BTC", -3, 0, 1, 1e16, 0.0),
        ("ETH", -2, 1, -1, 0.0, 0.2),
        ("BTC", -1, 1, 2, 1.0, 0.0),
        ("BTC", 1, 0, 1, 0.0, 0.0),
        ("BTC", 1.5, 1, 0, -1e16, 0.0),
        ("ETH", 2, -1, 1, 2.0, 0.3),
        ("ETH", 3, 1, 1e-10, 3.0, 0.1),
        ("BTC", 4, 0, 1, 0.0, 0.0),
    ]
    rows = [
        replace(
            base,
            user=USER,
            coin=coin,
            exchange_time=CUTOFF + timedelta(hours=hour),
            side="B" if post > start else "A",
            sz=abs(post - start),
            start_position=start,
            post_position=post,
            px=2**53 + 1,
            closed_pnl=pnl,
            fee=fee,
            tid=i,
            oid=i,
            source_line=i,
            event_index=i,
            event_id=str(i),
        )
        for i, (coin, hour, start, post, pnl, fee) in enumerate(specs)
    ]

    class Spool:
        def __init__(self):
            self.rows = []

        def add(self, *values):
            self.rows.append(values)

    def replay(fills, active, spool):
        for fill in fills:
            _episode(active, fill, leader_net_pnl(fill, semantics), semantics, spool)

    uninterrupted, expected = {}, Spool()
    replay(rows, uninterrupted, expected)
    partial, actual = {}, Spool()
    replay([f for f in rows if f.exchange_time < CUTOFF], partial, actual)
    payload = encode(partial, semantics=semantics)
    restored = decode(json.loads(json.dumps(payload)), semantics=semantics)
    replay([f for f in rows if f.exchange_time >= CUTOFF], restored, actual)
    assert actual.rows == expected.rows
    assert expected.rows[0][1] == 0.0  # Incremental episode cancellation, not sum=1.
    final_cutoff = CUTOFF + timedelta(days=1)
    assert encode(restored, cutoff=final_cutoff, semantics=semantics) == encode(
        uninterrupted, cutoff=final_cutoff, semantics=semantics
    )


def test_empty_checkpoint_resumes_with_new_episode():
    class Spool:
        def __init__(self):
            self.rows = []

        def add(self, *values):
            self.rows.append(values)

    opened = replace(history(1)[0], user=USER, exchange_time=CUTOFF)
    closed = replace(
        opened,
        exchange_time=CUTOFF + timedelta(hours=1),
        side="A",
        start_position=1.0,
        post_position=0.0,
        closed_pnl=2.0,
        event_id="close",
        tid=2,
        oid=2,
    )
    restored = decode(json.loads(json.dumps(encode({}))))
    actual, reference, uninterrupted = Spool(), Spool(), {}
    for fill in (opened, closed):
        net = leader_net_pnl(fill, SEMANTICS)
        _episode(restored, fill, net, SEMANTICS, actual)
        _episode(uninterrupted, fill, net, SEMANTICS, reference)
    assert restored == uninterrupted == {}
    assert actual.rows == reference.rows and len(actual.rows) == 1
