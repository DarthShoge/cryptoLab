from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from copy import deepcopy

import pytest

from arblab.hyperliquid_copy.episode_state import encode_episode_state
from arblab.hyperliquid_copy.episodes import closing_size, leader_net_pnl
from arblab.hyperliquid_copy.episodes import PositionEpisode
from arblab.hyperliquid_copy.streaming_wallet_metrics import _episode
from .test_ranking import history

BASE = history(1)[0]
DAY = BASE.exchange_time.replace(hour=0, minute=0, second=0, microsecond=0)
SEMANTICS = "gross_excludes_fee"


def run(rows, **kwargs):
    from arblab.hyperliquid_copy.wallet_day_features import derive_wallet_day

    args = dict(
        user=BASE.user,
        day=DAY,
        semantics=SEMANTICS,
        checkpoint=encode_episode_state(
            {}, user=BASE.user, cutoff=DAY, semantics=SEMANTICS
        ),
        on_fill=lambda row: None,
        on_episode=lambda row: None,
    )
    args.update(kwargs)
    return derive_wallet_day(rows, **args)


def test_empty_day_advances_explicit_checkpoint_without_mutation():
    checkpoint = encode_episode_state(
        {}, user=BASE.user, cutoff=DAY, semantics=SEMANTICS
    )
    original = deepcopy(checkpoint)
    result = run(iter(()), checkpoint=checkpoint)
    assert result.fill_count == result.episode_count == 0
    assert result.checkpoint == encode_episode_state(
        {}, user=BASE.user, cutoff=DAY + timedelta(days=1), semantics=SEMANTICS
    )
    assert checkpoint == original


@pytest.mark.parametrize("semantics", [SEMANTICS, "net_includes_fee"])
def test_resumed_days_match_uninterrupted_reference(semantics):
    shapes = [(0, 1), (1, -1), (-1, 0), (0, 1), (0, 2), (2, 0)]
    rows = [
        replace(
            BASE,
            coin="BTC" if i % 2 else "ETH",
            exchange_time=DAY + timedelta(hours=i // 2),
            start_position=start,
            post_position=post,
            sz=abs(post - start),
            side="B" if post > start else "A",
            px=2**53 + 1,
            fee=0.01,
            closed_pnl=[1e16, 1.0, -1e16][i % 3],
            source_line=i,
            event_id=str(i),
        )
        for i in range(144)
        for start, post in [shapes[(i // 2) % 6]]
    ]
    expected, active = [], {}

    class Sink:
        def add(self, kind, pnl, minutes, fragments):
            assert kind == "episode"
            expected.append(
                (self.fill.coin, self.fill.order_key, pnl, minutes, fragments)
            )

    sink = Sink()
    for fill in rows:
        sink.fill = fill
        _episode(active, fill, leader_net_pnl(fill, semantics), semantics, sink)
    checkpoint = encode_episode_state(
        {}, user=BASE.user, cutoff=DAY, semantics=semantics
    )
    numeric, episodes = [], []
    for index in range(3):
        day = DAY + timedelta(days=index)
        result = run(
            (f for f in rows if day <= f.exchange_time < day + timedelta(days=1)),
            day=day,
            semantics=semantics,
            checkpoint=checkpoint,
            on_fill=numeric.append,
            on_episode=episodes.append,
        )
        checkpoint = result.checkpoint
    assert checkpoint == encode_episode_state(
        active, user=BASE.user, cutoff=DAY + timedelta(days=3), semantics=semantics
    )
    assert [
        (e.coin, e.order_key, e.pnl, e.minutes, e.fragments) for e in episodes
    ] == expected
    assert len(numeric) == len(rows)
    for observation, fill in zip(numeric, rows):
        assert observation.order_key == fill.order_key
        assert observation.net_pnl == leader_net_pnl(fill, semantics)
        assert observation.closed_notional == closing_size(fill) * fill.px
        assert observation.gross_volume == fill.sz * fill.px
        assert type(observation.px) is int and observation.px == 2**53 + 1
        with pytest.raises(FrozenInstanceError):
            observation.net_pnl = 0


@pytest.mark.parametrize(
    "case", ["wallet", "early", "late", "duplicate", "reverse", "fee", "checkpoint"]
)
def test_rejects_invalid_stream_or_context(case):
    first = replace(BASE, exchange_time=DAY)
    rows, kwargs = [first], {}
    if case == "wallet":
        rows = [replace(first, user="0x" + "f" * 40)]
    if case == "early":
        rows = [replace(first, exchange_time=DAY - timedelta(seconds=1))]
    if case == "late":
        rows = [replace(first, exchange_time=DAY + timedelta(days=1))]
    if case == "duplicate":
        rows *= 2
    if case == "reverse":
        rows = [replace(first, exchange_time=DAY + timedelta(hours=1)), first]
    if case == "fee":
        rows = [replace(first, fee_token="OTHER")]
    if case == "checkpoint":
        kwargs["checkpoint"] = None
    with pytest.raises(ValueError):
        run(rows, **kwargs)


def test_large_stream_is_consumed_in_lockstep():
    observed = 0

    def consume(row):
        nonlocal observed
        observed += 1

    def source():
        for i in range(100001):
            assert observed == i
            yield replace(
                BASE,
                exchange_time=DAY + timedelta(microseconds=i),
                start_position=0,
                post_position=0,
                fee=0.0,
            )

    result = run(source(), on_fill=consume)
    assert result.fill_count == observed == 100001


@pytest.mark.parametrize("failure", ["source", "fill_sink", "episode_sink"])
def test_failure_does_not_mutate_checkpoint_or_continue_consuming(failure):
    checkpoint = encode_episode_state(
        {}, user=BASE.user, cutoff=DAY, semantics=SEMANTICS
    )
    original, consumed = deepcopy(checkpoint), []

    def fail(row):
        raise RuntimeError("injected")

    def source():
        consumed.append(1)
        yield replace(BASE, exchange_time=DAY, start_position=0, post_position=0)
        if failure == "source":
            raise RuntimeError("injected")
        pytest.fail("sink failure must stop input consumption")

    kwargs = (
        {"on_fill": fail}
        if failure == "fill_sink"
        else {"on_episode": fail}
        if failure == "episode_sink"
        else {}
    )
    with pytest.raises(RuntimeError, match="injected"):
        run(source(), checkpoint=checkpoint, **kwargs)
    assert checkpoint == original and consumed == [1]


@pytest.mark.parametrize(
    "day",
    [
        DAY.replace(tzinfo=None),
        DAY + timedelta(hours=1),
        "2026-08-01",
        datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_invalid_day_is_rejected_before_input_consumption(day):
    def source():
        pytest.fail("invalid context must not consume source")
        yield

    with pytest.raises(ValueError):
        run(source(), day=day)


def test_last_calendar_day_cannot_advance():
    day = datetime(9999, 12, 31, tzinfo=timezone.utc)
    checkpoint = encode_episode_state(
        {}, user=BASE.user, cutoff=day, semantics=SEMANTICS
    )
    with pytest.raises(ValueError, match="range"):
        run([], day=day, checkpoint=checkpoint)


@pytest.mark.parametrize(
    "field,value",
    [
        ("user", "0x" + "f" * 40),
        ("day", DAY + timedelta(days=1)),
        ("semantics", "net_includes_fee"),
        ("semantics", "unknown"),
        ("on_fill", None),
        ("on_episode", None),
    ],
)
def test_context_and_callback_validation(field, value):
    with pytest.raises(ValueError):
        run([], **{field: value})


@pytest.mark.parametrize("seeded", [False, True])
def test_fifty_market_bound_includes_restored_state(seeded):
    active = (
        {
            f"COIN{i}": PositionEpisode(
                BASE.user, f"COIN{i}", DAY - timedelta(days=400), fill_count=1
            )
            for i in range(50)
        }
        if seeded
        else {}
    )
    checkpoint = encode_episode_state(
        active, user=BASE.user, cutoff=DAY, semantics=SEMANTICS
    )
    original = deepcopy(checkpoint)
    rows = [
        replace(BASE, exchange_time=DAY + timedelta(seconds=i), coin=f"COIN{i}")
        for i in (range(50, 51) if seeded else range(51))
    ]
    with pytest.raises(ValueError, match="market limit"):
        run(iter(rows), checkpoint=checkpoint)
    assert checkpoint == original


def test_dormant_open_state_survives_empty_day():
    active = {
        "BTC": PositionEpisode(
            BASE.user,
            "BTC",
            DAY - timedelta(days=400),
            pnl=-0.0,
            fill_count=1,
            peak_notional=2**53 + 1,
        )
    }
    checkpoint = encode_episode_state(
        active, user=BASE.user, cutoff=DAY, semantics=SEMANTICS
    )
    result = run([], checkpoint=checkpoint)
    assert result.checkpoint["episodes"] == checkpoint["episodes"]
    assert result.checkpoint["episodes"][0]["pnl"].hex() == (-0.0).hex()


def test_net_fee_mode_does_not_add_reference_absent_token_restriction():
    observed = []
    semantics = "net_includes_fee"
    checkpoint = encode_episode_state(
        {}, user=BASE.user, cutoff=DAY, semantics=semantics
    )
    fill = replace(BASE, exchange_time=DAY, fee_token="OTHER", closed_pnl=-0.0)
    run([fill], semantics=semantics, checkpoint=checkpoint, on_fill=observed.append)
    assert observed[0].net_pnl.hex() == (-0.0).hex()


@pytest.mark.parametrize("epsilon", [0.0, 1e-10, 1e-9, 2e-9])
@pytest.mark.parametrize("semantics", [SEMANTICS, "net_includes_fee"])
def test_cancellation_sensitive_episode_spans_midnight(epsilon, semantics):
    rows = [
        replace(
            BASE,
            exchange_time=DAY + timedelta(days=1, minutes=i - 2),
            start_position=start,
            post_position=post,
            sz=abs(post - start),
            side="B" if post > start else "A",
            fee=0.0,
            closed_pnl=pnl,
        )
        for i, (start, post, pnl) in enumerate(
            [(0, 1, 1e16), (1, 2, 1.0), (2, epsilon, -1e16)]
        )
    ]
    checkpoint = encode_episode_state(
        {}, user=BASE.user, cutoff=DAY, semantics=semantics
    )
    episodes, numeric = [], []
    first = run(
        rows[:2],
        checkpoint=checkpoint,
        semantics=semantics,
        on_fill=numeric.append,
        on_episode=episodes.append,
    )
    assert not episodes
    second = run(
        rows[2:],
        day=DAY + timedelta(days=1),
        checkpoint=first.checkpoint,
        semantics=semantics,
        on_fill=numeric.append,
        on_episode=episodes.append,
    )
    assert sum(o.net_pnl for o in numeric) == 1.0
    if epsilon <= 1e-9:
        assert len(episodes) == 1 and episodes[0].pnl == 0.0
        assert episodes[0].minutes == 2.0 and episodes[0].fragments == 3
        assert second.checkpoint["episodes"] == []
    else:
        assert not episodes and second.checkpoint["episodes"][0]["pnl"] == 0.0
