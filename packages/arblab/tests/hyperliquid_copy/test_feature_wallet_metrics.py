from dataclasses import asdict, replace
from datetime import timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.episode_state import encode_episode_state
from arblab.hyperliquid_copy.feature_records import (
    OBSERVATION_SCHEMA,
    decode_observation,
    encode_observation,
)
from arblab.hyperliquid_copy.ranking import RankingConfig
from arblab.hyperliquid_copy.streaming_wallet_metrics import stream_wallet_metrics
from arblab.hyperliquid_copy.wallet_day_features import derive_wallet_day
from .test_ranking import history

BASE = history(1)[0]
DAY = BASE.exchange_time.replace(hour=0, minute=0, second=0, microsecond=0)


def observations(rows, semantics):
    checkpoint = encode_episode_state(
        {}, user=BASE.user, cutoff=DAY, semantics=semantics
    )
    output = []
    for offset in range(3):
        day = DAY + timedelta(days=offset)
        result = derive_wallet_day(
            (r for r in rows if day <= r.exchange_time < day + timedelta(days=1)),
            user=BASE.user,
            day=day,
            semantics=semantics,
            checkpoint=checkpoint,
            on_fill=output.append,
            on_episode=output.append,
        )
        checkpoint = result.checkpoint
    return output


def semantic(result):
    data = asdict(result)
    del data["spool_bytes"], data["peak_buffered_rows"]
    return data


@pytest.mark.parametrize("semantics", ["gross_excludes_fee", "net_includes_fee"])
@pytest.mark.parametrize("offset", [0, 8, 16])
@pytest.mark.parametrize("unclosed", [False, True])
def test_persisted_feature_metrics_match_exact_window_reference(
    tmp_path, semantics, offset, unclosed
):
    from arblab.hyperliquid_copy.feature_wallet_metrics import feature_wallet_metrics

    shapes = [(0, 1), (1, -1), (-1, 1), (-1, 0), (0, 1), (0, 2), (2, 0)]
    rows = []
    for i in range(144):
        start, post = shapes[(i // 2) % len(shapes)]
        if unclosed and i % 2 == 0:
            start, post = (1, 0) if i == 142 else (0, 1)
        rows.append(
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
                event_id=f"event-{i:04d}",
            )
        )
    path = tmp_path / "features.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [encode_observation(r) for r in observations(rows, semantics)],
            schema=OBSERVATION_SCHEMA,
        ),
        path,
    )
    start = DAY + timedelta(days=1, hours=offset)
    decision = start + timedelta(days=1)
    config = RankingConfig(
        lookback_days=1,
        min_active_days=1,
        min_episodes=2,
        min_notional=1,
        min_minutes=0,
    )

    def persisted():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=7):
            for row in batch.to_pylist():
                if start <= row["exchange_time"] < decision:
                    yield decode_observation(row)

    expected = stream_wallet_metrics(
        (r for r in rows if start <= r.exchange_time < decision),
        decision,
        config,
        semantics,
        temp_root=tmp_path,
    )
    actual = feature_wallet_metrics(
        persisted(),
        decision,
        config,
        semantics,
        temp_root=tmp_path,
    )
    assert semantic(actual) == semantic(expected)


@pytest.mark.parametrize(
    "fault", ["before", "cutoff", "user", "reverse", "duplicate", "dangling"]
)
def test_invalid_stream_never_returns_metrics(tmp_path, fault):
    from arblab.hyperliquid_copy.feature_wallet_metrics import feature_wallet_metrics

    rows = [
        replace(r, exchange_time=DAY + timedelta(hours=i))
        for i, r in enumerate(history(1))
    ]
    data = observations(rows, "gross_excludes_fee")
    assert len(data) == 3
    if fault in ("before", "cutoff"):
        at = (
            DAY - timedelta(microseconds=1)
            if fault == "before"
            else DAY + timedelta(days=1)
        )
        data[0] = replace(data[0], order_key=(at, *data[0].order_key[1:]))
    elif fault == "user":
        data[1] = replace(data[1], user="0x" + "f" * 40)
    elif fault == "reverse":
        data.reverse()
    elif fault == "duplicate":
        data.insert(1, data[0])
    else:
        data.pop()
    with pytest.raises(ValueError):
        feature_wallet_metrics(
            iter(data),
            DAY + timedelta(days=1),
            RankingConfig(lookback_days=1),
            "gross_excludes_fee",
            temp_root=tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_whale_reuses_suffix_without_replaying_every_fill(tmp_path, monkeypatch):
    from arblab.hyperliquid_copy import feature_wallet_metrics as module

    base = replace(history(1)[1], exchange_time=DAY)
    template = observations([base], "gross_excludes_fee")[0]
    count = 100_002
    calls = 0
    original = module._episode

    def replay(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(module, "_episode", replay)

    def fills():
        for i in range(count):
            yield replace(
                base,
                exchange_time=DAY + timedelta(microseconds=i),
                source_line=i,
                event_id=str(i),
            )

    def features():
        for fill in fills():
            yield replace(template, order_key=fill.order_key)

    config = RankingConfig(lookback_days=1)
    actual = module.feature_wallet_metrics(
        features(),
        DAY + timedelta(days=1),
        config,
        "gross_excludes_fee",
        temp_root=tmp_path,
    )
    expected = stream_wallet_metrics(
        fills(),
        DAY + timedelta(days=1),
        config,
        "gross_excludes_fee",
        temp_root=tmp_path,
    )
    assert semantic(actual) == semantic(expected)
    assert actual.fill_count == count
    assert calls == 1
    assert actual.peak_buffered_rows <= 4096


def test_empty_and_interrupted_streams(tmp_path):
    from arblab.hyperliquid_copy.feature_wallet_metrics import feature_wallet_metrics

    config = RankingConfig(lookback_days=1)
    decision = DAY + timedelta(days=1)
    actual = feature_wallet_metrics(
        iter(()), decision, config, "gross_excludes_fee", temp_root=tmp_path
    )
    expected = stream_wallet_metrics(
        iter(()), decision, config, "gross_excludes_fee", temp_root=tmp_path
    )
    assert semantic(actual) == semantic(expected)

    def interrupted():
        yield observations([replace(BASE, exchange_time=DAY)], "gross_excludes_fee")[0]
        raise RuntimeError("interrupted feature stream")

    with pytest.raises(RuntimeError, match="interrupted feature stream"):
        feature_wallet_metrics(
            interrupted(), decision, config, "gross_excludes_fee", temp_root=tmp_path
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("lookback", [0, 733, True, 1.5])
def test_invalid_lookback_rejected_before_reading(tmp_path, lookback):
    from arblab.hyperliquid_copy.feature_wallet_metrics import feature_wallet_metrics

    def forbidden():
        pytest.fail("invalid configuration consumed feature input")
        yield

    with pytest.raises(ValueError, match="lookback"):
        feature_wallet_metrics(
            forbidden(),
            DAY,
            RankingConfig(lookback_days=lookback),
            "gross_excludes_fee",
            temp_root=tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_unknown_fee_policy_rejected_before_reading(tmp_path):
    from arblab.hyperliquid_copy.feature_wallet_metrics import feature_wallet_metrics

    with pytest.raises(ValueError, match="fee"):
        feature_wallet_metrics(
            iter(()), DAY, RankingConfig(), "unknown", temp_root=tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_fifty_first_market_rejected(tmp_path):
    from arblab.hyperliquid_copy.feature_wallet_metrics import feature_wallet_metrics

    template = observations([replace(BASE, exchange_time=DAY)], "gross_excludes_fee")[0]

    def rows():
        for i in range(51):
            yield replace(
                template,
                coin=f"MARKET{i}",
                order_key=(DAY + timedelta(seconds=i), *template.order_key[1:]),
            )

    with pytest.raises(ValueError, match="market limit"):
        feature_wallet_metrics(
            rows(),
            DAY + timedelta(days=1),
            RankingConfig(lookback_days=1),
            "gross_excludes_fee",
            temp_root=tmp_path,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("post", [0.0, 1e-9, 1.00001e-9, -1e-12])
@pytest.mark.parametrize("semantics", ["gross_excludes_fee", "net_includes_fee"])
def test_tiny_boundary_and_notional_threshold_match_reference(
    tmp_path, post, semantics
):
    from arblab.hyperliquid_copy.feature_wallet_metrics import feature_wallet_metrics

    rows = [replace(BASE, exchange_time=DAY)]
    values = [1e12, 0.0001, 0.0001, 0.0001]
    for i, value in enumerate(values):
        rows.append(
            replace(
                BASE,
                exchange_time=DAY + timedelta(days=1, hours=i),
                source_line=i + 1,
                event_id=str(i),
                px=value,
                side="A",
                start_position=1.0,
                post_position=post,
                sz=1.0 - post,
                closed_pnl=[1e8, 1e-6, 1e-6, -1e8][i],
                fee=-0.001,
            )
        )
    start, decision = DAY + timedelta(days=1), DAY + timedelta(days=2)
    config = RankingConfig(
        lookback_days=1,
        min_active_days=1,
        min_episodes=1,
        min_notional=sum(values),
        min_minutes=0,
    )
    actual = feature_wallet_metrics(
        (
            r
            for r in observations(rows, semantics)
            if start <= r.order_key[0] < decision
        ),
        decision,
        config,
        semantics,
        temp_root=tmp_path,
    )
    expected = stream_wallet_metrics(
        iter(rows[1:]), decision, config, semantics, temp_root=tmp_path
    )
    assert semantic(actual) == semantic(expected)
