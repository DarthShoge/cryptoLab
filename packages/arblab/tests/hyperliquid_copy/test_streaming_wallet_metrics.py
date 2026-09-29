from dataclasses import replace
from datetime import timedelta

import pytest

from arblab.hyperliquid_copy.episodes import build_episodes, closing_size
from arblab.hyperliquid_copy.ranking import RankingConfig, wallet_metrics
from .test_ranking import history


def mixed_history():
    base = history(1)[0]
    specs = [
        ("BTC", 0, 2, "B", 2, 0),
        ("ETH", 3, 2, "A", 1, 2),
        ("BTC", 2, 1, "A", 1, 1),
        ("BTC", 1, -1, "A", 2, -2),
        ("ETH", 2, 0, "A", 2, -1),
        ("BTC", -1, 0, "B", 1, 1),
        ("ETH", 0, 1, "B", 1, 0),
        ("ETH", 1, 0, "A", 1, 3),
        ("BTC", 0, 1, "B", 1, 0),
    ]
    return [
        replace(
            base,
            coin=coin,
            start_position=float(start),
            post_position=float(end),
            side=side,
            sz=float(size),
            closed_pnl=float(pnl),
            fee=0.1 * size,
            crossed=i % 2 == 0,
            tid=i,
            oid=i,
            event_id=str(i),
            exchange_time=base.exchange_time + timedelta(hours=i * 5),
        )
        for i, (coin, start, end, side, size, pnl) in enumerate(specs)
    ]


@pytest.mark.parametrize(
    "semantics,smoke",
    [("gross_excludes_fee", False), ("net_includes_fee", False), ("unknown", True)],
)
@pytest.mark.parametrize("buffer_rows", [2, 4096])
def test_streaming_matches_reference_metrics_and_episode_rules(
    tmp_path, semantics, smoke, buffer_rows
):
    from arblab.hyperliquid_copy.streaming_wallet_metrics import stream_wallet_metrics

    rows = mixed_history()
    config = RankingConfig(
        min_active_days=3, min_episodes=3, min_notional=10, min_minutes=30
    )
    decision = rows[-1].exchange_time + timedelta(seconds=1)
    expected, reasons = wallet_metrics(rows, config, semantics, smoke)
    result = stream_wallet_metrics(
        iter(rows),
        decision,
        config,
        semantics,
        temp_root=tmp_path,
        smoke=smoke,
        buffer_rows=buffer_rows,
    )
    assert result.metrics == expected
    assert result.exclusions == reasons
    assert result.gross_volume == sum(f.sz * f.px for f in rows)
    assert result.closing_notional == sum(closing_size(f) * f.px for f in rows)
    assert result.complete_episodes == sum(
        e.complete for e in build_episodes(rows, semantics, smoke=smoke)
    )
    assert result.fill_count == len(rows)
    assert result.peak_buffered_rows <= buffer_rows
    assert list(tmp_path.iterdir()) == []


def test_no_closing_notional_and_empty_history_are_explicit(tmp_path):
    from arblab.hyperliquid_copy.streaming_wallet_metrics import stream_wallet_metrics

    row = history(1)[0]
    decision = row.exchange_time + timedelta(seconds=1)
    config = RankingConfig()
    result = stream_wallet_metrics(
        iter([row]), decision, config, "gross_excludes_fee", temp_root=tmp_path
    )
    expected, reasons = wallet_metrics([row], config, "gross_excludes_fee", False)
    assert result.metrics == expected
    assert result.exclusions == reasons
    assert result.closing_notional == 0
    empty = stream_wallet_metrics(
        iter(()), decision, config, "gross_excludes_fee", temp_root=tmp_path
    )
    assert empty.exclusions == ("no_activity_in_lookback",)
    assert all(v is None for v in empty.metrics.values())
    assert empty.fill_count == empty.gross_volume == 0


@pytest.mark.parametrize("fault", ["order", "wallet", "future", "expired", "fee"])
def test_invalid_stream_never_yields_partial_metrics(tmp_path, fault):
    from arblab.hyperliquid_copy.streaming_wallet_metrics import stream_wallet_metrics

    rows = mixed_history()
    decision = rows[-1].exchange_time + timedelta(seconds=1)
    if fault == "order":
        rows.reverse()
    elif fault == "wallet":
        rows[-1] = replace(rows[-1], user="0x" + "f" * 40)
    elif fault == "future":
        rows[-1] = replace(rows[-1], exchange_time=decision)
    elif fault == "expired":
        rows[0] = replace(rows[0], exchange_time=decision - timedelta(days=91))
    elif fault == "fee":
        rows[-1] = replace(rows[-1], fee_token="OTHER")
    with pytest.raises(ValueError):
        stream_wallet_metrics(
            iter(rows),
            decision,
            RankingConfig(),
            "gross_excludes_fee",
            temp_root=tmp_path,
            buffer_rows=2,
        )
    assert list(tmp_path.iterdir()) == []


def test_more_than_100000_fills_without_a_wallet_list(tmp_path):
    from arblab.hyperliquid_copy.streaming_wallet_metrics import stream_wallet_metrics

    base = history(1)[0]
    count = 100_002

    def rows():
        for i in range(count):
            closing = i % 2 == 1
            yield replace(
                base,
                exchange_time=base.exchange_time + timedelta(seconds=i),
                start_position=1.0 if closing else 0.0,
                post_position=0.0 if closing else 1.0,
                side="A" if closing else "B",
                closed_pnl=0.01 if closing else 0.0,
                fee=0.0,
                tid=i,
                oid=i,
                event_id=str(i),
            )

    result = stream_wallet_metrics(
        rows(),
        base.exchange_time + timedelta(seconds=count),
        RankingConfig(),
        "gross_excludes_fee",
        temp_root=tmp_path,
    )
    assert result.fill_count == count
    assert result.complete_episodes == count // 2
    assert result.closing_notional == count // 2 * base.px
    assert result.spool_bytes > 0
    assert result.peak_buffered_rows <= 4096
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("buffer_rows", [2, 4096])
def test_sparse_days_and_varied_episode_lengths_match_exactly(tmp_path, buffer_rows):
    from arblab.hyperliquid_copy.streaming_wallet_metrics import stream_wallet_metrics

    base = history(1)[0]
    rows = []
    for episode, length in enumerate([2, 3, 5, 7]):
        for i in range(length):
            closing = i == length - 1
            rows.append(
                replace(
                    base,
                    exchange_time=base.exchange_time
                    + timedelta(days=episode * 4, minutes=i * 17),
                    start_position=float(i),
                    post_position=0.0 if closing else float(i + 1),
                    sz=float(i) if closing else 1.0,
                    side="A" if closing else "B",
                    closed_pnl=[1e16, 1.0, -1e16, 3.0][episode] if closing else 0.0,
                    fee=0.125,
                    event_id=str(len(rows)),
                    tid=len(rows),
                    oid=len(rows),
                )
            )
    decision = rows[-1].exchange_time + timedelta(seconds=1)
    result = stream_wallet_metrics(
        iter(rows),
        decision,
        RankingConfig(),
        "gross_excludes_fee",
        temp_root=tmp_path,
        buffer_rows=buffer_rows,
    )
    expected, reasons = wallet_metrics(
        rows, RankingConfig(), "gross_excludes_fee", False
    )
    assert result.metrics == expected
    assert result.exclusions == reasons
    assert result.complete_episodes == 4


def test_market_and_lookback_bounds(tmp_path):
    from arblab.hyperliquid_copy.streaming_wallet_metrics import stream_wallet_metrics

    base = history(1)[0]
    decision = base.exchange_time + timedelta(seconds=100)
    rows = [
        replace(
            base, coin=f"C{i}", exchange_time=base.exchange_time + timedelta(seconds=i)
        )
        for i in range(51)
    ]
    with pytest.raises(ValueError, match="market limit"):
        stream_wallet_metrics(
            iter(rows),
            decision,
            RankingConfig(),
            "gross_excludes_fee",
            temp_root=tmp_path,
            buffer_rows=2,
        )
    for days in [0, 733, True]:
        with pytest.raises(ValueError, match="lookback"):
            stream_wallet_metrics(
                iter(()),
                decision,
                RankingConfig(lookback_days=days),
                "gross_excludes_fee",
                temp_root=tmp_path,
            )
    assert list(tmp_path.iterdir()) == []
