from datetime import datetime, timedelta, timezone

import pytest

from arblab.hyperliquid_copy.proxy_bars import ProxyBar
from arblab.hyperliquid_copy.proxy_funding import FundingEvent

T = datetime(2026, 8, 3, tzinfo=timezone.utc)


def at(hours):
    return T + timedelta(hours=hours)


def bar(hour, opening=100, close=100, coin="BTC", duration=1):
    return ProxyBar(
        coin,
        at(hour),
        at(hour + duration),
        opening,
        max(opening, close),
        min(opening, close),
        close,
    )


def funding(hours=3, rate_at_two=0, coins=("BTC",)):
    return [
        FundingEvent(c, at(h), at(h), rate_at_two if h == 2 else 0, 0)
        for c in coins
        for h in range(hours)
    ]


def run(signals, bars=None, rates=None, **changes):
    from arblab.hyperliquid_copy.proxy_simulator import (
        ProxySimulationConfig,
        simulate_proxy,
    )

    config = ProxySimulationConfig(
        **(
            dict(
                initial_equity=1000,
                fee_bps=0,
                slippage_bps=0,
                min_trade_usd=0,
                deadband=0,
            )
            | changes
        )
    )
    return simulate_proxy(
        signals,
        bars or [bar(-1), bar(0), bar(1, 100, 110), bar(2, 110, 120)],
        funding() if rates is None else rates,
        T,
        at(3),
        ["BTC"],
        config,
    )


@pytest.mark.parametrize("weight,final", [(1, 1189), (-1, 811)])
def test_long_short_funding_and_residual_reconcile(weight, final):
    result = run(
        [dict(time=T, coin="BTC", value=weight)], rates=funding(rate_at_two=0.01)
    )
    assert result.fills[0]["time"] == at(1)
    assert result.fills[0]["filled_qty"] == 10 * weight
    assert result.funding[1]["qty"] == 0  # settlement precedes coincident first open
    assert result.funding[2]["cash_delta"] == -11 * weight
    assert result.equity[-1]["equity"] == pytest.approx(final)
    position = result.positions["BTC"]
    assert result.cash + position.qty * (120 - position.entry) == pytest.approx(final)


def test_same_time_new_decision_cannot_cancel_already_pending_open():
    result = run(
        [dict(time=T, coin="BTC", value=1), dict(time=at(1), coin="BTC", value=0)]
    )
    assert [(f["time"], f["filled_qty"]) for f in result.fills] == [
        (at(1), 10),
        (at(2), -10),
    ]
    assert result.cash == 1100
    assert result.positions["BTC"].qty == 0


def test_latest_target_waits_for_actual_session_open():
    result = run(
        [dict(time=T, coin="BTC", value=1), dict(time=at(1), coin="BTC", value=-0.5)],
        bars=[bar(-1), bar(2.5, duration=0.5)],
        max_mark_age_seconds=4 * 3600,
    )
    assert len(result.fills) == 1
    assert result.fills[0]["time"] == at(2.5)
    assert result.fills[0]["signal_time"] == at(1)
    assert result.fills[0]["filled_qty"] == -5
    assert result.fills[0]["delay_seconds"] == 5400
    assert any(r["reason"] == "superseded" for r in result.requests)


def test_delay_is_strict_and_missing_open_is_disclosed():
    result = run([dict(time=T, coin="BTC", value=1)], latency_seconds=3600)
    assert result.fills[0]["time"] == at(2)
    empty = run([dict(time=at(2), coin="BTC", value=1)])
    assert not empty.fills
    assert empty.requests[-1]["reason"] == "no_open_before_end"


def test_missing_funding_and_stale_held_marks_reject():
    with pytest.raises(ValueError, match="funding coverage"):
        run([dict(time=T, coin="BTC", value=1)], rates=[])
    with pytest.raises(ValueError, match="stale"):
        run(
            [dict(time=T, coin="BTC", value=1)],
            bars=[bar(-1), bar(0.5, duration=0.5)],
            max_mark_age_seconds=1800,
        )


def test_slippage_fees_and_price_scale_invariance():
    signals = [dict(time=T, coin="BTC", value=1)]
    result = run(signals, fee_bps=10, slippage_bps=100)
    fill = result.fills[0]
    assert fill["vwap"] == 101
    assert fill["fee"] == pytest.approx(1.01)
    assert fill["spread_cost"] == 10
    assert result.cash == pytest.approx(998.99)
    assert result.equity[-1]["equity"] == pytest.approx(1188.99)
    scaled = run(
        signals,
        bars=[
            bar(-1, 10000, 10000),
            bar(0, 10000, 10000),
            bar(1, 10000, 11000),
            bar(2, 11000, 12000),
        ],
        fee_bps=10,
        slippage_bps=100,
    )
    assert scaled.equity[-1]["equity"] == pytest.approx(result.equity[-1]["equity"])
    assert scaled.fills[0]["filled_qty"] == pytest.approx(fill["filled_qty"] / 100)


def test_simultaneous_targets_share_budget_without_coin_order_bias():
    from arblab.hyperliquid_copy.proxy_simulator import (
        ProxySimulationConfig,
        simulate_proxy,
    )

    coins = ["BTC", "ETH"]
    bars = [bar(h, coin=c) for c in coins for h in range(-1, 3)]
    result = simulate_proxy(
        [dict(time=T, coin=c, value=1) for c in coins],
        bars,
        funding(coins=coins),
        T,
        at(3),
        coins,
        ProxySimulationConfig(initial_equity=1000, fee_bps=0, slippage_bps=0),
    )
    assert {c: p.qty for c, p in result.positions.items()} == {"BTC": 5, "ETH": 5}


def test_closed_market_exposure_consumes_budget_for_other_markets():
    from arblab.hyperliquid_copy.proxy_simulator import (
        ProxySimulationConfig,
        simulate_proxy,
    )

    coins = ["BTC", "xyz:TSLA"]
    bars = [bar(h) for h in range(-1, 3)] + [
        bar(-1, coin="xyz:TSLA"),
        bar(0.5, coin="xyz:TSLA", duration=0.5),
    ]
    signals = [
        dict(time=T, coin="xyz:TSLA", value=0.75),
        dict(time=at(1), coin="BTC", value=1),
    ]
    result = simulate_proxy(
        signals,
        bars,
        funding(coins=coins),
        T,
        at(3),
        coins,
        ProxySimulationConfig(initial_equity=1000, fee_bps=0, slippage_bps=0),
    )
    assert result.positions["xyz:TSLA"].qty == 7.5
    assert result.positions["BTC"].qty == 2.5
    assert result.equity[-1]["mark_ages_seconds"]["xyz:TSLA"] == 7200


@pytest.mark.parametrize(
    "thresholds",
    [{"deadband": 0.02, "min_trade_usd": 0}, {"deadband": 0, "min_trade_usd": 20}],
)
def test_skipped_reductions_do_not_release_gross_budget(thresholds):
    from arblab.hyperliquid_copy.proxy_simulator import (
        ProxySimulationConfig,
        simulate_proxy,
    )

    coins = ["BTC", "ETH"]
    bars = [bar(h, coin=c) for c in coins for h in range(-1, 3)]
    signals = [
        dict(time=T, coin="BTC", value=0.5),
        dict(time=at(1), coin="BTC", value=0.49),
        dict(time=at(1), coin="ETH", value=0.51),
    ]
    result = simulate_proxy(
        signals,
        bars,
        funding(coins=coins),
        T,
        at(3),
        coins,
        ProxySimulationConfig(
            initial_equity=1000, fee_bps=0, slippage_bps=0, **thresholds
        ),
    )
    assert result.positions["BTC"].qty == 5
    assert result.positions["ETH"].qty == 5
    assert result.equity[-1]["gross_exposure"] == 1000


def test_native_offset_funding_uses_position_after_prior_open():
    from dataclasses import replace

    rates = [
        replace(e, time=e.time + timedelta(milliseconds=80), rate=0.01)
        for e in funding()
    ]
    result = run([dict(time=T, coin="BTC", value=1)], rates=rates)
    assert result.funding[1]["time"] == at(1) + timedelta(milliseconds=80)
    assert result.funding[1]["qty"] == 10
    assert result.funding[1]["cash_delta"] == -10
    assert result.equity[-1]["equity"] == 1179


@pytest.mark.parametrize(
    "change",
    [
        {"fee_bps": float("nan")},
        {"slippage_bps": 10000},
        {"gross_cap": 2},
        {"max_wait_seconds": 0},
    ],
)
def test_invalid_risk_parameters_reject(change):
    with pytest.raises(ValueError):
        run([], **change)


def test_nonhourly_decisions_and_duplicate_funding_reject():
    with pytest.raises(ValueError, match="hourly"):
        run([dict(time=at(0.5), coin="BTC", value=1)])
    with pytest.raises(ValueError, match="duplicate funding"):
        run([], rates=funding() + funding()[:1])
