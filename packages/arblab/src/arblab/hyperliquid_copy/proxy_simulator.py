"""Hourly proxy execution with causal opens and collateral-based accounting.

Targets are portfolio notional weights, resized using equity at execution.
Execution sizing uses contemporaneous actual opens; hourly reporting and funding
use completed closes only. These are proxy marks, not native perpetual mark prices.
At equal times: completed closes are visible, funding settles, pending targets
execute, then hourly equity is recorded and new decisions replace pending targets.
"""

from dataclasses import dataclass, field
from datetime import timedelta

from .contracts import finite, symbol, utc
from .execution import Account, Position
from .proxy_bars import ProxyBars


@dataclass(frozen=True)
class ProxySimulationConfig:
    initial_equity: float = 10000
    gross_cap: float = 1
    asset_cap: float = 1
    fee_bps: float = 4.5
    slippage_bps: float = 5
    latency_seconds: float = 5
    min_trade_usd: float = 10
    deadband: float = 0.02
    max_mark_age_seconds: float = 4 * 86400
    max_wait_seconds: float = 4 * 86400

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, finite(getattr(self, name)))
        if (
            self.initial_equity <= 0
            or not 0 < self.gross_cap <= 1
            or not 0 < self.asset_cap <= 1
            or self.fee_bps < 0
            or not 0 <= self.slippage_bps < 10000
            or self.latency_seconds < 0
            or self.min_trade_usd < 0
            or self.deadband < 0
            or self.max_mark_age_seconds <= 0
            or self.max_wait_seconds <= 0
        ):
            raise ValueError("invalid proxy simulation configuration")


@dataclass
class ProxySimulationResult:
    sampling_interval_seconds: int = 3600
    equity: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    funding: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    positions: dict = field(default_factory=dict)
    cash: float = 0
    insolvent: bool = False


def _inputs(
    signals, bars, funding, start, end, coins, max_days=93, funding_starts=None
):
    if type(max_days) is not int or max_days not in (93, 366):
        raise ValueError("Invalid proxy simulation day limit")
    if not 0 < (end - start).total_seconds() <= max_days * 86400:
        raise ValueError(
            f"proxy simulation range must be positive and at most {max_days} days"
        )
    if len(coins) * int((end - start).total_seconds() / 3600) > 250_000:
        raise ValueError("proxy simulation asset-hour limit exceeded")
    if any(t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError("proxy simulation requires aligned UTC hours")
    if len(coins) > 50 or len(set(coins)) != len(coins):
        raise ValueError("at most 50 distinct proxy markets required")
    for coin in coins:
        symbol(coin)
    availability = {coin: utc(at) for coin, at in (funding_starts or {}).items()}
    if set(availability) - set(coins) or any(
        at.minute or at.second or at.microsecond for at in availability.values()
    ):
        raise ValueError("Invalid native funding availability")
    hours = [
        start + timedelta(hours=h)
        for h in range(int((end - start).total_seconds() / 3600))
    ]
    decisions, rates, opens = {}, {}, {}
    hour_set = set(hours)
    for index, row in enumerate(signals):
        if index >= 250_000:
            raise ValueError("proxy signal row limit exceeded")
        at, coin, weight = utc(row["time"]), row["coin"], finite(row["value"])
        if at not in hour_set or coin not in coins or abs(weight) > 1:
            raise ValueError("signal outside hourly proxy scope")
        if at < availability.get(coin, start):
            raise ValueError("Signal before native funding availability")
        if coin in decisions.setdefault(at, {}):
            raise ValueError("duplicate proxy signal")
        decisions[at][coin] = weight
    coverage = set()
    for event in funding:
        if event.instrument_id not in coins or not start <= utc(event.time) < end:
            continue
        hour = utc(event.time).replace(minute=0, second=0, microsecond=0)
        if hour < availability.get(event.instrument_id, start):
            raise ValueError("Funding contradicts native availability")
        key = event.instrument_id, hour
        if utc(event.hour) != hour or key in coverage:
            raise ValueError("invalid or duplicate funding coverage")
        coverage.add(key)
        rates.setdefault(utc(event.time), []).append(event)
        finite(event.rate)
    if coverage != {
        (coin, hour)
        for coin in coins
        for hour in hours
        if hour >= availability.get(coin, start)
    }:
        raise ValueError("missing hourly funding coverage")
    for bar in bars:
        if bar.instrument_id in coins and start <= bar.start < end:
            opens.setdefault(bar.start, {})[bar.instrument_id] = bar.open
    timeline = sorted(set(hours) | {end} | set(rates) | set(opens))
    return decisions, rates, opens, set(hours) | {end}, timeline


class _Replay:
    def __init__(self, market, config):
        self.market, self.config = market, config
        self.account = Account(config.initial_equity)
        self.result = ProxySimulationResult()
        self.pending = {}

    def mark(self, coin, at):
        return self.market.completed_mark(
            coin, at, max_age_seconds=self.config.max_mark_age_seconds
        )

    def held_marks(self, at, actual_opens=None):
        actual_opens = actual_opens or {}
        return {
            coin: actual_opens[coin]
            if coin in actual_opens
            else self.mark(coin, at).price
            for coin, p in self.account.positions.items()
            if p.qty
        }

    def settle(self, at, events):
        for event in sorted(events, key=lambda e: e.instrument_id):
            coin = event.instrument_id
            qty = self.account.positions.get(coin, Position()).qty
            mark = self.mark(coin, at) if qty else None
            delta = self.account.funding(coin, mark.price, event.rate) if mark else 0
            self.result.funding.append(
                dict(
                    time=at,
                    coin=coin,
                    qty=qty,
                    mark=mark.price if mark else None,
                    mark_age_seconds=mark.age_seconds if mark else None,
                    rate=event.rate,
                    cash_delta=delta,
                    treatment="proxy_completed_close_notional",
                )
            )

    def execute(self, at, opens):
        due = {
            c: r
            for c, r in self.pending.items()
            if c in opens and r["execution_time"] == at
        }
        if not due:
            return
        marks = self.held_marks(at, opens)
        equity = self.account.equity(marks)
        if equity <= 0:
            raise ValueError("Insolvent proxy portfolio; liquidation is not modeled")
        # Frozen batch equity and residual exposure prevent alphabetical allocation
        # bias and prevent a closed-market holding being ignored by the gross cap.
        targets = {
            c: max(-self.config.asset_cap, min(self.config.asset_cap, r["weight"]))
            * equity
            for c, r in due.items()
        }
        executable = set(due)
        while executable:
            residual = sum(
                abs(p.qty * marks[c])
                for c, p in self.account.positions.items()
                if p.qty and c not in executable
            )
            budget = max(0, self.config.gross_cap * equity - residual)
            gross = sum(abs(targets[c]) for c in executable)
            factor = min(1, budget / gross) if gross else 1
            deltas = {
                c: targets[c] * factor
                - self.account.positions.get(c, Position()).qty * opens[c]
                for c in executable
            }
            skipped = {
                c
                for c, delta in deltas.items()
                if abs(delta) <= self.config.deadband * equity
                or abs(delta) < self.config.min_trade_usd
            }
            if not skipped:
                break
            # A threshold-skipped reduction releases no capacity. Reallocate
            # before executing; each pass removes at least one candidate.
            executable -= skipped
        for coin, request in sorted(due.items()):
            del self.pending[coin]
            if coin not in executable:
                request["reason"] = "below_trade_threshold"
                continue
            px = opens[coin]
            delta = deltas[coin]
            qty = delta / px
            execution = px * (
                1 + (1 if qty > 0 else -1) * self.config.slippage_bps / 10000
            )
            fee = abs(qty * execution) * self.config.fee_bps / 10000
            self.account.fill(coin, qty, execution, fee)
            request["reason"] = "executed"
            self.result.fills.append(
                dict(
                    time=at,
                    book_time=at,
                    coin=coin,
                    signal_time=request["signal_time"],
                    delay_seconds=(at - request["signal_time"]).total_seconds(),
                    requested_qty=qty,
                    filled_qty=qty,
                    unfilled_qty=0,
                    vwap=execution,
                    arrival_mid=px,
                    requested_notional=abs(delta),
                    fee=fee,
                    spread_cost=abs(qty) * abs(execution - px),
                    reason="proxy_open_fill",
                )
            )

    def point(self, at):
        marks = self.held_marks(at)
        equity = self.account.equity(marks)
        if equity <= 0:
            raise ValueError("Insolvent proxy portfolio; liquidation is not modeled")
        exposure = {c: self.account.positions[c].qty * px for c, px in marks.items()}
        self.result.equity.append(
            dict(
                time=at,
                equity=equity,
                cash=self.account.cash,
                unrealized_pnl=equity - self.account.cash,
                realized_pnl=self.account.realized_pnl,
                gross_exposure=sum(abs(v) for v in exposure.values()),
                net_exposure=sum(exposure.values()),
                mark_ages_seconds={c: self.mark(c, at).age_seconds for c in marks},
            )
        )

    def decide(self, at, targets, end):
        for coin, weight in sorted(targets.items()):
            previous = self.pending.pop(coin, None)
            if previous is not None:
                previous["reason"] = "superseded"
            due = at + timedelta(seconds=self.config.latency_seconds)
            bar = self.market.next_open(
                coin, due, max_wait_seconds=self.config.max_wait_seconds
            )
            request = dict(
                coin=coin,
                signal_time=at,
                weight=weight,
                execution_time=bar.start if bar and bar.start < end else None,
                reason="pending" if bar and bar.start < end else "no_open_before_end",
            )
            self.result.requests.append(request)
            if request["execution_time"] is not None:
                self.pending[coin] = request


def simulate_proxy(
    signals,
    bars,
    funding,
    start,
    end,
    coins,
    config,
    *,
    max_days=93,
    funding_starts=None,
):
    start, end = utc(start), utc(end)
    bars = tuple(bars)
    if len(bars) > 250_000:
        raise ValueError("proxy bar input ceiling exceeded")
    market = ProxyBars(bars)
    decisions, rates, opens, points, timeline = _inputs(
        signals, bars, funding, start, end, coins, max_days, funding_starts
    )
    replay = _Replay(market, config)
    for at in timeline:
        replay.settle(at, rates.get(at, []))
        replay.execute(at, opens.get(at, {}))
        if at in points:
            replay.point(at)
        replay.decide(at, decisions.get(at, {}), end)
    replay.result.positions = replay.account.positions
    replay.result.cash = replay.account.cash
    return replay.result
