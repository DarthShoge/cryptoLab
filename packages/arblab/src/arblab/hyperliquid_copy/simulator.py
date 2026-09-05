"""Deterministic event ordering: funding, fills, marks, new requests."""

from dataclasses import asdict, dataclass
from datetime import timedelta
import heapq

from .contracts import finite, utc
from .execution import Account, Position, walk_book


@dataclass(frozen=True)
class SimulationConfig:
    initial_equity: float = 10000
    asset_cap: float = 0.25
    gross_cap: float = 1
    deadband: float = 0.02
    min_trade_usd: float = 10
    latency_seconds: int = 5
    fee_bps: float = 4.5
    max_mark_age: int = 60
    max_book_lag: int = 2


@dataclass
class SimulationResult:
    equity: list
    fills: list
    funding: list
    positions: dict
    cash: float
    insolvent: bool


def simulate(signals, market, start, end, coins, config, *, lifetimes=None):
    start, end = utc(start), utc(end)
    if end <= start or config.latency_seconds < 0 or not 0 < config.gross_cap <= 1:
        raise ValueError("invalid simulation range/config")
    if (
        config.initial_equity <= 0
        or config.asset_cap <= 0
        or config.deadband < 0
        or config.min_trade_usd < 0
    ):
        raise ValueError("invalid risk configuration")
    account, events = Account(config.initial_equity), []
    target_rows = {}
    for row in signals:
        at = utc(row["time"])
        coin = row["coin"]
        if not start <= at < end or coin not in coins:
            raise ValueError("signal outside simulation scope")
        if lifetimes is not None and not lifetimes[coin].active(at):
            raise ValueError("signal outside instrument lifetime")
        key = at, coin
        if key in target_rows:
            raise ValueError("duplicate signal")
        target_rows[key] = max(-1.0, min(1.0, finite(row["value"])))
    serial = 0

    def push(at, priority, coin, payload):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (at, priority, coin, serial, payload))

    for (coin, at), rate in market.funding.items():
        if coin in coins and start < at <= end:
            push(at, 0, coin, rate)
    at = start
    while at <= end:
        push(at, 2, "", None)
        at += timedelta(minutes=1)
    points, fills, funding, insolvent = [], [], [], False
    pending = set()
    while events:
        at, priority, coin, request_id, payload = heapq.heappop(events)
        if priority == 0:
            if (
                lifetimes is not None
                and not lifetimes[coin].active(at)
                and not account.positions.get(coin, Position()).qty
            ):
                raise ValueError("funding outside instrument lifetime")
            mark = market.mark(coin, at, max_age=config.max_mark_age)
            qty = account.positions.get(coin, Position()).qty
            delta = account.funding(coin, mark, payload)
            funding.append(
                dict(
                    time=at,
                    coin=coin,
                    qty=qty,
                    mark=mark,
                    rate=payload,
                    cash_delta=delta,
                )
            )
        elif priority == 1:
            pending.discard(coin)
            if insolvent:
                continue
            execution, signal_time, signal_mid = payload
            if execution.filled_qty:
                account.fill(coin, execution.filled_qty, execution.vwap, execution.fee)
            fills.append(
                dict(
                    coin=coin,
                    signal_time=signal_time,
                    signal_mid=signal_mid,
                    requested_notional=abs(execution.requested_qty) * signal_mid,
                    **asdict(execution),
                )
            )
        else:
            required = [
                c
                for c in coins
                if lifetimes is None
                or lifetimes[c].active(at)
                or account.positions.get(c, Position()).qty
                or c in pending
            ]
            marks = {
                c: market.mark(c, at, max_age=config.max_mark_age) for c in required
            }
            equity = account.equity(marks)
            insolvent |= equity <= 0
            exposure = {
                c: account.positions.get(c, Position()).qty * marks[c] for c in required
            }
            points.append(
                dict(
                    time=at,
                    equity=equity,
                    cash=account.cash,
                    unrealized_pnl=equity - account.cash,
                    realized_pnl=account.realized_pnl,
                    gross_exposure=sum(abs(x) for x in exposure.values()),
                    net_exposure=sum(exposure.values()),
                )
            )
            if insolvent or at == end:
                continue
            desired = {
                c: target_rows[(at, c)] * config.asset_cap * equity
                for c in coins
                if (at, c) in target_rows
            }
            gross = sum(abs(v) for v in desired.values())
            factor = min(1.0, config.gross_cap * equity / gross) if gross else 1.0
            for c, value in sorted(desired.items()):
                delta = value * factor - exposure[c]
                if (
                    c in pending
                    or abs(delta) <= config.deadband * equity
                    or abs(delta) < config.min_trade_usd
                ):
                    continue
                executable = at + timedelta(seconds=config.latency_seconds)
                book = market.execution_book(c, executable, max_lag=config.max_book_lag)
                if lifetimes is not None and (
                    not lifetimes[c].active(executable)
                    or book
                    and not lifetimes[c].active(book.time)
                ):
                    raise ValueError("execution outside instrument lifetime")
                if executable > end or (book and book.time > end):
                    continue
                execution = walk_book(book, delta / marks[c], fee_bps=config.fee_bps)
                if book is None:
                    fills.append(
                        dict(
                            coin=c,
                            signal_time=at,
                            signal_mid=marks[c],
                            requested_notional=abs(delta),
                            **asdict(execution),
                        )
                    )
                else:
                    pending.add(c)
                    push(book.time, 1, c, (execution, at, marks[c]))
    return SimulationResult(
        points, fills, funding, account.positions, account.cash, insolvent
    )
