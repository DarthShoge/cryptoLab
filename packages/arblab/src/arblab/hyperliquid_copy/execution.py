"""L2 taker execution and collateral-based perpetual accounting."""
from dataclasses import dataclass

from .contracts import finite


@dataclass(frozen=True)
class Execution:
    requested_qty: float
    filled_qty: float
    unfilled_qty: float
    vwap: float | None
    arrival_mid: float | None
    book_time: object
    fee: float
    spread_cost: float
    reason: str


def walk_book(book, qty, *, fee_bps=4.5):
    qty = finite(qty)
    if finite(fee_bps) < 0:
        raise ValueError("negative fee")
    if book is None:
        return Execution(qty, 0, qty, None, None, None, 0, 0, "stale_book")
    remaining, notional, filled = abs(qty), 0., 0.
    for px, size in book.asks if qty > 0 else book.bids:
        take = min(remaining, size)
        notional += take*px
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break
    sign = 1 if qty >= 0 else -1
    vwap = notional/filled if filled else None
    return Execution(qty, filled*sign, remaining*sign, vwap, book.mid, book.time,
                     notional*fee_bps/10000, sign*(notional-filled*book.mid),
                     "partial_depth" if remaining > 1e-12 else "filled")


@dataclass
class Position:
    qty: float = 0
    entry: float = 0


class Account:
    def __init__(self, collateral):
        self.cash = finite(collateral)
        self.positions = {}
        self.realized_pnl = 0.

    def fill(self, coin, qty, px, fee):
        qty, px, fee = finite(qty), finite(px), finite(fee)
        if px <= 0 or fee < 0:
            raise ValueError("invalid execution price/fee")
        position = self.positions.setdefault(coin, Position())
        old = position.qty
        new = old + qty
        if old*qty < 0:
            realized = min(abs(old), abs(qty))*(px-position.entry)*(1 if old > 0 else -1)
            self.cash += realized
            self.realized_pnl += realized
        if old*qty >= 0 and new:
            position.entry = (abs(old)*position.entry+abs(qty)*px)/abs(new)
        elif old*new < 0:
            position.entry = px
        elif abs(new) < 1e-12:
            new, position.entry = 0., 0.
        position.qty = new
        self.cash -= fee

    def funding(self, coin, mark, rate):
        delta = -self.positions.get(coin, Position()).qty*finite(mark)*finite(rate)
        self.cash += delta
        return delta

    def equity(self, marks):
        return self.cash + sum(p.qty*(marks[c]-p.entry) for c, p in self.positions.items() if p.qty)
