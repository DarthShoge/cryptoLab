"""Deterministic fictional portfolio for UI preview, never used as live fallback."""

import math

from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition

from .accounting import account_summary
from .timeframes import INTERVALS, aggregate_candles, candle_end, candle_start

DEMO_END = 1790726400  # 2026-09-30 00:00 UTC, last complete demo candle ends here.
BASE_PRICES = {"SOL": 118.62, "ETH": 2680.0, "BTC": 67250.0}


def demo_candles(asset="SOL", interval="1d", days=90, end_time=None, start_time=None):
    source_interval = "1h" if interval in ("1h", "4h") else "1d"
    step = INTERVALS[source_interval]
    end = candle_start(min(DEMO_END, end_time or DEMO_END), source_interval)
    start = candle_start(
        start_time
        if days == 0 and start_time is not None
        else end - (days or 90) * 86400,
        interval,
    )
    if source_interval == "1h":
        start = max(start, end - 600 * step)
    base = BASE_PRICES[asset]
    candles = []

    def price_at(timestamp):
        age = (DEMO_END - timestamp) / 86400
        return (
            base
            * math.exp(-0.0027 * age)
            * (1 + 0.041 * math.sin(age * 0.26) + 0.018 * (math.cos(age * 0.61) - 1))
        )

    for timestamp in range(start, end, step):
        close, opening = price_at(timestamp + step), price_at(timestamp)
        wick = close * (0.006 + 0.007 * abs(math.sin(timestamp / step * 5.7)))
        candles.append(
            {
                "time": timestamp,
                "endTime": timestamp + step,
                "open": opening,
                "close": close,
                "high": max(opening, close) + wick,
                "low": min(opening, close) - wick,
                "volume": 90000 + 70000 * abs(math.cos(timestamp / step * 0.61)),
            }
        )
    return (
        aggregate_candles(candles, interval, end, source_interval)
        if interval != source_interval
        else candles
    )


def demo_portfolio():
    snapshot = AccountSnapshot(
        [
            CollateralPosition("SOL", 209.91, 118.62, 0.7, 0.75),
            CollateralPosition("USDT", 500.15, 1, 0.8, 0.85),
        ],
        [DebtPosition("USDC", 13902.01, 1), DebtPosition("ETH", 1.2, 2680, 1.05)],
    )
    summary = account_summary(
        [("Demo · Main market", snapshot)],
        [
            {"symbol": "SOL", "amount": 7.82, "price": 118.62},
            {"symbol": "USDC", "amount": 682.4, "price": 1},
        ],
    )
    for position in summary["positions"]:
        position["apy"] = {
            ("supplied", "SOL"): 5.12,
            ("supplied", "USDT"): 3.90,
            ("borrowed", "USDC"): 8.23,
            ("borrowed", "ETH"): 1.54,
        }.get((position["kind"], position["symbol"]))
    history = []
    final = summary["netEquity"]
    for i in range(91):
        t = i / 90
        equity = final * (0.77 + 0.23 * t + 0.025 * math.sin(t * 19)) + (
            1200 if i >= 45 else 0
        )
        history.append(
            {
                "time": DEMO_END - (90 - i) * 86400,
                "equity": equity,
                "externalFlow": 1200 if i == 45 else 0,
                "solPrice": 96 + 22.62 * t,
            }
        )
    correction = history[-1]["equity"] - final
    for point in history:
        point["equity"] -= correction
    trades = []
    events = [
        (8, "buy", "SOL", 18.4, "Stablecoin-funded long"),
        (15, "borrow", "USDC", 2000, "Stablecoin loan"),
        (23, "sell", "ETH", 0.55, "Borrowed-token short"),
        (29, "buy", "SOL", 14.2, "Stablecoin-funded long"),
        (39, "sell", "SOL", 9.7, "Reduce long"),
        (45, "deposit", "USDC", 1200, "External deposit"),
        (54, "buy", "ETH", 0.25, "Cover short"),
        (64, "repay", "ETH", 0.25, "Repay token loan"),
        (72, "buy", "SOL", 12.6, "Stablecoin-funded long"),
        (84, "sell", "SOL", 8.5, "Take profit"),
    ]
    for day, kind, asset, amount, funding in events:
        price = 1 if asset == "USDC" else demo_candles(asset)[day]["close"]
        trades.append(
            {
                "id": f"demo-{day}",
                "time": DEMO_END - (90 - day) * 86400 + 43200,
                "type": kind,
                "asset": asset,
                "amount": amount,
                "price": price,
                "value": amount * price,
                "protocol": "Jupiter" if kind in ("buy", "sell") else "Kamino",
                "funding": funding,
                "feeSol": 0.000005,
                "signature": None,
                "quote": "USD",
            }
        )
    health_history = [
        {
            "time": p["time"],
            "health": round(1.12 + 0.17 * math.sin(i / 9), 4),
            "status": "available",
            "source": "Fictional demonstration daily risk",
        }
        for i, p in enumerate(history)
    ]
    return {
        "mode": "demo",
        "wallet": None,
        "asOf": DEMO_END,
        "summary": summary,
        "healthHistory": health_history,
        "kaminoSeries": [
            {
                "id": summary["loans"][0]["address"],
                "label": "Demo Kamino obligation",
                "history": history,
                "healthHistory": health_history,
            }
        ],
        "history": history,
        "trades": sorted(trades, key=lambda t: t["time"], reverse=True),
        "historyComplete": True,
        "warnings": [],
        "prices": {**BASE_PRICES, "USDC": 1, "USDT": 1},
        "source": "Fictional demonstration · not your account",
    }
