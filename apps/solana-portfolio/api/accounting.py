"""Portfolio valuation and explicitly cash-flow-adjusted history."""

import math
from collections import defaultdict


def finite(value):
    return value if type(value) in (float, int) and math.isfinite(value) else None


def account_summary(obligations, wallet):
    supplied = debt = known_wallet = 0.0
    exposure = defaultdict(float)
    loans, positions = [], []
    for address, snapshot in obligations:
        supplied += snapshot.total_collateral_value()
        debt += snapshot.total_debt_value()
        for kind, entries in (
            ("supplied", snapshot.collateral),
            ("borrowed", snapshot.debt),
        ):
            for position in entries:
                exposure[position.symbol] += position.amount * (
                    -1 if kind == "borrowed" else 1
                )
                positions.append(
                    {
                        "id": f"{address}:{kind}:{len(positions)}",
                        "symbol": position.symbol,
                        "amount": position.amount,
                        "price": position.price,
                        "value": position.value(),
                        "kind": kind,
                        "obligation": address,
                        "apy": None,
                    }
                )
        adjusted = snapshot.risk_adjusted_debt_value()
        loans.append(
            {
                "address": address,
                "supplied": snapshot.total_collateral_value(),
                "debt": snapshot.total_debt_value(),
                "ltv": finite(snapshot.borrow_ltv()),
                "liquidationLtv": snapshot.liquidation_ltv(),
                "health": finite(snapshot.liquidation_health_factor()),
                "borrowHealth": finite(snapshot.health_factor()),
                "liquidationBuffer": snapshot.liquidation_value() - adjusted,
            }
        )
    unpriced = 0
    for token in wallet:
        value = token["amount"] * token["price"] if token["price"] is not None else None
        if value is None:
            unpriced += 1
        else:
            known_wallet += value
        exposure[token["symbol"]] += token["amount"]
        positions.append(
            {
                **token,
                "id": f"wallet:{token.get('mint', token['symbol'])}",
                "value": value,
                "kind": "wallet",
                "obligation": None,
                "apy": None,
            }
        )
    healths = [loan["health"] for loan in loans if loan["health"] is not None]
    return {
        "supplied": supplied,
        "debt": debt,
        "walletValue": known_wallet,
        "knownEquity": supplied + known_wallet - debt,
        "netEquity": supplied + known_wallet - debt if not unpriced else None,
        "health": min(healths) if healths else None,
        "unpricedCount": unpriced,
        "loans": loans,
        "positions": positions,
        "exposure": dict(exposure),
    }


def history_metrics(history, complete=False):
    empty = {"pnl": None, "returnPct": None, "solPnl": None, "drawdownPct": None}
    if not complete or len(history) < 2:
        return empty
    first, last = history[0], history[-1]
    flows = sum(point["externalFlow"] for point in history[1:])
    pnl = last["equity"] - first["equity"] - flows
    # External flows occur at observation/period end; chain time-weighted returns.
    index = peak = 1.0
    drawdown = 0.0
    valid_returns = True
    for before, after in zip(history, history[1:]):
        if before["equity"] <= 0 or after["equity"] - after["externalFlow"] < 0:
            valid_returns = False
            break
        index *= (after["equity"] - after["externalFlow"]) / before["equity"]
        if not math.isfinite(index):
            valid_returns = False
            break
        peak = max(peak, index)
        drawdown = min(drawdown, index / peak - 1)
    sol_pnl = last["equity"] / last["solPrice"] - first["equity"] / first["solPrice"]
    sol_pnl -= sum(point["externalFlow"] / point["solPrice"] for point in history[1:])
    return {
        "pnl": pnl,
        "returnPct": (index - 1) * 100 if valid_returns else None,
        "solPnl": sol_pnl,
        "drawdownPct": drawdown * 100 if valid_returns else None,
    }


def validate_history(data):
    if (
        not isinstance(data, dict)
        or data.get("flowTiming") != "period-end"
        or type(data.get("complete")) is not bool
    ):
        raise ValueError("History must declare complete and flowTiming: period-end.")
    source = data.get("source")
    if not isinstance(source, dict) or any(
        not isinstance(source.get(key), str)
        or not source[key].strip()
        or len(source[key]) > 200
        for key in ("name", "account")
    ):
        raise ValueError(
            "Declare source.name and source.account for the imported history."
        )
    if source.get("equityScope") != "wallet-and-kamino" or source.get(
        "cashFlowCoverage"
    ) not in ("complete", "partial"):
        raise ValueError(
            "Declare source.equityScope: wallet-and-kamino and source.cashFlowCoverage: complete or partial."
        )
    if data["complete"] and source["cashFlowCoverage"] != "complete":
        raise ValueError(
            "Complete returns require declared complete external cash-flow coverage."
        )
    history = data.get("history")
    if not isinstance(history, list) or not 2 <= len(history) <= 10000:
        raise ValueError("History needs 2–10,000 priced equity observations.")
    previous = -1
    for point in history:
        if not isinstance(point, dict) or any(
            finite(point.get(key)) is None
            for key in ("time", "equity", "externalFlow", "solPrice")
        ):
            raise ValueError(
                "Each observation needs finite time, equity, externalFlow and solPrice."
            )
        if point["time"] <= previous or point["solPrice"] <= 0 or point["time"] <= 0:
            raise ValueError(
                "Use unique ascending Unix-second times and positive SOL prices."
            )
        if (
            point["time"] != int(point["time"])
            or point["time"] > 4102444800
            or any(
                abs(point[key]) > 1e18 for key in ("equity", "externalFlow", "solPrice")
            )
        ):
            raise ValueError(
                "Use Unix seconds before 2100 and financial amounts below 1e18."
            )
        previous = point["time"]
    if history[0]["externalFlow"] != 0:
        raise ValueError(
            "Opening equity includes initial capital; its externalFlow must be zero."
        )
    trades = data.get("trades", [])
    if not isinstance(trades, list) or len(trades) > 10000:
        raise ValueError("Trades must be an array of at most 10,000 records.")
    ids = set()
    for trade in trades:
        if not isinstance(trade, dict) or trade.get("type") not in (
            "buy",
            "sell",
            "borrow",
            "repay",
            "deposit",
            "withdraw",
        ):
            raise ValueError("Invalid trade type.")
        if (
            not isinstance(trade.get("id"), str)
            or not trade["id"]
            or trade["id"] in ids
        ):
            raise ValueError("Trade IDs must be unique nonempty strings.")
        ids.add(trade["id"])
        if (
            any(finite(trade.get(k)) is None for k in ("time", "amount", "price"))
            or trade["amount"] <= 0
            or trade["price"] <= 0
        ):
            raise ValueError("Trades need finite time and positive amount and price.")
        if not history[0]["time"] <= trade["time"] <= history[-1]["time"]:
            raise ValueError("Trade times must fall inside the imported history.")
        if trade.get("asset") not in ("SOL", "ETH", "BTC", "USDC", "USDT"):
            raise ValueError(
                "Imported trade asset must be SOL, ETH, BTC, USDC or USDT."
            )
        if any(trade[k] > 1e18 for k in ("amount", "price")):
            raise ValueError("Trade amounts and prices must be below 1e18.")
        for key in ("funding", "protocol", "quote", "signature"):
            if (
                key in trade
                and trade[key] is not None
                and (not isinstance(trade[key], str) or len(trade[key]) > 200)
            ):
                raise ValueError(f"Trade {key} must be a short string.")
        if trade.get("quote", "USD") != "USD":
            raise ValueError(
                "Imported execution prices must be in USD; quote must be USD."
            )
        if trade.get("feeSol") is not None and (
            finite(trade["feeSol"]) is None or not 0 <= trade["feeSol"] <= 1e6
        ):
            raise ValueError("feeSol must be finite and nonnegative.")
    # Do not allow arbitrary imported JSON to reach JSX or overwrite derived fields.
    allowed = (
        "id",
        "time",
        "type",
        "asset",
        "amount",
        "price",
        "funding",
        "protocol",
        "signature",
        "quote",
        "feeSol",
    )
    clean_trades = [
        {key: trade[key] for key in allowed if key in trade and trade[key] is not None}
        for trade in trades
    ]
    clean_history = [
        {key: point[key] for key in ("time", "equity", "externalFlow", "solPrice")}
        for point in history
    ]
    clean_source = {
        key: source[key]
        for key in ("name", "account", "equityScope", "cashFlowCoverage")
    }
    return {
        "history": clean_history,
        "trades": clean_trades,
        "complete": data["complete"],
        "flowTiming": "period-end",
        "source": clean_source,
    }
