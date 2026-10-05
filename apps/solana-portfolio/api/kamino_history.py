"""Official Kamino daily buckets: protocol equity only, never full-wallet returns."""

import hashlib
import json
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib import parse, request
from urllib.error import HTTPError

import base58

API_BASE = "https://api.kamino.finance"
CACHE_DIRECTORY = Path(__file__).resolve().parents[3] / "private" / "solana-portfolio"
SOURCE = "Kamino official API · daily obligation buckets"
CACHE_VERSION = 3
HEALTH_SOURCE = (
    "Kamino refreshed liquidation limit / borrow-factor-adjusted debt · daily bucket"
)


def api_request(url):
    # Kamino rejects Python's default User-Agent for some public data endpoints.
    req = request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    )
    for attempt in range(2):
        try:
            with request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 or attempt:
                raise
            time.sleep(2)


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _timestamp(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Historical timestamp must include its timezone.")
    return int(timestamp.timestamp())


def _event_time(event):
    if event.get("timestamp") is not None:
        return int(event["timestamp"]) // 1000
    return _timestamp(event["createdOn"])


def _event_type(name):
    if name.startswith("liquidateObligation"):
        return "liquidation"
    if name.startswith("borrowObligationLiquidity"):
        return "borrow"
    if name.startswith("repayObligationLiquidity"):
        return "repay"
    if name.startswith("depositReserveLiquidityAndObligationCollateral"):
        return "deposit"
    if name.startswith("withdrawObligationCollateralAndRedeemReserveCollateral"):
        return "withdraw"
    # Liquidations and composite repayment actions retain their exact names.
    return "kamino"


def _trade(event, obligation, index):
    signature = event.get("transactionSignature", "")
    name = event.get("transactionName", "unknown")
    identity = hashlib.sha256(
        f"{obligation}:{signature}:{name}:{index}".encode()
    ).hexdigest()[:24]
    return {
        "id": f"kamino-{identity}",
        "signature": signature,
        "time": _event_time(event),
        "type": _event_type(name),
        "asset": event.get("liquidityToken") or None,
        "mint": event.get("liquidityTokenMint") or None,
        "amount": _number(event.get("liquidityTokenAmount")),
        "price": _number(event.get("liquidityTokenPrice")),
        "value": _number(event.get("liquidityUsdValue")),
        "quote": "USD",
        "funding": "unknown",
        "feeSol": None,
        "protocol": "Kamino",
        "transactionName": name,
        "liquidationRole": (
            "collateral-seized"
            if event.get("isLiquidationWithdrawal") is True
            else "debt-repaid"
            if event.get("isLiquidationWithdrawal") is False
            else "unknown"
        )
        if name.startswith("liquidateObligation")
        else None,
        "obligation": obligation,
        "source": "Kamino official indexed instruction",
    }


def _health_point(timestamp, stats):
    # ObligationStats in https://api.kamino.finance/openapi/json?openapi=3.0.0
    # distinguishes raw debt from userTotalBorrowBorrowFactorAdjusted.
    raw_debt = stats.get("userTotalBorrowBorrowFactorAdjusted")
    raw_limit = stats.get("borrowLiquidationLimit")
    debt = None if isinstance(raw_debt, bool) else _number(raw_debt)
    limit = None if isinstance(raw_limit, bool) else _number(raw_limit)
    point = {
        "time": timestamp,
        "health": None,
        "status": "unavailable",
        "adjustedDebt": debt,
        "liquidationLimit": limit,
        "source": HEALTH_SOURCE,
    }
    if debt is None or limit is None or debt < 0 or limit < 0:
        return point
    if debt == 0:
        point["status"] = "no-debt"
    else:
        ratio = _number(limit / debt)
        if ratio is not None:
            point.update(health=ratio, status="available")
    return point


def _missing_active_bucket(signals, bucket):
    # Only a recorded close or empty snapshot proves that missing data can
    # represent inactivity. Same-day labels do not establish intraday ordering.
    earlier = [active for timestamp, active in signals if timestamp < bucket]
    same_day = [active for timestamp, active in signals if timestamp == bucket]
    active = earlier[-1] if earlier else False
    return active or any(same_day) or not signals


def build_history(markets, metrics, failed_obligations=0):
    series, trades, timelines, warnings = [], [], [], []
    health_timelines = []
    invalid_records = 0
    for market in markets:
        for obligation, events in market.get("obligations", {}).items():
            signals, points, health_points = [], {}, {}
            for index, event in enumerate(events):
                try:
                    trade = _trade(event, obligation, index)
                    trades.append(trade)
                    bucket = trade["time"] // 86400 * 86400
                    signals.append(
                        (bucket, event.get("transactionName") != "closeObligation")
                    )
                except (TypeError, ValueError, KeyError, OverflowError):
                    invalid_records += 1
            for item in metrics.get(obligation, []):
                try:
                    timestamp = _timestamp(item["timestamp"])
                    point = {
                        "time": timestamp,
                        "equity": _number(
                            item.get("refreshedStats", {}).get("netAccountValue")
                        ),
                        "solPrice": _number(
                            item.get("obligationSolValues", {}).get("solPrice")
                        ),
                        "externalFlow": None,
                    }
                    points[timestamp] = point
                    health_points[timestamp] = _health_point(
                        timestamp, item.get("refreshedStats", {})
                    )
                    # A zero net value alone is not proof of zero positions.
                    if item.get("deposits") == [] and item.get("borrows") == []:
                        signals.append((timestamp, False))
                    else:
                        signals.append((timestamp, True))
                except (TypeError, ValueError, KeyError, OverflowError):
                    invalid_records += 1
            signals.sort(key=lambda item: item[0])
            history = [points[t] for t in sorted(points)]
            series.append(
                {
                    "id": obligation,
                    "label": f"Kamino obligation {len(series) + 1}",
                    "market": market["market"],
                    "history": history,
                    "healthHistory": [health_points[t] for t in sorted(health_points)],
                }
            )
            timelines.append((points, signals))
            health_timelines.append(health_points)
    buckets = sorted({t for points, _ in timelines for t in points})
    missing_daily = sum(
        max(0, (b - a) // 86400 - 1) for a, b in zip(buckets, buckets[1:])
    )
    aggregate, omitted = [], 0
    health_history = []
    for bucket in buckets:
        equity, prices, complete = 0.0, [], True
        health_available, health_complete = [], not failed_obligations
        for index, (points, signals) in enumerate(timelines):
            point = points.get(bucket)
            if point is not None:
                health_point = health_timelines[index][bucket]
                if health_point["status"] == "unavailable":
                    health_complete = False
                elif health_point["status"] == "available":
                    health_available.append((health_point, series[index]["id"]))
                if point["equity"] is None:
                    complete = False
                else:
                    equity += point["equity"]
                    if point["solPrice"] is not None:
                        prices.append(point["solPrice"])
                continue
            if _missing_active_bucket(signals, bucket):
                complete = False
                health_complete = False
        if health_complete and health_available:
            worst, obligation = min(
                health_available, key=lambda item: item[0]["health"]
            )
            health_history.append({**worst, "obligation": obligation})
        else:
            health_history.append(
                {
                    "time": bucket,
                    "health": None,
                    "status": "no-debt" if health_complete else "unavailable",
                    "adjustedDebt": 0 if health_complete else None,
                    "liquidationLimit": None,
                    "source": HEALTH_SOURCE,
                }
            )
        if complete and not failed_obligations:
            aggregate.append(
                {
                    "time": bucket,
                    "equity": equity,
                    "solPrice": prices[0]
                    if prices and max(prices) - min(prices) < 1e-6
                    else None,
                    "externalFlow": None,
                }
            )
        else:
            omitted += 1
    warnings.append(
        "Kamino equity excludes wallet balances and other protocols. Dates are provider daily bucket labels, not verified point-in-time midnight balances."
    )
    warnings.append(
        "External cash flows and historical cost basis are unverified; returns and realized P&L are unavailable."
    )
    warnings.append(
        "Historical health uses refreshed liquidation limits divided by borrow-factor-adjusted debt. Combined health is the worst known obligation only when all active obligations have risk data; no-debt and unavailable are distinct states."
    )
    if missing_daily:
        warnings.append(
            f"{missing_daily} daily buckets are absent from all returned obligation series between their first and last dates; these dates are not interpolated."
        )
    if omitted:
        warnings.append(
            f"{omitted} available dates were omitted from combined Kamino equity because an active obligation lacked a valued bucket; missing values are not zero-filled or carried forward."
        )
    if any(
        points
        and signals
        and signals[-1][1]
        and max(points) < (buckets[-1] if buckets else 0)
        for points, signals in timelines
    ):
        warnings.append(
            "An obligation's history ends before another obligation's history; a withdrawal alone does not confirm closure, so its later equity remains unknown."
        )
    if invalid_records:
        warnings.append(
            f"{invalid_records} malformed historical records could not be normalized."
        )
    return {
        "history": aggregate,
        "healthHistory": health_history,
        "series": series,
        "trades": sorted(trades, key=lambda t: t["time"], reverse=True),
        "coverage": {
            "scope": "kamino",
            "bucket": "day",
            "pointInTime": False,
            "obligations": len(series),
            "eventCount": len(trades),
            "omittedBuckets": omitted,
            "missingDailyBuckets": missing_daily,
            "failedObligations": failed_obligations,
        },
        "warnings": warnings,
        "source": SOURCE,
        "historyComplete": False,
        "returnsComplete": False,
        "asOf": int(time.time()),
    }


def load_kamino_history(wallet):
    try:
        if (
            not isinstance(wallet, str)
            or not 32 <= len(wallet) <= 44
            or len(base58.b58decode(wallet)) != 32
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("Enter a valid Solana wallet address.") from None
    cache = CACHE_DIRECTORY / f"mainnet-{wallet}-kamino-history.json"
    try:
        saved = json.loads(cache.read_text())
        if (
            saved.get("cacheVersion") == CACHE_VERSION
            and 0 <= time.time() - saved["cachedAt"] < 3600
        ):
            return saved["data"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    markets = api_request(
        f"{API_BASE}/v3/kamino-market/users/{wallet}/transactions?env=mainnet-beta&sort=asc"
    )
    metrics, failed, failures = {}, 0, []

    def fetch(market, obligation):
        query = parse.urlencode(
            {"env": "mainnet-beta", "start": "1970-01-01T00:00:00.000Z"}
        )
        return api_request(
            f"{API_BASE}/v2/kamino-market/{market}/obligations/{obligation}/metrics/history?{query}"
        )["history"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {
            pool.submit(fetch, m["market"], obligation): obligation
            for m in markets
            for obligation in m.get("obligations", {})
        }
        for future in as_completed(jobs):
            try:
                metrics[jobs[future]] = future.result()
            except Exception as error:
                failed += 1
                status = (
                    f"HTTP {error.code}"
                    if getattr(error, "code", None)
                    else type(error).__name__
                )
                failures.append(
                    f"One historical obligation could not be loaded ({status}); combined Kamino equity is unavailable."
                )
    data = build_history(markets, metrics, failed)
    data["warnings"].extend(failures)
    try:
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=CACHE_DIRECTORY,
            prefix="kamino-history-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                {"cacheVersion": CACHE_VERSION, "cachedAt": time.time(), "data": data},
                handle,
                allow_nan=False,
            )
            temporary = Path(handle.name)
        temporary.replace(cache)
    except OSError:
        data["warnings"].append(
            "Historical data loaded, but the local one-hour cache could not be saved."
        )
    return data
