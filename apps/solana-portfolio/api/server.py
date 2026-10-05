"""Local HTTP bridge for the React app. Bind to loopback; never signs transactions."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .accounting import account_summary, history_metrics, validate_history
from .demo import DEMO_END, demo_candles, demo_portfolio
from .indicators import supertrend
from .timeframes import (
    HISTORY_START,
    aggregate_candles,
    candle_start,
    combined_supertrend,
    timeframe_supertrend,
)
from .providers import (
    ASSETS,
    INTERVALS,
    default_wallet,
    live_portfolio,
    market_candles,
    merge_activity,
    safe_error,
    validate_wallet,
)

_cache = {}
_candles_cache = {}
_lock = threading.Lock()


def _source_candles(mode, asset, interval, days, end_time, start_time):
    key = (mode, asset, interval, days, end_time, start_time)
    cached = _candles_cache.get(key)
    if cached and time.monotonic() - cached[0] < 300:
        return cached[1]
    candles = (
        demo_candles(asset, interval, days, end_time, start_time)
        if mode == "demo"
        else market_candles(asset, interval, days, end_time, start_time)
    )
    _candles_cache[key] = (time.monotonic(), candles)
    if len(_candles_cache) > 32:
        _candles_cache.pop(next(iter(_candles_cache)))
    return candles


def chart_data(
    mode,
    asset,
    interval,
    days,
    period,
    multiplier,
    end_time=None,
    start_time=None,
    combined=False,
):
    """Build one display series with optional causal, equal-weight consensus."""
    if (
        mode not in ("demo", "live", "imported")
        or asset not in ASSETS
        or interval not in INTERVALS
        or days not in (0, 7, 30, 90, 365)
    ):
        raise ValueError("Unsupported chart parameters.")
    supertrend([], period, multiplier)
    if end_time is not None and not 0 < end_time <= 4102444800:
        raise ValueError("Invalid historical chart end time.")
    as_of = min(
        DEMO_END if mode == "demo" else int(time.time()), end_time or 4102444800
    )
    # Normalize source cutoff for a reusable cache shared across display ranges
    # and timeframes. Daily history from 2020 gives monthly ATR adequate warmup.
    daily_end = candle_start(as_of, "1d")
    if daily_end <= HISTORY_START:
        raise ValueError("Historical chart end must be after January 1, 2020.")
    if start_time is not None and not HISTORY_START <= start_time < as_of:
        raise ValueError("Invalid historical chart start time.")
    if days == 0 and mode != "demo" and start_time is None:
        raise ValueError("An inception chart requires a start time.")
    daily = None
    if interval in ("1w", "1M") or combined:
        daily = _source_candles(mode, asset, "1d", 0, daily_end, HISTORY_START)
    if interval in ("1w", "1M") or (interval == "1d" and daily is not None):
        all_candles = (
            daily if interval == "1d" else aggregate_candles(daily, interval, daily_end)
        )
        indicators = timeframe_supertrend(all_candles, interval, period, multiplier)
        cutoff = candle_start(
            start_time if days == 0 and start_time else as_of - (days or 90) * 86400,
            interval,
        )
        candles = [c for c in all_candles if c["time"] >= cutoff]
        indicator = [p for p in indicators if p["time"] >= cutoff]
    else:
        candles = _source_candles(
            mode,
            asset,
            interval,
            days or (90 if mode == "demo" else 0),
            end_time,
            start_time,
        )
        indicator = timeframe_supertrend(candles, interval, period, multiplier)
    result = {
        "candles": candles,
        "indicator": indicator,
        "source": "Fictional demo candles"
        if mode == "demo"
        else f"Coinbase {asset}/USD · UTC · closed candles",
        "asset": asset,
        "interval": interval,
    }
    if combined:
        result["combined"] = combined_supertrend(
            candles, daily, interval, period, multiplier, as_of
        )
        result["combinedSource"] = (
            "Equal-weight 1d / Monday-UTC 1w / calendar 1M · completed candles only"
        )
    return result


def cached_live(wallet, refresh=False):
    # Serialize account refreshes and reuse the same snapshot for a minute.
    with _lock:
        entry = _cache.get(wallet)
        if not refresh and entry and time.monotonic() - entry[0] < 60:
            return entry[1]
        data = live_portfolio(wallet)
        _cache[wallet] = (time.monotonic(), data)
        if len(_cache) > 8:
            del _cache[next(iter(_cache))]
        return data


def enrich(data):
    history = data["history"]
    anchor = history[-1]["time"] if history else 0
    return {
        **data,
        "metrics": history_metrics(history, data["historyComplete"]),
        "metricsByRange": {
            str(days): history_metrics(
                [p for p in history if p["time"] >= anchor - days * 86400],
                data["historyComplete"],
            )
            for days in (7, 30, 90, 365)
        },
    }


def imported_portfolio(payload):
    data = validate_history(payload)
    summary = account_summary([], [])
    summary.update(
        netEquity=data["history"][-1]["equity"],
        knownEquity=data["history"][-1]["equity"],
        supplied=None,
        debt=None,
        walletValue=None,
    )
    trades = [
        {
            "protocol": "Imported",
            "funding": "unknown",
            "signature": None,
            "quote": "USD",
            **trade,
            "value": trade["amount"] * trade["price"],
        }
        for trade in data["trades"]
    ]
    return enrich(
        {
            "mode": "imported",
            "wallet": None,
            "asOf": data["history"][-1]["time"],
            "summary": summary,
            "history": data["history"],
            "trades": trades,
            "historyComplete": data["complete"],
            "prices": {"SOL": data["history"][-1]["solPrice"]},
            "source": f"Imported: {data['source']['name']} · {data['source']['account']} · user-declared coverage",
            "warnings": [
                "Imported equity and execution records do not establish current wallet positions or Kamino health. Returns assume external flows at each period end."
            ],
        }
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        # Do not log query strings containing private wallet addresses.
        pass

    def respond(self, value, status=200):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/api/health":
                self.respond({"ok": True})
            elif parsed.path == "/api/config":
                self.respond(
                    {"defaultWallet": default_wallet(), "cluster": "mainnet-beta"}
                )
            elif parsed.path == "/api/portfolio":
                mode = query.get("mode", "demo")
                refresh = query.get("refresh") == "1"
                if mode not in ("demo", "live"):
                    raise ValueError("Choose demo or live mode.")
                wallet = (
                    validate_wallet(query.get("wallet") or default_wallet())
                    if mode == "live"
                    else None
                )
                data = enrich(
                    demo_portfolio()
                    if mode == "demo"
                    else cached_live(wallet, refresh=refresh)
                )
                if refresh:
                    _candles_cache.clear()
                self.respond(data)
            elif parsed.path == "/api/history":
                from .history_index import read_history

                wallet = validate_wallet(query.get("wallet") or default_wallet())
                result = read_history(wallet)
                try:
                    from .kamino_history import load_kamino_history

                    protocol = load_kamino_history(wallet)["trades"]
                except Exception:
                    protocol = []
                self.respond(
                    {
                        "trades": merge_activity(result["records"], protocol),
                        "historyStatus": result["status"],
                    }
                )
            elif parsed.path == "/api/chart":
                mode = query.get("mode", "demo")
                asset, interval = query.get("asset", "SOL"), query.get("interval", "1d")
                days, period = int(query.get("days", 90)), int(query.get("period", 10))
                multiplier = float(query.get("multiplier", 3))
                end_time = int(query["end"]) if "end" in query else None
                start_time = int(query["start"]) if "start" in query else None
                if query.get("combined", "0") not in ("0", "1"):
                    raise ValueError("combined must be 0 or 1.")
                self.respond(
                    chart_data(
                        mode,
                        asset,
                        interval,
                        days,
                        period,
                        multiplier,
                        end_time,
                        start_time,
                        query.get("combined") == "1",
                    )
                )
            else:
                self.respond({"error": "Route not found."}, 404)
        except ValueError as error:
            self.respond({"error": str(error)}, 400)
        except Exception as error:
            self.respond(
                {
                    "error": f"Provider unavailable ({safe_error(error)}). Check server RPC settings or retry."
                },
                502,
            )

    def do_POST(self):
        if self.path != "/api/import":
            self.respond({"error": "Route not found."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("Upload a JSON history smaller than 2 MB.")
            self.respond(imported_portfolio(json.loads(self.rfile.read(length))))
        except (ValueError, KeyError, TypeError) as error:
            self.respond({"error": str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Portfolio API: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
