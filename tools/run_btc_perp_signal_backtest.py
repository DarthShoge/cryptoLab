"""Run a BTC pure perp signal backtest with exchange OHLCV data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arblab.backtest.data import OHLCVConfig, fetch_ohlcv
from arblab.perps.report import write_perp_report
from arblab.perps.signal import PureSignalConfig, generate_btc_signal
from arblab.perps.simulator import PerpSimulatorConfig, simulate_perp_account
from arblab.perps.venue import (
    PerpVenueConfig,
    binance_usdm_btcusdt_like,
    generic_linear_usdt_perp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01", help="inclusive start date")
    parser.add_argument("--end", default=None, help="optional inclusive end date")
    parser.add_argument("--exchange-id", default="binance", help="ccxt exchange id for OHLCV")
    parser.add_argument("--timeframe", default="1h", help="OHLCV timeframe")
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument("--max-gross-exposure", type=float, default=10.0)
    parser.add_argument("--fee-rate", type=float, default=0.0006)
    parser.add_argument("--funding-rate-per-bar", type=float, default=0.0)
    parser.add_argument("--funding-csv", default=None, help="event CSV with timestamp,funding_rate columns")
    parser.add_argument("--mark-price-csv", default=None, help="CSV with timestamp,mark_price columns")
    parser.add_argument(
        "--venue",
        choices=("none", "generic-linear-usdt", "binance-usdm-btcusdt-like"),
        default="none",
        help="optional generic perp venue model preset",
    )
    parser.add_argument("--rebalance-deadband", type=float, default=0.10)
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--take-profit-pct", type=float, default=None)
    parser.add_argument("--enable-trend-overlay", action="store_true")
    parser.add_argument("--trend-agreement-exposure", type=float, default=2.0)
    parser.add_argument("--trend-breakout-exposure", type=float, default=3.0)
    parser.add_argument("--trend-max-long-exposure", type=float, default=5.0)
    parser.add_argument("--trend-breakout-lookback", type=int, default=50)
    parser.add_argument("--enable-vol-flattening-overlay", action="store_true")
    parser.add_argument("--vol-flattening-leverage", type=float, default=1.05)
    parser.add_argument("--vol-flattening-window", type=int, default=168)
    parser.add_argument("--vol-flattening-lookback", type=int, default=2160)
    parser.add_argument("--vol-flattening-high-threshold", type=float, default=0.80)
    parser.add_argument("--vol-flattening-drop", type=float, default=0.25)
    parser.add_argument("--vol-flattening-recent-high-window", type=int, default=336)
    parser.add_argument("--enable-vol-target", action="store_true")
    parser.add_argument("--vol-target", type=float, default=0.50)
    parser.add_argument("--vol-target-window", type=int, default=336)
    parser.add_argument("--vol-target-long-cap", type=float, default=2.0)
    parser.add_argument("--vol-target-short-cap", type=float, default=1.0)
    parser.add_argument("--enable-vol-target-confidence", action="store_true")
    parser.add_argument("--vol-target-strong-mult", type=float, default=1.20)
    parser.add_argument("--vol-target-mixed-mult", type=float, default=1.00)
    parser.add_argument("--vol-target-weak-mult", type=float, default=0.70)
    parser.add_argument("--enable-vol-target-bull-floor", action="store_true")
    parser.add_argument("--vol-target-bull-floor", type=float, default=0.50)
    parser.add_argument("--vol-target-bull-floor-require-4h", action="store_true")
    parser.add_argument("--output-root", default="reports", help="artifact root directory")
    args = parser.parse_args()

    price_data = fetch_ohlcv(
        [OHLCVConfig(symbol="BTC/USDT", display_name="BTC")],
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        exchange_id=args.exchange_id,
    )
    signals = generate_btc_signal(
        price_data,
        PureSignalConfig(
            trend_overlay_enabled=args.enable_trend_overlay,
            trend_overlay_agreement_exposure=args.trend_agreement_exposure,
            trend_overlay_breakout_exposure=args.trend_breakout_exposure,
            trend_overlay_max_long_exposure=args.trend_max_long_exposure,
            trend_overlay_breakout_lookback=args.trend_breakout_lookback,
            vol_flattening_overlay_enabled=args.enable_vol_flattening_overlay,
            vol_flattening_overlay_leverage=args.vol_flattening_leverage,
            vol_flattening_vol_window=args.vol_flattening_window,
            vol_flattening_percentile_lookback=args.vol_flattening_lookback,
            vol_flattening_high_threshold=args.vol_flattening_high_threshold,
            vol_flattening_drop=args.vol_flattening_drop,
            vol_flattening_recent_high_window=args.vol_flattening_recent_high_window,
            vol_target_overlay_enabled=args.enable_vol_target,
            vol_target_annual_vol=args.vol_target,
            vol_target_window=args.vol_target_window,
            vol_target_long_cap=args.vol_target_long_cap,
            vol_target_short_cap=args.vol_target_short_cap,
            vol_target_confidence_enabled=args.enable_vol_target_confidence,
            vol_target_strong_trend_multiplier=args.vol_target_strong_mult,
            vol_target_mixed_trend_multiplier=args.vol_target_mixed_mult,
            vol_target_weak_trend_multiplier=args.vol_target_weak_mult,
            vol_target_bull_floor_enabled=args.enable_vol_target_bull_floor,
            vol_target_bull_floor=args.vol_target_bull_floor,
            vol_target_bull_floor_require_4h=args.vol_target_bull_floor_require_4h,
        ),
    )
    funding = _load_funding_rates(args.funding_csv, signals.index)
    if funding is not None:
        signals["funding_rate"] = funding
    price_input = signals[["btc_close"]].copy()
    mark_price = _load_mark_prices(args.mark_price_csv, signals.index)
    if mark_price is not None:
        price_input["mark_price"] = mark_price
    signal_columns = ["signal"]
    if "target_exposure" in signals.columns:
        signal_columns.append("target_exposure")
    if "funding_rate" in signals.columns:
        signal_columns.append("funding_rate")
    venue = _venue_from_args(args.venue, args.fee_rate)
    result = simulate_perp_account(
        price_input,
        signals[signal_columns],
        PerpSimulatorConfig(
            starting_equity=args.starting_equity,
            max_gross_exposure=args.max_gross_exposure,
            fee_rate=args.fee_rate,
            funding_rate_per_bar=args.funding_rate_per_bar,
            rebalance_deadband=args.rebalance_deadband,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            venue=venue,
        ),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"btc_pure_perp_signal_{timestamp}"
    written = write_perp_report(
        output_dir=output_dir,
        signals=signals,
        history=result.history,
        starting_equity=args.starting_equity,
    )
    print(written)


def _venue_from_args(name: str, fee_rate: float) -> PerpVenueConfig | None:
    if name == "generic-linear-usdt":
        return generic_linear_usdt_perp(taker_fee=fee_rate)
    if name == "binance-usdm-btcusdt-like":
        return binance_usdm_btcusdt_like()
    return None


def _load_funding_rates(path: str | None, index) -> object | None:
    if not path:
        return None
    import pandas as pd

    funding = pd.read_csv(path, parse_dates=["timestamp"])
    if not {"timestamp", "funding_rate"}.issubset(funding.columns):
        raise ValueError("funding CSV must include timestamp and funding_rate columns")
    funding = funding.set_index("timestamp")["funding_rate"].astype(float).sort_index()
    return funding.reindex(index).fillna(0.0)


def _load_mark_prices(path: str | None, index) -> object | None:
    if not path:
        return None
    import pandas as pd

    marks = pd.read_csv(path, parse_dates=["timestamp"])
    if not {"timestamp", "mark_price"}.issubset(marks.columns):
        raise ValueError("mark price CSV must include timestamp and mark_price columns")
    marks = marks.set_index("timestamp")["mark_price"].astype(float).sort_index()
    return marks.reindex(index, method="ffill")


if __name__ == "__main__":
    main()
