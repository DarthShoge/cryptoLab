"""Run a BTC pure perp signal backtest with Binance OHLCV data."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01", help="inclusive start date")
    parser.add_argument("--end", default=None, help="optional inclusive end date")
    parser.add_argument("--timeframe", default="1h", help="Binance OHLCV timeframe")
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument("--max-gross-exposure", type=float, default=10.0)
    parser.add_argument("--fee-rate", type=float, default=0.0006)
    parser.add_argument("--funding-rate-per-bar", type=float, default=0.0)
    parser.add_argument("--rebalance-deadband", type=float, default=0.10)
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--take-profit-pct", type=float, default=None)
    parser.add_argument("--enable-trend-overlay", action="store_true")
    parser.add_argument("--trend-agreement-exposure", type=float, default=2.0)
    parser.add_argument("--trend-breakout-exposure", type=float, default=3.0)
    parser.add_argument("--trend-max-long-exposure", type=float, default=5.0)
    parser.add_argument("--trend-breakout-lookback", type=int, default=50)
    parser.add_argument("--output-root", default="reports", help="artifact root directory")
    args = parser.parse_args()

    price_data = fetch_ohlcv(
        [OHLCVConfig(symbol="BTC/USDT", display_name="BTC")],
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        exchange_id="binance",
    )
    signals = generate_btc_signal(
        price_data,
        PureSignalConfig(
            trend_overlay_enabled=args.enable_trend_overlay,
            trend_overlay_agreement_exposure=args.trend_agreement_exposure,
            trend_overlay_breakout_exposure=args.trend_breakout_exposure,
            trend_overlay_max_long_exposure=args.trend_max_long_exposure,
            trend_overlay_breakout_lookback=args.trend_breakout_lookback,
        ),
    )
    signal_columns = ["signal"]
    if "target_exposure" in signals.columns:
        signal_columns.append("target_exposure")
    result = simulate_perp_account(
        signals[["btc_close"]],
        signals[signal_columns],
        PerpSimulatorConfig(
            starting_equity=args.starting_equity,
            max_gross_exposure=args.max_gross_exposure,
            fee_rate=args.fee_rate,
            funding_rate_per_bar=args.funding_rate_per_bar,
            rebalance_deadband=args.rebalance_deadband,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
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


if __name__ == "__main__":
    main()
