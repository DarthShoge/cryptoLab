# Initial Candidate Test Results

Data directory: `data\hyperliquid\ohlcv_3y_test_chunked`
Interval: `1h`
Fee assumption: `5.00` bps per unit turnover

| candidate | markets | total_return_pct | annualized_return_pct | annualized_volatility_pct | sharpe_ratio | sortino_ratio | max_drawdown_pct | calmar_ratio | avg_turnover | total_trades | exposure_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| btc_sma_24_96 | BTC | -21.8196 | -36.8752 | 29.2448 | -1.4264 | -1.3091 | 29.0481 | -1.2695 | 0.0141 | 66 | 48.0802 |
| btc_sma_50_200 | BTC | -14.1742 | -24.8495 | 27.3068 | -0.9093 | -0.7968 | 23.4295 | -1.0606 | 0.0068 | 32 | 44.7312 |
| xsec_mom_24h_top1 | BTC,xyz:AAPL | -39.8243 | -61.2976 | 36.1967 | -2.4409 | -3.2376 | 45.1115 | -1.3588 | 0.1999 | 469 | 99.4667 |
| xsec_mom_72h_top1 | BTC,xyz:AAPL | -25.4352 | -42.2215 | 35.6057 | -1.3624 | -1.7931 | 34.6410 | -1.2188 | 0.1145 | 269 | 98.4428 |

These are smoke-test results on currently available Hyperliquid history, not production research conclusions.
