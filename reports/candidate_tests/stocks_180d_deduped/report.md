# Initial Candidate Test Results

Data directory: `data\hyperliquid\stocks_180d`
Interval: `1h`
Fee assumption: `5.00` bps per unit turnover

| candidate | markets | total_return_pct | annualized_return_pct | annualized_volatility_pct | sharpe_ratio | sortino_ratio | max_drawdown_pct | calmar_ratio | avg_turnover | total_trades | exposure_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xsec_mom_24h_top1_reb24h | xyz:CRCL,xyz:GOOGL,xyz:HOOD,xyz:COIN,xyz:AMD,xyz:AMZN,xyz:AAPL,xyz:CRWV,xyz:BABA,flx:NVDA,flx:TSLA | 40.2576 | 161.7467 | 78.4702 | 1.6172 | 1.9994 | 25.2032 | 6.4177 | 0.0717 | 111 | 99.1886 |
| xsec_mom_72h_top1_reb24h | xyz:CRCL,xyz:GOOGL,xyz:HOOD,xyz:COIN,xyz:AMD,xyz:AMZN,xyz:AAPL,xyz:CRWV,xyz:BABA,flx:NVDA,flx:TSLA | 134.3505 | 1027.0837 | 82.4642 | 3.3469 | 4.3622 | 21.8669 | 46.9699 | 0.0425 | 66 | 97.6306 |
| xsec_mom_168h_top3_reb168h | xyz:CRCL,xyz:GOOGL,xyz:HOOD,xyz:COIN,xyz:AMD,xyz:AMZN,xyz:AAPL,xyz:CRWV,xyz:BABA,flx:NVDA,flx:TSLA | 16.9764 | 56.2005 | 53.1660 | 1.1043 | 1.3590 | 19.7445 | 2.8464 | 0.0081 | 18 | 94.5148 |
| xsec_mom_336h_top3_reb168h | xyz:CRCL,xyz:GOOGL,xyz:HOOD,xyz:COIN,xyz:AMD,xyz:AMZN,xyz:AAPL,xyz:CRWV,xyz:BABA,flx:NVDA,flx:TSLA | 50.6504 | 220.7564 | 50.9024 | 2.5432 | 3.1034 | 19.7445 | 11.1807 | 0.0047 | 16 | 89.0620 |
| xsec_mom_336h_top5_reb336h | xyz:CRCL,xyz:GOOGL,xyz:HOOD,xyz:COIN,xyz:AMD,xyz:AMZN,xyz:AAPL,xyz:CRWV,xyz:BABA,flx:NVDA,flx:TSLA | 31.3125 | 117.0106 | 39.0460 | 2.1787 | 2.6561 | 17.4286 | 6.7137 | 0.0031 | 9 | 89.0620 |
| xsec_mom_504h_top5_reb168h | xyz:CRCL,xyz:GOOGL,xyz:HOOD,xyz:COIN,xyz:AMD,xyz:AMZN,xyz:AAPL,xyz:CRWV,xyz:BABA,flx:NVDA,flx:TSLA | 30.2858 | 112.2193 | 38.5841 | 2.1424 | 2.5175 | 20.6528 | 5.4336 | 0.0032 | 15 | 83.6092 |

These are smoke-test results on currently available Hyperliquid history, not production research conclusions.
