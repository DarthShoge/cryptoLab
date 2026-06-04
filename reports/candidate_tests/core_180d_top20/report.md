# Initial Candidate Test Results

Data directory: `data\hyperliquid\core_180d`
Interval: `1h`
Fee assumption: `5.00` bps per unit turnover

| candidate | markets | total_return_pct | annualized_return_pct | annualized_volatility_pct | sharpe_ratio | sortino_ratio | max_drawdown_pct | calmar_ratio | avg_turnover | total_trades | exposure_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| btc_sma_24_96 | BTC | -19.1084 | -34.9498 | 28.1242 | -1.3879 | -1.2613 | 22.6669 | -1.5419 | 0.0139 | 60 | 46.2393 |
| btc_sma_50_200 | BTC | -3.9119 | -7.7730 | 26.0136 | -0.1810 | -0.1631 | 17.4444 | -0.4456 | 0.0060 | 26 | 42.7679 |
| xsec_mom_24h_top1_reb24h | BTC,ETH,SOL,XRP,SUI,DOGE,kPEPE,BNB,WLD,LINK,ZRO,AVAX,BCH,UNI,LTC,CRV,ARB,DOT,APT,INJ | -35.8650 | -59.3714 | 115.6431 | -0.2036 | -0.2882 | 61.1908 | -0.9703 | 0.0706 | 153 | 99.4214 |
| xsec_mom_72h_top1_reb24h | BTC,ETH,SOL,XRP,SUI,DOGE,kPEPE,BNB,WLD,LINK,ZRO,AVAX,BCH,UNI,LTC,CRV,ARB,DOT,APT,INJ | 22.2514 | 50.2903 | 116.7220 | 0.9290 | 1.3179 | 57.2299 | 0.8787 | 0.0424 | 92 | 98.3106 |
| xsec_mom_168h_top3_reb168h | BTC,ETH,SOL,XRP,SUI,DOGE,kPEPE,BNB,WLD,LINK,ZRO,AVAX,BCH,UNI,LTC,CRV,ARB,DOT,APT,INJ | -17.1697 | -31.7494 | 72.3932 | -0.1658 | -0.2354 | 46.9721 | -0.6759 | 0.0083 | 25 | 96.0889 |
| xsec_mom_336h_top3_reb168h | BTC,ETH,SOL,XRP,SUI,DOGE,kPEPE,BNB,WLD,LINK,ZRO,AVAX,BCH,UNI,LTC,CRV,ARB,DOT,APT,INJ | -7.0107 | -13.7044 | 69.4900 | 0.1350 | 0.1914 | 46.3212 | -0.2959 | 0.0058 | 23 | 92.2009 |
| xsec_mom_336h_top5_reb336h | BTC,ETH,SOL,XRP,SUI,DOGE,kPEPE,BNB,WLD,LINK,ZRO,AVAX,BCH,UNI,LTC,CRV,ARB,DOT,APT,INJ | -30.7434 | -52.5222 | 62.8093 | -0.8716 | -1.1834 | 46.8019 | -1.1222 | 0.0039 | 12 | 92.2009 |
| xsec_mom_504h_top5_reb168h | BTC,ETH,SOL,XRP,SUI,DOGE,kPEPE,BNB,WLD,LINK,ZRO,AVAX,BCH,UNI,LTC,CRV,ARB,DOT,APT,INJ | -31.6816 | -53.8173 | 62.2602 | -0.9292 | -1.2431 | 49.7828 | -1.0810 | 0.0044 | 23 | 88.3129 |

These are smoke-test results on currently available Hyperliquid history, not production research conclusions.
