# Traffic-Light Insurance Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Objective

Test whether traffic-light-dependent hedge-profit routing plus recovery-light rebuy improves the current raw traffic-light frontier.

Hard valid constraints:

- Final SOL-equivalent > `420.0 SOL`
- Minimum health factor >= `1.50`

## Search Space

- Controls: raw best traffic-light shape `yellow=0.50`, `orange=1.00`, `red=1.25`, reinvest allowed from 3 green votes, both with and without `traffic_light_add_min_hf=2.50`.
- Protection routing: mild, medium, strong realized hedge PnL fractions by traffic-light state.
- Recovery rebuy: min green votes `3` or `4`, rebuy fraction `0.25` or `0.50`, SOL-collateral cap `0.025` or `0.050`.

Total scenarios: `50`.

## Verdict

Reject this candidate for now. Traffic-light insurance did not improve the frontier. The raw traffic-light control remained best by both drawdown and final SOL-equivalent at `505.214 SOL`, `79.450%` max drawdown, and `1.423` min health factor. The health-guarded control improved min health factor to `1.665`, but fell to `406.340 SOL`. Every insurance/rebuy variant missed the `>420 SOL` and `>=1.50` min-health-factor hard gate.

The main failure mode is recovery churn. Mild/medium variants fired thousands of protected-rebuy events and usually ended with no protected book left, yet final SOL collapsed. Strong variants sometimes left protected USDC stranded, but still worsened drawdown or health factor. This suggests the reserve mechanism is not solving the core path problem; it buys back too early or preserves too little exposure when the trend resumes.

## Valid Configs Ranked By Drawdown

No configs passed the hard constraints.

## Near-Valid Configs Ranked By Drawdown

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | book | rebuy | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tlg_health_control | control | 406.340 | 50772.173 | 81.728 | 64.980 |  | 2.273 | 1.665 | 0.000 | 0 | False | True |

## Top By Drawdown Regardless Of Constraints

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | book | rebuy | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tlg_raw_control | control | 505.214 | 63126.506 | 79.450 | 64.605 |  | 2.343 | 1.423 | 0.000 | 0 | False | False |
| insurance_raw_mild_recovery3_frac0.25_cap0.050 | traffic_light_insurance | 162.937 | 20358.951 | 80.473 | 57.120 |  | 2.080 | 1.432 | 0.000 | 3394 | False | False |
| insurance_raw_mild_recovery3_frac0.25_cap0.025 | traffic_light_insurance | 163.206 | 20392.561 | 80.513 | 57.146 |  | 2.079 | 1.432 | 0.000 | 3391 | False | False |
| insurance_raw_mild_recovery3_frac0.50_cap0.050 | traffic_light_insurance | 156.129 | 19508.317 | 80.768 | 59.148 |  | 2.060 | 1.432 | 0.000 | 2484 | False | False |
| insurance_raw_mild_recovery3_frac0.50_cap0.025 | traffic_light_insurance | 158.440 | 19797.061 | 81.262 | 58.775 |  | 2.066 | 1.432 | 0.000 | 2475 | False | False |
| tlg_health_control | control | 406.340 | 50772.173 | 81.728 | 64.980 |  | 2.273 | 1.665 | 0.000 | 0 | False | True |
| insurance_health_mild_recovery3_frac0.25_cap0.025 | traffic_light_insurance | 140.041 | 17498.120 | 81.998 | 60.929 |  | 2.021 | 1.730 | 0.000 | 2906 | False | False |
| insurance_health_mild_recovery3_frac0.50_cap0.050 | traffic_light_insurance | 139.601 | 17443.095 | 82.011 | 61.871 |  | 2.015 | 1.739 | 0.000 | 1954 | False | False |
| insurance_health_mild_recovery3_frac0.25_cap0.050 | traffic_light_insurance | 142.862 | 17850.631 | 82.225 | 60.882 |  | 2.026 | 1.682 | 0.000 | 2910 | False | False |
| insurance_health_mild_recovery3_frac0.50_cap0.025 | traffic_light_insurance | 141.961 | 17737.971 | 82.411 | 62.132 |  | 2.021 | 1.678 | 0.000 | 1951 | False | False |
| insurance_raw_medium_recovery3_frac0.50_cap0.025 | traffic_light_insurance | 253.585 | 31685.456 | 84.093 | 63.238 |  | 2.182 | 1.432 | 0.000 | 2273 | False | False |
| insurance_raw_medium_recovery3_frac0.50_cap0.050 | traffic_light_insurance | 163.548 | 20435.311 | 84.146 | 65.161 |  | 2.077 | 1.432 | 0.000 | 2258 | False | False |
| insurance_raw_medium_recovery3_frac0.25_cap0.050 | traffic_light_insurance | 197.581 | 24687.722 | 84.743 | 61.539 |  | 2.122 | 1.432 | 0.000 | 3396 | False | False |
| insurance_raw_medium_recovery3_frac0.25_cap0.025 | traffic_light_insurance | 197.172 | 24636.692 | 84.811 | 61.546 |  | 2.121 | 1.432 | 0.000 | 3385 | False | False |
| insurance_health_medium_recovery3_frac0.25_cap0.025 | traffic_light_insurance | 231.830 | 28967.172 | 86.109 | 63.513 |  | 2.145 | 1.554 | 0.000 | 3702 | False | False |

## Top By SOL-Equivalent

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | book | rebuy | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tlg_raw_control | control | 505.214 | 63126.506 | 79.450 | 64.605 |  | 2.343 | 1.423 | 0.000 | 0 | False | False |
| tlg_health_control | control | 406.340 | 50772.173 | 81.728 | 64.980 |  | 2.273 | 1.665 | 0.000 | 0 | False | True |
| insurance_health_mild_recovery4_frac0.50_cap0.050 | traffic_light_insurance | 395.468 | 49413.760 | 86.827 | 66.213 |  | 2.245 | 1.707 | 0.000 | 1152 | False | False |
| insurance_health_mild_recovery4_frac0.25_cap0.050 | traffic_light_insurance | 393.250 | 49136.526 | 86.880 | 66.209 |  | 2.245 | 1.676 | 0.000 | 1320 | False | False |
| insurance_health_mild_recovery4_frac0.25_cap0.025 | traffic_light_insurance | 388.088 | 48491.643 | 86.730 | 66.142 |  | 2.243 | 1.722 | 0.000 | 1318 | False | False |
| insurance_health_mild_recovery4_frac0.50_cap0.025 | traffic_light_insurance | 387.657 | 48437.752 | 86.749 | 66.143 |  | 2.242 | 1.727 | 0.000 | 1155 | False | False |
| insurance_health_strong_recovery4_frac0.25_cap0.050 | traffic_light_insurance | 380.940 | 47598.486 | 90.881 | 66.749 |  | 2.208 | 1.270 | 0.000 | 514 | False | False |
| insurance_health_strong_recovery4_frac0.25_cap0.025 | traffic_light_insurance | 362.403 | 45282.208 | 91.932 | 66.789 |  | 2.190 | 1.203 | 0.000 | 508 | False | False |
| insurance_raw_strong_recovery4_frac0.50_cap0.025 | traffic_light_insurance | 362.216 | 45258.841 | 97.226 | 66.917 |  | 2.111 | 0.966 | 172.934 | 339 | False | False |
| insurance_raw_strong_recovery4_frac0.50_cap0.050 | traffic_light_insurance | 359.217 | 44884.122 | 96.703 | 66.903 |  | 2.121 | 1.000 | 183.128 | 341 | False | False |
| insurance_raw_strong_recovery4_frac0.25_cap0.050 | traffic_light_insurance | 357.285 | 44642.699 | 96.176 | 66.866 |  | 2.131 | 1.040 | 215.362 | 341 | False | False |
| insurance_raw_strong_recovery4_frac0.25_cap0.025 | traffic_light_insurance | 354.009 | 44233.478 | 95.331 | 66.848 |  | 2.141 | 1.095 | 283.793 | 354 | False | False |
| insurance_raw_medium_recovery4_frac0.50_cap0.025 | traffic_light_insurance | 340.958 | 42602.643 | 92.480 | 66.487 |  | 2.179 | 1.210 | 94.908 | 1335 | False | False |
| insurance_health_medium_recovery4_frac0.25_cap0.050 | traffic_light_insurance | 338.690 | 42319.299 | 90.081 | 66.564 |  | 2.195 | 1.445 | 0.000 | 1588 | False | False |
| insurance_raw_medium_recovery4_frac0.25_cap0.050 | traffic_light_insurance | 337.451 | 42164.556 | 92.484 | 66.467 |  | 2.177 | 1.213 | 93.659 | 1331 | False | False |

## Readout

Best drawdown candidate: `tlg_raw_control` with 79.450% max drawdown and 505.214 SOL-equivalent.

Best SOL-equivalent candidate: `tlg_raw_control` with 505.214 SOL-equivalent and 79.450% max drawdown.

Valid configs: `0` of `50`.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/traffic_light_insurance_20260615_080434/comparison.csv`
- Valid configs: `reports/sol_supertrend_3y/traffic_light_insurance_20260615_080434/valid_configs.csv`
- Near-valid configs: `reports/sol_supertrend_3y/traffic_light_insurance_20260615_080434/near_valid_configs.csv`
- Summary JSON: `reports/sol_supertrend_3y/traffic_light_insurance_20260615_080434/summary.json`
- Per-scenario history/events were generated during the run but removed after aggregation because the raw folders were 1.2 GB.
