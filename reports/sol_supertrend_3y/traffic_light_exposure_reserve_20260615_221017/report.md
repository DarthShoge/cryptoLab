# Traffic-Light Exposure Reserve Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Objective

Test whether direct traffic-light SOL exposure reduction improves drawdown while preserving the `>420 SOL` objective.

Hard valid constraints:

- Final SOL-equivalent > `420.0 SOL`
- Minimum health factor >= `1.50`

## Search Space

- Controls: raw and health-guarded traffic-light shapes.
- Sell routing: mild, medium, strong SOL reserve fractions by yellow/orange/red state.
- Reserve cap: `15%` or `30%` of SOL collateral value.
- Recovery rebuy: min green votes `3` or `4`, rebuy fraction `0.25` or `0.50`.

Total scenarios: `50`.

## Verdict

Reject this candidate for now. Direct traffic-light exposure reserve did not improve the frontier. The raw traffic-light control remained best by drawdown and final SOL-equivalent at `505.214 SOL`, `79.450%` max drawdown, and `1.423` min health factor. The health-guarded control remained the only near-valid result at `406.340 SOL`, `81.728%` max drawdown, and `1.665` min health factor.

The exposure-reserve variants sold too much SOL too often and did not rebuy fast enough. Many runs ended with large stranded reserves, commonly `$7k` to `$15k`, while final SOL-equivalent collapsed into the `118-194 SOL` range. The sell/rebuy counts also show heavy churn. This means the four-light sell rule is acting like a blunt cash rotation, not a drawdown governor.

## Valid Configs Ranked By Drawdown

No configs passed the hard constraints.

## Near-Valid Configs Ranked By Drawdown

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | reserve | sell/rebuy | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tlg_health_control | control | 406.340 | 50772.173 | 81.728 | 64.980 |  | 2.273 | 1.665 | 0.000 | 0/0 | False | True |

## Top By Drawdown Regardless Of Constraints

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | reserve | sell/rebuy | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tlg_raw_control | control | 505.214 | 63126.506 | 79.450 | 64.605 |  | 2.343 | 1.423 | 0.000 | 0/0 | False | False |
| tlg_health_control | control | 406.340 | 50772.173 | 81.728 | 64.980 |  | 2.273 | 1.665 | 0.000 | 0/0 | False | True |
| exposure_raw_strong_max0.15_recovery4_frac0.25 | traffic_light_exposure_reserve | 134.757 | 16837.886 | 82.935 | 75.291 |  | 2.010 | 1.326 | 8353.606 | 48/420 | False | False |
| exposure_raw_medium_max0.15_recovery4_frac0.25 | traffic_light_exposure_reserve | 132.060 | 16500.899 | 83.006 | 75.754 |  | 1.997 | 1.319 | 8077.898 | 89/504 | False | False |
| exposure_raw_mild_max0.15_recovery4_frac0.25 | traffic_light_exposure_reserve | 138.291 | 17279.444 | 83.120 | 75.047 |  | 2.011 | 1.332 | 8186.131 | 139/498 | False | False |
| exposure_raw_strong_max0.15_recovery3_frac0.25 | traffic_light_exposure_reserve | 127.984 | 15991.639 | 83.183 | 76.129 |  | 1.987 | 1.302 | 7582.511 | 103/989 | False | False |
| exposure_raw_medium_max0.15_recovery3_frac0.25 | traffic_light_exposure_reserve | 131.150 | 16387.246 | 83.204 | 76.141 |  | 1.994 | 1.295 | 7831.698 | 136/982 | False | False |
| exposure_health_strong_max0.30_recovery4_frac0.25 | traffic_light_exposure_reserve | 118.093 | 14755.723 | 83.231 | 75.484 |  | 1.964 | 1.395 | 12277.394 | 84/412 | False | False |
| exposure_raw_strong_max0.15_recovery4_frac0.50 | traffic_light_exposure_reserve | 150.900 | 18854.978 | 83.279 | 70.346 |  | 2.047 | 1.370 | 7554.919 | 51/352 | False | False |
| exposure_raw_mild_max0.30_recovery4_frac0.25 | traffic_light_exposure_reserve | 155.852 | 19473.748 | 83.288 | 71.696 |  | 2.055 | 1.411 | 14203.512 | 156/384 | False | False |
| exposure_raw_strong_max0.30_recovery4_frac0.25 | traffic_light_exposure_reserve | 190.683 | 23825.821 | 83.292 | 66.130 |  | 2.114 | 1.432 | 15121.064 | 91/489 | False | False |
| exposure_raw_mild_max0.15_recovery4_frac0.50 | traffic_light_exposure_reserve | 138.623 | 17320.990 | 83.313 | 75.328 |  | 2.010 | 1.327 | 8309.286 | 147/495 | False | False |
| exposure_raw_medium_max0.30_recovery4_frac0.25 | traffic_light_exposure_reserve | 156.193 | 19516.284 | 83.352 | 71.706 |  | 2.055 | 1.413 | 14152.996 | 105/413 | False | False |
| exposure_raw_medium_max0.15_recovery4_frac0.50 | traffic_light_exposure_reserve | 136.630 | 17071.864 | 83.425 | 75.408 |  | 2.006 | 1.325 | 8229.381 | 89/504 | False | False |
| exposure_raw_strong_max0.15_recovery3_frac0.50 | traffic_light_exposure_reserve | 146.059 | 18250.078 | 83.430 | 73.535 |  | 2.028 | 1.352 | 7657.091 | 93/934 | False | False |

## Top By SOL-Equivalent

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | reserve | sell/rebuy | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tlg_raw_control | control | 505.214 | 63126.506 | 79.450 | 64.605 |  | 2.343 | 1.423 | 0.000 | 0/0 | False | False |
| tlg_health_control | control | 406.340 | 50772.173 | 81.728 | 64.980 |  | 2.273 | 1.665 | 0.000 | 0/0 | False | True |
| exposure_raw_medium_max0.30_recovery4_frac0.50 | traffic_light_exposure_reserve | 193.624 | 24193.313 | 83.483 | 66.619 |  | 2.118 | 1.432 | 15584.932 | 119/484 | False | False |
| exposure_raw_strong_max0.30_recovery4_frac0.50 | traffic_light_exposure_reserve | 191.060 | 23872.938 | 83.456 | 66.418 |  | 2.113 | 1.432 | 15671.621 | 93/500 | False | False |
| exposure_raw_strong_max0.30_recovery4_frac0.25 | traffic_light_exposure_reserve | 190.683 | 23825.821 | 83.292 | 66.130 |  | 2.114 | 1.432 | 15121.064 | 91/489 | False | False |
| exposure_raw_mild_max0.30_recovery4_frac0.50 | traffic_light_exposure_reserve | 176.650 | 22072.380 | 83.544 | 68.908 |  | 2.091 | 1.432 | 14620.527 | 162/437 | False | False |
| exposure_raw_mild_max0.30_recovery3_frac0.50 | traffic_light_exposure_reserve | 168.614 | 21068.326 | 84.064 | 69.644 |  | 2.076 | 1.432 | 14102.579 | 258/942 | False | False |
| exposure_raw_strong_max0.30_recovery3_frac0.50 | traffic_light_exposure_reserve | 163.933 | 20483.453 | 83.895 | 69.444 |  | 2.067 | 1.432 | 13829.924 | 152/936 | False | False |
| exposure_raw_medium_max0.30_recovery3_frac0.50 | traffic_light_exposure_reserve | 161.331 | 20158.342 | 84.097 | 71.820 |  | 2.065 | 1.387 | 14747.704 | 172/779 | False | False |
| exposure_raw_medium_max0.30_recovery4_frac0.25 | traffic_light_exposure_reserve | 156.193 | 19516.284 | 83.352 | 71.706 |  | 2.055 | 1.413 | 14152.996 | 105/413 | False | False |
| exposure_raw_mild_max0.30_recovery4_frac0.25 | traffic_light_exposure_reserve | 155.852 | 19473.748 | 83.288 | 71.696 |  | 2.055 | 1.411 | 14203.512 | 156/384 | False | False |
| exposure_raw_medium_max0.30_recovery3_frac0.25 | traffic_light_exposure_reserve | 152.875 | 19101.766 | 83.809 | 71.655 |  | 2.050 | 1.413 | 13855.552 | 178/805 | False | False |
| exposure_raw_strong_max0.15_recovery4_frac0.50 | traffic_light_exposure_reserve | 150.900 | 18854.978 | 83.279 | 70.346 |  | 2.047 | 1.370 | 7554.919 | 51/352 | False | False |
| exposure_raw_mild_max0.30_recovery3_frac0.25 | traffic_light_exposure_reserve | 150.698 | 18829.736 | 83.952 | 70.904 |  | 2.047 | 1.419 | 13475.015 | 260/978 | False | False |
| exposure_raw_medium_max0.15_recovery3_frac0.50 | traffic_light_exposure_reserve | 149.498 | 18679.786 | 83.711 | 73.461 |  | 2.034 | 1.352 | 7855.232 | 136/930 | False | False |

## Readout

Best drawdown candidate: `tlg_raw_control` with 79.450% max drawdown and 505.214 SOL-equivalent.

Best SOL-equivalent candidate: `tlg_raw_control` with 505.214 SOL-equivalent and 79.450% max drawdown.

Valid configs: `0` of `50`.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/traffic_light_exposure_reserve_20260615_221017/comparison.csv`
- Valid configs: `reports/sol_supertrend_3y/traffic_light_exposure_reserve_20260615_221017/valid_configs.csv`
- Near-valid configs: `reports/sol_supertrend_3y/traffic_light_exposure_reserve_20260615_221017/near_valid_configs.csv`
- Summary JSON: `reports/sol_supertrend_3y/traffic_light_exposure_reserve_20260615_221017/summary.json`
