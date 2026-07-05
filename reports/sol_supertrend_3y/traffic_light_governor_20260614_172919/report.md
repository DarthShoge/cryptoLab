# Traffic-Light Governor Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Objective

Test the first traffic-light exposure-governor slice.

Hard valid constraints:

- Final SOL-equivalent > `420.0 SOL`
- Minimum health factor >= `1.50`

Investor target:

- Max drawdown <= `50%`

## Search Space

- Yellow hedge floor: `0.35`, `0.50`, `0.75`
- Orange hedge floor: `0.75`, `1.00`, `1.25`
- Red hedge floor: `1.00`, `1.25`, `1.50`
- Minimum green votes for surplus reinvestment: `3`, `4`
- Minimum green votes for USDC releverage: `4`
- Existing fast-break v2 overlay kept constant.

Total scenarios: `49`.

## Valid Configs Ranked By Drawdown

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |

## Near-Valid Configs Ranked By Drawdown

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |

## Best Per Family

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |
| tlg_y0.75_o1.25_r1.25_reinv3 | traffic_light_exposure | 492.141 | 61493.044 | 79.234 | 64.378 | 779.1 | 2.334 | 1.383 | False | False |

## Top By Drawdown Regardless Of Constraints

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tlg_y0.75_o1.25_r1.25_reinv3 | traffic_light_exposure | 492.141 | 61493.044 | 79.234 | 64.378 | 779.1 | 2.334 | 1.383 | False | False |
| tlg_y0.50_o1.00_r1.25_reinv3 | traffic_light_exposure | 505.214 | 63126.506 | 79.450 | 64.605 | 776.9 | 2.343 | 1.423 | False | False |
| tlg_y0.50_o1.25_r1.25_reinv3 | traffic_light_exposure | 473.166 | 59122.134 | 80.296 | 64.504 | 779.7 | 2.328 | 1.429 | False | False |
| tlg_y0.35_o1.25_r1.25_reinv3 | traffic_light_exposure | 448.556 | 56047.081 | 81.038 | 64.479 | 790.7 | 2.323 | 1.430 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv4 | traffic_light_exposure | 456.664 | 57060.141 | 81.049 | 64.490 | 775.4 | 2.312 | 1.430 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv4 | traffic_light_exposure | 458.360 | 57272.071 | 81.189 | 64.650 | 774.9 | 2.306 | 1.451 | False | False |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |
| tlg_y0.35_o0.75_r1.25_reinv4 | traffic_light_exposure | 450.201 | 56252.653 | 81.379 | 64.516 | 786.4 | 2.317 | 1.428 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv4 | traffic_light_exposure | 458.756 | 57321.563 | 81.468 | 64.715 | 775.0 | 2.306 | 1.448 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv3 | traffic_light_exposure | 438.078 | 54737.816 | 81.771 | 64.610 | 776.1 | 2.295 | 1.462 | False | False |
| tlg_y0.75_o1.00_r1.00_reinv3 | traffic_light_exposure | 430.939 | 53845.767 | 81.899 | 64.554 | 780.9 | 2.299 | 1.388 | False | False |
| tlg_y0.35_o1.00_r1.25_reinv3 | traffic_light_exposure | 416.138 | 51996.465 | 81.906 | 64.351 | 793.3 | 2.303 | 1.437 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv3 | traffic_light_exposure | 451.525 | 56418.084 | 81.909 | 64.738 | 775.1 | 2.301 | 1.442 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv3 | traffic_light_exposure | 436.234 | 54507.400 | 82.055 | 64.526 | 777.0 | 2.299 | 1.427 | False | False |
| tlg_y0.50_o0.75_r1.00_reinv3 | traffic_light_exposure | 422.863 | 52836.682 | 82.395 | 64.476 | 777.1 | 2.292 | 1.430 | False | False |

## Top By SOL-Equivalent

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tlg_y0.50_o1.00_r1.25_reinv3 | traffic_light_exposure | 505.214 | 63126.506 | 79.450 | 64.605 | 776.9 | 2.343 | 1.423 | False | False |
| tlg_y0.75_o1.25_r1.25_reinv3 | traffic_light_exposure | 492.141 | 61493.044 | 79.234 | 64.378 | 779.1 | 2.334 | 1.383 | False | False |
| tlg_y0.50_o1.25_r1.25_reinv3 | traffic_light_exposure | 473.166 | 59122.134 | 80.296 | 64.504 | 779.7 | 2.328 | 1.429 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv4 | traffic_light_exposure | 458.756 | 57321.563 | 81.468 | 64.715 | 775.0 | 2.306 | 1.448 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv4 | traffic_light_exposure | 458.360 | 57272.071 | 81.189 | 64.650 | 774.9 | 2.306 | 1.451 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv4 | traffic_light_exposure | 456.664 | 57060.141 | 81.049 | 64.490 | 775.4 | 2.312 | 1.430 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv3 | traffic_light_exposure | 451.525 | 56418.084 | 81.909 | 64.738 | 775.1 | 2.301 | 1.442 | False | False |
| tlg_y0.35_o0.75_r1.25_reinv4 | traffic_light_exposure | 450.201 | 56252.653 | 81.379 | 64.516 | 786.4 | 2.317 | 1.428 | False | False |
| tlg_y0.35_o1.25_r1.25_reinv3 | traffic_light_exposure | 448.556 | 56047.081 | 81.038 | 64.479 | 790.7 | 2.323 | 1.430 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv3 | traffic_light_exposure | 438.078 | 54737.816 | 81.771 | 64.610 | 776.1 | 2.295 | 1.462 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv3 | traffic_light_exposure | 436.234 | 54507.400 | 82.055 | 64.526 | 777.0 | 2.299 | 1.427 | False | False |
| tlg_y0.75_o1.00_r1.00_reinv3 | traffic_light_exposure | 430.939 | 53845.767 | 81.899 | 64.554 | 780.9 | 2.299 | 1.388 | False | False |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |
| tlg_y0.50_o0.75_r1.00_reinv3 | traffic_light_exposure | 422.863 | 52836.682 | 82.395 | 64.476 | 777.1 | 2.292 | 1.430 | False | False |
| tlg_y0.50_o1.00_r1.00_reinv3 | traffic_light_exposure | 420.901 | 52591.557 | 82.776 | 64.580 | 781.2 | 2.296 | 1.424 | False | False |

## Readout

The first traffic-light exposure-governor slice moved the frontier, but did not produce a promotable candidate under the full hard-gate set.

Only the control passed both hard validity gates. The traffic-light variants produced many `>420 SOL` outcomes and several improved drawdown, but all of those fell below the `1.50` minimum health-factor gate. This is important: the traffic-light idea is not dead, but the first implementation buys better SOL accumulation and mildly better drawdown by spending too much balance-sheet safety.

Best raw drawdown result:

- `tlg_y0.75_o1.25_r1.25_reinv3`
- Final SOL-equivalent: `492.141 SOL`
- Max drawdown: `79.234%`
- 2024+ drawdown: `64.378%`
- Sortino: `2.334`
- Min health factor: `1.383`

Best raw SOL-equivalent result:

- `tlg_y0.50_o1.00_r1.25_reinv3`
- Final SOL-equivalent: `505.214 SOL`
- Max drawdown: `79.450%`
- 2024+ drawdown: `64.605%`
- Sortino: `2.343`
- Min health factor: `1.423`

Compared with the control, the best raw traffic-light rows gained roughly `67-80 SOL` and improved max drawdown by about `1.9-2.1` percentage points, but they still left the strategy near `79%` drawdown. This is nowhere close to the institutional `50%` drawdown target.

The practical lesson is sharper than the headline result: traffic-light governance helps the compounding engine when it keeps reinvestment available at `3 green` and pushes stronger hedge floors into orange/red states. But because those stronger floors add ETH debt against a volatile SOL collateral base, the account later violates the health-factor safety gate. The next experiment should not simply raise floors further. It should add a traffic-light health governor:

- cap or skip new hedge adds when projected health factor is not high enough for the current light state;
- force USDC debt cleanup or partial ETH cover when live health factor falls below a warning band;
- keep the `>420 SOL` gate unchanged;
- rerun the best raw floor shapes with explicit health-defense behavior.

In short: the traffic-light model improved return and slightly improved drawdown, but it did so by leaning into unsafe debt. The next iteration needs to make the lights govern balance-sheet safety, not just target hedge ratio.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919/comparison.csv`
- Valid configs CSV: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919/valid_configs.csv`
- Near-valid configs CSV: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919/near_valid_configs.csv`
- All by drawdown CSV: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919/all_by_drawdown.csv`
- All by SOL CSV: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919/all_by_sol.csv`
- Family best CSV: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919/family_best.csv`
- Summary JSON: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919`
