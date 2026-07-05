# Traffic-Light Health-Constrained Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Objective

Retest the most promising raw traffic-light shapes with stricter projected-health-factor floors for traffic-light hedge additions.

Hard valid constraints:

- Final SOL-equivalent > `420.0 SOL`
- Minimum health factor >= `1.50`

## Search Space

- Seven promising traffic-light floor shapes from the first governor sweep.
- Traffic-light hedge-add projected min HF: `2.50`, `3.00`, `3.50`.
- Existing fast-break v2 overlay kept constant.

Total scenarios: `22`.

## Valid Configs Ranked By Drawdown

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |

## Near-Valid Configs Ranked By Drawdown

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |
| tlg_y0.50_o1.00_r1.25_reinv3_addhf2.50 | traffic_light_health | 406.340 | 50772.173 | 81.728 | 64.980 | 785.7 | 2.273 | 1.665 | False | True |

## Best Per Family

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |
| tlg_y0.50_o1.00_r1.25_reinv3_addhf2.50 | traffic_light_health | 406.340 | 50772.173 | 81.728 | 64.980 | 785.7 | 2.273 | 1.665 | False | True |

## Top By Drawdown Regardless Of Constraints

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |
| tlg_y0.50_o1.00_r1.25_reinv3_addhf2.50 | traffic_light_health | 406.340 | 50772.173 | 81.728 | 64.980 | 785.7 | 2.273 | 1.665 | False | True |
| tlg_y0.35_o1.25_r1.25_reinv3_addhf2.50 | traffic_light_health | 380.371 | 47527.380 | 82.341 | 64.937 | 786.0 | 2.258 | 1.714 | False | False |
| tlg_y0.50_o1.25_r1.25_reinv3_addhf2.50 | traffic_light_health | 390.852 | 48836.942 | 82.724 | 65.047 | 786.0 | 2.262 | 1.658 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv4_addhf3.50 | traffic_light_health | 322.462 | 40291.574 | 85.008 | 65.175 | 807.0 | 2.210 | 1.680 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv4_addhf3.00 | traffic_light_health | 311.485 | 38920.037 | 85.506 | 65.171 | 809.3 | 2.202 | 1.681 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv4_addhf3.00 | traffic_light_health | 309.664 | 38692.463 | 85.511 | 65.154 | 809.3 | 2.200 | 1.681 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv4_addhf3.00 | traffic_light_health | 309.664 | 38692.463 | 85.511 | 65.154 | 809.3 | 2.200 | 1.681 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv4_addhf3.50 | traffic_light_health | 306.842 | 38339.905 | 85.527 | 65.131 | 809.4 | 2.197 | 1.680 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv4_addhf3.50 | traffic_light_health | 306.842 | 38339.905 | 85.527 | 65.131 | 809.4 | 2.197 | 1.680 | False | False |
| tlg_y0.50_o1.00_r1.25_reinv3_addhf3.50 | traffic_light_health | 278.011 | 34737.531 | 86.520 | 65.045 | 810.6 | 2.173 | 1.680 | False | False |
| tlg_y0.50_o1.00_r1.25_reinv3_addhf3.00 | traffic_light_health | 283.888 | 35471.747 | 86.573 | 65.121 | 810.3 | 2.178 | 1.681 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv4_addhf2.50 | traffic_light_health | 281.766 | 35206.650 | 86.647 | 64.769 | 810.6 | 2.174 | 1.685 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv4_addhf2.50 | traffic_light_health | 245.331 | 30654.055 | 88.024 | 64.819 | 862.4 | 2.143 | 1.562 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv4_addhf2.50 | traffic_light_health | 245.331 | 30654.055 | 88.024 | 64.819 | 862.4 | 2.143 | 1.562 | False | False |

## Top By SOL-Equivalent

| scenario | family | SOL-eq | USD | maxDD | 2024DD | halfRec | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 785.6 | 2.290 | 1.598 | True | True |
| tlg_y0.50_o1.00_r1.25_reinv3_addhf2.50 | traffic_light_health | 406.340 | 50772.173 | 81.728 | 64.980 | 785.7 | 2.273 | 1.665 | False | True |
| tlg_y0.50_o1.25_r1.25_reinv3_addhf2.50 | traffic_light_health | 390.852 | 48836.942 | 82.724 | 65.047 | 786.0 | 2.262 | 1.658 | False | False |
| tlg_y0.35_o1.25_r1.25_reinv3_addhf2.50 | traffic_light_health | 380.371 | 47527.380 | 82.341 | 64.937 | 786.0 | 2.258 | 1.714 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv4_addhf3.50 | traffic_light_health | 322.462 | 40291.574 | 85.008 | 65.175 | 807.0 | 2.210 | 1.680 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv4_addhf3.00 | traffic_light_health | 311.485 | 38920.037 | 85.506 | 65.171 | 809.3 | 2.202 | 1.681 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv4_addhf3.00 | traffic_light_health | 309.664 | 38692.463 | 85.511 | 65.154 | 809.3 | 2.200 | 1.681 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv4_addhf3.00 | traffic_light_health | 309.664 | 38692.463 | 85.511 | 65.154 | 809.3 | 2.200 | 1.681 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv4_addhf3.50 | traffic_light_health | 306.842 | 38339.905 | 85.527 | 65.131 | 809.4 | 2.197 | 1.680 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv4_addhf3.50 | traffic_light_health | 306.842 | 38339.905 | 85.527 | 65.131 | 809.4 | 2.197 | 1.680 | False | False |
| tlg_y0.50_o1.00_r1.25_reinv3_addhf3.00 | traffic_light_health | 283.888 | 35471.747 | 86.573 | 65.121 | 810.3 | 2.178 | 1.681 | False | False |
| tlg_y0.50_o0.75_r1.25_reinv4_addhf2.50 | traffic_light_health | 281.766 | 35206.650 | 86.647 | 64.769 | 810.6 | 2.174 | 1.685 | False | False |
| tlg_y0.50_o1.00_r1.25_reinv3_addhf3.50 | traffic_light_health | 278.011 | 34737.531 | 86.520 | 65.045 | 810.6 | 2.173 | 1.680 | False | False |
| tlg_y0.75_o0.75_r1.25_reinv4_addhf2.50 | traffic_light_health | 245.331 | 30654.055 | 88.024 | 64.819 | 862.4 | 2.143 | 1.562 | False | False |
| tlg_y0.75_o0.75_r1.00_reinv4_addhf2.50 | traffic_light_health | 245.331 | 30654.055 | 88.024 | 64.819 | 862.4 | 2.143 | 1.562 | False | False |

## Readout

The health-constrained retest shows the trade-off clearly: raising the projected-health-factor floor for traffic-light hedge additions restores balance-sheet safety, but it destroys the `>420 SOL` objective.

Only the control passed the hard constraints. All traffic-light health variants passed the `1.50` minimum health-factor gate, but none finished above `420 SOL`.

Best traffic-light health row:

- `tlg_y0.50_o1.00_r1.25_reinv3_addhf2.50`
- Final SOL-equivalent: `406.340 SOL`
- Max drawdown: `81.728%`
- 2024+ drawdown: `64.980%`
- Sortino: `2.273`
- Min health factor: `1.665`

That result is near-valid but not promotable. It improves health factor versus the raw traffic-light frontier, but it gives up `18.562 SOL` versus the control and has worse drawdown.

The combined result of the two traffic-light sweeps is now clear:

- Raw traffic-light floors can improve SOL-equivalent value to `492-505 SOL` and slightly improve max drawdown to roughly `79.2%-79.4%`, but min health factor drops to `1.38-1.42`.
- Projected-HF-constrained traffic-light adds preserve min health factor above `1.50`, but the best traffic-light row falls to `406.340 SOL` and does not improve drawdown.
- Neither version comes remotely close to the `50%` institutional drawdown target.

Interpretation: the first traffic-light governor was useful as a diagnostic, not as a solution. It proved that the lights can change the return frontier, but the mechanism is still mostly "more or less ETH short." That is not enough. When allowed to act freely, the traffic-light governor leans too hard into debt. When constrained safely, it cannot add enough protection or compounding to beat the control.

Next research step should change mechanics rather than tune these floors:

1. Keep traffic lights as the state machine.
2. Stop relying on larger ETH hedge floors as the main drawdown lever.
3. Use the lights to govern a recovery/insurance book: protect realized gains in yellow/orange/red, then redeploy in bucketed SOL buys only when recovery lights confirm.
4. Consider SOL-native protection in a later phase if the insurance/recovery book still cannot reach `>420 SOL` with materially lower drawdown.

The investor-grade conclusion is blunt: traffic-light exposure governance alone cannot solve the drawdown problem under the `>420 SOL` constraint.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038/comparison.csv`
- Valid configs CSV: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038/valid_configs.csv`
- Near-valid configs CSV: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038/near_valid_configs.csv`
- All by drawdown CSV: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038/all_by_drawdown.csv`
- All by SOL CSV: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038/all_by_sol.csv`
- Family best CSV: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038/family_best.csv`
- Summary JSON: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038`
