# Constrained Drawdown Search

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Objective

Find configurations that reduce drawdown while still making money in SOL terms.

Hard valid constraints:

- Final SOL-equivalent >= `420.0 SOL`
- Minimum health factor >= `1.55`

Near-valid constraints:

- Final SOL-equivalent >= `400.0 SOL`
- Minimum health factor >= `1.55`

## Search Space

- Tiny protected-book allocations: `1%`, `2.5%`, `5%`, `7.5%`, `10%`, `15%`
- Hedge-failure overlay thresholds: `8%`, `10%`, `12%`; sell fractions: `0%`, `1%`, `2%`, `3%`, `5%`
- Combined micro-overlays: protected book `1%-10%` with hedge-failure sells `0%-3%`
- Fast-break shape variants around return threshold, volatility multiplier, and hold bars
- Profit-lock hedge-only variants around hedge floor and stateful exit gap

Total scenarios: `69`.

## Valid Configs Ranked By Drawdown

| scenario | family | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hedge_failure_thr0.10_sell0.01 | hedge_failure_micro | 420.951 | 52597.853 | 81.157 | 64.781 | 2.288 | 1.616 | True | True |
| hedge_failure_thr0.08_sell0.01 | hedge_failure_micro | 422.391 | 52777.788 | 81.206 | 64.799 | 2.288 | 1.610 | True | True |
| hedge_failure_thr0.12_sell0.03 | hedge_failure_micro | 421.001 | 52604.042 | 81.281 | 64.801 | 2.288 | 1.609 | True | True |
| hedge_failure_thr0.10_sell0.00 | hedge_failure_micro | 423.812 | 52955.345 | 81.318 | 64.827 | 2.289 | 1.601 | True | True |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold120 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.08_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.08_vol2.5_hold120 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.10_vol2.5_hold72 | fast_break_shape | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | True | True |
| fb_ret-0.10_vol2.5_hold120 | fast_break_shape | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | True | True |
| hedge_failure_thr0.12_sell0.02 | hedge_failure_micro | 424.737 | 53070.835 | 81.415 | 64.849 | 2.289 | 1.594 | True | True |
| hedge_failure_thr0.12_sell0.01 | hedge_failure_micro | 420.114 | 52493.291 | 81.651 | 64.854 | 2.286 | 1.593 | True | True |

## Near-Valid Configs Ranked By Drawdown

| scenario | family | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hedge_failure_thr0.12_sell0.05 | hedge_failure_micro | 413.476 | 51663.806 | 81.042 | 64.708 | 2.284 | 1.639 | False | True |
| combo_pb0.025_hf0.03 | combo_micro | 414.190 | 51753.019 | 81.050 | 64.715 | 2.284 | 1.637 | False | True |
| hedge_failure_thr0.10_sell0.01 | hedge_failure_micro | 420.951 | 52597.853 | 81.157 | 64.781 | 2.288 | 1.616 | True | True |
| hedge_failure_thr0.10_sell0.05 | hedge_failure_micro | 412.754 | 51573.650 | 81.162 | 64.722 | 2.284 | 1.635 | False | True |
| combo_pb0.025_hf0.02 | combo_micro | 404.010 | 50481.073 | 81.194 | 64.661 | 2.278 | 1.654 | False | True |
| hedge_failure_thr0.08_sell0.01 | hedge_failure_micro | 422.391 | 52777.788 | 81.206 | 64.799 | 2.288 | 1.610 | True | True |
| hedge_failure_thr0.12_sell0.03 | hedge_failure_micro | 421.001 | 52604.042 | 81.281 | 64.801 | 2.288 | 1.609 | True | True |
| hedge_failure_thr0.10_sell0.00 | hedge_failure_micro | 423.812 | 52955.345 | 81.318 | 64.827 | 2.289 | 1.601 | True | True |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold120 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.08_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.08_vol2.5_hold120 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.10_vol2.5_hold72 | fast_break_shape | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | True | True |
| fb_ret-0.10_vol2.5_hold120 | fast_break_shape | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | True | True |
| combo_pb0.010_hf0.03 | combo_micro | 410.736 | 51321.411 | 81.368 | 64.741 | 2.282 | 1.629 | False | True |
| fb_ret-0.06_vol2.0_hold120 | fast_break_shape | 419.723 | 52444.342 | 81.384 | 64.807 | 2.289 | 1.608 | False | True |
| hedge_failure_thr0.12_sell0.02 | hedge_failure_micro | 424.737 | 53070.835 | 81.415 | 64.849 | 2.289 | 1.594 | True | True |
| combo_pb0.025_hf0.01 | combo_micro | 404.225 | 50507.950 | 81.460 | 64.707 | 2.278 | 1.640 | False | True |
| hedge_failure_thr0.10_sell0.03 | hedge_failure_micro | 408.811 | 51080.911 | 81.575 | 64.760 | 2.281 | 1.623 | False | True |

## Best Per Family

| scenario | family | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hedge_failure_thr0.10_sell0.01 | hedge_failure_micro | 420.951 | 52597.853 | 81.157 | 64.781 | 2.288 | 1.616 | True | True |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| combo_pb0.025_hf0.03 | combo_micro | 414.190 | 51753.019 | 81.050 | 64.715 | 2.284 | 1.637 | False | True |
| pl_floor0.35_exit0.07 | profit_lock_hedge_only | 403.577 | 50426.895 | 82.700 | 64.978 | 2.282 | 1.553 | False | True |
| protected_book_pnl0.150 | protected_book_micro | 363.193 | 45381.015 | 81.899 | 64.514 | 2.253 | 1.684 | False | False |

## Top By Drawdown Regardless Of Constraints

| scenario | family | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hedge_failure_thr0.12_sell0.05 | hedge_failure_micro | 413.476 | 51663.806 | 81.042 | 64.708 | 2.284 | 1.639 | False | True |
| combo_pb0.025_hf0.03 | combo_micro | 414.190 | 51753.019 | 81.050 | 64.715 | 2.284 | 1.637 | False | True |
| hedge_failure_thr0.10_sell0.01 | hedge_failure_micro | 420.951 | 52597.853 | 81.157 | 64.781 | 2.288 | 1.616 | True | True |
| hedge_failure_thr0.10_sell0.05 | hedge_failure_micro | 412.754 | 51573.650 | 81.162 | 64.722 | 2.284 | 1.635 | False | True |
| combo_pb0.025_hf0.02 | combo_micro | 404.010 | 50481.073 | 81.194 | 64.661 | 2.278 | 1.654 | False | True |
| hedge_failure_thr0.08_sell0.01 | hedge_failure_micro | 422.391 | 52777.788 | 81.206 | 64.799 | 2.288 | 1.610 | True | True |
| hedge_failure_thr0.12_sell0.03 | hedge_failure_micro | 421.001 | 52604.042 | 81.281 | 64.801 | 2.288 | 1.609 | True | True |
| hedge_failure_thr0.10_sell0.00 | hedge_failure_micro | 423.812 | 52955.345 | 81.318 | 64.827 | 2.289 | 1.601 | True | True |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold120 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.08_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.08_vol2.5_hold120 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.10_vol2.5_hold72 | fast_break_shape | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | True | True |
| fb_ret-0.10_vol2.5_hold120 | fast_break_shape | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | True | True |

## Top By SOL-Equivalent

| scenario | family | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | valid | near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fb_ret-0.10_vol2.5_hold72 | fast_break_shape | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | True | True |
| fb_ret-0.10_vol2.5_hold120 | fast_break_shape | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | True | True |
| v2_best_no_overlays | control | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.06_vol2.5_hold120 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.08_vol2.5_hold72 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| fb_ret-0.08_vol2.5_hold120 | fast_break_shape | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | True | True |
| hedge_failure_thr0.12_sell0.02 | hedge_failure_micro | 424.737 | 53070.835 | 81.415 | 64.849 | 2.289 | 1.594 | True | True |
| hedge_failure_thr0.10_sell0.00 | hedge_failure_micro | 423.812 | 52955.345 | 81.318 | 64.827 | 2.289 | 1.601 | True | True |
| hedge_failure_thr0.08_sell0.01 | hedge_failure_micro | 422.391 | 52777.788 | 81.206 | 64.799 | 2.288 | 1.610 | True | True |
| hedge_failure_thr0.12_sell0.03 | hedge_failure_micro | 421.001 | 52604.042 | 81.281 | 64.801 | 2.288 | 1.609 | True | True |
| hedge_failure_thr0.10_sell0.01 | hedge_failure_micro | 420.951 | 52597.853 | 81.157 | 64.781 | 2.288 | 1.616 | True | True |
| hedge_failure_thr0.12_sell0.01 | hedge_failure_micro | 420.114 | 52493.291 | 81.651 | 64.854 | 2.286 | 1.593 | True | True |
| fb_ret-0.06_vol2.0_hold120 | fast_break_shape | 419.723 | 52444.342 | 81.384 | 64.807 | 2.289 | 1.608 | False | True |
| hedge_failure_thr0.12_sell0.00 | hedge_failure_micro | 415.878 | 51963.991 | 82.176 | 64.909 | 2.283 | 1.575 | False | True |

## Readout

The constrained search confirms the core dynamic: inside the current SOL-collateral plus ETH-short architecture, almost every meaningful drawdown reduction is paid for with SOL-equivalent loss.

There are `13` hard-valid configs out of `69`, but most are either the control or fast-break variants that effectively reproduce the control. The best hard-valid drawdown result is `hedge_failure_thr0.10_sell0.01`: final SOL-equivalent `420.951 SOL`, max drawdown `81.157%`, 2024+ peak-to-trough `64.781%`, Sortino `2.288`, and min HF `1.616`. That is a valid improvement, but it is tiny: max drawdown improves by only `0.187` percentage points versus the `81.344%` control, and 2024+ drawdown improves by only `0.058` points.

The near-valid frontier is more revealing. `hedge_failure_thr0.12_sell0.05` reaches max drawdown `81.042%`, but drops to `413.476 SOL`. `combo_pb0.025_hf0.03` is similar at `81.050%` max drawdown and `414.190 SOL`. The moment the search takes enough defensive action to visibly move drawdown, it falls below the `420 SOL` money-making gate.

Family readout:

- **Hedge-failure micro** is the only family worth keeping. It produced all non-control hard-valid improvements.
- **Fast-break shape** mostly aliases the current v2 benchmark; changing return threshold/hold bars did not create a better frontier.
- **Protected-book micro** failed even the `400 SOL` near-valid gate. Even `1%` realized-PnL protection finished at only `399.261 SOL`, which means the accumulation loop is extremely sensitive to compounding diversion.
- **Combo micro** had the best near-valid drawdown, but still missed the hard SOL gate.
- **Profit-lock hedge-only** did not help; it mostly worsened drawdown or missed SOL.

Conclusion: no tested current-architecture variant provides a radical drawdown reduction while preserving at least `420 SOL`. The only rational promotion candidate is the tiny hedge-failure overlay, and even that is incremental. A genuinely large drawdown reduction probably requires a SOL-native hedge or a different payoff source rather than further SOL selling, profit withholding, or ETH-short tuning.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238/comparison.csv`
- Valid configs CSV: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238/valid_configs.csv`
- Near-valid configs CSV: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238/near_valid_configs.csv`
- All by drawdown CSV: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238/all_by_drawdown.csv`
- All by SOL CSV: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238/all_by_sol.csv`
- Family best CSV: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238/family_best.csv`
- Summary JSON: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238`
