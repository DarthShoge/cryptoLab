# Fast-Break Overlay V2 Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can staged fast-break decay plus a fast-break-specific health-factor add cap preserve the final-value gains from v1 while reducing the drawdown and residual-risk damage?

## Scenario Set

The sweep keeps the current promoted stateful profit-lock incumbent as the control and varies only fast-break v2 parameters.

- Return triggers: `-8%`, `-10%`, `-12%` over 24h.
- Volatility confirmations: `1.50x`, `2.00x`, `2.50x` 24h realized volatility versus its 30d median.
- Donchian branches: `3d`, `7d`, `14d` channel breaks with `1.50x` or `2.00x` vol confirmation.
- Requested hedge floor: `1.00`.
- Default staged decay: `1.00 -> 0.75 -> 0.35`.
- Health-factor add cap: `1.50`, `2.00`, `2.25`, `2.50` for return/vol branches and `2.00`, `2.25`, `2.50` for Donchian branches.
- Extra hold-shape checks: `48h` two-stage decay and `96h` three-stage decay.

## Ranking By Sortino

| scenario | ret | vol | donchian | floor | hold | addHF | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | up | down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_ret8%_vol2.00_floor1.00_hold72_addhf2.00 | -0.080 | 2.000 |  | 1.000 | 72 | 2.000 | 410.681 | 51314.621 | 82.765 | 64.880 | 2.293 | 1.446 | 10.869 | 18 | 2 |
| v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50 | -0.080 | 2.500 |  | 1.000 | 72 | 2.500 | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 7.262 | 15 | 1 |
| v2_ret10%_vol1.50_floor1.00_hold72_addhf1.50 | -0.100 | 1.500 |  | 1.000 | 72 | 1.500 | 406.382 | 50777.470 | 82.715 | 64.906 | 2.290 | 1.353 | 16.448 | 12 | 6 |
| v2_ret10%_vol2.50_floor1.00_hold72_addhf2.50 | -0.100 | 2.500 |  | 1.000 | 72 | 2.500 | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | 5.088 | 15 | 1 |
| v2_ret12%_vol2.50_floor1.00_hold72_addhf2.50 | -0.120 | 2.500 |  | 1.000 | 72 | 2.500 | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | 4.963 | 15 | 1 |
| v2_ret10%_vol2.00_floor1.00_hold72_addhf1.50 | -0.100 | 2.000 |  | 1.000 | 72 | 1.500 | 405.469 | 50663.324 | 82.720 | 64.900 | 2.289 | 1.353 | 9.727 | 12 | 6 |
| v2_ret12%_vol2.00_floor1.00_hold72_addhf1.50 | -0.120 | 2.000 |  | 1.000 | 72 | 1.500 | 399.250 | 49886.316 | 83.060 | 64.468 | 2.289 | 1.390 | 8.062 | 14 | 6 |
| v2_ret12%_vol1.50_floor1.00_hold72_addhf1.50 | -0.120 | 1.500 |  | 1.000 | 72 | 1.500 | 390.542 | 48798.213 | 82.431 | 64.156 | 2.286 | 1.390 | 13.303 | 14 | 6 |
| v2_donchian3d_vol1.50_floor1.00_hold72_addhf2.25 | -0.990 | 1.500 | 72 | 1.000 | 72 | 2.250 | 408.622 | 51057.287 | 82.318 | 65.025 | 2.286 | 1.516 | 22.888 | 22 | 2 |
| v2_donchian7d_vol1.50_floor1.00_hold72_addhf2.25 | -0.990 | 1.500 | 168 | 1.000 | 72 | 2.250 | 408.622 | 51057.287 | 82.318 | 65.025 | 2.286 | 1.516 | 18.869 | 22 | 2 |
| v2_donchian3d_vol2.00_floor1.00_hold72_addhf2.00 | -0.990 | 2.000 | 72 | 1.000 | 72 | 2.000 | 405.573 | 50676.375 | 82.936 | 64.875 | 2.286 | 1.456 | 10.334 | 12 | 2 |
| v2_donchian7d_vol2.00_floor1.00_hold72_addhf2.00 | -0.990 | 2.000 | 168 | 1.000 | 72 | 2.000 | 405.573 | 50676.375 | 82.936 | 64.875 | 2.286 | 1.456 | 8.895 | 12 | 2 |

## Ranking By Max Drawdown

| scenario | ret | vol | donchian | floor | hold | addHF | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | up | down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50 | -0.080 | 2.500 |  | 1.000 | 72 | 2.500 | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 7.262 | 15 | 1 |
| v2_ret10%_vol2.50_floor1.00_hold72_addhf2.50 | -0.100 | 2.500 |  | 1.000 | 72 | 2.500 | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | 5.088 | 15 | 1 |
| v2_ret12%_vol2.50_floor1.00_hold72_addhf2.50 | -0.120 | 2.500 |  | 1.000 | 72 | 2.500 | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | 4.963 | 15 | 1 |
| v2_donchian14d_vol1.50_floor1.00_hold72_addhf2.25 | -0.990 | 1.500 | 336 | 1.000 | 72 | 2.250 | 401.212 | 50131.413 | 82.256 | 64.773 | 2.284 | 1.516 | 17.087 | 19 | 1 |
| v2_donchian3d_vol1.50_floor1.00_hold72_addhf2.25 | -0.990 | 1.500 | 72 | 1.000 | 72 | 2.250 | 408.622 | 51057.287 | 82.318 | 65.025 | 2.286 | 1.516 | 22.888 | 22 | 2 |
| v2_donchian7d_vol1.50_floor1.00_hold72_addhf2.25 | -0.990 | 1.500 | 168 | 1.000 | 72 | 2.250 | 408.622 | 51057.287 | 82.318 | 65.025 | 2.286 | 1.516 | 18.869 | 22 | 2 |
| incumbent_stateful_best |  |  |  |  |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0.000 | 0 | 0 |
| v2_ret10%_vol2.50_floor1.00_hold72_addhf2.00 | -0.100 | 2.500 |  | 1.000 | 72 | 2.000 | 383.104 | 47868.789 | 82.394 | 64.791 | 2.260 | 1.613 | 5.088 | 15 | 2 |
| v2_ret12%_vol2.50_floor1.00_hold72_addhf2.00 | -0.120 | 2.500 |  | 1.000 | 72 | 2.000 | 383.104 | 47868.789 | 82.394 | 64.791 | 2.260 | 1.613 | 4.963 | 15 | 2 |
| v2_ret12%_vol1.50_floor1.00_hold72_addhf1.50 | -0.120 | 1.500 |  | 1.000 | 72 | 1.500 | 390.542 | 48798.213 | 82.431 | 64.156 | 2.286 | 1.390 | 13.303 | 14 | 6 |
| v2_ret10%_vol1.50_floor1.00_hold72_addhf1.50 | -0.100 | 1.500 |  | 1.000 | 72 | 1.500 | 406.382 | 50777.470 | 82.715 | 64.906 | 2.290 | 1.353 | 16.448 | 12 | 6 |
| v2_ret10%_vol2.00_floor1.00_hold72_addhf1.50 | -0.100 | 2.000 |  | 1.000 | 72 | 1.500 | 405.469 | 50663.324 | 82.720 | 64.900 | 2.289 | 1.353 | 9.727 | 12 | 6 |

## Ranking By 2024+ Peak-To-Trough

| scenario | ret | vol | donchian | floor | hold | addHF | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | up | down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_ret12%_vol2.50_floor1.00_hold72_addhf1.50 | -0.120 | 2.500 |  | 1.000 | 72 | 1.500 | 109.099 | 13631.883 | 95.826 | 64.008 | 1.893 | 1.324 | 4.963 | 6 | 3 |
| v2_ret12%_vol1.50_floor1.00_hold72_addhf1.50 | -0.120 | 1.500 |  | 1.000 | 72 | 1.500 | 390.542 | 48798.213 | 82.431 | 64.156 | 2.286 | 1.390 | 13.303 | 14 | 6 |
| v2_ret10%_vol1.50_floor1.00_hold72_addhf2.00 | -0.100 | 1.500 |  | 1.000 | 72 | 2.000 | 120.673 | 15078.039 | 94.948 | 64.330 | 1.934 | 1.704 | 16.448 | 18 | 1 |
| v2_ret12%_vol2.00_floor1.00_hold72_addhf1.50 | -0.120 | 2.000 |  | 1.000 | 72 | 1.500 | 399.250 | 49886.316 | 83.060 | 64.468 | 2.289 | 1.390 | 8.062 | 14 | 6 |
| v2_ret8%_vol1.50_floor1.00_hold72_addhf1.50 | -0.080 | 1.500 |  | 1.000 | 72 | 1.500 | 263.093 | 32873.448 | 87.003 | 64.507 | 2.187 | 1.352 | 21.872 | 14 | 7 |
| v2_ret8%_vol2.50_floor1.00_hold72_addhf2.00 | -0.080 | 2.500 |  | 1.000 | 72 | 2.000 | 114.223 | 14272.141 | 95.247 | 64.511 | 1.916 | 1.673 | 7.262 | 12 | 1 |
| v2_ret10%_vol2.00_floor1.00_hold72_addhf2.00 | -0.100 | 2.000 |  | 1.000 | 72 | 2.000 | 119.717 | 14958.638 | 94.948 | 64.550 | 1.933 | 1.635 | 9.727 | 18 | 3 |
| v2_ret10%_vol2.50_floor1.00_hold72_addhf1.50 | -0.100 | 2.500 |  | 1.000 | 72 | 1.500 | 113.835 | 14223.629 | 95.826 | 64.624 | 1.901 | 1.324 | 5.088 | 5 | 3 |
| v2_ret8%_vol2.50_floor1.00_hold72_addhf2.25 | -0.080 | 2.500 |  | 1.000 | 72 | 2.250 | 113.504 | 14182.285 | 95.143 | 64.654 | 1.917 | 1.615 | 7.262 | 15 | 2 |
| v2_ret12%_vol2.50_floor1.00_hold72_addhf2.25 | -0.120 | 2.500 |  | 1.000 | 72 | 2.250 | 113.451 | 14175.704 | 95.171 | 64.670 | 1.917 | 1.626 | 4.963 | 14 | 0 |
| v2_ret8%_vol2.00_floor1.00_hold72_addhf1.50 | -0.080 | 2.000 |  | 1.000 | 72 | 1.500 | 284.185 | 35508.947 | 87.424 | 64.688 | 2.205 | 1.352 | 10.869 | 14 | 8 |
| v2_ret10%_vol2.50_floor1.00_hold72_addhf2.25 | -0.100 | 2.500 |  | 1.000 | 72 | 2.250 | 113.567 | 14190.236 | 95.171 | 64.689 | 1.917 | 1.613 | 5.088 | 14 | 2 |

## Ranking By Final SOL-Equivalent

| scenario | ret | vol | donchian | floor | hold | addHF | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | active% | up | down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_ret10%_vol2.50_floor1.00_hold72_addhf2.50 | -0.100 | 2.500 |  | 1.000 | 72 | 2.500 | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | 5.088 | 15 | 1 |
| v2_ret12%_vol2.50_floor1.00_hold72_addhf2.50 | -0.120 | 2.500 |  | 1.000 | 72 | 2.500 | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | 4.963 | 15 | 1 |
| v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50 | -0.080 | 2.500 |  | 1.000 | 72 | 2.500 | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 7.262 | 15 | 1 |
| v2_ret8%_vol2.00_floor1.00_hold72_addhf2.00 | -0.080 | 2.000 |  | 1.000 | 72 | 2.000 | 410.681 | 51314.621 | 82.765 | 64.880 | 2.293 | 1.446 | 10.869 | 18 | 2 |
| v2_donchian3d_vol1.50_floor1.00_hold72_addhf2.25 | -0.990 | 1.500 | 72 | 1.000 | 72 | 2.250 | 408.622 | 51057.287 | 82.318 | 65.025 | 2.286 | 1.516 | 22.888 | 22 | 2 |
| v2_donchian7d_vol1.50_floor1.00_hold72_addhf2.25 | -0.990 | 1.500 | 168 | 1.000 | 72 | 2.250 | 408.622 | 51057.287 | 82.318 | 65.025 | 2.286 | 1.516 | 18.869 | 22 | 2 |
| v2_ret10%_vol1.50_floor1.00_hold72_addhf1.50 | -0.100 | 1.500 |  | 1.000 | 72 | 1.500 | 406.382 | 50777.470 | 82.715 | 64.906 | 2.290 | 1.353 | 16.448 | 12 | 6 |
| v2_donchian3d_vol1.50_floor1.00_hold72_addhf2.00 | -0.990 | 1.500 | 72 | 1.000 | 72 | 2.000 | 405.644 | 50685.167 | 83.746 | 65.184 | 2.284 | 1.455 | 22.888 | 20 | 4 |
| v2_donchian7d_vol1.50_floor1.00_hold72_addhf2.00 | -0.990 | 1.500 | 168 | 1.000 | 72 | 2.000 | 405.644 | 50685.167 | 83.746 | 65.184 | 2.284 | 1.455 | 18.869 | 20 | 4 |
| v2_donchian3d_vol2.00_floor1.00_hold72_addhf2.00 | -0.990 | 2.000 | 72 | 1.000 | 72 | 2.000 | 405.573 | 50676.375 | 82.936 | 64.875 | 2.286 | 1.456 | 10.334 | 12 | 2 |
| v2_donchian7d_vol2.00_floor1.00_hold72_addhf2.00 | -0.990 | 2.000 | 168 | 1.000 | 72 | 2.000 | 405.573 | 50676.375 | 82.936 | 64.875 | 2.286 | 1.456 | 8.895 | 12 | 2 |
| v2_donchian14d_vol2.00_floor1.00_hold72_addhf2.00 | -0.990 | 2.000 | 336 | 1.000 | 72 | 2.000 | 405.573 | 50676.375 | 82.936 | 64.875 | 2.286 | 1.456 | 8.402 | 12 | 2 |

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | Active % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | incumbent_stateful_best | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0.000 |
| Best Sortino | v2_ret8%_vol2.00_floor1.00_hold72_addhf2.00 | 410.681 | 51314.621 | 82.765 | 64.880 | 2.293 | 1.446 | 10.869 |
| Best Max DD | v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50 | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 7.262 |
| Best 2024 DD | v2_ret12%_vol2.50_floor1.00_hold72_addhf1.50 | 109.099 | 13631.883 | 95.826 | 64.008 | 1.893 | 1.324 | 4.963 |
| Best SOL-eq | v2_ret10%_vol2.50_floor1.00_hold72_addhf2.50 | 425.311 | 53142.638 | 81.350 | 64.843 | 2.289 | 1.596 | 5.088 |

## Forensic Notes

Decision: provisionally promote `v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50` as the current best fast-break configuration, but do not consider the red-mark problem solved.

The best balanced v2 candidate improves the incumbent on the promotion metrics: final SOL-equivalent rises from `402.06` to `424.90`, final USD rises from `$50,237.89` to `$53,091.48`, max drawdown improves from `82.39%` to `81.34%`, Sortino improves from `2.274` to `2.290`, and min HF improves from `1.591` to `1.598`. This is a cleaner result than v1 because v1's best Sortino candidate had worse max drawdown and worse min HF.

The stricter health-factor add cap mattered. The `addhf2.50` winners carry much less final ETH debt than the v1 best Sortino candidate: about `$28.5k` versus about `$35.2k`. They also activate for only `5.1%` to `7.3%` of bars, versus `20.3%` for the v1 `-6% / 1.25x` trigger. That makes v2 feel more like an emergency overlay and less like a second always-on regime.

The biggest improvement comes from the old 2021-2023 drawdown window, not from the 2025 red-mark window. The incumbent peaked at `$22,927.54` on `2021-11-06` and troughed at `$4,038.47` on `2023-06-10`. The best v2 candidate peaks at `$23,034.73` and troughs at `$4,297.29` over the same window. That is enough to lower full-period max drawdown, but it is not a structural fix for the later bull-market giveback.

The 2025 peak-to-trough barely moves. The incumbent falls `64.86%` from the `2025-01-19` peak to the `2025-04-07` trough. The best v2 candidate falls `64.84%`. That is technically better, but economically negligible. It also finishes `48.47%` below its 2025 peak, essentially the same as the incumbent at `48.49%`.

The key diagnostic is that the fast-break signal was active during the early 2025 decline, but it did not actually resize the hedge. In the best v2 candidate, fast-break is active from `2025-01-20`, with an effective hedge target of `1.00`, while crisis mode is also active. However, history shows `action_count = 0` through that initial fast-break stretch and `under_hedged_crisis = true`. At `2025-01-20 00:00 UTC`, SOL collateral value is about `$88.7k`, ETH debt is only about `$7.9k`, and the requested 1.00x hedge is too large to add atomically under the account constraints. Because fast-break currently requests the full target and does not do its own partial fill, the strategy remains badly under-hedged.

That explains the strange combination of results: v2 can improve max drawdown where the account is small enough for hedge additions to execute, but it still fails when the portfolio has compounded and the required hedge jump becomes too large. The signal is not the only problem; execution sizing is now the bottleneck.

Some apparent 2024+ drawdown winners are traps. For example, `v2_ret12%_vol2.50_floor1.00_hold72_addhf1.50` shows the best 2024+ peak-to-trough at `64.01%`, but full-period max drawdown explodes to `95.83%` and final SOL-equivalent collapses to `109.10`. That is not a real improvement; it is a damaged path that happens to have a slightly shallower 2025 measured trough.

The Donchian branch is directionally interesting but not the best current candidate. The best Donchian rows are cleaner than many return-trigger rows, but they do not beat the best return/vol v2 candidate. `v2_donchian14d_vol1.50_floor1.00_hold72_addhf2.25` modestly improves full-period max drawdown to `82.26%`, but final SOL-equivalent is only `401.21`, below the incumbent. Keep Donchian as a research branch, not the promoted default.

Next hypothesis: add fast-break partial hedge filling. The report strongly suggests the next mechanism should not be another signal tweak. We need a fast-break-specific partial add path that buys as much ETH hedge as safely possible up to the requested floor, bounded by a configurable health-factor target and existing cooldown. This is different from lowering min HF globally; the goal is to avoid the all-or-nothing failure where the strategy knows it needs protection but cannot execute the full target.

Promotion note: if we update UI defaults from this report, use `v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50` rather than the marginally higher final-SOL `v2_ret10%_vol2.50_floor1.00_hold72_addhf2.50`. The `-8%` candidate has slightly better max drawdown, Sortino, and min HF, with almost identical final SOL-equivalent.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349/comparison.csv`
- Top-by-drawdown CSV: `reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349/top_by_drawdown.csv`
- Top-by-2024-drawdown CSV: `reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349/top_by_2024_drawdown.csv`
- Top-by-SOL-equivalent CSV: `reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349/top_by_sol_equiv.csv`
- Summary JSON: `reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/fast_break_v2_sweep_20260529_163349`
