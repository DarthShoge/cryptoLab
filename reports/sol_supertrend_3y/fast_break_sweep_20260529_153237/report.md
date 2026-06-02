# Fast-Break Overlay Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can the new fast-break overlay improve downside positioning by raising the ETH hedge floor only when SOL has both a short-lookback downside break and realized volatility expansion?

## Scenario Set

The sweep keeps the current promoted stateful profit-lock incumbent as the control and varies only fast-break overlay parameters.

- Return trigger: `-6%`, `-8%`, `-10%` over 24h.
- Volatility confirmation: `1.25x`, `1.50x`, `2.00x` 24h realized volatility versus its 30d median.
- Hedge floor: `0.50`, `0.75`, `1.00`.
- Hold / decay: `48h`, `72h`.
- Extra Donchian smoke test: `close < 7d Donchian low`, `1.50x` vol, `72h` hold.
- No reinvestment blocker.

## Ranking By Sortino

| scenario | ret | vol | floor | hold | SOL-eq | USD | maxDD | Sortino | minHF | 2024+ giveback | active% | upEvents |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fb_ret6%_vol1.25_floor1.00_hold72 | -0.060 | 1.250 | 1.000 | 72 | 421.968 | 52724.947 | 83.387 | 2.294 | 1.487 | -46.476 | 20.285 | 75 |
| fb_ret6%_vol2.00_floor1.00_hold72 | -0.060 | 2.000 | 1.000 | 72 | 414.798 | 51829.024 | 82.750 | 2.291 | 1.477 | -46.016 | 6.324 | 23 |
| fb_donchian7d_vol1.50_floor1.00_hold72 | -0.990 | 1.500 | 1.000 | 72 | 414.635 | 51808.621 | 82.685 | 2.291 | 1.445 | -46.069 | 8.295 | 20 |
| fb_ret6%_vol1.50_floor1.00_hold72 | -0.060 | 1.500 | 1.000 | 72 | 413.463 | 51662.165 | 82.715 | 2.291 | 1.477 | -46.319 | 13.668 | 30 |
| fb_ret10%_vol1.50_floor1.00_hold72 | -0.100 | 1.500 | 1.000 | 72 | 413.998 | 51728.994 | 82.754 | 2.289 | 1.562 | -46.011 | 8.139 | 19 |
| fb_ret10%_vol2.00_floor1.00_hold72 | -0.100 | 2.000 | 1.000 | 72 | 413.998 | 51728.994 | 82.754 | 2.289 | 1.562 | -46.011 | 4.750 | 19 |
| fb_ret6%_vol2.00_floor0.75_hold72 | -0.060 | 2.000 | 0.750 | 72 | 408.115 | 50994.017 | 83.140 | 2.286 | 1.446 | -46.044 | 6.324 | 23 |
| fb_ret8%_vol2.00_floor1.00_hold72 | -0.080 | 2.000 | 1.000 | 72 | 398.295 | 49767.002 | 82.708 | 2.285 | 1.451 | -45.893 | 5.563 | 18 |
| fb_ret8%_vol2.00_floor1.00_hold48 | -0.080 | 2.000 | 1.000 | 48 | 410.949 | 51348.034 | 82.766 | 2.284 | 1.499 | -45.992 | 4.424 | 17 |
| fb_ret8%_vol1.50_floor1.00_hold72 | -0.080 | 1.500 | 1.000 | 72 | 393.776 | 49202.332 | 82.708 | 2.282 | 1.451 | -46.507 | 11.097 | 19 |
| fb_ret8%_vol1.50_floor1.00_hold48 | -0.080 | 1.500 | 1.000 | 48 | 408.555 | 51048.958 | 82.766 | 2.282 | 1.499 | -46.307 | 8.365 | 18 |
| fb_ret8%_vol2.00_floor0.75_hold72 | -0.080 | 2.000 | 0.750 | 72 | 392.712 | 49069.346 | 83.188 | 2.281 | 1.461 | -45.948 | 5.563 | 17 |

## Ranking By Max Drawdown

| scenario | ret | vol | floor | hold | SOL-eq | USD | maxDD | Sortino | minHF | 2024+ giveback | active% | upEvents |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_stateful_best |  |  |  |  | 402.064 | 50237.887 | 82.386 | 2.274 | 1.591 | -48.486 | 0.000 | 0 |
| fb_ret10%_vol1.50_floor0.50_hold72 | -0.100 | 1.500 | 0.500 | 72 | 403.401 | 50404.896 | 82.573 | 2.280 | 1.558 | -48.529 | 8.139 | 4 |
| fb_ret10%_vol2.00_floor0.50_hold72 | -0.100 | 2.000 | 0.500 | 72 | 403.401 | 50404.896 | 82.573 | 2.280 | 1.558 | -48.529 | 4.750 | 4 |
| fb_ret10%_vol1.25_floor0.50_hold72 | -0.100 | 1.250 | 0.500 | 72 | 402.062 | 50237.699 | 82.650 | 2.278 | 1.552 | -48.534 | 11.608 | 6 |
| fb_donchian7d_vol1.50_floor1.00_hold72 | -0.990 | 1.500 | 1.000 | 72 | 414.635 | 51808.621 | 82.685 | 2.291 | 1.445 | -46.069 | 8.295 | 20 |
| fb_ret8%_vol2.00_floor1.00_hold72 | -0.080 | 2.000 | 1.000 | 72 | 398.295 | 49767.002 | 82.708 | 2.285 | 1.451 | -45.893 | 5.563 | 18 |
| fb_ret8%_vol1.50_floor1.00_hold72 | -0.080 | 1.500 | 1.000 | 72 | 393.776 | 49202.332 | 82.708 | 2.282 | 1.451 | -46.507 | 11.097 | 19 |
| fb_ret6%_vol1.50_floor1.00_hold72 | -0.060 | 1.500 | 1.000 | 72 | 413.463 | 51662.165 | 82.715 | 2.291 | 1.477 | -46.319 | 13.668 | 30 |
| fb_ret6%_vol1.50_floor0.50_hold72 | -0.060 | 1.500 | 0.500 | 72 | 394.841 | 49335.421 | 82.732 | 2.280 | 1.450 | -48.492 | 13.668 | 6 |
| fb_ret6%_vol2.00_floor0.50_hold72 | -0.060 | 2.000 | 0.500 | 72 | 392.039 | 48985.295 | 82.748 | 2.278 | 1.448 | -48.502 | 6.324 | 5 |

## Ranking By Final SOL-Equivalent

| scenario | ret | vol | floor | hold | SOL-eq | USD | maxDD | Sortino | minHF | 2024+ giveback | active% | upEvents |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fb_ret6%_vol1.25_floor1.00_hold72 | -0.060 | 1.250 | 1.000 | 72 | 421.968 | 52724.947 | 83.387 | 2.294 | 1.487 | -46.476 | 20.285 | 75 |
| fb_ret6%_vol2.00_floor1.00_hold72 | -0.060 | 2.000 | 1.000 | 72 | 414.798 | 51829.024 | 82.750 | 2.291 | 1.477 | -46.016 | 6.324 | 23 |
| fb_donchian7d_vol1.50_floor1.00_hold72 | -0.990 | 1.500 | 1.000 | 72 | 414.635 | 51808.621 | 82.685 | 2.291 | 1.445 | -46.069 | 8.295 | 20 |
| fb_ret10%_vol1.50_floor1.00_hold72 | -0.100 | 1.500 | 1.000 | 72 | 413.998 | 51728.994 | 82.754 | 2.289 | 1.562 | -46.011 | 8.139 | 19 |
| fb_ret10%_vol2.00_floor1.00_hold72 | -0.100 | 2.000 | 1.000 | 72 | 413.998 | 51728.994 | 82.754 | 2.289 | 1.562 | -46.011 | 4.750 | 19 |
| fb_ret6%_vol1.50_floor1.00_hold72 | -0.060 | 1.500 | 1.000 | 72 | 413.463 | 51662.165 | 82.715 | 2.291 | 1.477 | -46.319 | 13.668 | 30 |
| fb_ret8%_vol2.00_floor1.00_hold48 | -0.080 | 2.000 | 1.000 | 48 | 410.949 | 51348.034 | 82.766 | 2.284 | 1.499 | -45.992 | 4.424 | 17 |
| fb_ret8%_vol1.50_floor1.00_hold48 | -0.080 | 1.500 | 1.000 | 48 | 408.555 | 51048.958 | 82.766 | 2.282 | 1.499 | -46.307 | 8.365 | 18 |
| fb_ret6%_vol1.25_floor0.75_hold72 | -0.060 | 1.250 | 0.750 | 72 | 408.225 | 51007.728 | 84.429 | 2.279 | 1.554 | -45.595 | 20.285 | 64 |
| fb_ret6%_vol2.00_floor0.75_hold72 | -0.060 | 2.000 | 0.750 | 72 | 408.115 | 50994.017 | 83.140 | 2.286 | 1.446 | -46.044 | 6.324 | 23 |

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | Sortino | Min HF | 2024+ Giveback | Active % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | incumbent_stateful_best | 402.064 | 50237.887 | 82.386 | 2.274 | 1.591 | -48.486 | 0.000 |
| Best Sortino | fb_ret6%_vol1.25_floor1.00_hold72 | 421.968 | 52724.947 | 83.387 | 2.294 | 1.487 | -46.476 | 20.285 |
| Best Drawdown | incumbent_stateful_best | 402.064 | 50237.887 | 82.386 | 2.274 | 1.591 | -48.486 | 0.000 |
| Best SOL-eq | fb_ret6%_vol1.25_floor1.00_hold72 | 421.968 | 52724.947 | 83.387 | 2.294 | 1.487 | -46.476 | 20.285 |

## True 2025 Peak-To-Trough Check

The `2024+ giveback` column above is final value versus the 2024/2025 peak. That is useful, but it does not measure how deep the red-mark drawdown got before the portfolio recovered. The table below measures the true peak-to-trough move after `2024-01-01`.

| scenario | Final SOL-eq | Max DD | Sortino | Min HF | Final ETH debt | 2025 peak-to-trough | 2025 final-vs-peak | Active % | Up events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incumbent_stateful_best | 402.06 | 82.39% | 2.274 | 1.591 | $27,013 | -64.86% | -48.49% | 0.00% | 0 |
| fb_ret6%_vol1.25_floor1.00_hold72 | 421.97 | 83.39% | 2.294 | 1.487 | $35,178 | -64.95% | -46.48% | 20.28% | 75 |
| fb_ret6%_vol2.00_floor1.00_hold72 | 414.80 | 82.75% | 2.291 | 1.477 | $34,483 | -64.89% | -46.02% | 6.32% | 23 |
| fb_donchian7d_vol1.50_floor1.00_hold72 | 414.63 | 82.68% | 2.291 | 1.445 | $34,084 | -64.91% | -46.07% | 8.29% | 20 |
| fb_ret10%_vol1.50_floor1.00_hold72 | 414.00 | 82.75% | 2.289 | 1.562 | $34,415 | -64.88% | -46.01% | 8.14% | 19 |
| fb_ret10%_vol1.50_floor0.50_hold72 | 403.40 | 82.57% | 2.280 | 1.558 | $27,151 | -64.92% | -48.53% | 8.14% | 4 |

## Forensic Notes

Decision: no promotion yet.

The fast-break overlay is directionally useful for final compounding, but it did not solve the actual drawdown problem. The best Sortino and best SOL-equivalent candidate, `fb_ret6%_vol1.25_floor1.00_hold72`, improves final SOL-equivalent from `402.06` to `421.97` and Sortino from `2.274` to `2.294`, but it worsens max drawdown from `82.39%` to `83.39%` and drops min HF from `1.591` to `1.487`. That is not a clean promotion under the current guardrails.

The most important finding is the difference between final-vs-peak giveback and peak-to-trough drawdown. Several candidates improve final-vs-peak by roughly 2.0-2.5 percentage points, but the true 2025 peak-to-trough drawdown remains around `-64.9%`, essentially unchanged from the incumbent. In other words, the overlay often helps the portfolio recover to a better final value, but it does not materially prevent the 2025 red-mark trough.

The best max-drawdown row is still the incumbent. The least intrusive fast-break variant by drawdown, `fb_ret10%_vol1.50_floor0.50_hold72`, slightly improves final SOL-equivalent and Sortino, but its max drawdown is still worse (`82.57%` versus `82.39%`) and it does not improve the true 2025 peak-to-trough drawdown. This is useful evidence, but not enough to promote.

The most aggressive successful variants all use a `1.00` hedge floor and `72h` hold. Those improve final SOL-equivalent but materially increase final ETH debt, often into the `$34k-$35k` range versus the control at about `$27k`. The overlay is adding profitable defensive exposure in some places, but it is doing so by carrying much larger residual ETH debt.

The trigger frequency matters. The `-6% / 1.25x vol / 1.00 floor / 72h` case is active for `20.28%` of bars and generates `75` fast-break hedge-up events. That is not a narrow emergency overlay; it is close to becoming a second regime system. The stricter `-10% / 1.50x / 1.00 / 72h` case is active `8.14%` of bars with `19` hedge-up events and has a healthier min HF (`1.562`), while retaining most of the final SOL-equivalent improvement. If we keep iterating, the stricter trigger family is the more credible branch.

The `48h` hold variants are fragile. Averaged across return/vol settings, shorter holds produce worse outcomes for several floor groups, and some 48h cases collapse final SOL-equivalent into the low `100s`. This suggests the current exit/decay logic can cut protection too early, then leave the system badly positioned during continuing drawdowns. The overlay needs staged decay rather than a binary expiry.

The Donchian smoke test is worth keeping in the research queue. `fb_donchian7d_vol1.50_floor1.00_hold72` is not promotable because min HF falls to `1.445`, but it lands near the top by Sortino and final SOL-equivalent with fewer hedge-up events than the most sensitive `-6%` trigger. That supports testing Donchian/channel breaks more seriously, ideally with an HF-aware sizing cap.

## Next Hypotheses

1. Add staged decay instead of a hard fast-break exit: `1.00 -> 0.75 -> 0.35`, with each step requiring either time elapsed or 4h/8h recovery.
2. Add an HF-aware add cap for fast-break hedge increases. The good candidates breach the min-HF guardrail mainly when the floor is `1.00`; the next version should test whether we can keep the signal but size additions to preserve `min HF >= 1.50`.
3. Prioritize stricter triggers for the next sweep: `-10% / 1.50x` and Donchian `7d / 1.50x`, both with `1.00` requested floor but capped by projected HF.
4. Add event-log visibility for `in_fast_break_overlay` and `fast_break_hedge_floor`; history has the fields, but strategy event logs do not currently include them.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/fast_break_sweep_20260529_153237/comparison.csv`
- Top-by-drawdown CSV: `reports/sol_supertrend_3y/fast_break_sweep_20260529_153237/top_by_drawdown.csv`
- Top-by-SOL-equivalent CSV: `reports/sol_supertrend_3y/fast_break_sweep_20260529_153237/top_by_sol_equiv.csv`
- Summary JSON: `reports/sol_supertrend_3y/fast_break_sweep_20260529_153237/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/fast_break_sweep_20260529_153237`
