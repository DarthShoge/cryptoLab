# SOL Supertrend Post-Reinvestment Report

Run window: 2021-01-01 to 2025-12-31, 1h cached SOL/ETH data, 100 initial SOL collateral.

This run includes the new hedge realized PnL ledger, surplus USDC reinvestment, and green-regime USDC debt cleanup. USDC releverage remains disabled.

## Headline

The new surplus reinvestment logic solved the most obvious prior failure: the strategy no longer ends with exactly 100 SOL. It finishes with 254.01 SOL collateral and 267.29 SOL-equivalent net value.

That said, the result is not clean victory. The strategy still experiences a 95.20% max drawdown from the 2021 peak into late 2022, only slightly better than SOL buy-and-hold's inferred 96.80% drawdown over the same cycle. This is still a brutally cyclical profile.

## Final Account

| Metric | Value |
|---|---:|
| Final net portfolio value | $33,397.26 |
| Final SOL-equivalent value | 267.29 SOL |
| SOL buy-and-hold final value | $12,495.00 |
| Strategy / SOL benchmark | 2.67x |
| Total return | 21,527.55% |
| Sortino | 2.106 |
| Max drawdown | 95.20% |
| Min health factor | 1.366 |
| Liquidations | 0 |
| Bars below HF 1.5 | 254 |

Final positions reconcile:

| Position | Amount | USD value |
|---|---:|---:|
| SOL collateral | 254.0118 SOL | $31,738.78 |
| USDC collateral | 25,306.06 USDC | $25,306.06 |
| ETH debt | 7.9718 ETH | -$23,647.58 |
| USDC debt | 0.00 USDC | $0.00 |
| Net |  | $33,397.26 |

## Year-End Snapshots

| Date | Net USD | SOL-equivalent | Actual SOL | USDC collateral | ETH debt value | HF |
|---|---:|---:|---:|---:|---:|---:|
| 2021-12-31 | $14,633.68 | 84.76 | 100.00 | $6,156.81 | $8,788.13 | 2.104 |
| 2022-12-31 | $1,423.28 | 142.90 | 183.04 | $1,270.96 | $1,670.79 | 1.503 |
| 2023-12-31 | $17,286.91 | 168.54 | 183.04 | $3,211.58 | $4,699.42 | 3.611 |
| 2024-12-31 | $34,442.49 | 181.22 | 183.04 | $16,873.49 | $17,220.20 | 2.397 |
| 2025-12-31 | $33,397.26 | 267.29 | 254.01 | $25,306.06 | $23,647.58 | 1.970 |

## What Improved

The strategy now actually accumulates SOL. It bought 154.01 additional SOL through surplus reinvestment:

| Year | SOL bought |
|---|---:|
| 2022 | 83.04 |
| 2025 | 70.97 |

This is exactly the behavior we wanted philosophically: hedge profits during/after drawdowns can become additional SOL collateral without borrowing new USDC.

The 2022 reinvestment timing is promising. The first surplus reinvestment occurred on 2022-06-25, after the crash had already repriced SOL dramatically lower. That is much closer to the intended "defense creates dry powder, then buy SOL when recovery begins" shape.

## What Still Looks Wrong

The strategy did not accumulate any SOL during 2023 or 2024. Actual SOL stayed at 183.04 from mid-2022 until April 2025. That means the 2024 bull market was mostly ridden with the same SOL stack while hedge/short mechanics built USDC and ETH debt around it.

The 2021-2022 protection still starts late. From the 2021 SOL high on 2021-11-06 at $258.44:

| Event | Timestamp | SOL price | Drop from ATH |
|---|---|---:|---:|
| First hedge-up after ATH | 2021-11-12 08:00 UTC | $231.31 | -10.50% |
| First full-short-up after ATH | 2022-01-08 00:00 UTC | $140.03 | -45.82% |

That confirms the same issue we saw before: full-short escalation is probably too slow if the goal is to guard the account during a regime break.

The drawdown remains too large. A 95.20% drawdown technically beats SOL's 96.80% inferred drawdown, but it is not a robust risk-control profile. The strategy's SOL-equivalent value fell as low as 70.87 SOL, meaning it temporarily underperformed simply holding 100 SOL by almost 30% in SOL terms.

The ending account is still heavily short ETH: ETH debt value is 74.5% of SOL collateral value at the end. Because the final weekly regime is still bearish, this may be intended by the model, but it means the final reported net value depends heavily on an open short that has not been realized.

The average health factor metric is not useful in this report. It is distorted by periods with no debt where health factor is effectively infinite. Use min HF, bars below thresholds, and final LTV instead.

## Event Log Findings

| Event reason | Count |
|---|---:|
| cooldown_skip | 2,461 |
| hedge_up | 541 |
| full_short_cover | 223 |
| full_short_up | 97 |
| hedge_down | 29 |
| surplus_reinvestment | 20 |
| green_regime_usdc_debt_cleanup | 0 |

There were no green-regime USDC debt cleanup events because USDC releverage was disabled and USDC debt stayed zero. That path is implemented but not exercised by this default run.

The surplus reinvestment events happened in two clusters only:

| Cluster | Dates | Interpretation |
|---|---|---|
| 2022-06-25 to 2022-06-27 | 13 events | Good: converted crash-era hedge profit into cheap SOL. |
| 2025-04-20 to 2025-04-21 | 7 events | Mixed: bought after a later hedge profit event, but missed the 2024 bull accumulation window. |

## Scientific Read

Hypothesis tested: "A realized hedge-profit ledger plus gated surplus reinvestment will accumulate SOL without enabling USDC releverage."

Result: supported. The strategy ended with 254.01 actual SOL instead of 100 SOL, and SOL-equivalent value improved to 267.29 SOL.

But the stronger hypothesis, "this materially controls downside while compounding SOL," is only weakly supported. Final value is excellent, but risk remains extremely high, and the model still allows deep underperformance in SOL terms during the cycle.

## Next Experiments

1. Move full-short escalation earlier.
   Test lower entry requirements than `green == 0 and bearish_3d`, such as `green <= 1 and bearish_3d`, or require 1w confirmation only for sizing rather than entry. This directly targets the late 2022 protection issue.

2. Add a moderate bullish accumulation path that does not require realized hedge profit.
   The strategy missed 2023-2024 SOL accumulation. Consider allowing a small DCA-style conversion of excess USDC collateral when 1w is green and ETH short is at target, separate from hedge-profit surplus.

3. Make surplus reinvestment less cluster-bound.
   The current 5% per-rebalance cap plus existing cooldown buys in tight bursts. Test a slower per-day cap or a volatility-adjusted cap so buying spreads across recovery regimes rather than a handful of adjacent bars.

4. Report realized versus unrealized hedge PnL separately.
   The final account carries a large open ETH short. Future reports should split final net value into realized equity and open short PnL sensitivity.

5. Run a grid sweep over surplus reinvestment parameters.
   Meaningful parameters: profit gate 5/10/15%, cap 2.5/5/10%, ladder 15/30 vs 25/50 vs 35/70, and min HF 1.75/2.0/2.25.

## Artifacts

- `reports/sol_supertrend_3y/post_reinvestment_report/history.csv`
- `reports/sol_supertrend_3y/post_reinvestment_report/strategy_events.csv`
- `reports/sol_supertrend_3y/post_reinvestment_report/summary.json`
