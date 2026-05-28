# SOL Supertrend Short Strategy 3-Year Forensic Report

Run date: 2026-05-28

Dataset: Binance `SOL/USDT` and `ETH/USDT`, 1h candles, `2023-05-28 00:00 UTC` through `2026-05-28 00:00 UTC`.

Important note: the first diagnostic run exposed a Supertrend implementation bug. At the first ATR-valid candle the indicator copied previous `NaN` bands, flipped bearish, and then stayed bearish almost permanently. That invalidated the first result. I added a regression test for a steady uptrend and fixed the warmup logic before producing the numbers below.

## Verdict

The corrected strategy is directionally working in the sense that it runs, accounts for collateral/debt consistently, and no longer treats every market as bearish. But it does **not** meet the SOL accumulation objective.

Over this 3-year window:

| Metric | Strategy | Buy-and-hold SOL |
|---|---:|---:|
| Initial USD value | $2,063 | $2,063 |
| Final USD value | $2,286.61 | $8,271 |
| USD return | +10.84% | +300.92% |
| Final SOL-equivalent | 27.65 SOL | 100 SOL |
| SOL-equivalent return | -72.35% | 0.00% |
| Max drawdown | 98.78% | 73.62% |
| Liquidation events | 23 | n/a |

Honest conclusion: this version should not be trusted as a production strategy. It adds complexity, liquidation risk, and short exposure while badly underperforming plain SOL ownership in the unit we care about.

## Accounting Verification

The engine accounting ties out exactly after adding mark-to-market position value columns.

| Check | Max absolute error |
|---|---:|
| Sum of collateral position values equals `collateral_value` | 0.0 |
| Sum of debt position values equals `debt_value` | 0.0 |
| `collateral_value - debt_value + cash_reserve` equals `portfolio_value` | 0.0 |

The Position Values chart should now reconcile visually with Portfolio Value: collateral lines are positive USD values, debt lines are negative USD values, and Net Portfolio is shown on the same chart.

## Signal Verification

After fixing Supertrend warmup, signal distribution is plausible:

| Green votes | Hours |
|---:|---:|
| 0 | 3,886 |
| 1 | 5,927 |
| 2 | 5,538 |
| 3 | 6,300 |
| 4 | 4,654 |

Higher-regime signals:

| Signal | Hours bearish |
|---|---:|
| 3d bearish | 8,881 |
| 1w bearish | 9,001 |

This is a major correction from the invalid pre-fix result, where nearly every hour was classified bearish.

## Exposure And Risk

| Exposure metric | Value |
|---|---:|
| Hours with ETH debt | 26,265 / 26,305 |
| Hours with USDC debt | 17,299 / 26,305 |
| Full-short-like hours (`ETH debt / SOL collateral > 95%`) | 0 |
| Average ETH short ratio | 46.8% |
| Max ETH short ratio | 79.94% |
| Max ETH debt value | $87,320 |
| Max USDC debt value | $117,426 |
| Max SOL collateral | 1,298.11 SOL |
| Final SOL collateral | 83.64 SOL |
| Final ETH debt | $4,630.98 |
| Final health factor | 1.12 |

Key point: despite having full-short mode configured at 100%-150%, the strategy never truly reached full-short-like exposure. The health-factor cap and existing debt load prevented it. So the "full short" branch exists in code, but in this run it barely expressed itself.

## Drawdown Path

The strategy peaked at:

- `2025-01-19 11:00 UTC`
- Portfolio value: `$134,777.76`
- SOL collateral: `966.72 SOL`
- SOL collateral value: `$276,714.65`
- ETH debt: `$67,305.76`
- USDC debt: `$74,631.13`
- Health factor: `1.42`

The post-peak trough was:

- `2026-05-05 03:00 UTC`
- Portfolio value: `$1,651.01`
- Drawdown from peak: `-98.78%`
- SOL collateral: `83.64 SOL`
- ETH debt: `$5,439.71`
- Health factor: `0.98`

This is the core failure mode: the strategy aggressively grew SOL and debt during the bull phase, then rode a massive deleveraging path that destroyed SOL-equivalent value.

## Events

| Event reason | Count |
|---|---:|
| cooldown_skip | 744 |
| bullish_relever | 174 |
| hedge_up | 137 |
| hedge_down | 20 |
| full_short_cover | 9 |
| full_short_up | 2 |

Most active event days clustered in late 2023, when the strategy was actively re-levering and hedging during SOL's explosive rally.

Important issue: event names are not always semantically precise. For example, some `full_short_cover` events increased ETH short exposure because the full-short cover ladder target was above current exposure. That is logically explainable, but the event label is misleading. We should split event labels into `mode` and `action_direction`.

## Liquidations

There were 23 liquidation penalty rows. The strategy frequently operated below borrow health factor 1.0:

| HF metric | Value |
|---|---:|
| Min health factor | 0.938 |
| Average health factor | 1.134 |
| Bars below HF 1.5 | 26,292 |
| Bars below HF 1.0 | 3,528 |

This is too close to the edge. The minimum rebalance health factor is not sufficient as currently applied because the strategy can spend long periods with unsafe borrow HF and rely on liquidation health factor not immediately breaking.

## Parameter Sweep

The first sweep only varied indicator defaults:

- ATR period: `7, 10, 14`
- Supertrend multiplier: `2.0, 3.0, 4.0`
- Rebalance threshold: `0.05, 0.10, 0.15`

Top by Sortino:

| ATR | Mult | Threshold | USD return | Final SOL-equiv | Max DD | Sortino | Liquidations |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.0 | 0.15 | +47.62% | 36.82 SOL | 98.76% | 1.528 | 20 |
| 14 | 4.0 | 0.10 | +25.23% | 31.24 SOL | 98.88% | 1.517 | 20 |
| 10 | 3.0 | 0.15 | +37.26% | 34.24 SOL | 98.80% | 1.509 | 21 |
| 10 | 2.0 | 0.10 | +40.19% | 34.97 SOL | 98.79% | 1.499 | 20 |
| 14 | 3.0 | 0.10 | +18.39% | 29.53 SOL | 98.85% | 1.494 | 22 |

Even the best swept setting only ends at 36.82 SOL-equivalent from 100 starting SOL. That is better than the base setting's 27.65 SOL-equivalent, but still fails the SOL accumulation objective.

That was not enough context. The meaningful controls are economic: ETH hedge ladder, full-short sizing, USDC debt budget, bullish target health factor, minimum rebalance health factor, rebalance threshold, and cooldown. I then ran a targeted 30-combination economic sweep across three signal regimes:

- Responsive signal: ATR `10`, multiplier `2.0`
- Base signal: ATR `10`, multiplier `3.0`
- Slow signal: ATR `14`, multiplier `4.0`

The best results were completely different from the default run:

| Signal | Economic profile | Final SOL-equiv | SOL return | Max DD | Sortino | Min HF | HF < 1 bars | Liquidations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ATR 10 / 3.0 | Soft ETH hedge, no USDC debt | 210.13 SOL | +110.13% | 46.66% | 1.825 | 1.735 | 0 | 0 |
| ATR 10 / 3.0 | Soft ETH hedge, full-short enabled, no USDC debt | 206.72 SOL | +106.72% | 43.15% | 1.817 | 1.653 | 0 | 0 |
| ATR 10 / 3.0 | Base ETH hedge, no USDC debt | 172.47 SOL | +72.47% | 52.65% | 1.720 | 1.601 | 0 | 0 |
| ATR 10 / 3.0 | Aggressive ETH hedge, no USDC debt | 146.98 SOL | +46.98% | 59.09% | 1.620 | 1.560 | 0 | 0 |
| ATR 14 / 4.0 | Base ETH hedge, no USDC debt | 117.16 SOL | +17.16% | 67.82% | 1.480 | 1.608 | 0 | 0 |
| ATR 10 / 2.0 | Conservative USDC + soft ETH hedge | 102.15 SOL | +2.15% | 92.64% | 1.457 | 0.941 | 838 | 1 |

Profile medians tell the story more clearly:

| Profile family | Median final SOL-equiv | Median max DD | Median liquidations | Median HF < 1 bars |
|---|---:|---:|---:|---:|
| Base ETH hedge, no USDC debt | 117.16 SOL | 67.82% | 0 | 0 |
| Aggressive ETH hedge, no USDC debt | 103.41 SOL | 70.70% | 0 | 0 |
| Hold-like control | 100.20 SOL | 73.50% | 0 | 0 |
| Conservative USDC + soft ETH hedge | 84.57 SOL | 93.54% | 2 | 839 |
| Moderate USDC + soft ETH hedge | 57.15 SOL | 96.97% | 8 | 2,879 |
| Aggressive USDC + base/full-short | 31.24 SOL | 98.79% | 20 | 3,678 |

Median by USDC permission:

| USDC debt allowed? | Median final SOL-equiv | Median max DD | Median liquidations | Median HF < 1 bars | Median Sortino |
|---|---:|---:|---:|---:|---:|
| No | 100.21 SOL | 73.09% | 0 | 0 | 1.339 |
| Yes | 64.28 SOL | 95.88% | 6 | 2,013 | 1.434 |

This is the more useful conclusion: the strategy is not uniformly bad. The **USDC releverage module** is the dominant failure. ETH-only hedging, especially the soft ladder with ATR 10 / multiplier 3.0, improved SOL-equivalent outcome and drawdown materially in this sample. But adding USDC debt converts the system into a fragile leveraged account.

There is one critical modeling caveat: the current Kamino defaults set ETH borrow APY to `0.0`. That makes ETH shorts too cheap unless the real deployment can borrow ETH at negligible cost. I reran the best ETH-only profile with explicit ETH borrow carry:

| ETH borrow APY | Final SOL-equiv | SOL return | Max DD | Sortino | Interest paid | Liquidations |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 210.13 SOL | +110.13% | 46.66% | 1.825 | $0 | 0 |
| 5% | 203.29 SOL | +103.29% | 47.18% | 1.802 | $564 | 0 |
| 10% | 196.88 SOL | +96.88% | 47.70% | 1.780 | $1,129 | 0 |
| 20% | 182.98 SOL | +82.98% | 49.72% | 1.730 | $2,263 | 0 |

The ETH-only result survives realistic carry better than I expected. It still needs live Kamino borrow-rate modeling, but the performance is not solely an artifact of zero ETH interest.

Important caveat: trying to disable full-short by setting lower/upper bounds to zero does not truly disable that mode, because once `in_full_short_mode` is entered, the default cover ladder can still target ETH exposure. The clean implementation needs an explicit `enable_full_short_mode` switch and event labels that separate `mode` from `action_direction`.

## Likely Root Problems

1. **USDC releverage is the main risk engine.**
   The default run failed mostly because it borrowed USDC to buy more SOL while also carrying ETH debt. The targeted sweep shows USDC-enabled profiles had much worse SOL-equivalent outcomes, deeper drawdowns, more liquidations, and thousands of bars below borrow HF 1.0.

2. **The optimizer ranks USD Sortino, not SOL accumulation.**
   The strategy objective says maximize SOL, but our scorecard rewards USD path quality. USDC-enabled runs can have decent Sortino while ending with poor SOL-equivalent value and severe health-factor stress.

3. **USDC leverage is too aggressive in a bull market.**
   The strategy grew from 100 SOL to a peak of 1,298 SOL collateral. That sounds good, but it carried huge USDC and ETH debt. When the regime shifted, the debt stack dominated.

4. **The health-factor target is too low for this style.**
   `target_bullish_hf = 1.35` creates a fragile account for 24/7 crypto volatility. Bars below HF 1.0 and 23 liquidation events are unacceptable for a strategy intended to survive multi-year regimes.

5. **Full short mode is not cleanly separable from ordinary hedging.**
   The code has full-short mode, but it needs an explicit enable/disable switch. Otherwise "no full short" experiments can still inherit cover-ladder targets after the mode is entered.

6. **ETH hedge works better than expected here, but remains relative-value risk.**
   ETH-only hedging beat buy-and-hold in the best tested profiles. Still, ETH debt can become a second source of risk if ETH strengthens relative to SOL. This strategy is not simply "long SOL with insurance"; it is a SOL/ETH relative-value structure.

7. **Borrow-rate modeling needs to become first-class.**
   The current default ETH borrow APY is zero. A carry sensitivity still left the best ETH-only profile above buy-and-hold, but production backtests should ingest or parameterize ETH and USDC borrow rates rather than relying on fixed defaults.

8. **Event logs need sharper semantics.**
   A `full_short_cover` event can borrow more ETH if current exposure is below the cover-ladder target. The log should separate signal regime from executed direction.

9. **Liquidation health and borrow health are both needed in the risk rules.**
   The engine liquidates on liquidation health, but the strategy's safety logic references borrow health. The report shows many hours below borrow HF 1.0; this needs explicit policy.

## Recommended Next Experiments

These are not cosmetic tweaks; they are likely design changes.

1. Change the optimization objective from USD Sortino to SOL-equivalent final value or SOL-equivalent Sortino.

2. Add a hard maximum drawdown or minimum SOL-equivalent survival constraint to grid search. Any run with 98% drawdown should be rejected regardless of Sortino.

3. Treat USDC borrowing as opt-in and disabled by default until it passes SOL-equivalent and health-factor gates.

4. Split ETH hedge and USDC leverage into separate permission systems:
   - ETH hedge can react early.
   - USDC leverage should require persistent bullish confirmation and maybe a volatility filter.

5. Add a SOL-denominated benchmark and report every run in SOL terms by default.

6. Revisit whether borrowing ETH is the right hedge asset. A hedge that does not reliably preserve SOL-equivalent value may be worse than holding USDC collateral or simply reducing leverage.

7. Add explicit `enable_full_short_mode` and `enable_usdc_releverage` controls so experiments can isolate hedge-only, full-short, and USDC leverage behavior.

8. Add event-log fields for pre/post exposure, action direction, and blocked target amount due to HF cap.

9. Add borrow-rate scenarios to the harness and rank candidates both before and after carry cost.

## Files Generated

- `reports/sol_supertrend_3y/history_fixed_supertrend.csv`
- `reports/sol_supertrend_3y/strategy_events_fixed_supertrend.csv`
- `reports/sol_supertrend_3y/grid_fixed_supertrend.csv`
- `reports/sol_supertrend_3y/targeted_economic_sweep.csv`
- `reports/sol_supertrend_3y/targeted_economic_sweep_summary.json`
- `reports/sol_supertrend_3y/eth_borrow_rate_sensitivity.csv`
- `reports/sol_supertrend_3y/eth_borrow_rate_sensitivity.json`
- `reports/sol_supertrend_3y/summary_fixed_supertrend.json`
