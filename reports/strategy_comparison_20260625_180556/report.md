# High-Sortino Candidate vs Highest-Return Strategy

Generated: 2026-06-25T18:06:57

## Strategies Compared

- `static_long1.50`: the highest-return branch from the earlier USD-first/multi-asset sweep. It is long-only, uses three-green traffic-light entry, borrows USDC to hold 1.50x long exposure when a qualifying long is present, and has no short hedge or drawdown/volatility governor.
- `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012`: the highest-Sortino row from the latest return-recovery microsearch. It keeps the SOL/ETH traffic-light engine but adds the RS-gated ETH protective short, a 336-hour realized-volatility governor, drawdown exposure tiers, and a recovery boost when drawdown is improving.

## Headline

- Highest-return strategy finished with $247,243.51 (1978.740 SOL) but accepted 73.663% max DD and Sortino 1.937.
- High-Sortino candidate finished with $75,273.09 (602.426 SOL), 55.594% max DD, and Sortino 2.314.
- The production candidate gives up extreme terminal upside versus 1.50x static leverage, but it turns the strategy into a materially cleaner capital-raising profile: lower drawdown, higher Sortino, higher minimum health factor, and far less constant leverage.

## Portfolio Characteristics

| Strategy | Final USD | Final SOL | Max DD | Post-2024 DD | Sortino | Min HF | Bars HF <1.5 | Avg Long | Avg Short | Interest Paid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_long1.50` | $247,243.51 | 1978.740 | 73.663% | 51.837% | 1.937 | 1.239 | 6681 | 1.500 | 0.000 | $12,440.89 |
| `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012` | $75,273.09 | 602.426 | 55.594% | 37.930% | 2.314 | 1.392 | 81 | 0.534 | 0.005 | $1,668.47 |

## Regime Handling

| Strategy | Regime | Return | Drawdown | Avg Long | Avg Short | Min HF | Boost Share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_long1.50` | full_2021_2025 | 160,091.24% | 73.663% | 1.500 | 0.000 | 1.239 | 0.000 |
| `static_long1.50` | bull_2021 | 9,715.23% | 73.022% | 1.500 | 0.000 | 1.239 | 0.000 |
| `static_long1.50` | crash_2022 | -54.91% | 73.663% | 1.500 | 0.000 | 1.246 | 0.000 |
| `static_long1.50` | recovery_2023 | 808.04% | 54.645% | 1.500 | 0.000 | 1.242 | 0.000 |
| `static_long1.50` | post_2024 | 310.58% | 51.837% | 1.500 | 0.000 | 1.247 | 0.000 |
| `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012` | full_2021_2025 | 48,649.35% | 55.594% | 0.534 | 0.005 | 1.392 | 0.002 |
| `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012` | bull_2021 | 4,691.45% | 47.822% | 0.670 | 0.006 | 1.483 | 0.004 |
| `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012` | crash_2022 | -34.81% | 55.400% | 0.314 | 0.003 | 1.392 | 0.003 |
| `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012` | recovery_2023 | 414.98% | 40.051% | 0.657 | 0.004 | 1.503 | 0.003 |
| `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012` | post_2024 | 208.92% | 37.930% | 0.514 | 0.005 | 1.480 | 0.001 |

## Similarities

- Both strategies start from 100 SOL collateral and use the same cached hourly SOL/ETH data from 2021-01-01 to 2025-12-31.
- Both use the multi-timeframe traffic-light model to decide whether there is a qualifying long candidate.
- Both can borrow USDC to add long exposure, so both are path-dependent and sensitive to sharp SOL/ETH drawdowns.
- Neither run liquidated in the backtest.

## Differences

- `static_long1.50` is simple and always targets 1.50x when the long signal is active. It maximizes upside in sustained bull regimes but does little to protect capital when the path turns against it.
- The high-Sortino candidate is explicitly stateful. It reduces exposure through drawdown tiers, trims exposure under high realized volatility, uses a small RS-gated ETH protective short, and only re-risks through a recovery boost when drawdown is improving.
- The high-return strategy has much higher terminal USD/SOL because it remains aggressively exposed through the whole path. The cost is a 73.663% max drawdown and weaker Sortino.
- The high-Sortino candidate accepts lower terminal wealth in exchange for a 18.069 percentage point drawdown reduction versus `static_long1.50` and a higher Sortino.

## Regime Notes

- In strong upside regimes, `static_long1.50` should dominate because it keeps more gross long exposure. That is exactly what happens in the full-period final value.
- In crash or high-volatility regimes, the production candidate should preserve capital better because its drawdown tiers and vol governor reduce long exposure. The ETH short is small on average, so most protection comes from exposure control rather than hedge PnL.
- In recovery regimes, the production candidate tries to avoid the old failure mode of staying de-risked too long. The recovery boost is deliberately sparse, but it is enough in the microsearch to recover return without reverting to a 70%+ drawdown profile.
- The highest-return strategy is better viewed as a benchmark for upside appetite, not as an investor-ready model: its health factor minimum is lower, it spends more time below HF 1.5, and its drawdown would likely be difficult to raise institutional capital around.

## Production Implications

- Treat `rv336_dd_e_tight_2_rec_t1.250_dd0.22_w0.012` as the productionisation candidate if the objective is risk-adjusted return and capital-raising viability.
- Keep `static_long1.50` as the upside benchmark and stress-test comparator, not as the recommended production policy.
- Before production, validate the high-Sortino candidate out-of-sample and with execution assumptions: borrow availability, borrow rates, slippage, rebalance latency, stale oracle behavior, and Kamino health-factor constraints.

## Artifacts

- Summary CSV: `reports/strategy_comparison_20260625_180556/summary.csv`
- Regime CSV: `reports/strategy_comparison_20260625_180556/regime_summary.csv`
- High-Sortino history: `reports/strategy_comparison_20260625_180556/high_sortino_history.csv`
- Highest-return history: `reports/strategy_comparison_20260625_180556/highest_return_history.csv`
