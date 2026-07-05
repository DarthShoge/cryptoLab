# Protected Book Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Does splitting realized hedge profits into an accumulation book and protected USDC book reduce drawdown without selling SOL collateral directly?

## Ranking

| scenario | fraction | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | maxProtected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_protected_book |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0.000 |
| protected_book_pnl0.25 | 0.250 | 219.361 | 27409.170 | 81.349 | 60.259 | 2.138 | 1.702 | 1950.394 |
| protected_book_pnl0.50 | 0.500 | 150.059 | 18749.861 | 76.504 | 58.673 | 2.034 | 1.563 | 3914.062 |

## Readout

This candidate is structurally promising, but the tested allocation sizes are too aggressive for the SOL-accumulation objective.

The protected book does reduce drawdown without directly selling SOL collateral. Allocating `25%` of realized hedge PnL to protected USDC improves 2024+ peak-to-trough from `64.839%` to `60.259%`, but leaves full-window max drawdown essentially unchanged and cuts final SOL-equivalent to `219.361 SOL`. Allocating `50%` is much more defensive: max drawdown improves from `81.344%` to `76.504%` and 2024+ peak-to-trough improves to `58.673%`. The cost is severe: final SOL-equivalent falls to `150.059 SOL` and Sortino drops to `2.034`.

This is less destructive than CPPI in mechanism, because it protects realized hedge profits rather than rotating core SOL collateral, but it still starves the accumulation loop. The result is a clear dial: higher protected-book allocation buys drawdown reduction by giving up SOL compounding.

Conclusion: keep this as the best structural candidate for further tuning, but do not promote the tested `25%` or `50%` settings as final. The next useful grid should test much smaller allocations, probably `5%`, `10%`, and `15%`, possibly combined with the hedge-failure circuit breaker.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/protected_book_sweep_20260602_115854/comparison.csv`
- Ranked CSV: `reports/sol_supertrend_3y/protected_book_sweep_20260602_115854/ranked.csv`
- Summary JSON: `reports/sol_supertrend_3y/protected_book_sweep_20260602_115854/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/protected_book_sweep_20260602_115854`
