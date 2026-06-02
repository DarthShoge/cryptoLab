# Radical Drawdown Candidate Results

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

Baseline for comparison is `v2_best_no_reserve`:

- Final SOL-equivalent: `424.902 SOL`
- Final USD: `$53,091.484`
- Max drawdown: `81.344%`
- 2024+ peak-to-trough: `64.839%`
- Sortino: `2.290`
- Min health factor: `1.598`

## Candidate Comparison

| Candidate | Best tested variant | Final SOL-eq | Final USD | Max DD | 2024+ DD | Sortino | Min HF | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1. TIPP high-watermark reserve | `tipp_sell0.05_esc0.10_max0.15_rebuy0.50_cool168_reset0.02` by 2024+ DD | `222.613` | `$27,815.436` | `83.858%` | `60.199%` | `2.153` | `1.702` | Reject tested settings: modest local DD improvement, severe SOL opportunity cost. |
| 2. CPPI/TIPP cushion cap | `cppi_protect0.55_mult1.5_green3` by max DD | `66.280` | `$8,281.626` | `74.869%` | `70.167%` | `1.795` | `1.350` | Reject: reduces headline max DD by permanently de-risking into the core floor. |
| 3. SOL/ETH hedge-failure circuit breaker | `hedge_failure_sell5pct` | `412.754` | `$51,573.650` | `81.162%` | `64.722%` | `2.284` | `1.635` | Keep for further tuning: small but constructive improvement with modest cost. |
| 4. Confirmed-crash partial fill | `confirmed_crash_partial_fill` | `239.423` | `$29,915.912` | `81.754%` | `70.403%` | `2.157` | `1.377` | Reject: stricter trigger helps old partial-fill variant but still damages recovery. |
| 5. Protected book | `protected_book_pnl0.50` by drawdown | `150.059` | `$18,749.861` | `76.504%` | `58.673%` | `2.034` | `1.563` | Promising structure, tested sizes too aggressive; retest smaller `5%-15%` allocations. |

## Readout

The best risk/reward result is Candidate 3, the SOL/ETH hedge-failure circuit breaker. It barely moves drawdown, but it improves both full-window and 2024+ drawdown while preserving most SOL-equivalent value. It is not radical enough yet, but it is the cleanest overlay to keep.

The best structural drawdown reducer is Candidate 5, the protected book. It can materially reduce drawdown without selling core SOL collateral, but the tested `25%` and `50%` realized-PnL protection settings starve compounding too much. The next pass should test smaller allocations and combine them with Candidate 3.

The reserve-heavy candidates did reduce certain drawdown windows, but the opportunity cost dominated. CPPI in particular confirmed the portfolio-insurance trap from the research: the floor protects the account by preventing participation in the recovery.

## Reports

- Candidate 1: `reports/sol_supertrend_3y/tipp_profit_lock_reserve_sweep_20260602_111944/report.md`
- Candidate 2: `reports/sol_supertrend_3y/cppi_exposure_cap_sweep_20260602_114111/report.md`
- Candidate 3: `reports/sol_supertrend_3y/hedge_failure_circuit_breaker_sweep_20260602_115413/report.md`
- Candidate 4: `reports/sol_supertrend_3y/confirmed_crash_partial_fill_sweep_20260602_114944/report.md`
- Candidate 5: `reports/sol_supertrend_3y/protected_book_sweep_20260602_115854/report.md`

## Recommended Next Experiment

Test the combination:

- `enable_hedge_failure_circuit_breaker = true`
- `hedge_failure_sell_fraction = 0.05`
- `enable_protected_book = true`
- `protected_book_realized_pnl_fraction` in `[0.05, 0.10, 0.15]`

This combination targets the only two tested mechanisms that improved drawdown without immediately destroying the book.
