# Fast-Break Crisis-Gated Partial-Fill Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can fast-break partial hedge filling help when it is gated to crisis overlap, avoiding the broad false positives that damaged the generic partial-fill sweep?

## Scenario Set

The sweep keeps the stateful-profit-lock incumbent and the best v2 fast-break no-partial-fill candidate as controls, then enables partial fill only when crisis mode is active.

- Base trigger: `-8%` 24h SOL return with `2.50x` realized-vol expansion.
- Requested hedge floor: `1.00`.
- Add HF for normal fast-break adds: `2.50`.
- Staged decay: `1.00 -> 0.75 -> 0.35`.
- Partial-fill gate: crisis mode must be active.
- Partial-fill min HF grid: `1.75`, `2.00`, `2.25`, `2.50`.
- Partial-fill episode budget grid: `10%`, `20%`, `35%`, `50%` of SOL collateral value.
- Cross-check: stricter `-10% / 2.50x` trigger family.

## Ranking By Sortino

| scenario | pfHF | budget | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | ETH debt | active% | pf events | pf max $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 28530.476 | 7.262 | 0 | 0.000 |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 27013.323 | 0.000 | 0 | 0.000 |
| pf_ret10%_vol2.50_hf1.75_budget0.20 | 1.750 | 0.200 | 244.074 | 30497.043 | 82.858 | 67.368 | 2.178 | 1.357 | 27452.593 | 5.088 | 49 | 18540.694 |
| pf_ret8_vol2.50_hf2.50_budget0.35 | 2.500 | 0.350 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 56 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.50 | 2.500 | 0.500 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 56 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.20 | 2.500 | 0.200 | 238.081 | 29748.238 | 82.703 | 70.023 | 2.164 | 1.376 | 22661.591 | 7.262 | 55 | 8153.282 |
| pf_ret8_vol2.50_hf1.75_budget0.20 | 1.750 | 0.200 | 214.550 | 26808.016 | 83.201 | 69.501 | 2.153 | 1.343 | 24904.325 | 7.262 | 88 | 14759.229 |
| pf_ret8_vol2.50_hf1.75_budget0.10 | 1.750 | 0.100 | 218.219 | 27266.521 | 84.238 | 67.395 | 2.150 | 1.375 | 24024.550 | 7.262 | 89 | 8198.015 |
| pf_ret8_vol2.50_hf1.75_budget0.50 | 1.750 | 0.500 | 210.569 | 26310.596 | 82.942 | 71.282 | 2.145 | 1.290 | 25075.075 | 7.262 | 88 | 20470.165 |
| pf_ret8_vol2.50_hf2.00_budget0.20 | 2.000 | 0.200 | 204.836 | 25594.308 | 83.074 | 71.404 | 2.134 | 1.309 | 21205.420 | 7.262 | 68 | 12536.890 |
| pf_ret8_vol2.50_hf2.25_budget0.35 | 2.250 | 0.350 | 206.842 | 25844.862 | 82.949 | 71.484 | 2.133 | 1.352 | 21348.274 | 7.262 | 74 | 9261.068 |
| pf_ret8_vol2.50_hf2.25_budget0.50 | 2.250 | 0.500 | 206.842 | 25844.862 | 82.949 | 71.484 | 2.133 | 1.352 | 21348.274 | 7.262 | 74 | 9261.068 |

## Ranking By Max Drawdown

| scenario | pfHF | budget | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | ETH debt | active% | pf events | pf max $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 28530.476 | 7.262 | 0 | 0.000 |
| pf_ret8_vol2.50_hf2.50_budget0.35 | 2.500 | 0.350 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 56 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.50 | 2.500 | 0.500 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 56 | 8137.138 |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 27013.323 | 0.000 | 0 | 0.000 |
| pf_ret8_vol2.50_hf2.50_budget0.10 | 2.500 | 0.100 | 209.882 | 26224.795 | 82.673 | 71.974 | 2.128 | 1.313 | 21564.858 | 7.262 | 59 | 7141.094 |
| pf_ret8_vol2.50_hf2.50_budget0.20 | 2.500 | 0.200 | 238.081 | 29748.238 | 82.703 | 70.023 | 2.164 | 1.376 | 22661.591 | 7.262 | 55 | 8153.282 |
| pf_ret10%_vol2.50_hf1.75_budget0.20 | 1.750 | 0.200 | 244.074 | 30497.043 | 82.858 | 67.368 | 2.178 | 1.357 | 27452.593 | 5.088 | 49 | 18540.694 |
| pf_ret8_vol2.50_hf1.75_budget0.50 | 1.750 | 0.500 | 210.569 | 26310.596 | 82.942 | 71.282 | 2.145 | 1.290 | 25075.075 | 7.262 | 88 | 20470.165 |
| pf_ret8_vol2.50_hf2.25_budget0.35 | 2.250 | 0.350 | 206.842 | 25844.862 | 82.949 | 71.484 | 2.133 | 1.352 | 21348.274 | 7.262 | 74 | 9261.068 |
| pf_ret8_vol2.50_hf2.25_budget0.50 | 2.250 | 0.500 | 206.842 | 25844.862 | 82.949 | 71.484 | 2.133 | 1.352 | 21348.274 | 7.262 | 74 | 9261.068 |
| pf_ret10%_vol2.50_hf2.25_budget0.35 | 2.250 | 0.350 | 207.019 | 25867.015 | 82.952 | 71.721 | 2.126 | 1.344 | 21356.781 | 5.088 | 56 | 16330.664 |
| pf_ret8_vol2.50_hf2.00_budget0.20 | 2.000 | 0.200 | 204.836 | 25594.308 | 83.074 | 71.404 | 2.134 | 1.309 | 21205.420 | 7.262 | 68 | 12536.890 |

## Ranking By 2024+ Peak-To-Trough

| scenario | pfHF | budget | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | ETH debt | active% | pf events | pf max $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 28530.476 | 7.262 | 0 | 0.000 |
| pf_ret10%_vol2.50_hf1.75_budget0.35 | 1.750 | 0.350 | 172.399 | 21541.220 | 87.406 | 64.846 | 2.083 | 1.371 | 19105.001 | 5.088 | 57 | 12542.631 |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 27013.323 | 0.000 | 0 | 0.000 |
| pf_ret10%_vol2.50_hf1.75_budget0.20 | 1.750 | 0.200 | 244.074 | 30497.043 | 82.858 | 67.368 | 2.178 | 1.357 | 27452.593 | 5.088 | 49 | 18540.694 |
| pf_ret8_vol2.50_hf1.75_budget0.10 | 1.750 | 0.100 | 218.219 | 27266.521 | 84.238 | 67.395 | 2.150 | 1.375 | 24024.550 | 7.262 | 89 | 8198.015 |
| pf_ret8_vol2.50_hf1.75_budget0.20 | 1.750 | 0.200 | 214.550 | 26808.016 | 83.201 | 69.501 | 2.153 | 1.343 | 24904.325 | 7.262 | 88 | 14759.229 |
| pf_ret8_vol2.50_hf2.00_budget0.10 | 2.000 | 0.100 | 208.367 | 26035.487 | 84.324 | 69.965 | 2.133 | 1.348 | 19483.225 | 7.262 | 63 | 8414.990 |
| pf_ret8_vol2.50_hf2.50_budget0.35 | 2.500 | 0.350 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 56 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.50 | 2.500 | 0.500 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 56 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.20 | 2.500 | 0.200 | 238.081 | 29748.238 | 82.703 | 70.023 | 2.164 | 1.376 | 22661.591 | 7.262 | 55 | 8153.282 |
| pf_ret8_vol2.50_hf2.25_budget0.10 | 2.250 | 0.100 | 149.438 | 18672.303 | 87.067 | 70.633 | 2.031 | 1.347 | 14375.834 | 7.262 | 67 | 6122.550 |
| pf_ret8_vol2.50_hf1.75_budget0.50 | 1.750 | 0.500 | 210.569 | 26310.596 | 82.942 | 71.282 | 2.145 | 1.290 | 25075.075 | 7.262 | 88 | 20470.165 |

## Ranking By Final SOL-Equivalent

| scenario | pfHF | budget | SOL-eq | USD | maxDD | 2024DD | Sortino | minHF | ETH debt | active% | pf events | pf max $ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_best_no_partial_fill |  |  | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 28530.476 | 7.262 | 0 | 0.000 |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 27013.323 | 0.000 | 0 | 0.000 |
| pf_ret10%_vol2.50_hf1.75_budget0.20 | 1.750 | 0.200 | 244.074 | 30497.043 | 82.858 | 67.368 | 2.178 | 1.357 | 27452.593 | 5.088 | 49 | 18540.694 |
| pf_ret8_vol2.50_hf2.50_budget0.20 | 2.500 | 0.200 | 238.081 | 29748.238 | 82.703 | 70.023 | 2.164 | 1.376 | 22661.591 | 7.262 | 55 | 8153.282 |
| pf_ret8_vol2.50_hf2.50_budget0.35 | 2.500 | 0.350 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 56 | 8137.138 |
| pf_ret8_vol2.50_hf2.50_budget0.50 | 2.500 | 0.500 | 237.677 | 29697.740 | 81.982 | 69.985 | 2.166 | 1.376 | 22606.627 | 7.262 | 56 | 8137.138 |
| pf_ret8_vol2.50_hf1.75_budget0.10 | 1.750 | 0.100 | 218.219 | 27266.521 | 84.238 | 67.395 | 2.150 | 1.375 | 24024.550 | 7.262 | 89 | 8198.015 |
| pf_ret8_vol2.50_hf1.75_budget0.20 | 1.750 | 0.200 | 214.550 | 26808.016 | 83.201 | 69.501 | 2.153 | 1.343 | 24904.325 | 7.262 | 88 | 14759.229 |
| pf_ret8_vol2.50_hf1.75_budget0.50 | 1.750 | 0.500 | 210.569 | 26310.596 | 82.942 | 71.282 | 2.145 | 1.290 | 25075.075 | 7.262 | 88 | 20470.165 |
| pf_ret8_vol2.50_hf2.50_budget0.10 | 2.500 | 0.100 | 209.882 | 26224.795 | 82.673 | 71.974 | 2.128 | 1.313 | 21564.858 | 7.262 | 59 | 7141.094 |
| pf_ret8_vol2.50_hf2.00_budget0.10 | 2.000 | 0.100 | 208.367 | 26035.487 | 84.324 | 69.965 | 2.133 | 1.348 | 19483.225 | 7.262 | 63 | 8414.990 |
| pf_ret10%_vol2.50_hf2.25_budget0.35 | 2.250 | 0.350 | 207.019 | 25867.015 | 82.952 | 71.721 | 2.126 | 1.344 | 21356.781 | 5.088 | 56 | 16330.664 |

## Control Versus Best Candidates

| Case | Scenario | Final SOL-eq | Final USD | Max DD | 2024+ Peak-To-Trough | Sortino | Min HF | Partial-fill events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent | incumbent_stateful_best | 402.064 | 50237.887 | 82.386 | 64.859 | 2.274 | 1.591 | 0 |
| V2 no partial fill | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |
| Best Sortino | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |
| Best Max DD | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |
| Best 2024 DD | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |
| Best SOL-eq | v2_best_no_partial_fill | 424.902 | 53091.484 | 81.344 | 64.839 | 2.290 | 1.598 | 0 |

## Forensic Notes

Decision: no promotion. The best v2 no-partial-fill fast-break configuration remains the current best across Sortino, max drawdown, 2024+ peak-to-trough drawdown, and final SOL-equivalent value.

The crisis gate helped conceptually, but it did not fix the execution problem. Crisis mode is still too broad to authorize extra ETH borrowing by itself. The best crisis-gated partial-fill Sortino candidate, `pf_ret10%_vol2.50_hf1.75_budget0.20`, finished at 244.07 SOL-equivalent versus 424.90 for v2 no partial fill, while also producing worse max drawdown, worse 2024+ drawdown, and a lower minimum health factor.

The best crisis-gated partial-fill max-drawdown candidates, `pf_ret8_vol2.50_hf2.50_budget0.35` and `pf_ret8_vol2.50_hf2.50_budget0.50`, only reduced max drawdown relative to most other partial-fill rows. They still lost to v2 no partial fill: 81.98% max drawdown versus 81.34%, 69.99% 2024+ peak-to-trough versus 64.84%, and only 237.68 SOL-equivalent versus 424.90.

The 2024/2025 red-mark problem is also not solved. The closest 2024+ peak-to-trough challenger, `pf_ret10%_vol2.50_hf1.75_budget0.35`, essentially tied the no-partial-fill 2024+ drawdown at 64.85%, but at the cost of a severe collapse in final SOL-equivalent value to 172.40 SOL and a much worse full-period max drawdown of 87.41%.

Event inspection shows the core failure mode: partial fills still fire repeatedly while the broader regime is not clearly broken. The crisis gate allowed fills in 2021, 2022, 2024, and 2025 because crisis mode can remain active or overlap with mixed trend votes. In particular, 2025 fills can occur while the standard vote is still mostly green. That means the strategy is adding ETH debt into conditions where the drawdown has started, but the market has not confirmed a durable downside regime strongly enough to justify the extra borrow.

Next hypothesis: keep partial fill disabled by default, and test a stricter execution gate rather than a larger budget. The most useful next sweep should require crisis overlap plus much weaker trend confirmation, for example:

- `fast_break_partial_fill_max_green <= 1`, or
- 3d or 1w SOL Supertrend bearish confirmation, or
- both crisis active and under-hedged with 0-1 green votes.

The broader conclusion is that extra ETH borrowing is not the right first response unless the strategy is very confident the market has already flipped. For the large 2025 giveback, a separate SOL de-risk-to-USDC rule may still be necessary because it directly reduces the SOL beta that ETH hedging cannot fully offset.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/fast_break_crisis_gated_partial_fill_sweep_20260530_004941/comparison.csv`
- Top-by-drawdown CSV: `reports/sol_supertrend_3y/fast_break_crisis_gated_partial_fill_sweep_20260530_004941/top_by_drawdown.csv`
- Top-by-2024-drawdown CSV: `reports/sol_supertrend_3y/fast_break_crisis_gated_partial_fill_sweep_20260530_004941/top_by_2024_drawdown.csv`
- Top-by-SOL-equivalent CSV: `reports/sol_supertrend_3y/fast_break_crisis_gated_partial_fill_sweep_20260530_004941/top_by_sol_equiv.csv`
- Summary JSON: `reports/sol_supertrend_3y/fast_break_crisis_gated_partial_fill_sweep_20260530_004941/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/fast_break_crisis_gated_partial_fill_sweep_20260530_004941`
