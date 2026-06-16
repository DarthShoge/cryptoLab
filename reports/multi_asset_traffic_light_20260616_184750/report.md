# Multi-Asset Traffic-Light Sweep

Generated: 2026-06-16T18:50:35

## Scope

- Window: `2021-01-01` to `2025-12-31` hourly.
- Directional assets with local cached data: `SOL`, `ETH`.
- Starting capital: 100 SOL collateral, matching the SOL-equivalent benchmark frame.
- Strategy: rotate long collateral to strongest traffic-light asset and short weakest asset.

## Checkpoint Comparison

| Reference | Final SOL | Max DD |
| --- | ---: | ---: |
| Current best DD checkpoint | 496.353 | 78.991% |
| Current Pareto checkpoint | 506.680 | 79.349% |

## Best Candidates

| Selection | Scenario | Final SOL | Max DD | Post-2024 DD | Liquidations |
| --- | --- | ---: | ---: | ---: | ---: |
| Lowest DD | `mlg3_msg0_long1.00_short0.00` | 408.001 | 57.399% | 37.466% | 0 |
| Highest SOL | `mlg3_msg0_long1.50_short0.00` | 1978.740 | 73.663% | 51.837% | 0 |
| Selected viable | `mlg3_msg0_long1.50_short0.00` | 1978.740 | 73.663% | 51.837% | 0 |

## Initial Read

This is a first mechanics test, not a tuned result. The important questions are whether cross-asset selection reduces drawdown, whether it preserves enough SOL-equivalent upside, and whether the chosen short leg creates unacceptable liquidation or debt-path risk.

The selected row is the lowest-drawdown candidate that clears the current best-DD checkpoint final SOL value. If no row clears that gate, selection falls back to the lowest-drawdown row.

## Findings

- The first viable multi-asset candidate is `mlg3_msg0_long1.50_short0.00`: `1978.740 SOL`, `73.663%` max drawdown, `51.837%` post-2024 drawdown, and no liquidations.
- This improves max drawdown by `5.328` percentage points versus the current best-DD checkpoint while producing far more final SOL.
- It still fails the investor-quality drawdown target because max drawdown remains above `50%`.
- The weak-asset short overlay hurt this pass. At the same 1.5x long exposure, adding a 0.30 short target reduced final SOL from `1978.740` to `1450.760` and increased max drawdown from `73.663%` to `77.091%`.
- A stricter four-green entry reduced participation too much. It lowered some post-2024 drawdown numbers but missed the recovery and failed the SOL objective.

## Next Experiment

The next test should focus on the viable long-only branch, not the short overlay. The obvious frontier is between `1.0x` and `1.5x` long exposure with a drawdown governor that steps exposure down after portfolio drawdown, then steps back up only after three-green recovery. That directly targets the gap between the 1.0x row (`408 SOL`, `57.399%` DD) and the 1.5x row (`1978.740 SOL`, `73.663%` DD).

## Selected Final Exposure

- Final SOL collateral: `0.000000`
- Final ETH collateral: `0.000000`
- Final USDC collateral: `247243.51`
- Final SOL debt value: `$0.00`
- Final ETH debt value: `$0.00`

## Artifacts

- Summary CSV: `reports/multi_asset_traffic_light_20260616_184750/summary.csv`
- Summary JSON: `reports/multi_asset_traffic_light_20260616_184750/summary.json`
- Selected history CSV: `reports/multi_asset_traffic_light_20260616_184750/selected_history.csv`
- Selected events JSON: `reports/multi_asset_traffic_light_20260616_184750/selected_events.json`
