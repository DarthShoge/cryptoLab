# USD-First Governed Multi-Asset Sweep

Generated: 2026-06-25T02:26:31

## Objective

Primary objective is final USD equity. SOL-equivalent is tracked as a secondary outcome to test whether maximizing USD also maximizes SOL by effect.

## Setup

- Window: `2021-01-01` to `2025-12-31` hourly.
- Directional assets with local cached data: `SOL`, `ETH`.
- Starting capital: 100 SOL collateral.
- Signal: three-green multi-timeframe traffic-light entry.
- Short overlay disabled because the prior sweep showed it hurt both USD/SOL outcome and drawdown.

## Checkpoint Comparison

| Reference | Final USD | Final SOL equiv | Max DD |
| --- | ---: | ---: | ---: |
| Prior static 1.5x multi-asset | $247,243.51 | 1978.740 | 73.663% |
| Prior SOL checkpoint | $62,003.62 | 496.353 | 78.991% |

## Winners

| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Min HF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Lowest DD | `governor_strict` | $11,270.25 | 90.198 | 54.515% | 40.403% | 1.269 |
| Highest USD | `static_long1.50` | $247,243.51 | 1978.740 | 73.663% | 51.837% | 1.239 |
| Highest SOL | `static_long1.50` | $247,243.51 | 1978.740 | 73.663% | 51.837% | 1.239 |
| Best USD <=60% DD | `static_long1.05` | $65,410.66 | 523.495 | 59.354% | 38.459% | 1.466 |

## Does USD Maximization Also Maximize SOL?

Yes for this sweep.
The highest-USD scenario is `static_long1.50` and the highest-SOL scenario is `static_long1.50`.

## Findings

- USD and SOL-equivalent rank together in this limited SOL/ETH universe. The highest-USD row is also the highest-SOL row: `static_long1.50`.
- The best sub-60% drawdown row is `static_long1.05`: `$65,410.66`, `523.495 SOL`, `59.354%` max DD, `38.459%` post-2024 DD, and no liquidations.
- `static_long1.05` beats the prior SOL checkpoint on both objectives: +`$3,407.04` final USD, +`27.142` SOL-equivalent, and `19.636` percentage points lower max DD.
- `static_long1.075` is the next useful boundary row: `$70,853.34`, `567.054 SOL`, and `60.246%` max DD. This is attractive if a roughly-60% cap is acceptable, but it misses a strict <=60% DD gate.
- The reactive drawdown governor is dominated in this pass. `governor_mild` ends near the 1.05x USD level but has worse DD (`66.546%`), while `governor_strict` lowers DD to `54.515%` but destroys final capital.
- The next governor should be proactive rather than reactive: reduce exposure before equity drawdown compounds, using traffic-light deterioration, volatility, or trend-break conditions rather than portfolio drawdown alone.

## Artifacts

- Summary CSV: `reports/usd_first_multi_asset_governor_20260625_022251/summary.csv`
- Summary JSON: `reports/usd_first_multi_asset_governor_20260625_022251/summary.json`
- Winner history/event files were generated locally in `reports/usd_first_multi_asset_governor_20260625_022251`; large history CSVs are not intended for versioned commits by default.
