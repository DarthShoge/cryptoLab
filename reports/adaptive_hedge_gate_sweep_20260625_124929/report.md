# Adaptive Protective Hedge Gate Sweep

Generated: 2026-06-25T12:54:51

## Objective

Test path-aware gates for the current SOL/ETH light protective hedge.

## Reference

| Reference | Final USD | Final SOL | Max DD |
| --- | ---: | ---: | ---: |
| `static_light_hedge` | $68,291.65 | 546.552 | 57.440% |

## Winners

| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Hedge | Gate Active |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Highest USD | `dd_slope_dd0.050_w0.0050` | $69,640.31 | 557.345 | 57.545% | 37.942% | 0.002x | 0.771 |
| Lowest DD | `rs_lb24_u0.000` | $69,619.57 | 557.179 | 57.431% | 37.945% | 0.005x | 0.836 |
| Strict reference improvement | `rs_lb24_u0.000` | $69,619.57 | 557.179 | 57.431% | 37.945% | 0.005x | 0.836 |

## Findings

- Strict reference improvement: `rs_lb24_u0.000` with $69,619.57, 557.179 SOL, and 57.431% max DD.
- Drawdown-slope gates test whether the hedge should activate only while portfolio drawdown is worsening.
- Relative-strength gates test whether ETH should only be shorted when it is underperforming the SOL basis.

## Top Variants By Final USD

| Scenario | Final USD | Final SOL | Max DD | Avg Hedge | Gate Active |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dd_slope_dd0.050_w0.0050` | $69,640.31 | 557.345 | 57.545% | 0.002x | 0.771 |
| `rs_lb24_u0.000` | $69,619.57 | 557.179 | 57.431% | 0.005x | 0.836 |
| `dd_slope_dd0.100_w0.0000` | $69,533.00 | 556.487 | 57.438% | 0.005x | 0.841 |
| `dd_slope_dd0.050_w0.0025` | $69,526.72 | 556.436 | 57.517% | 0.004x | 0.803 |
| `dd_slope_dd0.100_w0.0025` | $69,428.42 | 555.650 | 57.480% | 0.003x | 0.795 |
| `rs_lb72_u0.000` | $69,383.44 | 555.290 | 57.440% | 0.006x | 0.859 |
| `static_light_hedge` | $68,291.65 | 546.552 | 57.440% | 0.013x | 1.000 |
| `rs_lb168_u0.000` | $67,751.05 | 542.225 | 58.827% | 0.007x | 0.868 |

## Artifacts

- Summary CSV: `reports/adaptive_hedge_gate_sweep_20260625_124929/summary.csv`
- Summary JSON: `reports/adaptive_hedge_gate_sweep_20260625_124929/summary.json`
