# Adaptive Protective Hedge Gate Refinement

Generated: 2026-06-25T13:04:45

## Objective

Refine the useful single-gate region from the first adaptive hedge sweep.

## Reference

| Reference | Final USD | Final SOL | Max DD |
| --- | ---: | ---: | ---: |
| `static_light_hedge` | $68,291.65 | 546.552 | 57.440% |

## Winners

| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Hedge | Gate Active |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Highest USD | `rs_refine_lb24_u0.005` | $69,809.83 | 558.702 | 57.431% | 37.945% | 0.005x | 0.821 |
| Lowest DD | `rs_refine_lb24_u0.005` | $69,809.83 | 558.702 | 57.431% | 37.945% | 0.005x | 0.821 |
| Strict reference improvement | `rs_refine_lb24_u0.005` | $69,809.83 | 558.702 | 57.431% | 37.945% | 0.005x | 0.821 |

## Findings

- Strict reference improvement: `rs_refine_lb24_u0.005` with $69,809.83, 558.702 SOL, and 57.431% max DD.
- Versus the static light hedge checkpoint, this adds $1,518.18, adds 12.150 SOL-equivalent, and reduces max DD by 0.009pp.
- The winning gate keeps the same light hedge floors but only permits the ETH short when ETH has underperformed the SOL basis by at least 0.5% over the prior 24 hours.
- Drawdown-slope gates improved final USD in the 5%-7% drawdown / 0.3%-0.5% worsening region, but their max DD was slightly worse than the static checkpoint.
- Combining gates was not useful in the coarse sweep; it filtered out too much of the hedge and damaged compounding.

## Top Variants By Final USD

| Scenario | Final USD | Final SOL | Max DD | Avg Hedge | Gate Active |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rs_refine_lb24_u0.005` | $69,809.83 | 558.702 | 57.431% | 0.005x | 0.821 |
| `rs_refine_lb18_u0.005` | $69,755.69 | 558.269 | 57.432% | 0.004x | 0.817 |
| `dd_refine_dd0.070_w0.0050` | $69,675.75 | 557.629 | 57.529% | 0.002x | 0.767 |
| `dd_refine_dd0.070_w0.0030` | $69,665.59 | 557.548 | 57.501% | 0.003x | 0.791 |
| `rs_refine_lb36_u0.000` | $69,653.46 | 557.451 | 57.434% | 0.006x | 0.840 |
| `dd_refine_dd0.050_w0.0050` | $69,640.31 | 557.345 | 57.545% | 0.002x | 0.771 |
| `rs_refine_lb24_u0.000` | $69,619.57 | 557.179 | 57.431% | 0.005x | 0.836 |
| `dd_refine_dd0.050_w0.0030` | $69,597.78 | 557.005 | 57.509% | 0.003x | 0.795 |
| `dd_refine_dd0.050_w0.0060` | $69,593.60 | 556.972 | 57.531% | 0.002x | 0.765 |
| `dd_refine_dd0.030_w0.0050` | $69,578.72 | 556.852 | 57.551% | 0.002x | 0.770 |

## Artifacts

- Summary CSV: `reports/adaptive_hedge_gate_refinement_20260625_125612/summary.csv`
- Summary JSON: `reports/adaptive_hedge_gate_refinement_20260625_125612/summary.json`
