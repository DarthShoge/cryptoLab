# RS Hedge Gate Robustness

Generated: 2026-06-25T14:51:41

## Objective

Check whether the relative-strength hedge gate remains useful across adjacent parameters, hedge floors, and subperiods.

## Headline

- Best full-sample variant: `rs_lb24_u0.0050` with $69,809.83, 558.702 SOL, and 57.431% max DD.
- Most stable variant by strict wins: `rs_lb24_u0.0050_g3_0.050_g2_0.050` with 4/4 subperiod wins versus the static hedge.
- The full-sample winner and the 4/4 robust winner are economically identical over 2021-2025, but differ in 2025-only behavior.
- The 4/4 robust winner gives up some 2025-only final value versus the plain full-sample winner, but it is the only tested variant that also lowers 2025-only drawdown versus static.
- Keep the RS gate structure: `24h` lookback, ETH must underperform SOL by at least `0.5%`, green-3 hedge floor `5%`. The robustness pass suggests raising green-2 from `2.5%` to `5%` if we prioritize cross-window drawdown consistency.

## Stability Ranking

| Scenario | Strict Wins | Avg USD Gap | Min USD Gap | Avg DD Gap | Max DD Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rs_lb24_u0.0050_g3_0.050_g2_0.050` | 4/4 | $763.13 | $334.75 | -0.118pp | -0.009pp |
| `rs_lb24_u0.0050_g3_0.050_g2_0.000` | 3/4 | $874.94 | $427.56 | 0.049pp | 0.248pp |
| `rs_lb24_u0.0050` | 3/4 | $855.72 | $427.56 | 0.052pp | 0.264pp |
| `rs_lb24_u0.0050_g3_0.050_g2_0.025` | 3/4 | $855.72 | $427.56 | 0.052pp | 0.264pp |
| `rs_lb24_u0.0025` | 3/4 | $803.45 | $363.12 | 0.051pp | 0.256pp |
| `rs_lb24_u0.0000` | 3/4 | $801.80 | $363.12 | 0.065pp | 0.311pp |
| `rs_lb36_u0.0000` | 3/4 | $775.62 | $381.30 | 0.154pp | 0.661pp |
| `rs_lb18_u0.0075` | 3/4 | $602.88 | $94.89 | 0.059pp | 0.260pp |
| `rs_lb18_u0.0050` | 3/4 | $576.93 | $85.22 | 0.180pp | 0.747pp |
| `rs_lb18_u0.0025` | 3/4 | $551.80 | $85.22 | 0.180pp | 0.747pp |

## Full-Sample Top Variants

| Scenario | Final USD | Final SOL | Max DD | USD Gap | DD Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rs_lb24_u0.0050` | $69,809.83 | 558.702 | 57.431% | $1,518.18 | -0.009pp |
| `rs_lb24_u0.0050_g3_0.050_g2_0.000` | $69,809.83 | 558.702 | 57.431% | $1,518.18 | -0.009pp |
| `rs_lb24_u0.0050_g3_0.050_g2_0.025` | $69,809.83 | 558.702 | 57.431% | $1,518.18 | -0.009pp |
| `rs_lb24_u0.0050_g3_0.050_g2_0.050` | $69,809.83 | 558.702 | 57.431% | $1,518.18 | -0.009pp |
| `rs_lb18_u0.0075` | $69,776.15 | 558.433 | 57.432% | $1,484.49 | -0.008pp |
| `rs_lb18_u0.0025` | $69,763.26 | 558.329 | 57.432% | $1,471.61 | -0.008pp |
| `rs_lb18_u0.0050` | $69,755.69 | 558.269 | 57.432% | $1,464.04 | -0.008pp |
| `rs_lb36_u0.0000` | $69,653.46 | 557.451 | 57.434% | $1,361.81 | -0.006pp |

## Artifacts

- Summary CSV: `reports/rs_gate_robustness_20260625_143914/summary.csv`
- Stability CSV: `reports/rs_gate_robustness_20260625_143914/stability.csv`
- Summary JSON: `reports/rs_gate_robustness_20260625_143914/summary.json`
