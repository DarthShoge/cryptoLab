# USD Protective Hedge Refinement

Generated: 2026-06-25T11:01:49

## Objective

Refine the ETH protective short hedge around the only promising low-floor region.

## Anchor

| Anchor | Final USD | Final SOL | Max DD |
| --- | ---: | ---: | ---: |
| `signal_g3_1.025_g4_1.075` | $66,475.12 | 532.014 | 59.294% |

## Winners

| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Hedge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Highest USD | `refine_g3_0.050_g2_0.025_r_0.000` | $68,291.65 | 546.552 | 57.440% | 37.945% | 0.013x |
| Lowest DD | `refine_g3_0.050_g2_0.025_r_0.000` | $68,291.65 | 546.552 | 57.440% | 37.945% | 0.013x |
| Best USD <=58% DD | `refine_g3_0.050_g2_0.025_r_0.000` | $68,291.65 | 546.552 | 57.440% | 37.945% | 0.013x |
| Strict anchor improvement | `refine_g3_0.050_g2_0.025_r_0.000` | $68,291.65 | 546.552 | 57.440% | 37.945% | 0.013x |

## Findings

- Best drawdown candidate: `refine_g3_0.050_g2_0.025_r_0.000` with $68,291.65, 546.552 SOL, and 57.440% max DD.
- Highest final-value candidate: `refine_g3_0.050_g2_0.025_r_0.000` with $68,291.65 and 57.440% max DD.
- Strict anchor improvement: `refine_g3_0.050_g2_0.025_r_0.000` with $68,291.65, 546.552 SOL, and 57.440% max DD.

## Artifacts

- Summary CSV: `reports/usd_protective_hedge_refinement_20260625_105110/summary.csv`
- Summary JSON: `reports/usd_protective_hedge_refinement_20260625_105110/summary.json`
