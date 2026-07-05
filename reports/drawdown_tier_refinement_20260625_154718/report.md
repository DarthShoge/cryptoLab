# Drawdown Tier Refinement

Generated: 2026-06-25T15:52:49

## Objective

Refine the high-return 336-hour realized-volatility profile with drawdown tiers, looking for the lowest DD while preserving at least 420 SOL and checking for strict improvement over the checkpoint.

## Headline

- Best strict improvement: `rv336_tv0.018_min0.70_dd_soft` with $70,588.21, 564.932 SOL, and 57.075% DD.
- Lowest DD above 420 SOL: `rv336_dd_hard_lite_a` with $58,503.38, 468.214 SOL, and 52.247% DD.

## Ranked Results

| Scenario | Final USD | Final SOL | Max DD | USD Gap | SOL Gap | DD Gap | Avg Long | Capped Share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rv336_tv0.018_min0.70_dd_hard` | $52,164.04 | 417.479 | 51.266% | $-17,645.79 | -141.223 | -6.165pp | 0.508 | 0.220 |
| `rv336_dd_hard_lite_a` | $58,503.38 | 468.214 | 52.247% | $-11,306.45 | -90.488 | -5.184pp | 0.516 | 0.200 |
| `rv336_dd_hard_lite_b` | $55,501.50 | 444.190 | 52.971% | $-14,308.33 | -114.512 | -4.460pp | 0.521 | 0.199 |
| `rv336_dd_medium_plus_b` | $61,874.56 | 495.195 | 54.632% | $-7,935.27 | -63.507 | -2.799pp | 0.527 | 0.151 |
| `rv336_dd_hard_lite_c` | $58,464.89 | 467.906 | 54.686% | $-11,344.94 | -90.796 | -2.745pp | 0.520 | 0.188 |
| `rv336_dd_medium_plus_a` | $63,664.77 | 509.522 | 54.865% | $-6,145.06 | -49.180 | -2.566pp | 0.526 | 0.169 |
| `rv336_dd_hard_lite_d` | $54,883.11 | 439.241 | 54.889% | $-14,926.72 | -119.461 | -2.542pp | 0.521 | 0.167 |
| `rv336_tv0.018_min0.70_dd_medium` | $63,302.14 | 506.620 | 55.591% | $-6,507.69 | -52.082 | -1.840pp | 0.531 | 0.147 |
| `rv336_dd_medium_plus_c` | $61,718.55 | 493.946 | 55.621% | $-8,091.28 | -64.756 | -1.810pp | 0.531 | 0.137 |
| `rv336_tv0.018_min0.70_dd_soft` | $70,588.21 | 564.932 | 57.075% | $778.38 | 6.230 | -0.356pp | 0.538 | 0.116 |
| `rs_lb24_u0.0050_g3_0.050_g2_0.050` | $69,809.83 | 558.702 | 57.431% | $-0.00 | 0.000 | -0.000pp | 0.555 | 0.000 |
| `rv336_tv0.018_min0.70` | $76,356.74 | 611.098 | 58.558% | $6,546.91 | 52.396 | 1.127pp | 0.546 | 0.050 |

## Artifacts

- Summary CSV: `reports/drawdown_tier_refinement_20260625_154718/summary.csv`
- Summary JSON: `reports/drawdown_tier_refinement_20260625_154718/summary.json`
