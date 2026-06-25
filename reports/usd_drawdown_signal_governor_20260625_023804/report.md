# USD Drawdown Signal-Governor Sweep

Generated: 2026-06-25T02:44:37

## Objective

Minimize USD drawdown while preserving the USD-first advantage from the static 1.05x row.

## Anchors

| Anchor | Final USD | Final SOL | Max DD |
| --- | ---: | ---: | ---: |
| Static 1.05x | $65,410.66 | 523.495 | 59.354% |
| Static 1.075x | $70,853.34 | 567.054 | 60.246% |
| Prior SOL checkpoint | $62,003.62 | 496.353 | 78.991% |

## Winners

| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Exposure |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Highest USD | `signal_g3_1.050_g4_1.150` | $84,895.51 | 679.436 | 61.830% | 39.898% | 0.582x |
| Lowest DD | `signal_g3_0.750_g4_1.075` | $43,682.24 | 349.598 | 56.480% | 34.398% | 0.484x |
| Best USD <=60% DD | `signal_g3_1.000_g4_1.100` | $69,526.14 | 556.432 | 59.921% | 38.714% | 0.555x |
| Best anchor improvement | `signal_g3_1.025_g4_1.075` | $66,475.12 | 532.014 | 59.294% | 38.686% | 0.555x |

## Findings

- Best anchor-improvement candidate: `signal_g3_1.025_g4_1.075` with $66,475.12, 532.014 SOL, and 59.294% max DD.
- This is a valid improvement over static 1.05x: +`$1,064.45` final USD, +`8.519` SOL-equivalent, and `0.061` percentage points lower max DD.
- The best row under a 60% DD cap is `signal_g3_1.000_g4_1.100`: `$69,526.14`, `556.432 SOL`, `59.921%` max DD. It offers more wealth but accepts `0.567` percentage points more DD than static 1.05x.
- Highest USD in this sweep is `signal_g3_1.050_g4_1.150`: `$84,895.51`, `679.436 SOL`, `61.830%` max DD. This is attractive only if the DD budget is relaxed above 60%.
- Proactive traffic-light deterioration works better than reactive equity drawdown gating in this test: it improves the frontier before portfolio damage becomes the control variable.
- Next test should combine this signal governor with a fast-break/volatility cut, aiming to bring the best anchor-improvement row below `57%` DD without giving back its USD advantage.

## Artifacts

- Summary CSV: `reports/usd_drawdown_signal_governor_20260625_023804/summary.csv`
- Summary JSON: `reports/usd_drawdown_signal_governor_20260625_023804/summary.json`
- Winner event files generated locally: `reports/usd_drawdown_signal_governor_20260625_023804`
