# USD Protective Hedge Sweep

Generated: 2026-06-25T10:48:59

## Objective

Adapt the SOL strategy's hedge-floor shorting mechanism to the USD-first multi-asset strategy.

## Anchor

| Anchor | Final USD | Final SOL | Max DD |
| --- | ---: | ---: | ---: |
| `signal_g3_1.025_g4_1.075` | $66,475.12 | 532.014 | 59.294% |

## Winners

| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Hedge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Highest USD | `anchor_no_hedge` | $66,475.12 | 532.014 | 59.294% | 38.686% | 0.000x |
| Lowest DD | `hedge_g3_0.05_g2_0.05_r_0.20` | $66,203.01 | 529.836 | 57.440% | 37.926% | 0.105x |
| Best USD <=58% DD | `hedge_g3_0.05_g2_0.05_r_0.20` | $66,203.01 | 529.836 | 57.440% | 37.926% | 0.105x |

## Findings

- Best drawdown candidate: `hedge_g3_0.05_g2_0.05_r_0.20` with $66,203.01, 529.836 SOL, and 57.440% max DD.
- Highest final-value candidate remains `anchor_no_hedge` with $66,475.12 and 59.294% max DD.
- No protective hedge variant in this grid improved max DD while also preserving final USD at or above the signal-governor anchor.
- This sweep uses ETH debt as the protective short instrument and sizes it against selected or existing long collateral.

## Artifacts

- Summary CSV: `reports/usd_protective_hedge_20260625_104136/summary.csv`
- Summary JSON: `reports/usd_protective_hedge_20260625_104136/summary.json`
