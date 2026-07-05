# BTC Directional + SOL/BTC Protective Hedge Test

Generated: 2026-06-25T11:56:46

## Objective

Test BTC in the directional long universe and SOL/BTC as dynamic protective short candidates.

## Reference

| Reference | Final USD | Final SOL | Max DD |
| --- | ---: | ---: | ---: |
| `current_best_sol_eth_eth_hedge` | $68,291.65 | 546.552 | 57.440% |

## Winners

| Selection | Scenario | Final USD | Final SOL | Max DD | Post-2024 DD | Avg Hedge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Highest USD | `current_best_sol_eth_eth_hedge` | $68,291.65 | 546.552 | 57.440% | 37.945% | 0.013x |
| Lowest DD | `current_best_sol_eth_eth_hedge` | $68,291.65 | 546.552 | 57.440% | 37.945% | 0.013x |

## Findings

- No BTC/SOL protective candidate variant beat the current reference on both final USD and max DD.
- BTC was added to the modeled research market with conservative simple lending parameters; this is a research assumption, not a live Kamino availability claim.
- `protective_short_symbols` chooses the weakest eligible hedge candidate from SOL/BTC and excludes the selected long asset.
- Adding BTC to the directional universe was the main source of degradation. Keeping ETH as the hedge with BTC direction still fell to $25,950.45 and 62.454% max DD.
- The best SOL/BTC-only directional variant reached $34,772.12 with 60.981% max DD; adding the light SOL/BTC hedge improved DD to 58.164% but lowered final USD to $33,567.13.

## Tested Variants

| Scenario | Direction | Protective | Final USD | Final SOL | Max DD | BTC Long Share |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `current_best_sol_eth_eth_hedge` | SOL, ETH | ETH | $68,291.65 | 546.552 | 57.440% | 0.000 |
| `sol_btc_direction_no_protective_hedge` | SOL, BTC | none | $34,772.12 | 278.288 | 60.981% | 0.162 |
| `sol_btc_direction_sol_btc_candidates_light` | SOL, BTC | SOL, BTC | $33,567.13 | 268.644 | 58.164% | 0.162 |
| `btc_direction_eth_hedge_light` | SOL, ETH, BTC | ETH | $25,950.45 | 207.687 | 62.454% | 0.093 |
| `btc_direction_no_protective_hedge` | SOL, ETH, BTC | none | $25,792.44 | 206.422 | 63.369% | 0.093 |
| `btc_direction_sol_btc_candidates_red10` | SOL, ETH, BTC | SOL, BTC | $23,545.68 | 188.441 | 61.189% | 0.093 |
| `btc_direction_sol_btc_candidates_light` | SOL, ETH, BTC | SOL, BTC | $23,253.17 | 186.100 | 61.189% | 0.093 |
| `btc_direction_sol_btc_candidates_red20` | SOL, ETH, BTC | SOL, BTC | $22,137.85 | 177.174 | 61.189% | 0.093 |

## Artifacts

- Summary CSV: `reports/btc_direction_sol_btc_protective_20260625_115413/summary.csv`
- Summary JSON: `reports/btc_direction_sol_btc_protective_20260625_115413/summary.json`
