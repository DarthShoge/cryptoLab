# Drawdown Containment Isolated Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

The first containment sweep bundled hedge-floor risk reduction with capital-deployment blockers. This report isolates those effects:

- floor-only containment: raise the ETH hedge floor after portfolio drawdown, but keep reinvestment/rebuy behavior enabled;
- blocker isolation: use the same `20% drawdown / 1.00 hedge floor` candidate and toggle each blocker separately.

## Ranking By Max Drawdown

| scenario | group | trigger | floor | block_reinvest | final_sol_eq | max_dd | sortino | min_hf | 2024_giveback | trough_sol | active_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor_only_dd0.20_floor1.25 | floor_only | 0.200 | 1.250 | False | 434.456 | 80.721 | 2.330 | 1.407 | -48.415 | 388.459 | 74.737 |
| incumbent_stateful_best | control |  |  |  | 402.064 | 82.386 | 2.274 | 1.591 | -48.486 | 355.692 | 0.000 |
| floor_only_dd0.25_floor1.00 | floor_only | 0.250 | 1.000 | False | 384.415 | 82.526 | 2.282 | 1.452 | -49.498 | 349.348 | 66.054 |
| floor_only_dd0.20_floor1.00 | floor_only | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| dd20_floor1_block_rebuy_only | blocker_isolation | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| dd20_floor1_block_releverage_only | blocker_isolation | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| floor_only_dd0.20_floor0.50 | floor_only | 0.200 | 0.500 | False | 382.382 | 82.966 | 2.277 | 1.462 | -48.486 | 338.233 | 74.248 |
| floor_only_dd0.25_floor0.50 | floor_only | 0.250 | 0.500 | False | 373.283 | 83.288 | 2.260 | 1.559 | -48.456 | 329.839 | 66.316 |
| floor_only_dd0.25_floor1.25 | floor_only | 0.250 | 1.250 | False | 371.886 | 83.293 | 2.271 | 1.452 | -49.466 | 337.773 | 66.054 |
| floor_only_dd0.15_floor0.75 | floor_only | 0.150 | 0.750 | False | 398.551 | 83.513 | 2.271 | 1.480 | -44.914 | 329.078 | 80.576 |
| floor_only_dd0.25_floor0.75 | floor_only | 0.250 | 0.750 | False | 380.506 | 83.591 | 2.274 | 1.470 | -48.088 | 333.182 | 66.419 |
| dd20_floor1_block_reinvestment_only | blocker_isolation | 0.200 | 1.000 | True | 301.929 | 83.747 | 2.232 | 1.444 | -49.132 | 270.102 | 73.618 |

## Ranking By SOL-Equivalent

| scenario | group | trigger | floor | block_reinvest | final_sol_eq | max_dd | sortino | min_hf | 2024_giveback | trough_sol | active_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor_only_dd0.20_floor1.25 | floor_only | 0.200 | 1.250 | False | 434.456 | 80.721 | 2.330 | 1.407 | -48.415 | 388.459 | 74.737 |
| floor_only_dd0.20_floor1.00 | floor_only | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| dd20_floor1_block_rebuy_only | blocker_isolation | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| dd20_floor1_block_releverage_only | blocker_isolation | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| incumbent_stateful_best | control |  |  |  | 402.064 | 82.386 | 2.274 | 1.591 | -48.486 | 355.692 | 0.000 |
| floor_only_dd0.15_floor0.75 | floor_only | 0.150 | 0.750 | False | 398.551 | 83.513 | 2.271 | 1.480 | -44.914 | 329.078 | 80.576 |
| floor_only_dd0.25_floor1.00 | floor_only | 0.250 | 1.000 | False | 384.415 | 82.526 | 2.282 | 1.452 | -49.498 | 349.348 | 66.054 |
| floor_only_dd0.20_floor0.50 | floor_only | 0.200 | 0.500 | False | 382.382 | 82.966 | 2.277 | 1.462 | -48.486 | 338.233 | 74.248 |

## Readout

Best by max drawdown: `floor_only_dd0.20_floor1.25` at 80.721% max drawdown, 434.456 SOL-equivalent, Sortino 2.330.

Best by SOL-equivalent: `floor_only_dd0.20_floor1.25` at 434.456 SOL-equivalent, 80.721% max drawdown, Sortino 2.330.

Incumbent control: 402.064 SOL-equivalent, 82.386% max drawdown, Sortino 2.274.

Promotion decision: do not promote yet. The raw champion improves the headline metrics, but its minimum health factor is 1.407, below the 1.50 safety line we have been using for promoted configurations. Treat this as a strong hypothesis, not a production-ready default.

## Floor-Only Findings

| scenario | group | trigger | floor | block_reinvest | final_sol_eq | max_dd | sortino | min_hf | 2024_giveback | trough_sol | active_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor_only_dd0.20_floor1.25 | floor_only | 0.200 | 1.250 | False | 434.456 | 80.721 | 2.330 | 1.407 | -48.415 | 388.459 | 74.737 |
| floor_only_dd0.25_floor1.00 | floor_only | 0.250 | 1.000 | False | 384.415 | 82.526 | 2.282 | 1.452 | -49.498 | 349.348 | 66.054 |
| floor_only_dd0.20_floor1.00 | floor_only | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| floor_only_dd0.20_floor0.50 | floor_only | 0.200 | 0.500 | False | 382.382 | 82.966 | 2.277 | 1.462 | -48.486 | 338.233 | 74.248 |
| floor_only_dd0.25_floor0.50 | floor_only | 0.250 | 0.500 | False | 373.283 | 83.288 | 2.260 | 1.559 | -48.456 | 329.839 | 66.316 |
| floor_only_dd0.25_floor1.25 | floor_only | 0.250 | 1.250 | False | 371.886 | 83.293 | 2.271 | 1.452 | -49.466 | 337.773 | 66.054 |
| floor_only_dd0.15_floor0.75 | floor_only | 0.150 | 0.750 | False | 398.551 | 83.513 | 2.271 | 1.480 | -44.914 | 329.078 | 80.576 |
| floor_only_dd0.25_floor0.75 | floor_only | 0.250 | 0.750 | False | 380.506 | 83.591 | 2.274 | 1.470 | -48.088 | 333.182 | 66.419 |

Floor-only containment behaves much better than the bundled policy. The best raw candidate, `floor_only_dd0.20_floor1.25`, improves final SOL-equivalent, Sortino, and global max drawdown at the same time. This confirms the earlier suspicion: the containment idea was not inherently broken; the damaging part was blocking reinvestment.

The catch is risk capacity. The best raw candidate reaches a 1.407 minimum health factor, which is too close to liquidation mechanics for promotion. The safety-qualified rows with min HF above 1.50 do not beat the incumbent on max drawdown. So the next experiment should keep the floor-only concept but constrain health factor, partial fill, or hedge floor so that the improvement does not come from unsafe leverage.

## Blocker Isolation

| scenario | group | trigger | floor | block_reinvest | final_sol_eq | max_dd | sortino | min_hf | 2024_giveback | trough_sol | active_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dd20_floor1_block_rebuy_only | blocker_isolation | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| dd20_floor1_block_releverage_only | blocker_isolation | 0.200 | 1.000 | False | 406.704 | 82.963 | 2.311 | 1.392 | -48.254 | 362.705 | 74.933 |
| dd20_floor1_block_reinvestment_only | blocker_isolation | 0.200 | 1.000 | True | 301.929 | 83.747 | 2.232 | 1.444 | -49.132 | 270.102 | 73.618 |
| dd20_floor1_block_all | blocker_isolation | 0.200 | 1.000 | True | 301.929 | 83.747 | 2.232 | 1.444 | -49.132 | 270.102 | 73.618 |

Blocking reinvestment is the damaging part of the first sweep. Rebuy and releverage blockers are inert or near-inert here because froth reserve and USDC releverage are disabled in the incumbent. Reinvestment blocking prevents the strategy from converting hedge gains into extra SOL, which lowers the eventual trough value.

## Trough Reconciliation

| scenario | trough_value | trough_sol | trough_usdc | trough_eth_debt | max_dd |
| --- | --- | --- | --- | --- | --- |
| incumbent_stateful_best | 4038.470 | 355.692 | 3183.291 | 4138.743 | 82.386 |
| floor_only_dd0.20_floor1.00 | 3916.878 | 362.705 | 4331.365 | 5506.870 | 82.963 |
| dd20_floor1_block_reinvestment_only | 3736.621 | 270.102 | 3782.744 | 3838.355 | 83.747 |
| dd20_floor1_block_all | 3736.621 | 270.102 | 3782.744 | 3838.355 | 83.747 |

The drawdown math is not obviously broken. The losing containment variants really do end the max-drawdown trough with less SOL collateral. A defensive rule that blocks SOL accumulation can lower volatility in one local window while making the deepest long-cycle valley worse.

## Conclusion

Do not promote the current drawdown containment defaults.

The implementation should remain available as an experimental toggle. The floor-only version deserves another pass, but only with a safety constraint:

- preserve reinvestment unless a separate test proves it should be blocked;
- use floor-only containment if testing this family further;
- sweep `drawdown_containment_hedge_floor` near 1.05-1.25 and require min HF >= 1.50;
- consider raising `partial_fill_min_hf` or adding a containment-specific min HF cap before increasing ETH debt;
- continue investigating pre-peak SOL-to-USDC rotation and short-lived crash circuit breakers, because the later red marks remain only slightly improved.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/drawdown_containment_isolated_sweep_20260529_140056/comparison.csv`
- Unsorted CSV: `reports/sol_supertrend_3y/drawdown_containment_isolated_sweep_20260529_140056/comparison_unsorted.csv`
- Summary JSON: `reports/sol_supertrend_3y/drawdown_containment_isolated_sweep_20260529_140056/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/drawdown_containment_isolated_sweep_20260529_140056`
