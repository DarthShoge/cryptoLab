# Autoresearch Drawdown Loop - 2026-06-16

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

Original hard gates:

- Final SOL-equivalent > `420 SOL`

The `1.50` minimum health-factor gate was removed after review because the current priority is max drawdown and final SOL amount. Health factor is now retained as a diagnostic, not a pass/fail gate.

Primary objective: lowest max drawdown among configs that preserve strong final SOL.

## Result

The revised no-HF-gate loop found two better frontier choices:

| scenario | final SOL-eq | max DD | 2024 DD | min HF | Sortino | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw traffic-light baseline | 505.214 | 79.450% | 64.605% | 1.423 | 2.343 | baseline |
| best no-HF drawdown | 496.353 | 78.991% | 64.436% | 1.432 | 2.342 | keep |
| Pareto SOL improvement | 506.680 | 79.349% | 64.595% | 1.421 | 2.343 | keep |
| previous HF-valid compromise | 449.588 | 80.385% | 64.667% | 1.506 | 2.311 | obsolete under revised objective |

Config for the best no-HF drawdown candidate:

```python
traffic_light_hedge_floors = {
    4: 0.0,
    3: 0.10,
    2: 0.50,
    1: 1.20,
    0: 1.20,
}
traffic_light_min_reinvestment_green = 3
traffic_light_min_releverage_green = 4
traffic_light_add_min_hf = None
```

Config for the Pareto SOL improvement:

```python
traffic_light_hedge_floors = {
    4: 0.0,
    3: 0.10,
    2: 0.52,
    1: 1.05,
    0: 1.10,
}
traffic_light_min_reinvestment_green = 3
traffic_light_min_releverage_green = 4
traffic_light_add_min_hf = None
```

## What Failed

1. Volatility-budget overlay at a `0.50` hedge floor:
   - `433.354 SOL`, `82.269%` max DD, `1.425` min HF.
   - Kept the SOL gate but worsened drawdown and failed the HF gate.

2. Standalone traffic-light add-health floor at `2.25`:
   - `409.043 SOL`, `82.212%` max DD, `1.533` min HF.
   - Fixed HF but missed the SOL gate and worsened drawdown.

3. Floor-only local grid around the raw winner:
   - Best drawdown near-miss: `497.426 SOL`, `79.095%` max DD, `1.431` min HF.
   - Useful signal: `y=0.50, o=1.10, r=1.10` slightly improves drawdown, but cannot satisfy HF alone.

## Interpretation

Removing the HF gate changes the decision. The prior `449.588 SOL` compromise is no longer attractive because it gave up too much SOL to satisfy a secondary diagnostic.

The best drawdown candidate found so far is `y=0.50, o=1.20, r=1.20`, which improves max drawdown by about `0.46 percentage points` versus the raw baseline while giving up about `8.86 SOL`.

The best Pareto candidate is `y=0.52, o=1.05, r=1.10`, which improves both final SOL and max drawdown versus the raw baseline: `+1.47 SOL` and about `0.10 percentage points` lower max drawdown.

This is still not close to the investor target of sub-50% drawdown. It is a better no-HF frontier checkpoint, not the final answer.

## Autoresearch Log

The untracked `results.tsv` contains the experiment loop log for this run.
