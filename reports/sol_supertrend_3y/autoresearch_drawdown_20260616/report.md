# Autoresearch Drawdown Loop - 2026-06-16

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

Hard gates:

- Final SOL-equivalent > `420 SOL`
- Minimum health factor >= `1.50`

Primary objective: lowest max drawdown among configs that pass both hard gates.

## Result

The first autoresearch loop found a new valid compromise:

| scenario | final SOL-eq | max DD | 2024 DD | min HF | Sortino | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw traffic-light baseline | 505.214 | 79.450% | 64.605% | 1.423 | 2.343 | invalid HF |
| previous best valid checkpoint | 420.951 | 81.157% | 64.781% | 1.616 | 2.288 | valid |
| autoresearch valid compromise | 449.588 | 80.385% | 64.667% | 1.506 | 2.311 | valid |

Config for the new valid compromise:

```python
traffic_light_hedge_floors = {
    4: 0.0,
    3: 0.10,
    2: 0.50,
    1: 1.10,
    0: 1.10,
}
traffic_light_min_reinvestment_green = 3
traffic_light_min_releverage_green = 4
traffic_light_add_min_hf = 2.20
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

The valid improvement came from combining the better local floor shape with a moderate projected-HF guard. The floor change alone improves drawdown slightly but leaves health factor too low. The HF guard alone can cross `1.50`, but usually costs too much SOL. The combination gives up about `55.6 SOL` versus the raw invalid frontier while improving the prior valid checkpoint by about `28.6 SOL` and reducing max drawdown by about `0.77 percentage points`.

This is still not close to the investor target of sub-50% drawdown. It is a better valid checkpoint, not the final answer.

## Autoresearch Log

The untracked `results.tsv` contains the experiment loop log for this run.
