# Drawdown Containment Sweep

Window: `2021-01-01` to `2025-12-31` at 1h bars from cached Binance SOL/ETH data.

## Question

Can a stateful drawdown containment mode reduce the large post-peak givebacks by raising the hedge floor and blocking bullish capital deployment while the portfolio is materially below its rolling high?

## Ranking

| scenario | trigger | hedge_floor | final_sol_equiv | final_portfolio_value_usd | max_drawdown_pct | sortino_ratio | min_health_factor | giveback_from_2024_peak_pct | containment_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_stateful_best |  |  | 402.064 | 50237.887 | 82.386 | 2.274 | 1.591 | -48.486 | 0 |
| contain_dd0.20_floor1.00_exit0.10 | 0.200 | 1.000 | 301.929 | 37726.052 | 83.747 | 2.232 | 1.444 | -49.132 | 56 |
| contain_dd0.15_floor0.75_exit0.10 | 0.150 | 0.750 | 303.155 | 37879.272 | 84.064 | 2.203 | 1.480 | -44.166 | 135 |
| contain_dd0.25_floor1.00_exit0.10 | 0.250 | 1.000 | 283.934 | 35477.561 | 84.143 | 2.205 | 1.452 | -49.067 | 35 |
| contain_dd0.25_floor0.75_exit0.10 | 0.250 | 0.750 | 298.267 | 37268.426 | 84.927 | 2.213 | 1.470 | -46.698 | 40 |
| contain_dd0.20_floor0.75_exit0.10 | 0.200 | 0.750 | 281.125 | 35126.509 | 85.651 | 2.205 | 1.542 | -46.633 | 54 |
| contain_dd0.15_floor0.50_exit0.10 | 0.150 | 0.500 | 271.915 | 33975.841 | 86.232 | 2.171 | 1.624 | -44.855 | 31 |
| contain_dd0.20_floor0.50_exit0.10 | 0.200 | 0.500 | 267.256 | 33393.622 | 86.723 | 2.182 | 1.462 | -48.274 | 10 |
| contain_dd0.25_floor0.50_exit0.10 | 0.250 | 0.500 | 268.172 | 33508.138 | 86.782 | 2.173 | 1.559 | -48.286 | 9 |
| contain_dd0.10_floor0.50_exit0.10 | 0.100 | 0.500 | 255.115 | 31876.567 | 86.890 | 2.154 | 1.580 | -45.092 | 62 |
| contain_dd0.10_floor0.75_exit0.10 | 0.100 | 0.750 | 191.278 | 23900.225 | 88.115 | 2.083 | 1.350 | -46.443 | 163 |
| contain_dd0.15_floor1.00_exit0.10 | 0.150 | 1.000 | 170.882 | 21351.674 | 89.568 | 2.058 | 1.460 | -45.894 | 132 |
| contain_dd0.10_floor1.00_exit0.10 | 0.100 | 1.000 | 156.654 | 19573.857 | 90.261 | 2.027 | 1.350 | -45.649 | 172 |

## Readout

Best by max drawdown: `incumbent_stateful_best` at 82.386% max drawdown, 402.064 SOL-equivalent, Sortino 2.274.

Incumbent control: 82.386% max drawdown, 402.064 SOL-equivalent, Sortino 2.274.

Decision: no promotion. The implementation is useful as a toggleable experiment, but this first simple containment policy does not improve the incumbent.

## Forensic Notes

This sweep ranks by max drawdown first because the hypothesis is defensive: reduce the repeated losses after the first large drawdown, then check whether the SOL-equivalent cost is acceptable.

Drawdown containment is intentionally simple in this run. It enters when portfolio value falls below the selected rolling-high drawdown threshold, exits only after a four-green non-bearish recovery near the rolling high, raises the ETH hedge target to a configured floor, and blocks froth reserve rebuys, surplus reinvestment, and bullish USDC releverage while active.

The result argues against this exact rule. The best challenger by max drawdown, `contain_dd0.20_floor1.00_exit0.10`, still worsened max drawdown from 82.39% to 83.75% and cut final SOL-equivalent from 402.06 to 301.93. The more responsive 15% trigger with a 75% floor improved the 2024/2025 giveback number, but it still worsened max drawdown to 84.06% and reduced final SOL-equivalent to 303.16.

The max-drawdown trough remained around 2023-06-10 in the tested challengers, which means the rule did not solve the 2021-2023 bear-market valley. In the strongest containment rows, containment was active for roughly 74-81% of bars, so the strategy spent a large amount of time carrying extra ETH short exposure. That appears to reduce upside and can still leave the portfolio exposed to sustained SOL beta.

The practical lesson is that reacting to portfolio drawdown after the peak is too late and too sticky. The next defensive hypothesis should either:

- act before the drawdown, using froth/peak conditions to reduce SOL beta into USDC;
- use a short-lived circuit breaker with forced time decay instead of a near-high exit;
- or split containment into two independent states: a brief crash response and a separate recovery redeployment gate.

## Debug Follow-Up

After reviewing the surprising result, the key problem is not that containment failed to engage. It did engage around the 2021 peak and raised the target hedge.

The bundled policy was too broad: it raised the hedge floor and also blocked reinvestment. By the 2023-06-10 max-drawdown trough:

| scenario | portfolio trough | SOL collateral at trough | USDC collateral at trough | ETH debt value at trough |
|---|---:|---:|---:|---:|
| incumbent | $4,038.47 | 355.69 SOL | $3,183.29 | $4,138.74 |
| containment 20% / 1.00 floor / blocked reinvestment | $3,736.62 | 270.10 SOL | $3,782.74 | $3,838.35 |

So the containment run carried less SOL at the bottom. It reduced some risk mechanically, but it also prevented the strategy from converting hedge profits into the SOL collateral that made the incumbent trough higher.

A quick diagnostic separating the hedge floor from the blocking gates showed:

| diagnostic | max DD | Sortino | final SOL-equiv | note |
|---|---:|---:|---:|---|
| incumbent | 82.386% | 2.274 | 402.064 | current best |
| floor only, 20% trigger, 1.00 floor | 82.963% | 2.311 | 406.704 | better final value, still worse max DD |
| same floor with reinvestment blocked | 83.747% | 2.232 | 301.929 | blocker caused most damage |

This changes the diagnosis. The implementation is not obviously failing to turn on, but the experiment design was flawed because it combined two effects. Future sweeps should isolate hedge-floor behavior from capital-deployment blocking, and should not assume blocking reinvestment is automatically defensive in a strategy whose trough is helped by accumulated SOL.

## Artifacts

- Comparison CSV: `reports/sol_supertrend_3y/drawdown_containment_sweep_20260529_134448/comparison.csv`
- Summary JSON: `reports/sol_supertrend_3y/drawdown_containment_sweep_20260529_134448/summary.json`
- Scenario folders: `reports/sol_supertrend_3y/drawdown_containment_sweep_20260529_134448`
