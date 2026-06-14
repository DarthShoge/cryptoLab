# Traffic-Light Research Checkpoint

Date: `2026-06-14`

## Current Best Valid Strategy

The best strategy that currently satisfies both hard gates remains:

`hedge_failure_thr0.10_sell0.01`

| Metric | Value |
| --- | ---: |
| Final SOL-equivalent | `420.951 SOL` |
| Max drawdown | `81.157%` |
| 2024+ drawdown | `64.781%` |
| Sortino | `2.288` |
| Min health factor | `1.616` |

This is the valid benchmark to beat. It improves drawdown only marginally.

## Current Best Iteration Target

The current best research frontier strategy is:

`tlg_y0.50_o1.00_r1.25_reinv3`

| Metric | Value |
| --- | ---: |
| Final SOL-equivalent | `505.214 SOL` |
| Max drawdown | `79.450%` |
| 2024+ drawdown | `64.605%` |
| Average major-drawdown half recovery | `776.9 bars` |
| Sortino | `2.343` |
| Min health factor | `1.423` |

This is not promotable because it violates the `1.50` min-health-factor gate. It is still the best strategy to iterate because it materially changed the frontier: it produced much higher SOL-equivalent value and slightly lower drawdown than the control.

## Interpretation

Traffic-light exposure governance helped, but the first implementation relied too much on stronger ETH hedge floors. That created more compounding and a small drawdown improvement, but spent too much health-factor safety.

The health-constrained retest proved the inverse: once traffic-light hedge adds are forced to respect stricter projected-HF floors, safety improves but the strategy loses the `>420 SOL` objective.

## Next Idea

The next idea should keep the traffic lights as the state machine but stop using larger ETH hedge floors as the main lever.

Next experiment: **Traffic-Light Insurance And Recovery Book**.

Core thesis:

- Green/lime states keep compounding into SOL.
- Yellow/orange/red states route a portion of realized hedge profit and surplus USDC into a protected USDC book.
- Protected USDC can defend health factor or repay debt during red/orange states.
- Protected USDC can re-enter SOL only through bucketed recovery rules when the lights improve.

This targets the actual failure from the latest sweeps: the model needs a safer balance-sheet buffer that does not permanently starve SOL accumulation.

## Artifacts

- Raw traffic-light sweep: `reports/sol_supertrend_3y/traffic_light_governor_20260614_172919/report.md`
- Health-constrained traffic-light sweep: `reports/sol_supertrend_3y/traffic_light_health_20260614_175038/report.md`
- Prior constrained benchmark: `reports/sol_supertrend_3y/constrained_drawdown_search_20260602_225238/report.md`
