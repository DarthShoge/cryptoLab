# Real hourly copy-strategy acceptance — 2026-09-07

The real-data workflow now runs end to end. This is an integration/research
checkpoint, **not evidence that copying these traders has an edge**. Both tested
scenarios lost money and underperformed their BTC controls. No exchange orders
were placed. No additional paid acquisition is required to reproduce these runs.

## Frozen experiment

Evaluation: 2026-08-04 00:00 UTC through 2026-08-06 00:00 UTC, 48 hours,
USD 10,000 initial capital. Native fills cover August 2–7 exclusive with full
adjacent source days retained. Prices use Binance BTCUSDT and Yahoo TSLA, GLD and
^GSPC; funding rates/timestamps are native Hyperliquid. These are explicit return
proxies, not proof of exact historical contract equivalence.

The preset was chosen before inspecting portfolio results: top five eligible
traders per market, two-day lookback, daily reselection, equal trader and market
weights, direction copying. Minimum eligibility: one active day, two episodes,
USD 10,000 closing notional and gross volume. Five equally weighted ranking
metrics: PnL efficiency, profit factor, positive-day rate, drawdown efficiency and
copyability. These fill-based metrics are not deposit-adjusted account returns.
Hourly follower decisions; 5-second minimum delay followed by strictly later real
bar opens; 4.5bps fees; gross budget 1, per-asset cap 0.5. Mark/wait limits four
days; regular-session proxies cannot trade while closed.

The sensitivity changes only slippage from 5 to 25bps. Frozen report configs
confirm no other configuration difference. Signals and cohort artifact hashes
match between runs; subsequent equity-dependent sizing can differ.

| Metric | Strategy 5bps | BTC 5bps | Strategy 25bps | BTC 25bps |
| --- | ---: | ---: | ---: | ---: |
| Final equity, USD | 9,952.80 | 10,193.41 | 9,894.17 | 10,173.40 |
| Return | -0.4720% | +1.9341% | -1.0583% | +1.7340% |
| Maximum drawdown | 0.8426% | 0.6975% | 1.0847% | 0.6989% |
| Fees, USD | 13.26 | 4.50 | 13.23 | 4.51 |
| Modeled slippage, USD | 14.74 | 5.00 | 73.50 | 25.00 |
| Final gross exposure, USD | 3,494.32 | 10,204.67 | 3,477.15 | 10,204.67 |

Cash controls remain USD 10,000. Strategy Sharpe/Sortino are -7.56/-9.78 at 5bps
and -17.18/-19.70 at 25bps, using hourly sampling. With only 48 return intervals,
these annualized ratios are mechanically calculated diagnostics, not reliable
skill estimates. Neither scenario is an out-of-sample winner.

## Accounting and forensic checks

Independently reconstructed every saved equity point from initial capital minus
signed fill consideration and fees, plus accumulated native funding cashflows
and net proxy positions marked at the latest completed source close. All 49
points per scenario/control reconcile: maximum error below USD 3.3e-11. Funding
was independently recalculated from pre-event holdings, source close and native
rate; maximum error below USD 7e-18. Residual quantities equal summed fills;
collateral plus unrealized PnL equals final equity. All declared report artifact
SHA-256 hashes were checked. No immutable saved report was overwritten.

Baseline final collateral is USD 9,978.54 and unrealized PnL is -25.74; sensitivity
collateral is USD 9,926.73 and unrealized PnL is -32.56. Baseline open positions:

| Native market label | Proxy units held | Final completed proxy mark |
| --- | ---: | ---: |
| BTC | -0.01558881 BTC | 64,633.90 |
| xyz:GOLD | -1.28114155 GLD shares | 389.570007 |
| xyz:SP500 | +0.06436029 index units | 7,722.120117 |
| xyz:TSLA | +4.63716129 shares | 321.459991 |

These quantities are **not native Hyperliquid contract balances**. Baseline net
per-market PnL including costs/funding is BTC -33.23, gold -0.42, S&P +5.11 and
Tesla -18.65 USD. Funding contributed just +0.0172 USD. BTC-short exposure in a
rising BTC window contributed to benchmark underperformance. Maximum observed
strategy gross leverage was 0.653. A zero liquidation count is not evidence of
liquidation safety: this approximate engine does not model native margin or
liquidations.

Largest baseline hourly equity declines ended August 5 15:00 UTC (-50.06 USD),
August 5 16:00 (-25.99) and August 4 19:00 (-17.20). The first interval contains
14:30 UTC S&P/Tesla fills; the next contains 15:30 gold/S&P fills. Scheduled
session bars and completed-close valuation concentrate proxy repricing. Maximum mark age was
18 hours: this is important overnight stale exposure, not continuous exchange
pricing. Each strategy has 34 fills, 13 unexecuted requests, 90 superseded
requests and 55 below-threshold requests. Costs and unexecuted targets therefore
matter; do not interpret target weights as continuously achieved holdings.

Five wallets were selected per market on each reselection date. Eligible counts
August 4/5: BTC 2,218/3,052; gold 51/89; S&P 246/398; Tesla 72/100. Next-day
membership turnover was respectively 80%, 60%, 60%, 80% (mean 70%). This is a very
short and unstable trader-selection window. Baseline contribution weights sum
exactly to each of its 192 saved signals; contributions explain positioning, not
individual wallet PnL attribution.

Legacy report qualification remains `research_eligible=false`, with the older
`smoke_only_unreconciled` marker in `reconciliation.json`. That marker is not an
independent test of the cashflow reconciliation above; it means exact/native
research qualification has not been granted. Do not relabel these immutable
reports as exchange-reconciled or production-qualified.

## Acceptance evidence and boundaries

- Real dataset registered and visible in the existing strategy UI; its preset
  passes preflight. Native activity is disk-backed with bounded ingestion/replay.
- Actual HTTP-submitted baseline completed; actual cloned sensitivity completed.
  Both remain in Saved backtests with descriptive names and frozen inputs.
- Browser checks cover the real preset, historical wallet membership, real market
  mappings (including GLD, with no browser page errors) and saved
  comparison. Comparison visibly flags the proxy-assumption difference and shows
  equity, drawdown, Sharpe/Sortino, costs and universe turnover.
- Market/trader selection, prior-only ranking, session/mark limits, missing-data
  rejection, funding, saved reports and legacy routes are regression-tested:
  fresh full core/API suite **277 passed**, 15 dependency deprecation warnings.
  Frontend: 12 unit tests, TypeScript compilation and generated API type check
  passed. The type generator required execution outside the sandbox to spawn
  its Python exporter; this did not require application changes.
- Dataset is the mapped, observed four-market universe, not an all-market census.
  Adjacent source days and observed timestamp bracketing qualify retained
  boundaries; they do not independently prove every exchange event was archived.
  Mappings, USD/USDT parity, source action evidence and closed-session marks remain
  research assumptions. No historical present-day leaderboard is substituted.

Useful next experiments (not performed or selected as winners): longer independently
qualified history; top-N/quantile and eligibility stability; turnover controls;
mark-age/session sensitivity; walk-forward evaluation. Longer paid acquisition
needs a new explicit scope and budget. No auto-trading is enabled.

## Local evidence

All paths are relative to the `hyperliquid-trader-ensemble` worktree.

- Dataset: `.hyperliquid_lab/datasets/real_cross_class_aug2026/`.
- Baseline experiment: `af8f0ec48be5458797c6e3b493f56f98`.
- Sensitivity experiment: `fbc95ab3678945a1b04b54ca48a485b0`.
- Immutable reports: `.hyperliquid_lab/reports/hyperliquid_trader_ensemble_<experiment-id>/`.
  Each contains config, summary, equity/fill/funding/request and cohort artifacts.
- Historical drilldowns: `.hyperliquid_lab/results/<experiment-id>/`.
- Browser captures: `.hyperliquid_cache/real_builder_20260907.png`,
  `real_trader_drilldown_20260907.png`, `real_market_drilldown_20260907.png`,
  `real_comparison_20260907.png`.
- Acquisition identities and approved 5.2193 GiB download evidence:
  [proxy data log](hyperliquid-proxy-data.md). Actual AWS billing was not verified.
