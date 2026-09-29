# Saved strategy diagnostics

## Objective and approved scope

Add a Diagnostics tab to a completed saved Hyperliquid backtest, initially verified
against weekly experiment `b32e5f1ed7db44008a927c6d965e7c3f`. Show a system card,
return profiles, statistical analysis, and trader-copy diagnostics. The user
approved this scope on 2026-09-26. This is descriptive analysis of existing
artifacts, not a new simulation, a strategy change, or research qualification.

## Evidence and boundaries

The report is in `.hyperliquid_lab/reports/hyperliquid_trader_ensemble_b32e5f1ed7db44008a927c6d965e7c3f/`.
It covers 2025-09-01 through 2026-09-01, starting with $10,000. Frozen configuration
selects BTC, xyz:GOLD, xyz:SP500 and xyz:TSLA; ranks traders per asset over 90 days;
selects the top 5% with cohort bounds 5–25; and updates cohorts and positions weekly.
Ranking is an equal-weight blend of copyability, drawdown efficiency, PnL
efficiency, positive-day rate and profit factor, not cash-flow-adjusted wallet ROI.

Current artifacts contain 8,761 hourly strategy equity observations, 17,522
control observations, 98 strategy fills and 117 asset/cohort records. Stored
return is -16.3198784%, BTC perpetual control -32.4670555%, and max drawdown
21.3327791%. These are verification references, never hardcoded UI results.
The report has `research_eligible=false`; reconciliation says
`accepted=false`, `issues=["smoke_only_unreconciled"]`. Approximate proxy prices,
non-native funding marks, observed/mapped markets only and incomplete account
state must remain visible. GOLD/SP500/TSLA quantities are proxy units.

Do not edit the simulation core, running worker, cached source data, frozen
experiment metadata or source report. Do not rerun or restart the daily job.
Build derived diagnostics in isolated API modules and render in the existing
React application. No external data requests, new trading capabilities, or
automatic parameter selection.

## Page design

Place Diagnostics in `ResultPanels` alongside Performance and Trader universe.
Reuse existing panel, table, metric-card, chart and status patterns. The page is
read-only and scoped to the experiment's configured strategy and matching-delay
BTC control; a selection on Performance must not silently change its subject.

1. **Evidence status** at the top: qualification, reconciliation issues, proxy
   limitations, data window and diagnostic availability. Explain that smaller
   losses than BTC do not establish positive alpha or profitability.
2. **System card**: experiment/run identifiers; frozen config/dataset identity;
   requested markets and allocation; trader scope, lookback, selection bounds,
   eligibility and ranking weights; weekly schedules; aggregation, risk caps,
   initial capital; latency, fees, slippage, stale-mark/wait limits; development
   split and benchmark. Show configured values, not editor defaults. Expandable
   frozen configuration supplies the complete detail without crowding the card.
3. **Return profiles**: common-window equity growth versus BTC and cash;
   drawdown chart; calendar-month returns for strategy/BTC; full-week return
   distribution and best/worst weeks. Show observation counts and partial
   periods explicitly. Include accessible tabular equivalents of chart data.
4. **Statistical analysis**: stored hourly Sharpe, Sortino, annualized volatility,
   CAGR and full-grid drawdown with conventions; calculated full-week win rate,
   mean, median, volatility, extrema, empirical 5th percentile and average of
   observations at/below that percentile. Benchmark correlation/beta and
   tracking error use exactly aligned weekly returns. Label excess total return
   as percentage-point outperformance, not regression alpha.
5. **Uncertainty**: descriptive 95% block-bootstrap interval for mean weekly
   strategy return and paired weekly excess return. Use deterministic seed,
   2,000 circular moving-block samples of length four weeks, truncated to the
   observed number of weeks. Require at least 20 full weeks; report method,
   block length and sample count. No IID hourly significance tests or claims of
   trader skill. Mention dependence assumptions, one development window,
   selection/multiple-testing risk and proxy pricing as limitations. Do not
   imply that statistical intervals incorporate missing data/model risk.
6. **Strategy diagnostics**: gross/net exposure, cost/funding totals, final
   collateral and unrealized PnL, residual positions in proxy units, observed
   maximum mark age, and available liquidation/execution diagnostics. Show
   dated cohort sizes, eligible/candidate counts and turnover by asset. Reuse
   existing Trader universe drilldown for date/asset and wallet membership.
   Explain missing cohort dates rather than interpreting them as zero members.

## Calculation and data contract

Implement a typed read-only `GET /api/lab/experiments/{id}/diagnostics` endpoint.
It resolves a completed backtest through the existing store and repository;
clients cannot supply arbitrary file paths. The response has explicit sections
for system metadata, warnings/availability, profiles, statistics, uncertainty,
accounting and cohort summaries. Each unavailable statistic has a reason and
JSON null rather than NaN, Infinity or invented zero. Missing optional artifacts
disable their section, not the entire page. Failed/incomplete experiments return
an actionable safe application error. Missing/invalid equity prevents derived
return statistics while leaving system metadata and evidence warnings visible.

Keep loading/validation, pure statistical calculations, and API models separate.
Reuse DuckDB's bounded connection and repository path validation. Compute on
full-resolution source observations before chart downsampling; never calculate
returns from the existing 2,000-point chart response. Filter strategy and control
by exact scenario and latency before checking time uniqueness, expected hourly
or minute spacing, finite values and positive equity. Reject gaps/duplicates;
do not forward-fill missing history or implicitly pair unequal timestamps.

Full weeks use Monday 00:00 UTC boundaries with endpoint-to-endpoint simple
returns; both endpoints and every expected intervening sample must exist.
Partial first/last weeks are displayed separately and excluded from weekly
statistics and bootstrap. Calendar months use month-start UTC boundaries;
clipped edge months are labelled partial. Cash is the saved cash control where
available, not an invented replacement. A constant cash return series has
undefined correlation/Sharpe, not zero. Benchmark-dependent statistics use only
exactly matching complete periods, with the paired count displayed.

Weekly volatility/tracking error use sample standard deviation and sqrt(365/7)
annualization. Beta is sample covariance(strategy,BTC)/sample variance(BTC);
correlation requires positive variance in both. Bootstrap resamples paired
strategy/control blocks together for excess returns. Weekly frequency must be
labelled separately from stored hourly risk metrics. Avoid silently converting
the saved annualized Sharpe into a weekly Sharpe.

Use linearly interpolated empirical quantiles and percentile bootstrap intervals
(2.5th and 97.5th percentiles). Missing or invalid benchmark history must leave
valid strategy-only statistics available. Display saved cohort turnover rather
than silently recomputing it; document its producer's denominator and first-cohort
semantics during implementation.

Local reconciliation checks final equity against final collateral plus unrealized
PnL, sums modeled fill fees and funding cash deltas, and compares with stored
summary values within documented floating tolerances (absolute USD 0.01).
Also check every equity row's cash plus unrealized PnL. Surface mismatches;
do not rewrite the original `accepted=false` even if these limited checks pass.
No assertion of independently verified native marks or liquidation modeling.
Costs must not be subtracted twice from already net equity. Only make asset PnL
attribution claims supported by available ledgers; otherwise state unavailable.
Worst-week context uses dated fills/funding and exposure, not inferred causality.

## Verification and delivery

- Pure Python tests: known return series, exact weekly/monthly boundaries,
  partial weeks, gaps/duplicates, invalid equity, zero variance, missing benchmark,
  deterministic paired bootstrap, accounting mismatches and unavailable data.
- API tests: completed fixture resolves correct scenario; incomplete/unknown
  experiment and missing optional artifacts behave safely; response is finite
  JSON and includes original qualification flags. Temporary fixture roots only.
- Frontend tests/browser checks: tab navigation; system card; warning visibility;
  chart/table rendering; metric conventions and unavailable reasons; cohort
  drilldown; error/loading states; no horizontal page overflow on mobile.
- Run TypeScript checking/build and focused backend regression tests. No broad
  costly suite or rerun of the annual strategy is necessary.
- Exercise diagnostics against the actual completed weekly run, independently
  reconcile the headline figures, and inspect the rendered page. Record evidence
  and screenshots under a new timestamped diagnostics report directory; never
overwrite the original backtest or previous reports.

## Review and preliminary accounting evidence

Independent spec review approved this design on 2026-09-26. Read-only checks
confirmed final equity $8,368.012159543725, summed fees $60.45045387848561,
and net funding cost $45.678127159021365. Every hourly equity row satisfied
equity = cash + unrealized PnL exactly in the saved data. These limited checks
do not change the original research-qualification or reconciliation status.
The asset cohorts start on different dates: BTC in September 2025, TSLA in
February 2026, GOLD in March 2026 and SP500 in June 2026. The page must not
present this as twelve months of simultaneous four-asset cohort coverage.

## Alternatives

A standalone HTML report is cheaper to publish once but separates the diagnostic
view from saved backtests and historical membership drilldown. A new general
research dashboard expands scope unnecessarily. The approved saved-run tab
preserves the strategy-specific workflow and supports later saved-run reuse.
