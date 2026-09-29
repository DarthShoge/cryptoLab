# Hyperliquid copy-strategy lab

## Approved product direction

This is a self-contained research tool for one strategy family: selecting
historically successful Hyperliquid traders and combining their positions into a
simulated follower portfolio. The user approved this direction and saving every
backtest for comparison on 2026-09-05.

The central workflow is **configure → backtest → save → compare → inspect the
traders behind the result**. A saved artifact browser is a supporting capability,
not the product's organising principle. This spec supersedes the navigation,
GET-only restriction and backtest-launch exclusion in the earlier report UI spec.
Its portfolio-metric definitions, safe artifact access and synthetic-data warnings
remain applicable. This is a design update, not a claim that the new tool exists.

## Boundaries and reuse

Extend the existing React/TypeScript and Python/FastAPI apps in place. Keep
reusable ranking, position replay, aggregation and simulation code in
`packages/arblab/src/arblab/hyperliquid_copy`; do not introduce a general strategy
plugin framework or modify the Kamino applications.

Local writes and local simulation jobs are in scope. Exchange orders, signing
keys, live trading, automatic paid downloads, cloud deployment and multi-user
accounts are not. Dataset acquisition/reconciliation remains a separate explicitly
approved workflow. A new experiment must use an existing registered local dataset.

Alternatives considered: a configuration sidebar on the old report browser would
reuse the most UI but leave experiment history secondary; a separate new platform
would duplicate the existing API, charts and simulator. Restructuring the existing
apps around strategy experiments provides the required workflow with focused reuse.

## Strategy definition

The strategy definition is visible above every result, not buried in Data. A
human-readable summary is generated from the exact versioned configuration, e.g.
“SOL + ETH · per-asset cohorts · top 5% of eligible wallets · trailing 90 days ·
weekly reselection · equal trader votes · equal asset budgets · BTC perp benchmark”.

Keep these settings distinct:

| Setting | Required behaviour |
|---|---|
| Copied markets | Select a nonempty subset of BTC, ETH and SOL initially. |
| Ranking scope | Per-asset cohorts by default, or one pooled cohort scored across the selected copied markets. |
| Candidate universe | All wallets observed in the registered dataset's declared coverage, never today's winners projected backwards. Show coverage limitations. |
| Eligibility | Minimum active days, completed episodes, activity notional and holding duration; show defaults and exclusions. |
| Ranking window | Trailing 30/90/custom days. “90 days” is not labelled as a calendar quarter. |
| Ranking rule | Select supported metrics, direction and nonnegative weights; weights normalised explicitly. Existing five-factor composite remains a named legacy preset. |
| Selection | Exactly one of top N or top fraction of eligible wallets, with visible minimum/maximum cohort guards. |
| Reselection | Daily, weekly or monthly, with recorded UTC decision timestamps. |
| Trader aggregation | Equal direction votes, score-weighted direction votes, or the existing score-weighted normalised/trimmed conviction method. |
| Asset allocation | Separate per-asset budget weights, initially equal or explicit fixed weights summing to one; no redistribution into missing cohorts. |
| Follower settings | Initial capital, gross/asset caps, target update frequency, deadband, minimum trade size, execution delay and fee assumptions. |
| Evaluation | Simulation start/end, research split, dataset version and benchmark definition. |

“Return” must not conceal a change of denominator. The existing engine computes
realised PnL per closing notional, not wallet investment return. Expose that as
**PnL efficiency**, alongside its existing profit factor, positive-day rate,
drawdown efficiency and copyability metrics. Add gross traded volume (sum of
absolute fill quantity × price) as a ranking metric or eligibility filter;
distinguish it from the existing closing-notional filter. Explain that volume is
activity, not automatically evidence of skill.

An actual wallet-return ranking is a separately named capability requiring a
specified, cash-flow-adjusted historical equity series and verified coverage.
It is disabled with an explanation when that evidence is absent. Do not relabel
PnL efficiency or raw PnL as return to make the user's example appear supported.
The initial runnable engine supports the explicit efficiency/volume alternatives;
historical wallet ROI is a subsequent data-dependent increment.

For a weighted ranking, compute metric percentiles within the eligible cohort at
the decision time, reversing lower-is-better fields. Combine the percentiles using
the declared weights. Persist raw metrics, component percentiles and total score.
Missing required ranking metrics exclude a wallet with a reason, never an implicit
zero. Ties resolve by wallet address. Top fraction selects ceil(fraction × eligible
count), subject to displayed guards; show requested and actual cohort size. Fewer
than the minimum eligible wallets yields an empty cohort, not weaker eligibility.

Equal direction weights mean equal wallet votes, not equal dollars copied from
each wallet. Conviction normalisation retains the existing trailing-notional
denominator and is named accordingly; it is not account-equity normalisation.
Store weighting and missing-position coverage conventions with the experiment.

## Historical causality and portfolio construction

All ranking inputs must precede the selection timestamp. Warmup includes the full
ranking lookback and any position/normalisation history needed for reconstruction;
missing history is a visible data gate, not a silently shortened window.

The simulation starts on a UTC day boundary with an initial cohort decision.
Subsequent daily decisions occur at UTC midnight, weekly decisions Monday 00:00,
and monthly decisions on the first at 00:00. Cohorts remain fixed between decisions;
copied positions may update more frequently on the configured minute-grid cadence.
The engine records the actual schedule so it cannot depend on the host timezone.

A wallet may belong to both per-asset cohorts; its votes are applied independently
inside the corresponding asset budgets. Pooled membership is reused across copied
assets, but each asset's position state and coverage are still evaluated separately.
Unknown positions remain in the weight denominator under the existing coverage
policy. An empty/insufficiently observed cohort produces a zero target for that
asset, subject to executable rebalancing, not an invented instant liquidation.
Departed wallets lose their contribution at reselection; the combined target is
then executed through the normal simulator, with delays, partial fills and costs.

Separate copied markets from data requirements. A SOL/ETH strategy may require
BTC market data solely for its benchmark: BTC must not leak into the copied universe.
Keep the existing BTC perpetual buy-and-hold convention visibly named (not spot
BTC), with the same initial capital and simulation period and declared execution/
funding costs. Additional existing controls can remain optional comparisons.

## Product structure

### Strategy builder

The default landing screen contains a compact strategy summary and editable
sections for universe/eligibility, ranking/selection, copying/allocation, and
backtest assumptions. The ranking universe and selection settings appear first.
Show the local dataset's market coverage, date coverage, evidence status and
resource estimate beside Run backtest. Invalid or unsupported combinations are
explained before submission. No decorative live exchange controls.

Previewing the cohort at a selected date uses the same ranking implementation as
the backtest, strictly prior information and explicit scope. It is a bounded
preview job, not a second browser-side ranking formula; label its decision date.

### Saved backtests

Every submitted configuration receives a stable experiment ID immediately and is
saved automatically; users do not need to remember to click Save after completion.
The library lists name, strategy summary, date range, status, data mode, selected
headline metrics and optional sweep group. Users can name, annotate, reopen and
clone an experiment, and select it for comparison. A clone is an editable draft;
running it creates a new ID. No overwrite or delete action in the initial version.

### Backtest detail

Keep the immutable strategy definition and evidence badge visible. Use primary
views **Performance**, **Trader universe**, **Execution**, and **Data & assumptions**.
Reuse existing portfolio analytics and charts in Performance. Historic membership
is not hidden in an ancillary raw-artifact tab.

Trader universe provides:

- A date/scope timeline of observed candidates, eligible and selected wallets,
  additions, removals, retention and membership turnover.
- A decision-date drilldown into selected, eligible-but-not-selected and excluded
  wallets, rank, raw metrics, percentile components, score, weights and reasons.
- A wallet drilldown across decisions: membership episodes, score trajectory,
  inclusion/exclusion reasons and position contributions while selected.
- A timestamp/asset drilldown connecting selected wallets, their known positions,
  normalised signal inputs and weights to the aggregate signal and follower target.

Store decision reasons including threshold failure, rank below cutoff and missing
metric; do not guess reasons by comparing only membership lists. Membership turnover
is (entries + exits) / (previous size + current size), with zero when both are empty;
the initial decision has no prior-cohort turnover. Retention is intersection /
previous size, unavailable when previous size is zero. Keep these distinct from
executed portfolio turnover. Historical exclusion tables cover observed candidates,
not wallets absent from the dataset.

Position contribution explains target formation, not a claim of additive per-wallet
realised PnL. Netting, execution and fees occur at follower-portfolio level; precise
wallet PnL attribution is outside this first version.

### Compare backtests

Select up to six saved completed experiments. Show configuration differences
before a table of performance, BTC-relative return, costs, exposure and universe
turnover. Overlay equity and drawdown; offer dollar equity and rebased growth
of one, visibly labelled. Rebase full-series values on the backend, not chart samples.
Compare cohort histories by date and scope when definitions permit it.

Flag different periods, starting capital, dataset versions/coverage, ranking metric
definitions, execution assumptions, benchmarks and research splits. Same-window
comparisons are the default. Mismatched windows may be inspected with warnings and
their original metrics, but are not automatically ranked or labelled winners.
Do not silently truncate, forward-fill or recompute statistics over an intersection.
Synthetic/smoke and research-qualified results remain visibly separated.

### Bounded variations

Permit an explicit grid of supported strategy settings (top N/fraction, lookback,
ranking weights, reselection and aggregation). Display expanded configurations and
trial count before submission, capped at 32 variants initially and one executing
job at a time. A sweep group has its own ID and immutable variant manifest; every
variant is a separately saved experiment, including failures. This is not an
unbounded optimiser. Deduplicate identical expanded configurations visibly.

Preserve existing development/validation/locked-test safeguards. Sweeps operate on
development only; comparison must not automatically promote a winner or unlock a
test split. Repeated exploratory comparisons do not establish out-of-sample skill.

## Persistence, jobs and API boundaries

Use a strategy-specific local SQLite store for experiment/group IDs, frozen
submission specs, status transitions and editable annotations. Large evidence
remains immutable JSON/Parquet artifacts in unique experiment directories. This
is a small local store, not a distributed job system. Separate modules own:

1. Configuration/capability validation and normalisation.
2. Dataset catalogue and preflight coverage/resource checks.
3. Experiment persistence, annotations and comparison queries.
4. A single local worker invoking the existing Python research pipeline.
5. Historical universe/position-contribution artifact queries.
6. Typed HTTP routes and the strategy-specific frontend views.

An experiment freezes config/schema version, dataset manifest and checksums,
simulation/warmup dates, source revision/engine version, dependency versions,
benchmark definition, research split and random seeds if used. Completed output
artifacts and their hashes never change. Name/notes are editable metadata outside
that reproducibility payload. Referenced inputs are checked before execution;
changed checksums fail the job rather than silently using new data. Retain dataset
references for reproduction; if source data later disappears, results remain
viewable but reruns are explicitly unavailable.

Job lifecycle: queued → running → completed/failed/cancelled. Write results to
private staging and publish only after validation and checksum generation. Recover
interrupted running jobs as interrupted failures on restart, never completed;
queued jobs remain saved and require explicit resume. Cancellation stops the owned
worker and preserves configuration/status, without publishing partial success.
Failed variants do not prevent other queued variants completing.

Expose typed local endpoints for capabilities/datasets, cohort preview, experiment
submission/status/detail, metadata updates, cloning, cancellation, sweeps and
comparison, alongside bounded existing artifact-query endpoints. The browser uses
generated OpenAPI types; strategy math remains Python-only. Server-selected IDs
resolve datasets/artifacts: requests cannot supply filesystem paths or commands.

Because the API now mutates local state, bind to loopback and enforce an explicit
Host/Origin allowlist plus JSON-only mutation requests and a per-launch local UI
request token. No permissive CORS. The token is bootstrap state, not an exchange
credential. Keep GETs non-mutating, reject cross-origin writes, use fixed process
arguments/no shell interpolation, bound queue size (32 outstanding runs), and
return sanitised errors. Authentication/public hosting remain out of scope.

Legacy report folders remain readable through an adapter, identified as imported
artifacts rather than newly executed experiments. Their existing scenarios can be
selected as individual comparison series using report ID plus scenario identity.
Missing configuration/evidence stays unknown. Only fully specified supported
configurations can be cloned into runnable drafts; never silently fill research
assumptions. Do not modify original reports or backfill invented membership data.

## Data readiness and staged delivery

The current fixture is synthetic and the current runner is not qualified for an
all-wallet quarter. The existing fixed market validator, daily-only schedule,
five-factor score and shared strategy/benchmark market set need explicit extension.
Do not assume configuration widgets already control those behaviours.

Deliver the complete configure/save/compare/universe workflow first against bounded
local datasets and deterministic multi-reselection fixtures. Expand saved artifacts
to include the raw ranking components, counts/reasons, effective weights and signal
contributions required by drilldowns. Persist/query these in partitioned Parquet;
never send an entire all-wallet history to the browser. Retain the runner's hard
row/memory guard until bounded replay is qualified; oversized runs fail preflight.

The first delivery increment is the single-backtest loop, including cloning and
comparison of separately saved runs. Bounded sweeps follow as a second increment
using the same configuration, persistence and worker contracts; they must not
delay the core strategy research workflow.

Real quarterly research requires verified historical source coverage, fee and
position reconciliation, sufficient warmup and bounded replay qualification. Those
are explicit readiness gates, not satisfied by a synthetic demo or relaxed tests.
Show them in the UI. This design neither purchases those datasets nor promises a
quarterly result before that work. Live paper collection is a later, separate stage.

## Acceptance and verification

- A user can configure SOL/ETH copying with BTC benchmark data without copying BTC.
- Every supported input changes a persisted effective configuration and the engine
  behaviour it claims to control; unknown/inapplicable fields fail validation.
- Hand-calculated fixtures verify top N/fraction, eligible denominators, ties,
  ranking weights, missing metrics, guards and per-asset versus pooled membership.
- Multi-decision fixtures verify joins/exits, weekly/monthly UTC boundaries,
  wallet drilldowns, missing-state weights and contribution-to-target reconciliation.
  Appending future fills cannot change earlier selections or targets.
- Backtests are saved across restart; reopening reproduces the original config,
  results and cohort history. Renaming cannot change hashes. Cloning/rerunning
  creates new IDs. Changed inputs, worker failure, cancellation and publication
  interruption cannot masquerade as a completed experiment.
- A bounded sweep saves each configuration/result, including failures, and cannot
  bypass resource limits or development-only restrictions.
- Comparison shows config differences, full-series metrics and aligned overlays,
  with explicit mismatches and unchanged original evidence. N/A remains distinct
  from zero. Existing Sharpe/Sortino/drawdown availability policies remain in force.
- Browser tests exercise configure → submit → completion → reopen → clone →
  compare, then inspect historical cohort dates and a wallet. Cover errors, absent
  datasets, incompatible comparisons, keyboard access and narrow-screen layout.
- API tests cover local write protection, safe IDs/artifacts, bounded queries,
  persistence and job recovery. Existing unrelated apps retain their behaviour.

Success means a focused copy-strategy research loop, not merely new labels on the
report explorer. No claim of profitable copying or real-data validation follows
from implementation or passing fixture tests.
