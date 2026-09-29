# Hyperliquid copy-strategy lab

A self-contained React/TypeScript frontend and Python/FastAPI backend for one
question: which historical trader-selection and copying rules are worth testing?
The application runs local development simulations. It never connects a wallet,
places exchange orders, or downloads data. Other strategy apps remain independent.

## Rebalance cadence

Proxy datasets now offer **Portfolio rebalance cadence**: hourly, daily, or weekly.
Weekly aligns market and trader selection to Monday 00:00 UTC and locks those
controls to the same schedule. Quantities are held between decisions; valuation
and native funding remain hourly. Execution waits for an actual proxy open strictly
after the decision plus configured delay. A midweek start stays in cash until Monday.
Cadence survives save/clone and appears in comparison configuration differences.
Existing hourly scenarios are unchanged. The real two-day preset has no Monday
inside its evaluation interval, so switching only that preset to weekly yields
cash, not a meaningful weekly test. Scheduled daily/weekly runs now support up to
366 evaluation days using disk-backed ranking output, subject to dataset coverage
and resource checks. Legacy hourly runs retain the 93-day cap. The real annual
dataset and results are still pending acquisition and full-scale validation.

## Try the saved annual infrastructure example

At http://127.0.0.1:8010, **Saved backtests** contains two clearly labelled
`SYNTHETIC annual · BTC` runs: weekly and daily selection/positions, 90-day warmup,
August 3, 2026–August 3, 2027 exclusive. Select both to compare. To create another,
choose **Synthetic annual weekly copy strategy** and load its synthetic preset.
This fabricated single-wallet BTC fixture tests persistence, schedules, ledgers,
history drilldown and accounting—not trader skill or annual real-data coverage.
Each run retains 8,761 hourly equity points; charts may downsample for display.

Rankings stream to Parquet under a 250-million-row / 16 GiB file ceiling, avoiding an
all-year Python ranking list. Publication can temporarily hold three copies
(up to 12 GiB for rankings alone), plus input data, query spill and other reports;
free-space guards reject without sampling/truncation. Other existing per-query,
asset-hour, contribution and input limits still apply.

## Qualified full-history datasets

The operator registration function `register_annual_dataset` now accepts an
explicit `cache_reference={"path": absolute_path, "identity": existing_identity}`.
This selects `qualified_v1`; omitting it retains the legacy `sharded_v1` path.
The new path requires a completed, qualified archive job and verified price/native
funding evidence. It retains canonical source files byte-for-byte rather than
cropping away early positions or dormant trader observations. Metric lookbacks
and the registered evaluation boundaries still apply to each query.

Qualified datasets require weekly/daily scheduled configurations. Their canonical
input ceiling is 64 GiB; the legacy shared derived cache is capped at 8 GiB,
including unfinished reservations. The real capacity-probe cache has an explicitly
approved, receipt-backed 16 GiB policy (see below). These are limits, not a total disk-footprint
estimate: reports, source archives and temporary query work also occupy storage.
The loader requires an already initialized cache and never creates a replacement
when it is busy. Weekly/daily runs and previews reuse identical decision results.

The original qualified archive paths and shared cache are durable dependencies.
Renaming the registered dataset directory is supported while those dependencies
remain available; copying just that directory does not make a portable dataset.
Do not edit its manifest or source files during a run. Loading and closing verify
their bindings, and failures release the owned cache lease without deleting data
or refunding reservations.

Registration, saved-run comparison and preview integration have passed fixture
tests. This does not yet establish acceptable real annual throughput, complete
annual source coverage, or strategy performance. Preflight estimates remain
advisory; the real all-wallet daily ranking-output capacity audit is outstanding.

Qualified scheduled rankings now use reusable feature history. The chronological
controller builds missing qualified days from the source origin, reuses published
days across requests/restarts, and supplies exact-cutoff lookback windows to the
feature-ranking interface. A fixed source-market chain supports changing active
markets, while each ranking query uses only its configured scope. The controller
keeps bounded day descriptors rather than loading wallet histories into Python.
It does not yet retire old feature files. Real annual cache capacity and throughput
remain unproven; this is not a claim that the annual run is ready end to end.

A verified resume-anchor primitive can now persist and reopen an explicit retained
feature interval under a fresh lease without revisiting pre-boundary days. It
retains complete checkpoint evidence and uses the same shared cache accounting.
FeatureHistory now accepts explicit anchor inputs under a fresh lease, advances
only missing days, and rejects lookbacks before the retained boundary. The actual
source origin and exact query cutoffs remain unchanged. There is no automatic
anchor selection or expiry policy; this does not authorize deleting old history.
The interrupted real capacity probe retained86full-scope days. Its approved
16 GiB migration has now passed full integrity audit and reopened all87 existing
feature publications (including the separate BTC-only day) unchanged. The new
`open_expanded_cache` factory requires the durable migration receipt inputs;
legacy8 GiB opening rejects this cache. Registered qualified datasets may now
include `expansion_receipt` alongside `path` and `identity` in their cache
reference. Supply the exact saved receipt-input object, not a new byte limit or
an approval flag. The scheduled loader verifies that receipt under its single
cache lease on opening and closing; it never migrates, creates or substitutes a
cache. Legacy two-field references remain8 GiB-only. Focused saved weekly/daily
and preview tests cover both policies; the full related regression suite passed
83 tests. A separate capacity audit proves the planned daily comparison exceeds
the former50-million ranking-row limit even for BTC alone. Approved per-run ceilings
of250 million rows and16 GiB are now implemented. Publication can temporarily
hold two ranking-file copies (up to32 GiB per run), apart from other retained data.
These ceilings do not prove that the annual comparison fits. The real90-day
feature check matched all108,185 reference ranking rows exactly and verified
cadence-only weekly reuse, with peak RSS1,076,404,224bytes. The shared-cache and
AWS caps are unchanged; the real annual saved comparison remains unfinished.
Migration does not establish annual capacity or
strategy performance. Recovery and verification evidence are documented
in `hyperliquid-90day-capacity-audit-2026-09-09.md`.

The standalone `QualifiedSourceSession` and `SavedFeatureRankings` APIs now support
explicit verified ranking receipts. A fresh source session fully verifies the
canonical corpus; subsequent operations check unchanged source identities,
report, caller and code context. Capture invokes the real feature producer and
shares its existing ranking payload rather than copying it. Load verifies the
receipt, exact query, engine and payload without consulting intermediate feature,
candidate or score publications. Cadence-only reuse is supported. Changed source,
strategy or engine context is rejected, with no fallback replay. These APIs have
passed89 related regression tests and narrow review. The scheduled loader now
uses these receipts before opening feature history; automatic physical retirement remains
outstanding. These APIs deleted no real derived-cache files.

`SavedFeatureRankings.find` now supports exact-query discovery from the existing
catalogue. It authenticates bounded metadata before classifying receipts, rejects
ambiguous/corrupt matches, and verifies matching payloads without reading unrelated
payloads. Hit and miss paths recheck source, query, engine, lease and catalogue
identity, with SQLite data-version checks independent of filesystem timestamp
precision. The scheduled facade lazily verifies the source once, then finds an
exact receipt or captures a new ranking through the real producer. Reopened
weekly/daily requests can reuse a result without intermediate publications or
new allocations. Source and ranking identities remain guarded through final
facade checks. Existing canonical hardlinks from registration are fully verified
and their initial topology pinned; later alias writes or topology changes reject.
Owned cache files and reports still require single links. The final expanded
integration suite passed100 tests, including saved API runs and previews under
both cache policies. Automatic physical retirement remains outstanding.

The explicit owned-retirement transaction is now implemented and fixture-tested.
It journals ownership before detaching a publication, preserves shared payloads,
keeps missing bytes charged across interruptions, and supports explicit recovery
under either cache policy. Actual intermediate-file disposal followed by identical
weekly/daily saved-ranking reuse is tested with replay forbidden. The final related
suite passed159 tests. No successful real cache object has been expired: automatic
lookback/anchor policy and exact real-target authorization remain separate gates.

Anchor-owned recovery now has a bounded read-only discovery step. It distinguishes
prepared, detached and completed operations, refuses other unfinished owners and
stale prepared baselines, and preserves the complete retained checkpoint interval.
Recovery checks source/scope/fee semantics and source membership before acting;
protected source and anchor identities are checked immediately before disposal.
Registered canonical hardlinks are preserved, and unrelated journal payloads are
not opened. Temporary qualified fixtures cover interruption and fresh-lease
recovery; that related regression suite passed181 tests.

The explicit rolling-history controller is now implemented and fixture-tested.
It verifies and reopens each replacement anchor before retiring obsolete derived
days, recovers interrupted owned work before new publication, and preserves active
checkpoints, canonical sources and saved rankings. Final related regressions passed
182 tests. Registered/scheduled opt-in is now wired through the explicit
`feature_history_policy="rolling_feature_anchor_v1"` metadata option. Absence
preserves existing non-retiring behaviour. Fixture API worker checks passed for
both cache limits with and without the policy, including saved weekly/daily runs,
preview and comparison. The broader integration regression passed113 tests.
No existing real dataset has expiry enabled. Earlier cache misses now route to
the bounded canonical-history scorer after verifying the retained frontier and
settling owned retirement. Exact saved-ranking hits still bypass both producers.
Raw results bind the original candidate universe and exact lookback, and use the
same standalone receipt lookup. This avoids recreating retired feature keys;
it can cost additional CPU and retained metric/score intermediates. Annual cache
sufficiency is not established by this fallback. The postformat receipt/scheduled
regression passed61 tests, expanded lifecycle checks passed20, and registration/
annual-publication/owned-loading/API-worker checks passed37. Policy/config changes are checked before physical deletion,
not only when the reader closes.

A separate opt-in, `ranking_staging_policy="bounded_ranking_staging_v1"`, is
now wired through annual registration, manifest validation and scheduled readers.
It leaves default published producers unchanged and does not enable feature expiry.
On a ranking miss it retains the complete final ranking and authenticated saved
receipt, but removes its own verified temporary metric/score files after independent
receipt reload. Exact saved hits run first and never clean up unfinished work.
Any pending allocation blocks new staged production before feature preparation;
interrupted feature retirement must therefore be recovered separately under its
existing authorization. No automatic retry, adoption or failed-work refund exists.
Admission reserves4.5GiB+64KiB above retained objects and metadata; it is not a
claim about actual bytes or annual capacity. Fixture tests cover weekly/daily
saved comparison, historical raw misses, active market changes and final cleanup
validation. The postformat combined backend/API regression passed93 tests in501.56s.
No real dataset has
this staging policy enabled, and annual shared-cache capacity remains unproven.

A read-only footprint check of the existing90-day research cache found about5.29GB
retained, including5.08GB of feature observations and0.10GB of checkpoints. This
supports focusing rolling retention on expired observation days; it does not prove
the full annual comparison fits the approved16GiB shared-cache limit.

Qualified registration now retains verified, compact daily candidate-count history
from the complete source origin, including dormant and ineligible wallets.
Preflight uses this evidence across the actual trader/market selection schedule;
the old100,000-trader clamp has been removed. Legacy datasets without this evidence
use an explicitly conservative fill-count bound. Qualified datasets without it,
or with insufficient count-history coverage, require re-registration or a covered
interval before running. Corrupt evidence is rejected, not rebuilt during a UI
request.

The run-readiness panel displays **Ranking rows (upper bound)** and its basis.
An upper bound above the row cap means capacity is unproven, not that actual output
necessarily exceeds the cap. A bound below the cap proves neither ranking-file
bytes nor shared-cache/total-disk sufficiency. Batched prefix-count queries verify
retained provenance at the calculation boundaries without rescanning raw fills.
This integration passed66 backend tests and14 frontend tests, production build and
generated-type checks. Actual annual capacity estimates and saved annual results
remain pending complete acquisition and the remaining storage gates.

## Open the registered real-data example

The current worktree instance is served at http://127.0.0.1:8010. In **Saved
backtests**, open either `Real cross-class · top 5 equal · 2d lookback · Aug 4–6`
scenario (5bps or 25bps proxy slippage), or select both to compare. Trader universe
shows historical membership and wallet drilldowns; Market universe shows mappings.

For a new hypothesis, select **Real cross-class hourly proxies · Aug 2026 · short
research window** in the builder and click **Load dataset preset** before editing
and running. Its two-day lookback and August 4–6 exclusive evaluation fit the
registered history; the normal quarterly defaults do not. This dataset contains
BTC, gold (GLD proxy), S&P 500 and Tesla, not every Hyperliquid market. General
selection ranks only the mapped, observed universe in that dataset.

These are hourly approximate results with real native trader activity, not exact
exchange replay. The first strategy loses money and underperforms BTC; the sample
is only 48 hours. See the [acceptance report](hyperliquid-real-proxy-acceptance-2026-09-07.md).
Data and saved runs live in `.hyperliquid_lab` in the feature worktree, not the
main checkout. Existing synthetic examples are still available.

## Try the local prototype

From this checkout, with the existing workspace dependencies installed:

```bash
uv run --all-packages python tools/generate_hyperliquid_lab_dataset.py \
  --output .hyperliquid_lab/datasets/demo
pnpm build:hyperliquid
HYPERLIQUID_LAB_ROOT="$PWD/.hyperliquid_lab" \
HYPERLIQUID_REPORTS_ROOT="$PWD/reports" \
HYPERLIQUID_WEB_ROOT="$PWD/apps/hyperliquid-explorer-web/dist" \
uv run --package hyperliquid-explorer-api uvicorn hyperliquid_explorer_api.app:app \
  --host 127.0.0.1 --port 8010 --ws none
```

Open http://127.0.0.1:8010. The dataset writer refuses to overwrite an existing
directory; run it only once, or choose a new dataset ID. No API keys are needed.

The builder checks the draft before enabling Run. Initial quarterly settings do
not fit the short demo: the readiness panel shows the required history, available
dates and specific coverage errors. **Load synthetic preset** is the opt-in way to
load compatible dates and relaxed eligibility; the app does not silently change
your configuration. **Ready to submit** is an advisory metadata check, not a promise
of a successful simulation: submission still verifies checksums, and the worker
checks actual market data. Submission shows **Saving…**, guards repeated clicks,
and focuses any error while preserving the draft. Preflight does not save a job.

1. Choose **Load synthetic preset**. This explicitly replaces the normal quarterly
   settings with a short, relaxed demonstration configuration.
2. Preview the historical trader universe, then **Run and save backtest**.
3. Inspect performance, historical selection dates, wallet membership and target
   contributions. Contributions explain positioning, not wallet PnL attribution.
4. Clone the completed run, change top N or another setting, and run again.
5. Select both in **Saved backtests** and compare configuration differences,
   portfolio statistics, equity/drawdown and average membership turnover.

The fabricated dataset contains six wallets across BTC, ETH and SOL, four days of
coverage and a two-day evaluation period. It demonstrates changing cohorts, not
trader skill. Sharpe/Sortino can correctly be unavailable on this short sample.

## Implemented boundaries

- Universe-first settings: copied assets, pooled/per-asset ranking, lookback,
  eligibility, metric weights/directions, top N/fraction, cohort guards and schedule.
- Direction/equal, direction/score-weighted and trimmed-conviction copying, asset
  budgets, caps, cadence, latency, fees and known-position thresholds.
- Prior-only selection evidence, including dormant observed wallets and explicit
  exclusions. The observable universe means wallets present in the registered
  history, not a claim of all Hyperliquid wallets.
- Independent BTC perpetual buy-and-hold benchmark and a cash control. Existing
  portfolio analysis includes Sharpe, Sortino, max drawdown, volatility, fees,
  funding, turnover and exposure with unavailable statistics shown honestly.
- Every submitted backtest has a persisted identity, canonical config/hash,
  dataset provenance and artifact checksums. Names/notes are editable; a changed
  hypothesis requires cloning. Preview jobs do not enter the saved backtest list.
- Comparisons support 2–6 completed lab experiments and warn about differing
  periods/data/assumptions. Curves retain their dates; no automatic winner is chosen.
- Legacy reports remain available at `/reports`; they are not silently converted
  into runnable configurations or admitted to saved-lab comparison.

Not yet implemented/qualified: parameter sweeps, real all-wallet quarterly data,
account-equity ROI ranking, validation/test promotion, live paper trading, and
cross-run membership timeline overlays. Membership history is currently drilled
into per run; comparison includes aggregate turnover. This runner is deliberately
development-only and cannot establish real strategy profitability.

## Dataset registration

### Cross-class market-universe prototype

Generate the additional fixture once (the writer refuses existing destinations):

```bash
.venv/bin/python tools/generate_hyperliquid_lab_cross_class_dataset.py \
  --output .hyperliquid_lab/datasets/cross_class_demo
```

Select **Synthetic cross-class** in Local dataset, then **Load synthetic preset**.
The copied-markets section now supports Crypto, Commodities, Equities and Indices.
**General** is mutually exclusive with selected classes and automatically ranks
the combined supported pool by traded USD notional. Within selected classes,
choose explicit instrument checkboxes or top-N volume selection. Market reselection
and trader reselection are independent. Changing classes/mode clears explicit
IDs and custom budgets with a visible notice; changing instruments resets budgets
to equal. Unavailable classes and volume capability are disabled for old datasets.

Daily volume uses a fixed one-day lag: at Jan 5, a two-day window covers Jan 2–3,
with every bucket published strictly before Jan 5. Missing volume is not zero.
Equal allocation divides by selected markets before checking trader availability;
an empty trader cohort does not redistribute its budget to other assets.

Results include **Market universe** timeline, candidates, exclusions, volume
windows, budgets, entries/exits and links to trader decisions. Comparisons separate
market membership turnover from trader membership and portfolio trading turnover.
Unscheduled previews are explicitly hypothetical; they do not change the actual
selection schedule. Old v1 runs remain unchanged and readable; cloning in the UI
creates a v2 draft with preserved scalar settings and explicit budgets.

The fixture adds seven fabricated continuous USD-linear instruments across four
classes, including two namespaced STOCK IDs. It is not real commodity/equity/index
market history and does not model exchange sessions, corporate actions or settlement.

For trusted v2 registration use `schema: hyperliquid_lab_dataset_v2`, the three
original files plus `instruments.parquet` and optional `market_volume.parquet`.
Declare `snapshot_at`, `catalogue_hash`, all file row counts/hashes, and volume
provenance (`source`, `currency: USD`, `conversion`, `counting: market_once`,
`interval: utc_day`). The fixture generator is the executable schema example.
Instrument versions require strictly prior `known_at`, effective intervals and
consistent lifetimes/execution specifications. Classification may change; base,
quote, settlement, multiplier and execution model may not change in this increment.
Only unit-multiplier continuous USD-linear contracts are executable. Unsupported
models remain visible but excluded. Mid-run listings require no invented earlier
marks; delisting during a participating run is rejected. Exiting a selected market
requests zero through normal delayed/depth-limited execution; remaining exposure
continues to be marked and funded. Market evidence has a one-million-row ceiling
counting excluded-class candidates too.

### Original v1 registration

An operator places a trusted dataset in `<lab-root>/datasets/<safe-id>/` containing
`manifest.json`, `fills.parquet`, `books.parquet`, and `funding.parquet`. The generated
demo provides a concrete schema example using the existing `FillEvent` and
`MarketData` contracts. The manifest declares:

- `schema: hyperliquid_lab_dataset_v1`, `name`, boolean `synthetic`;
- UTC date `coverage_start` / `coverage_end`, `coins`, `coverage_note`;
- `fee_semantics` (`gross_excludes_fee`, `net_includes_fee`, or `unknown`);
- `files`, each with fixed `name`, exact `rows` and `sha256`;
- optional `default_config` for an explicitly loaded preset.

Do not mutate registered files while experiments use them. Preflight verifies
hashes, row counts, copied-market/BTC scope, full ranking/normalization warmup and
resource bounds. The worker checks minute marks, hourly funding and executable
books at the configured cadence/delay. Missing funding is not imputed as zero.
There is no remote fetch, silent wallet sampling or automatic range shortening.

Limits: one million input rows, 250,000 copied-asset simulation minutes, one million
selected-wallet contribution rows (conservative estimate), and one million ranking
rows conservatively estimated as fill rows × evaluation days. These intentionally
restrict development runs; passing them is not research-data qualification.

## Local persistence and worker lifecycle

`HYPERLIQUID_LAB_ROOT` defaults to `<reports-root>/.copy_lab`. SQLite stores experiment
state; `staging/`, `results/` and `reports/` hold worker artifacts. Only completed
publication becomes visible to the report reader. A single process lock protects
one worker per lab root; at most 32 jobs may be queued/running.

Restart marks interrupted running work failed and leaves queued work requiring an
explicit Resume. Cancel terminates the owned worker without publishing results.
Failed/cancelled identities remain saved. Clone to retry a failed hypothesis.
Preserve the entire lab root when backing up saved experiments.

Keep the server on loopback. Local mutation endpoints require the bootstrapped
session token, JSON, allowed Host/Origin and bounded request bodies. These are
browser request protections, not multi-user authentication or public deployment
support. Registered datasets and local report files must be trusted.

## Checks

```bash
uv run --all-packages pytest packages/arblab/tests apps/hyperliquid-explorer-api/tests -q
pnpm --filter @cryptolab/hyperliquid-explorer build
pnpm --filter @cryptolab/hyperliquid-explorer test
pnpm --filter @cryptolab/hyperliquid-explorer types:check
pnpm --filter @cryptolab/hyperliquid-explorer test:browser
```

Browser tests create a temporary synthetic dataset and exercise the real local
worker/API, persistence, preview, clone/compare, historical wallets and mobile UI.
