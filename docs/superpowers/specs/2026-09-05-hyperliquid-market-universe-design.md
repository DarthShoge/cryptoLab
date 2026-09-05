# Hyperliquid market-universe selection and actionable run validation

## Purpose and status

Next increment of the existing copy-strategy lab, following the user's request
to remove BTC/ETH/SOL restrictions and select liquid crypto, commodity, equity
and index markets. This is a design for review, not implemented capability.
It extends the 2026-09-05 copy-strategy-lab design; all local-only, synthetic-data,
immutable-experiment and no-exchange-order boundaries remain in force.

The workflow becomes **select markets → select traders → combine positions →
simulate → save and compare both historical universes**. Market liquidity does
not become a trader-skill metric. No live data acquisition is included here.

## Chosen approach

Use a dataset-backed instrument catalogue with point-in-time market-volume
observations. Retain the existing simulator and saved-experiment workflow.

Alternatives: enlarging the fixed ticker list is cheap but cannot express classes
or historical top-volume selection. Live metadata discovery would help future
ingestion but cannot reconstruct historical liquidity. Neither solves this request
alone. A generic multi-strategy asset framework would exceed the app's boundary.

## Builder controls

Market universe is the first section, above trader selection:

- Class checkboxes: **Crypto**, **Commodities**, **Equities**, **Indices**.
- **General — all supported classes** is a separate mutually exclusive checkbox.
  Selecting General clears class selections; selecting any class clears General.
  At least one class or General is required. General is selection mode metadata,
  not a fifth instrument classification.
- With explicit classes, choose **Specific instruments** or **Top by volume**.
  Specific instruments are searchable checkboxes, grouped by class and labelled
  with market name and venue. Selected instruments must belong to selected classes.
- General uses automatic **Top by volume** selection across the four supported
  classes. It cannot carry specific instrument IDs or class selections.
- Automatic controls: top N assets, minimum trailing dollar volume, liquidity
  lookback days and asset reselection schedule (daily/weekly/monthly).
- Rank top N across the combined selected pool, not N per class. Show this wording.
  Additional ranking metrics are disabled until they have a historical data
  contract; initial supported metric is trailing traded dollar volume.
- Trader controls remain separate: per-asset/pooled, trader lookback, eligibility,
  top N/percentage of wallets, score weights and trader reselection schedule.
- Dataset capability/coverage controls availability. Unsupported classes are
  visible with an explanation, never populated with invented real instruments.
  General means all supported classes in this dataset, not all of Hyperliquid.

Changing class or instrument mode explicitly clears incompatible selections and
custom allocations; explain the reset next to the controls. Do not hide stale
settings in the request. Legacy defaults can remain ETH/SOL for old datasets;
the available menu must come from the catalogue, not hard-coded symbols.

## Versioned configuration and catalogue

Introduce a v2 lab configuration with a discriminated market-universe definition:

- `mode`: `explicit` or `liquidity`.
- `general`: boolean; when true, mode must be liquidity and classes must be empty.
- `classes`: nonempty subset of crypto/commodity/equity/index unless general.
- Explicit-only: nonempty unique `instrument_ids` and equal or custom asset budgets.
- Both modes: `reselection` (daily/weekly/monthly), default daily. For explicit
  selection this is labelled **Instrument eligibility refresh**, not liquidity
  ranking; the requested ID set stays fixed, but effective eligibility can change.
- Liquidity-only: `top_n` (1–25), `min_volume_usd` (nonnegative finite),
  `lookback_days` (1–3650); equal asset budgets.
  The metric is explicitly versioned `traded_notional_usd`, not a free-form string.
- Reject fields belonging to the inactive mode, boolean-as-number, unknown fields,
  invalid classes, unsupported IDs and General mixed with classes or explicit IDs.

Existing v1 configuration/hash/artifacts are never rewritten. A v1 adapter maps
BTC/ETH/SOL to explicit crypto instruments for execution without changing the
stored payload. Cloning v1 into v2 is an explicit draft migration with visible
equivalent settings and parent ID. Completed v1 reports stay readable regardless
of dataset availability; unsupported reruns fail with a clear reason.

Dataset v2 manifest freezes a catalogue hash and these instrument attributes:
canonical instrument ID, display name, venue, class, base/quote/settlement identity,
contract type/multiplier, listing/delisting effective times and supported execution
model. Preserve namespaced venue-qualified IDs end-to-end; never merge markets
merely because their display tickers match. Treat IDs as values, not file paths.
The existing `contracts.symbol` rejects colons; explicitly extend the shared
instrument-ID boundary and its tests, preserving accepted core names and rejecting
slashes, traversal, whitespace and malformed namespaces. Audit fills, books,
funding, API filters and report paths so namespace support is end-to-end, not
only a new dropdown. Do not uppercase or strip the venue portion.

This increment executes only the existing linear, USD-valued perpetual simulation
model with continuous minute marks and declared hourly funding. Unsupported
collateral valuation, multipliers, session gaps, settlement or payoff models fail
capability validation; adding an equity/commodity label is not model qualification.
Synthetic fixtures cover all four classes using this explicitly supported model.
Real non-crypto market qualification remains later work, not assumed by the UI.

## Historical market-volume data

Add immutable `market_volume.parquet` to v2 datasets supporting automatic selection.
Each observation contains instrument ID, UTC interval start/end, publication
(`available_at`) time, and nonnegative finite traded notional in common USD units.
Use complete UTC daily buckets initially, unique per instrument/day; reject
duplicates, overlaps, future publication relative to dataset snapshot, invalid
notional, and ambiguous currency/conversion conventions. Manifest provenance
declares source and USD conversion convention. Count each trade once at market
level, not once for maker and again for taker.

Do not derive market liquidity from the selected trader sample. Do not sum rolling
24-hour snapshots as if they were non-overlapping daily buckets. Unknown volume
is not zero; valid zero-volume buckets are explicit.

At decision T (UTC midnight), the initial daily-volume model uses a fixed one-day
publication lag: window [T-1 day-lookback, T-1 day). Show **1-day volume-data lag**
in the builder, saved definition and evidence; store it explicitly in v2 with
the sole supported value 1. Use only buckets published strictly before T. Require
publication no earlier than bucket end. Missing or late buckets exclude the
instrument with a missing-volume reason, never a zero. This avoids assuming the
day ending exactly at T was completely published before that same decision.
Warmup includes liquidity lookback plus this lag. Synthetic fixtures use honest
publication times after bucket end, and real datasets must follow the same rule.

An instrument becomes a candidate only once its listing/catalogue information was
known before T. The catalogue must include `known_at` for listing/classification
records, with effective intervals for changes; use the version effective and known
at T. Include delisted historical instruments; today's inventory is not the past
universe. Eligibility requires active listing at T, supported model, full liquidity
lookback and the configured volume threshold. Record every observed candidate's
exclusion reason. An unclassified instrument is not silently assigned a class.

Sort eligible instruments by descending volume, break ties by canonical ID, select
up to N. Fewer than N remain visible as a smaller selection; zero means a zero
target portfolio. No threshold relaxation or universe sampling. Explicit selection
does not need volume data but still checks listing, classification and model.

## Schedule, budgets and exit behaviour

Both selection stages run initially at simulation start. Thereafter each uses its
own existing UTC daily/weekly/monthly convention. Process market decisions before
trader decisions when timestamps coincide.

Explicit mode reevaluates its fixed requested IDs at each configured market
refresh. Not-yet-known/listed IDs remain inactive with reasons; newly eligible
IDs enter at the first refresh after becoming known and active. Classification
changes can remove IDs that no longer match requested classes at the next refresh.
This is causal eligibility checking, not reranking the manually chosen list.
Funding, actual tradability and valuation follow instrument lifetimes independently
of refresh times; never submit a trade outside an instrument's active lifetime.

- Per-asset mode: newly entering assets receive an immediate trader ranking;
  retained assets keep their cohort until its scheduled trader reselection.
- Pooled mode: any market-set change immediately reranks the pooled cohort using
  only fills in the new selected asset set and the trader lookback. Record the
  decision trigger; otherwise a pooled score could retain excluded-market evidence.
- Normal scheduled trader decisions continue independently. No duplicate decision
  rows when both triggers occur at one timestamp.
- Automatic budgets are 1 / actual selected asset count before trader availability
  is evaluated. Explicit equal budgets use selected eligible assets; explicit
  custom weights keep their configured weights, with inactive allocations in cash.
  No redistribution from assets whose trader cohorts/signals are unavailable.
- Removed assets get a zero target at the next configured target-update tick.
  Entries and new weights likewise take effect at the next tick. Execute through
  normal latency, fees, depth, deadband and minimum-trade rules, not instant fills.
  Continue issuing zero targets for exited assets so residual exposure can be
  unwound. Continue marks and funding until the end even if an exit cannot fill.
- Preserve the full dataset-candidate set for position reconstruction and
  conservative resource bounds; the selected set controls new copied targets.
  Missing required execution/valuation evidence is a data failure, never an
  ex-post reason to replace a selected market with the next winner.

Extend the simulator with an optional instrument-lifetime contract for v2; legacy
calls retain their behaviour. Do not request a mark or funding for an instrument
before listing when it has no position or pending order. Require minute marks
while listed during the evaluation window, and at every valuation/funding event
for nonzero residual positions after any universe removal. Never fabricate
pre-listing marks, erase residual positions or forward-fill across missing data.
Only require normalization warmup marks during an instrument's active lifetime;
if its available active history is shorter than a selected normalization window,
exclude it as insufficient normalization history before asset admission. Volume
lookback and trader eligibility retain their independent coverage rules.
Reject contradictory lifetimes, fills outside declared tradability and unsupported
delisting/settlement. Scope this extension to lifecycle-aware data access and
target admission; do not rewrite account math or implement settlement here.

BTC remains an independent named benchmark, not automatically a copied asset.
Insufficient benchmark data prevents the configured run. Delisting/settlement
inside a run requires a supported model and continued valuation; otherwise reject
the run rather than fabricate a settlement price or erase a position.

## Historical evidence and comparison

Add **Market universe** beside Trader universe in saved results:

- Decision timeline: candidate/eligible/selected counts, requested N, entries,
  exits, retention and membership turnover using the existing cohort convention.
- Date drilldown: instrument/venue/class, trailing volume, rank, eligibility,
  selection, reasons and allocated budget. Show effective target-update time.
- Instrument history across decisions. Trader cohort and contribution rows link
  to the market decision that admitted their instrument; display forced-rerank
  triggers separately from scheduled trader decisions.

Persist `market_rankings.parquet` and `market_cohorts.parquet` with explicit schemas,
including typed empty outputs. Extend paginated exact-date/instrument/class filters
with the existing 200-row cap and safe bound SQL parameters. Preview replays market
decisions from start through the chosen date to obtain the actual active market
set, then shows trader ranking as of that date. Label an unscheduled trader
preview as hypothetical; at a recorded trader decision it must match the run.

Comparison includes the configured market rule, selected-asset counts and market
turnover, separate from trader membership turnover. Legacy missing market evidence
is unavailable, not zero. Keep existing config differences and no-winner warnings.
No cross-run market timeline overlay is required in this increment.

## Run validation and feedback

The reproduced issue: default 90-day lookback and requested dates exceed the
four-day demo; API catches the coverage ValueError but treats the slash in
`lookback/warmup` as a path and replaces the explanation with a generic message.

Introduce typed public validation issues (`code`, `field`, safe `message`, optional
required/available dates). Only explicitly constructed application messages are
returned. Parser, OS and unexpected exception text stays private. Do not simply
remove path sanitisation for arbitrary exception messages.

Add token-protected, non-persisting `POST /api/lab/preflight` using the same canonical
validation as submission. Return issue list, required coverage, resource estimates
and normalized config identity. Debounce requests after draft changes, discard
stale responses, and show checking/blocked/ready state near Run. Preflight scans
manifest and bounded metadata only; expensive checksum and data validation remain
submission/worker checks, so Ready means ready to submit, not guaranteed success.
Submission reruns authoritative checks; never trust the browser's earlier result.

Blocked Run shows specific dates/lookback or unsupported capabilities and a clear
recovery action. Load synthetic preset remains opt-in; never silently change dates
or eligibility. During submission show **Saving…** and prevent duplicate clicks;
on success show saved identity/queued state and navigate to the result. Failures
use an accessible, focused alert, preserving the draft. Expired launch token tells
the user to refresh; do not automatically retry mutation requests.

## Bounds and module ownership

Keep current one-worker, 32-outstanding-jobs and development-only restrictions.
Preserve one-million input-row, 250,000 copied-asset-minute, one-million ranking
and contribution limits. Include catalogue/volume rows and market-history output
in conservative preflight bounds. Estimate using all candidate assets that can
participate, not merely simultaneous top N; never bypass limits with rotation.

- arblab: instrument contracts, market selection, typed validation issues, v1/v2
  config adaptation, scheduled two-stage replay and evidence generation.
- API: versioned dataset parsing, shared preflight, jobs, artifact publication,
  pagination and generated OpenAPI contracts. No independent ranking formula.
- Frontend: a focused MarketUniverseBuilder component, a preflight/status component
  and a MarketUniverse result view. Extract these rather than growing the existing
  large StrategyBuilder with another conditional block.
- Preserve the descriptive scenario-name changes currently uncommitted in this
  worktree. No edits to unrelated apps or user reports; no merge/push in this scope.

## Acceptance tests and delivery order

1. Validation regressions: the screenshot's invalid period and 90-day warmup produce
   safe, specific issues; preset clears them; duplicate-click and stale-response
   browser cases preserve exactly one saved job or an unchanged invalid draft.
2. Catalogue/config tests: namespaced duplicate tickers stay distinct, all four
   classes, mode exclusivity, unknown IDs, unsupported models, v1 compatibility.
3. Hand-calculated market selection: volume units/counting, known-at and listing
   boundaries, ties, zero/missing volume, top N and thin/empty pools. Appending
   future observations or future listings cannot alter earlier selections.
4. Replay: independent schedules, forced new-asset/pooled reranking, simultaneous
   decisions, budget sums, no missing-cohort redistribution, lagged entry/exit
   targets and residual funding/exposure. Explicit-mode admission/classification
   changes follow its refresh schedule; mid-run listings need no fake earlier
   marks, but held positions always require valuation. Benchmark stays outside
   selection. Existing fixed-universe simulator tests remain unchanged and green.
5. Immutable storage: market evidence hashes, typed empty tables, restart/read,
   clone, preview parity at recorded decisions and paginated filters.
6. Browser: General/class mutual exclusion, explicit instruments versus liquidity,
   synthetic cross-class preset, preflight, run/save, historical assets and traders,
   clone/change market rule/compare; mobile and legacy report regressions.

Deliver clearer validation first, then versioned catalogue/config, causal market
selection/replay, and finally integrated controls/history against a new synthetic
cross-class fixture. Do not overwrite the existing demo or saved experiments.
Real market ingestion, additional liquidity metrics, sessions/settlement models,
account ROI and sweeps remain separate subsequent work.
