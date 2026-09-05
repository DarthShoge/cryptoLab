# Hyperliquid Market Universe Implementation Plan

> **For agentic workers:** Use @executing-plans inline, with @test-driven-development for each task and @verification-before-completion at checkpoints. This honours the user's earlier lower-cost execution preference. Use only the bounded plan/correctness reviews required by skills, not an implementation agent per task.

**Goal:** Replace fixed copied markets with historical, class-aware explicit/liquidity selection while making rejected submissions actionable.

**Architecture:** Python owns versioned instrument/configuration contracts, market selection and two-stage replay. FastAPI owns safe preflight, trusted datasets, immutable artifacts and bounded queries. React presents the generated contracts, separate market/trader controls and historical evidence; the existing v1 workflow stays readable and runnable.

**Tech Stack:** Existing Python dataclasses, FastAPI/Pydantic, SQLite, PyArrow/DuckDB, React/TypeScript/Vite, Vitest and Playwright. No new runtime dependencies or downloads.

**Approved spec:** `docs/superpowers/specs/2026-09-05-hyperliquid-market-universe-design.md` (approved after commit `e046914`).

## Execution status — 2026-09-05

- [x] Plan received bounded review: Approved.
- [x] Task 0: baseline verified; prior naming change committed as `bd187e4`.
- [x] Task 1 implementation: shared advisory inspection, typed safe coverage
  issues, authoritative checksum validation and generated API contracts. Three
  new API regressions observed failing before implementation, then passing.
- [x] Task 2 implementation: debounced exact-draft readiness, stale-response
  rejection, synchronous duplicate-click guard, Saving state and focused errors.
  Three new browser regressions observed failing before implementation, then all
  nine browser tests passed. No configuration is silently changed.
- [x] Task 1–2 final review: no material findings. Final regression: 477 Python
  package/app tests passed, 8 deselected; 6 frontend unit tests and 9 browser tests
  passed; production build and OpenAPI drift passed. Existing dependency and
  chart-bundle warnings remain. Owned local server refreshed after confirming
  zero active jobs; no saved run or dataset was overwritten.
- [ ] Tasks 3–10: market-universe extension not yet implemented.

The fine-grained task checklists below describe the execution recipe. This status
section records completed checkpoint work without implying later tasks are done.

## Working directory and verification commands

All paths below are relative to `/home/lshoge/code/cryptoLab/.worktrees/hyperliquid-trader-ensemble`.
Preserve existing uncommitted descriptive-name changes. Do not modify main-worktree
reports, overwrite either demo, merge, push, or purchase data.

Commands used throughout:

```bash
.venv/bin/python -m pytest packages/arblab/tests/hyperliquid_copy -q
timeout 60s .venv/bin/python -m pytest apps/hyperliquid-explorer-api/tests -q --tb=short
npm_config_cache=/tmp/cryptolab-npm-cache npx --offline pnpm@10.30.3 --filter @cryptolab/hyperliquid-explorer test
npm_config_cache=/tmp/cryptolab-npm-cache npx --offline pnpm@10.30.3 --filter @cryptolab/hyperliquid-explorer build
npm_config_cache=/tmp/cryptolab-npm-cache npx --offline pnpm@10.30.3 --filter @cryptolab/hyperliquid-explorer types
npm_config_cache=/tmp/cryptolab-npm-cache npx --offline pnpm@10.30.3 --filter @cryptolab/hyperliquid-explorer types:check
npm_config_cache=/tmp/cryptolab-npm-cache npx --offline pnpm@10.30.3 --filter @cryptolab/hyperliquid-explorer test:browser
```

API event loops, browser subprocesses and local servers need approved escalation
in this environment. Do not leave a hanging sandbox TestClient process. The
browser harness owns a temporary root at port 8011; the user's lab is port 8010.
Uvicorn must retain `--ws none`. Use `apply_patch` for edits, pinned formatters for
formatting, and force-add ignored plan/spec documents when making local commits.

## Task 0 — Preserve and establish the baseline

**Files:** existing name edits in `apps/hyperliquid-explorer-web/src/{Experiments,Overview,Records}.tsx`, `src/labApi.ts`, `src/scenarioLabel.test.ts`, `tests/lab.spec.ts`.

- [ ] Inspect `git status --short` and verify these are the prior task's edits.
- [ ] Run frontend unit/build/browser commands above. Expect 6 unit and 6 browser tests green before new tests. Preserve existing chart-size warning.
- [ ] Commit these exact paths separately as `fix(hyperliquid): describe saved scenario identities` after verification; do not include unrelated changes.

## Task 1 — Safe validation issues and shared lightweight preflight

**Create:** `packages/arblab/src/arblab/hyperliquid_copy/lab_validation.py`; `apps/hyperliquid-explorer-api/tests/test_lab_preflight.py`.
**Modify:** `apps/hyperliquid-explorer-api/src/hyperliquid_explorer_api/lab_datasets.py`, `lab_models.py`, `lab_routes.py`, `lab_jobs.py`; generated `apps/hyperliquid-explorer-web/src/api.generated.ts`.

- [ ] Add failing API tests using `lab_fixture.write_fixture` and TestClient's context manager. Bootstrap token, post the screenshot config (start 2025-01-05, end 2026-01-08, lookback90) to `/api/lab/preflight`: expect 200 with `ready=false`, codes `insufficient_warmup` and `end_after_coverage`, safe dates and no saved experiment. A valid preset gives `ready=true`.
- [ ] Add tests that direct `/experiments` submission still rejects invalid coverage with the same issue codes; checksum checks still run on submission, not advisory preflight. Force a parser error containing an absolute path and assert no path/exception text is returned. Invalid token/Origin/JSON must remain blocked.
- [ ] Run `timeout 60s .venv/bin/python -m pytest apps/hyperliquid-explorer-api/tests/test_lab_preflight.py -q --tb=short`; expect missing endpoint/contracts failures.
- [ ] Implement trusted public issue types and a typed exception; no arbitrary exception message reflection:

```python
@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str
    required: str | None = None
    available: str | None = None

class LabValidationError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("; ".join(i.message for i in self.issues))
```

- [ ] Extract `DatasetCatalog.inspect(identifier, config)` returning `ready`, `issues`, `config_hash`, required coverage and estimates. It validates manifest and Parquet metadata without loading rows/checksumming bytes. Keep `preflight` as authoritative: call inspect, raise typed issues if blocked, then verify every input checksum and return existing frozen provenance shape. `load` continues rechecking frozen identity.
- [ ] Return separate start/warmup/end/scope/resource issues with fixed message templates and validated date strings. Preserve conservative ranking-output bound. Expected warmup is `start - max(trader lookback, applicable scale lookback)` for v1; v2 extends this in Task 4.
- [ ] Add typed response models and `POST /preflight`; token guard already applies. Catch `LabValidationError` before ValueError, returning `detail` plus typed `issues` and status422 for direct submission. Unexpected parser/OS errors return only a fixed generic unavailable message. Config/Pydantic errors retain existing safe field validation.
- [ ] Run new and all API tests; expect green, unchanged valid job creation. Regenerate TypeScript and run drift. Commit as `fix(hyperliquid): expose safe actionable preflight issues`.

## Task 2 — Visible browser preflight and submission states

**Create:** `apps/hyperliquid-explorer-web/src/RunPreflight.tsx`, `src/usePreflight.ts`.
**Modify:** `src/StrategyBuilder.tsx`, `src/labApi.ts`, `tests/lab.spec.ts`.

- [ ] Add failing browser cases: initial demo/default settings show warmup/end issues and disabled Run; preset makes Ready to submit; delayed obsolete response cannot replace current preset readiness; two rapid clicks produce one experiment; submission422 displays server error without losing draft. Existing successful flow remains unchanged.
- [ ] Run browser tests and confirm the new readiness assertion fails on existing UI.
- [ ] Implement a 250ms debounced hook keyed by dataset/config/token. Abort stale fetches and independently check a monotonically increasing generation before applying results. Clear readiness synchronously for changed inputs; Run requires a successful response for the exact current draft key. Unmount cancels timer/fetch. Fetch mutation protection token but never persist an experiment during preflight.
- [ ] Present checking, blocked, Ready to submit, and error states beside Run with issue dates. Keep Load synthetic preset opt-in. Readiness explicitly does not guarantee full data validation success.
- [ ] Add a synchronous `useRef` submission lock in addition to disabled state; show Saving… and release in finally. Focus a `tabIndex=-1`, `role=alert` submission error; keep draft intact. Catch expired token with refresh instruction; never retry POST automatically.
- [ ] Run unit/build/types/browser checks. Update `docs/hyperliquid-copy-lab.md` with readiness meaning. Commit as `fix(hyperliquid): show preflight and saving feedback`.
- [ ] Checkpoint: restart owned local server only when no jobs are active, verify health/datasets, and hand off this independently useful fix if stopping before Task 3. State that market selection remains pending.

## Task 3 — Versioned config, namespaced instruments and historical catalogue

**Create:** `packages/arblab/src/arblab/hyperliquid_copy/lab_config_v2.py`, `lab_instruments.py`, `lab_config_codec.py`; tests `packages/arblab/tests/hyperliquid_copy/test_lab_config_v2.py`, `test_lab_instruments.py`.
**Modify:** domain `contracts.py`; API `lab_models.py`, `lab_jobs.py`, `lab_worker.py`.

- [ ] Write tests for `parse_lab_config(payload)` dispatching by schema, strict inactive-field rejection, General exclusivity, finite bounds, custom budget sums, fixed publication lag1 and namespaces. `symbol('demo:ABC')` succeeds and differs from `other:ABC`; path-like, multiple-colon and whitespace IDs fail. V1 canonical payload/hash unchanged.
- [ ] Run those files; expect missing implementation/namespace failures.
- [ ] Preserve `LabConfig` v1 unchanged. Extract reusable scalar-field validation only if required, retaining all v1 golden tests. Define v2 as a composition: `schema_version`, `market_universe`, `trader` and `follower` sections with current effective scalar controls, plus evaluation dates/benchmark/split. No redundant writable `coins` or top-level asset_weights in v2.
- [ ] Define strict dataclass unions `ExplicitUniverse(mode, general=False, classes, instrument_ids, allocation, weights, reselection)` and `LiquidityUniverse(mode, general, classes, top_n, min_volume_usd, lookback_days, reselection, metric='traded_notional_usd', publication_lag_days=1)`. Equal allocation has no custom weights; liquidity has neither weights nor explicit IDs. Limits and exact supported values come from approved spec.
- [ ] Implement `migrate_v1_to_v2` preserving explicit IDs/weights and every effective setting; store migration notice in clone draft response, not original evidence. Existing v1 execution may use the legacy runner to guarantee output stability; no persisted conversion of old runs.
- [ ] Define instrument metadata/lifetime records with ID/display/venue/class/base/quote/settlement/linear multiplier/model/effective_from/effective_to/known_at/listed_at/delisted_at. Catalogue parsing rejects conflicting overlapping effective records rather than guessing correction precedence. Require common USD valuation and multiplier1 for this increment; unsupported records have explicit capability reasons.
- [ ] Catalogue resolution at T uses only records known strictly before T and effective at T. Exclusion history retains previously observed IDs when no current version exists. Requested explicit future IDs may be displayed as requested-but-not-yet-known, without exposing future classification/volume fields. Never include unknown future IDs in automatic candidates.
- [ ] Thread config union through Pydantic request/response/clone models without coercing unknown fields. Worker dispatches version; v2 submission remains capability-blocked until Task 7. Run domain/API regressions and regenerate types. Commit contracts checkpoint.

## Task 4 — Versioned local dataset with market-wide daily volume

**Create:** domain `packages/arblab/src/arblab/hyperliquid_copy/lab_volume.py`; tests `packages/arblab/tests/hyperliquid_copy/test_lab_volume.py`, API `apps/hyperliquid-explorer-api/tests/test_lab_datasets_v2.py`.
**Modify:** API `lab_datasets.py`; domain `lab_instruments.py`.

- [ ] Test UTC daily nonoverlapping buckets, unique ID/day, finite nonnegative USD notional, publication >= bucket end and <= manifest snapshot, valid catalogue references, and clear missing-versus-zero semantics. V1 three-file manifests still load.
- [ ] Test malformed/oversized catalogues, unknown metadata fields, path traversal, hash mismatch and missing market_volume capability. Assert advisory inspect does not scan complete volume rows.
- [ ] Implement v2 manifest fixed filenames: fills/books/funding plus `instruments.parquet`, optional `market_volume.parquet`. Require `snapshot_at`, versioned volume provenance/source and USD conversion/count-once convention when volume exists. Existing manifest file containment/metadata/hash mechanisms apply to every declared file.
- [ ] Extend public dataset capabilities: supported classes, liquidity availability, catalogue summary/hash, per-model availability and preset union. Expose paginated `/datasets/{id}/instruments` for searchable checkboxes; never send an unbounded catalogue in bootstrap.
- [ ] Extend inspect warmup by `liquidity.lookback_days+1`, applicable trader/normalization warmup; derive candidate count from all potentially participating catalogue IDs, not top_n. Count all file rows toward1m, candidate minutes toward250k, all decision output rows toward1m. Cache only immutable hash-keyed parsed catalogue metadata; never trust a stale live path after submission.
- [ ] Return a `LoadedDataset` value containing fills, market data, manifest, catalogue and optional volume; adapt v1 call sites explicitly rather than expanding positional tuples differently by version.
- [ ] Run new dataset/domain/API tests and commit `feat(hyperliquid): register historical instrument and volume datasets`.

## Task 5 — Pure causal market selector and typed evidence

**Create:** `packages/arblab/src/arblab/hyperliquid_copy/lab_market_selection.py`, `lab_market_evidence.py`; tests `packages/arblab/tests/hyperliquid_copy/test_lab_market_selection.py`.

- [ ] Write hand-calculated tests with 4 classes and duplicate display tickers: at Jan5 with lookback2/lag1, use buckets Jan2 and Jan3, ignore Jan4 and publication >= Jan5. Volumes100/300/300 select deterministic IDs for top2. Append future catalogue/volume rows and assert earlier rankings byte-equivalent.
- [ ] Test classification refresh, explicit future admission, positive threshold, validzero, missing days, delisted/unsupported/unclassified records, no eligible markets and normalization-history exclusion.
- [ ] Implement `select_markets(dataset, universe, decision, previous, normalization_days)` returning all observed ranking rows, selected IDs/budgets and one cohort snapshot. Reuse existing retention/turnover convention. Automatic rank is volume descending then exact ID ascending; explicit rows have nullable volume/rank, not invented liquidity values.
- [ ] Window implementation is fixed and shared by preflight evidence/preview:

```python
window_end = decision - timedelta(days=universe.publication_lag_days)
window_start = window_end - timedelta(days=universe.lookback_days)
# Require each full UTC day, publication strictly < decision.
# Equal budget denominator is selected assets before trader availability.
```

- [ ] Declare Arrow schemas for market_rankings/cohorts, including empty outputs. Store decision_time, window dates, ID/venue/class, eligible/selected/reasons/rank/volume/budget, counts/members/entries/exits/retention/turnover and effective target tick. Cross-class selection ranks the combined pool, not each class separately.
- [ ] Run selector tests and old ranking tests; commit `feat(hyperliquid): select markets from prior liquidity evidence`.

## Task 6 — Optional instrument lifetimes in simulation

**Modify:** `packages/arblab/src/arblab/hyperliquid_copy/simulator.py`, `execution.py` only if needed for zero-position valuation handling; `lab_instruments.py`.
**Create tests:** `packages/arblab/tests/hyperliquid_copy/test_lab_lifetimes.py`.

- [ ] Write failures for mid-run listing with no earlier marks/funding; valid postlisting trade; rejected outside-lifetime signal/fill; missing held-position mark; residual funding after universe exit. Snapshot legacy outputs and assert no change when optional lifetime argument absent.
- [ ] Add keyword-only `lifetimes=None` to `simulate`. For v2 compute mark requirements from currently listed instruments plus nonzero account positions and pending orders. Skip prelisting unheld instruments entirely; reject a target outside tradability instead of inventing its mark. Account equity/exposure must not require dictionary marks for truly zero positions, but must require marks for any nonzero position.
- [ ] Validate pending execution timestamp against lifetime, not only signal timestamp. Zero-target requests outside active lifetime cannot liquidate a delisted position; reject unsupported delisting during a participating run in preflight/worker. Funding events for unlisted unheld assets are invalid data; never impute zero funding for held assets.
- [ ] Run `.venv/bin/python -m pytest packages/arblab/tests/hyperliquid_copy/test_lab_lifetimes.py packages/arblab/tests/hyperliquid_copy/test_simulator.py -q`, then whole domain directory. Commit only after unchanged legacy results.

## Task 7 — Two-stage replay, worker publication and preview parity

**Create:** `packages/arblab/src/arblab/hyperliquid_copy/lab_pipeline_v2.py`, `lab_schedule.py`; test `packages/arblab/tests/hyperliquid_copy/test_lab_pipeline_v2.py`.
**Modify:** API `lab_worker.py`, `lab_jobs.py`; domain `lab_ranking.py` only for a narrow explicit market-scope adapter, not legacy metric changes.

- [ ] Test asset daily/trader weekly and the reverse, all coincident decisions, per-asset entry forced ranking, pooled asset-change forced ranking, and unscheduled preview distinction. Assert one decision per scope/time.
- [ ] Implement `decision_timeline(config,start,end)` and `selection_state_at(dataset,config,T)` shared by replay/preview. At each minute process asset decision first; rank new per-asset cohorts or the new pooled set immediately; retain unaffected cohorts until scheduled trigger. Record `market_decision_time`, `trader_decision_time`, trigger and effective update time.
- [ ] Reuse v1 ranking/aggregation/position replay through an internal effective-settings adapter accepting arbitrary IDs without constructing a validating v1 fixed-market config. Shared trader scalar validation must remain identical.
- [ ] At each target tick emit targets for every active selected market and zero targets for every previously selected market still tradable. Position state/scale updates continue independently of universe membership where necessary. Never redistribute budgets from missing trader cohorts. Scale warmup uses only admitted active history.
- [ ] Test selected2→selected1 exit with execution latency and insufficient depth: zero request persists, residual quantity continues funding/marking, contributions reconcile only to targets (not attributed PnL). Cash/empty selection produces a valid empty trader evidence table. BTC benchmark remains independent.
- [ ] Worker validates all required active-lifetime marks/funding/books without replacing data-deficient selected assets. Write v2 report with original submitted canonical config and one strategy scenario; new market artifacts alongside trader artifacts with explicit schemas. Existing source fingerprint includes new domain modules and engine version becomes copy_lab_v2 for v2 jobs only.
- [ ] Preview replays selection schedule through T but does not simulate portfolio returns. At actual trader decisions return identical rankings/market-decision link; otherwise compute hypothetical as-of trader ranks under currently effective market set and label them hypothetical.
- [ ] Remove v2 runtime capability block only when pipeline tests pass. Verify checksum publication, cancellation and clone/read preserve old experiments. Commit replay checkpoint.

## Task 8 — Bounded market history and comparison APIs

**Create:** API `apps/hyperliquid-explorer-api/src/hyperliquid_explorer_api/lab_market_queries.py`; tests `apps/hyperliquid-explorer-api/tests/test_lab_market_api.py`.
**Modify:** API `lab_models.py`, `lab_routes.py`, `lab_queries.py`.

- [ ] Add failed tests for completed v2 market history, page size201 rejection, exactdate/class/instrument filtering, namespaces, empty results, unknown artifacts and v1 unavailable evidence. Assert artifact query cannot accept a path or arbitrary column.
- [ ] Add `/experiments/{id}/market-universe?table=rankings|cohorts` with fixed artifact map, stable sort,200 cap, bound SQL and typed row unions. Existing trader history gains decision linkage fields without breaking v1 rows.
- [ ] Extend comparison with nullable market counts/mean turnover and canonical market-rule differences. Distinguish unavailable v1 from empty-zero v2. Do not build cross-run membership overlays or silently normalize mismatched windows.
- [ ] Run new/all API tests; regenerate TypeScript/drift. Commit API checkpoint.

## Task 9 — Cross-class synthetic fixture and focused frontend components

**Create:** domain `packages/arblab/src/arblab/hyperliquid_copy/lab_fixture_v2.py`; tool `tools/generate_hyperliquid_lab_cross_class_dataset.py`; web `src/MarketUniverseBuilder.tsx`, `src/MarketUniverse.tsx`, `src/marketConfig.ts` and `src/marketConfig.test.ts`.
**Modify:** web `src/StrategyBuilder.tsx`, `src/Experiments.tsx`, `src/Universe.tsx`, `src/Compare.tsx`, `src/labApi.ts`, `src/styles.css`, `scripts/browser-server.mjs`, `tests/lab.spec.ts`.

- [ ] Fabricate BTC plus 2crypto/1commodity/2equity/1index namespaced instruments, changing daily liquidity ranks and multi-decision trader positions. Give all normal fixture listings enough historical warmup; add a separate new-listing test fixture. Manifest says fabricated continuous USD-linear contracts, not real market specifications. Writer refuses existing destination.
- [ ] Keep old fixture/default dataset intact. Browser harness registers both, exposing v1 regression and v2 class-selection flows. All volume is independently fabricated market data, not derived from wallet fills.
- [ ] Test pure `marketConfig` transitions: General clears class/IDs/weights, class clears General, mode switch drops inapplicable fields, custom weights reset visibly, no empty selections accepted. Test summaries include pool,topN,volume-window/lag and separate trader settings; preserve descriptive-name behavior.
- [ ] Build class/mode/ID controls in MarketUniverseBuilder using typed catalogue endpoint. Paginate/search, show unavailable classes and supported execution model. StrategyBuilder composes this component and RunPreflight; do not embed ranking math in React. Explicit ID budgets and liquidity equal allocations render differently.
- [ ] Add result Market universe tab with timeline,date/filter drilldown and instrument history; link to trader evidence. Preview shows effective assets and scheduled/hypothetical labels. Compare displays market turnover separately from trader turnover.
- [ ] Add actual-API browser test: select new dataset/preset → General → topN → run → market history → trader link → clone into class-restricted configuration → run → compare. Assert General cannot coexist with class/IDs. Add narrow-screen result/detail tests as well as builder tests; no horizontal document overflow.
- [ ] Run unit/build/drift/browser commands and inspect screenshots. Commit integrated UI checkpoint.

## Task 10 — Review, full regression and local handoff

**Modify:** `docs/hyperliquid-copy-lab.md`, this plan status; no prior report contents.

- [ ] Request one bounded correctness review of causality, lifetime handling, safe validation and immutable v1/v2 compatibility using @requesting-code-review. Reproduce material findings with failing tests, fix inline, and re-review affected scope.
- [ ] Run `timeout 60s .venv/bin/python -m pytest packages/arblab/tests apps/hyperliquid-explorer-api/tests apps/kamino-simulator/tests apps/report-explorer/tests apps/strategy-backtester/tests -q --tb=short`, all frontend checks, and `git diff --check`. Missing unrelated root arblab.perps collection remains out of scope.
- [ ] Document v2 registration, publication lag, supported linear model, non-crypto qualification limitations, lifecycle, explicit budget refresh, preflight readiness and reproduction commands.
- [ ] Generate new ignored `.hyperliquid_lab/datasets/cross_class_demo` only if absent; never overwrite demo. Stop/restart only the owned port8010 server after confirming zero active jobs. Verify health and both datasets, then inspect UI at localhost8010.
- [ ] Update checked tasks with observed verification results, commit exact task files, and retain branch/worktree. Hand off capabilities and remaining real-data/ROI/sweeps work without claiming profitability.

## Checkpoint discipline

Task 2 is an independently useful stopping point. Tasks 3–9 together implement
the approved market extension; do not report it complete while only selectors or
widgets exist. After each task run its listed tests before marking the checkbox.
Unexpected data/model requirements must be surfaced, not silently implemented as
a financial-model expansion. All arithmetic remains covered by causal fixtures.
