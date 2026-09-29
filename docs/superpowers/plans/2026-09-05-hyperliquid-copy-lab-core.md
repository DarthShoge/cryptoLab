# Hyperliquid copy-lab core implementation plan

> **For agentic workers:** Use executing-plans inline, task by task, following the user's cost preference. Use test-driven-development for implementation. Keep the existing feature worktree. Do not implement sweeps in this increment.

**Goal:** A working local strategy builder → saved backtest → compare → historical universe loop, reusing the existing Hyperliquid engine and report UI.

**Architecture:** Versioned configuration and causal ranking/replay live in arblab. A local dataset catalogue, SQLite experiment store and single cancellable worker belong to the FastAPI app. The existing report reader remains compatible; new completed experiments publish reports into a dedicated local lab root and expose richer universe artifacts. React is configuration and presentation only.

**Tech Stack:** Existing Python/FastAPI/Pydantic, SQLite stdlib, DuckDB/PyArrow, React/TypeScript/Vite/Recharts and generated OpenAPI types. No new runtime service or external data download.

**Spec:** `docs/superpowers/specs/2026-09-05-hyperliquid-copy-strategy-lab-design.md`.

## Scope and evidence

This increment covers single configurations and manual comparison, not sweeps or
ROI-data acquisition. Launch only development/smoke experiments until the new
runner's research qualification and existing registry integration are verified;
advertise that capability restriction in the UI rather than bypassing study gates.
Legacy CLI/research gates are unchanged. Synthetic data remains visibly synthetic.
Use a deterministic multi-day, multi-market fixture for demonstration, never the
old three-minute report as if it were a runnable quarterly dataset.

## File map

Under `packages/arblab/src/arblab/hyperliquid_copy/`:

- `lab_config.py`: typed frozen dataclass settings and strict validation, effective
  config normalisation and human-readable summary; no Pydantic dependency in arblab.
- `lab_ranking.py`: weighted percentile ranking using existing episode metrics,
  additional gross volume, top-N/fraction guards, ranks/reasons and cohort deltas.
- `lab_pipeline.py`: scheduled cohort decisions, existing replay/aggregation/
  simulator reuse, independent BTC benchmark, contribution trace; bounded inputs.
- `lab_fixture.py`: explicit synthetic dataset writer for development/demo/tests,
  multi-reselection position histories with changing ranks; no network.
- Existing `ranking.py`: expose reusable per-wallet metrics without changing legacy
  five-factor behaviour. Existing simulator/report modifications, if necessary,
  must preserve their default output and tests.

Under `apps/hyperliquid-explorer-api/src/hyperliquid_explorer_api/`:

- `lab_models.py`: Pydantic request/response contracts; no duplicate strategy math.
- `lab_datasets.py`: manifest-first trusted local catalogue and bounded loading.
- `lab_store.py`: SQLite immutable experiment records, annotations and lifecycle.
- `lab_jobs.py`, `lab_worker.py`: one owned worker process, explicit queue/resume,
  cancellation, staged artifact publication and restart recovery.
- `lab_queries.py`: bounded cohort/ranking/contribution queries and comparison.
- `lab_routes.py`: typed lab router and local-write guard integration.
- `app.py`, `repository.py`, `models.py`: mount lab routes before SPA, lifespan,
  opt-in lab root and allowlisted new artifacts; preserve legacy GET API contracts.

Under `apps/hyperliquid-explorer-web/src/`:

- `Lab.tsx`: primary navigation and selected experiment/draft state.
- `StrategyBuilder.tsx`: grouped strategy inputs, coverage/preflight and launch.
- `Experiments.tsx`: saved library, metadata editing, progress, reopen and clone.
- `Universe.tsx`: dated cohort summaries, selected/excluded rankings, wallet history
  and signal-contribution drilldown.
- `Compare.tsx`: explicit config differences, metric table and multi-run overlays.
- `labApi.ts`: generated-type aliases, token-protected mutations and job polling.
- `App.tsx`: retain original explorer as supporting legacy view and export reusable
  report workspace; `styles.css`: focused strategy-first layout additions.

Tests: `packages/arblab/tests/hyperliquid_copy/test_lab_config.py`,
`test_lab_ranking.py`, `test_lab_pipeline.py`;
`apps/hyperliquid-explorer-api/tests/test_lab_store.py`, `test_lab_api.py`;
frontend unit tests plus `apps/hyperliquid-explorer-web/tests/lab.spec.ts`.
Docs/tools: `tools/generate_hyperliquid_lab_dataset.py`, update startup docs and
browser-server fixture launch. Generated artifacts remain ignored.

## Task 1 — Effective configuration and causal ranking

Implementation record (2026-09-05): Tasks 1–5 are implemented for the first
development increment. The original fine-grained checklist below is retained as
the planned acceptance detail, not a claim that every proposed test/commit was
executed verbatim. Work was tested inline and consolidated into one implementation
checkpoint rather than separate per-task commits. Comparison currently accepts
completed lab experiments only; legacy reports remain a separate supporting view.
Cross-run membership timeline overlays and library headline performance columns
remain follow-ups; per-run universe history and comparison turnover are available.
Task 6: correctness review and regression fixes completed, documentation and local
synthetic dataset prepared; final verification and running handoff recorded in
the task response. Review regressions cover dormant wallets, undefined efficiency
and conservative ranking-output bounds. No real profitability claim is made.

Final verification: 474 Python package/app tests passed (8 deselected by existing
configuration), 4 frontend unit tests passed, 6 real-API browser tests passed,
production TypeScript/Vite build and OpenAPI drift check passed. Existing dependency
deprecation warnings and the chart bundle-size warning remain. Loopback health and
registered synthetic dataset verified at port 8010. Worktree and branch retained;
no merge/push. The broader root collection issue involving missing arblab.perps is
outside this package/app suite and was not changed.

- [ ] Write failing tests: unknown fields, invalid weights, ROI unsupported,
  fractional/N selection exclusivity, copied SOL/ETH with BTC benchmark, fee and
  exposure bounds. Config roundtrip and hash are stable; every displayed field is
  effective, not an ignored UI setting.
- [ ] Run `.venv/bin/python -m pytest packages/arblab/tests/hyperliquid_copy/test_lab_config.py -q`; expect missing implementation failure.
- [ ] Implement a strict `LabConfig.from_dict` boundary, rejecting unknown keys,
  nonfinite/bool-as-number inputs and unsupported metrics; dataclass output is
  canonical and all numeric settings have finite bounds. Fixed version/split/
  benchmark fields are explicit. Normalise weights and market ordering once.
- [ ] Write ranking tests with existing `history()` fills: top two selected by
  PnL efficiency; volume reverses a constructed ranking; future fills do not
  affect an earlier decision; empty/undersized cohorts and ties are deterministic.
- [ ] Implement `rank_universe(fills, decision, config, scope)` returning raw
  metrics, percentiles, score, rank, selected, reasons and nominal trader weight.
  Filter copied markets and `[decision-lookback, decision)` before all metrics.
  Reuse the existing per-wallet episode calculation; add volume outside its
  legacy five-factor metric dictionary to preserve old scores.
- [ ] Run new and existing ranking tests, then commit this boundary.

## Task 2 — Configured replay, benchmark and historical traces

- [ ] Add tests before code for daily/weekly/monthly decision schedule, unchanged
  earlier decisions after future input, per-asset/pooled membership, target weights
  and BTC benchmark separation. Hand-check contributions sum to aggregate targets.
- [ ] Implement a new configured pipeline without rewriting the old CLI runner:
  keep full bounded fills for position replay; rank only prior-window copied-market
  activity. Reuse PositionReplay, RollingScale, aggregate and simulate. Effective
  portfolio targets are aggregate signal × asset budget × gross budget, capped per
  asset, passed to the simulator with an unambiguous scale of one.
- [ ] At scheduled decisions record counts, membership, entries/exits, retention,
  turnover, cutoff and reasons. Record per-wallet signal input, nominal/effective
  aggregation weight, known state and target contribution at each target update.
  Aggregate traces account for trimming and missing-state shrinkage explicitly.
- [ ] Return one selected strategy scenario at one delay, BTC perpetual buy-and-hold
  at that delay and cash; no silent fixed sweep. Preserve costs/funding and normal
  rebalancing when membership changes. Reject insolvency and incomplete funding.
- [ ] Add a synthetic fixture writer with BTC/ETH/SOL market marks, execution books
  and hourly funding plus warmup and multiple decision dates. Label its relaxed
  eligibility preset and fabricated source. Use a hard one-million input-row guard
  and a bounded output/contribution estimate before allocating arrays.
- [ ] Run pipeline, simulator and report regressions; inspect hand-calculated
  fixture scenarios. Commit engine increment.

## Task 3 — Datasets, immutable experiments and worker lifecycle

- [ ] Add failing temp-directory tests for catalogue containment/checksums, missing
  warmup/markets, oversized rows, changed inputs, saved identity across restart,
  metadata edits not changing config hash, clone-new-ID, queue cap and recovery.
- [ ] Catalogue: one operator-owned lab root with `datasets/<id>/manifest.json`
  containing schema, mode, coverage, fee semantics and relative allowlisted Parquet
  files with hashes. Validate containment and Parquet metadata before loading.
  Clients select IDs only. Do not register or fetch datasets through HTTP.
- [ ] SQLite store: generate UUID experiment IDs; freeze effective config and
  input/source/dependency fingerprints at submission. Keep name/notes separately.
  Use transaction-protected status transitions and at most 32 outstanding runs.
- [ ] A lifespan-owned coordinator starts one fixed Python worker subprocess with
  a store-controlled ID/root; no shell or caller-selected commands. Worker rechecks
  input hashes, runs the pipeline and writes unique staging artifacts. Coordinator
  publishes only successful validated output and records hashes. Interrupted jobs
  fail explicitly on restart; saved queued jobs need resume. Cancellation terminates
  the owned process and cannot publish partial success. Shutdown joins/terminates.
- [ ] Test real bounded worker completion, cancellation/recovery and failure paths,
  with no exchange/network access. Existing report files remain unchanged.
- [ ] Use the same queue/worker for a `cohort_preview` job kind, with frozen config,
  explicit decision date/scope and dataset/resource checks. Its output is a bounded
  ranking artifact, not a completed portfolio experiment. Preview jobs are excluded
  from backtest comparisons and the default saved-backtest library.

## Task 4 — Safe typed lab API and bounded research queries

- [ ] Write failing API tests for capabilities/datasets, launch/status/detail,
  annotations, clone/resume/cancel, completed-report lookup, comparison and historical
  universe filters. Require local Host/Origin, JSON mutation content and per-launch
  token; reject hostile/missing credentials, traversal, unknown config and bad IDs.
- [ ] Mount `/api/lab` router. Bootstrap exposes defaults, capabilities and local
  token with no-store. New lab report paths are resolved through the experiment
  store; legacy `/api/runs` remains read-only. GETs cannot enqueue or mutate state.
- [ ] Universe endpoints use paginated SQL scans, max 200 rows, exact date/scope/
  wallet filters. Typed rows include historical counts, ranks, percentiles/reasons
  and contribution traces. Missing legacy evidence is explicitly unavailable.
- [ ] Compare 2–6 completed experiment/report scenario identities: canonical config
  differences, existing full-series analytics, BTC excess return, cohort turnover,
  bounded curve points in dollar/rebased units. Mismatched periods/data/assumptions
  produce warnings; no silent intersection/re-ranking of incomparable results.
- [ ] Export OpenAPI and regenerate committed TypeScript; run drift and API tests.
- [ ] Add token-protected preview submission/status/result endpoints; paginate
  preview results with the same row cap. Test that preview at a saved backtest
  decision returns identical metrics, scores and membership for that scope.

## Task 5 — Strategy-first frontend and real end-to-end workflow

- [ ] Write failing browser smoke using actual temporary lab dataset/API: landing
  shows Strategy builder, choose SOL/ETH, top N and volume, launch, wait, reopen,
  clone/change setting, run again, compare, inspect decision date and wallet.
- [ ] Implement builder grouped by universe, ranking, copying and evaluation.
  Defaults come from typed backend capabilities; demo preset is explicitly relaxed.
  Surface unsupported ROI and real research readiness. Distinguish changed draft
  from immutable selected result. Handle rejected preflight/job failures visibly.
- [ ] Add a builder preview action with decision date and scope, job progress,
  eligible/selected counts and ranked rows. Explicitly label it historical preview,
  not portfolio performance. Browser tests compare it with the corresponding run.
- [ ] Implement saved library with polling while active, statuses, rename/notes,
  clone into draft, cancellation and explicit resume for persisted queued work.
  Completed details retain full strategy summary above reused portfolio metrics.
- [ ] Implement first-class universe drilldowns with counts/timeline, exact selected
  decision, rank table filters and clickable wallet history; contribution queries
  are scoped to timestamp/asset. Do not claim wallet PnL attribution.
- [ ] Implement comparison selection and config differences above chart/metric
  results; changing display units requests backend rebasing. Keep legacy reports
  accessible with an import label and no fabricated runnable config.
- [ ] Run TypeScript build, unit tests, generated drift and browser tests. Inspect
  desktop/mobile screenshots and verify chart/table loading/error states.

## Task 6 — Review, documentation and running handoff

- [ ] Use one bounded correctness review, fix material findings with regressions.
- [ ] Document dataset manifest/registration, synthetic fixture command, lab root,
  safe startup, worker lifecycle, saved experiment semantics and capabilities not
  yet qualified (ROI, real research, sweeps). Explain independent apps remain intact.
- [ ] Run all package/app offline tests, new browser workflow and existing explorer
  browser tests, build/unit/type drift. Report unrelated root collection errors
  separately; do not modify arblab.perps to obtain a green root suite.
- [ ] Generate a clearly synthetic local dataset into a new dedicated lab directory,
  launch the UI on loopback and verify the actual URL. Preserve user reports.
- [ ] Commit only task files, keep the worktree/branch; no merge/push. Hand off a
  functioning first increment and explicitly list the next increment (sweeps and
  data qualification), without claiming real performance.
