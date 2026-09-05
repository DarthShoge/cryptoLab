# Hyperliquid Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Inline execution is the user's existing cost preference. Steps use checkbox syntax for tracking.

**Goal:** Serve an actually usable local, read-only React dashboard for Hyperliquid report artifacts, including portfolio analytics.

**Architecture:** A FastAPI app owns safe report discovery, bounded Parquet queries and descriptive analytics. A React/TypeScript/Vite app uses generated OpenAPI types, renders four tabs and can be served from the Python app's configured static-build directory. Preserve the existing strategy engine and raw reports.

**Tech Stack:** Existing uv/pnpm workspaces; FastAPI/Pydantic/Uvicorn/DuckDB; React 18, TypeScript, Vite 5, Recharts; pytest, Vitest and Playwright. Use `npx --yes pnpm@10.30.3` if pnpm is absent. Resolve and lock dependencies; do not upgrade unrelated apps.

**Approved spec:** `docs/superpowers/specs/2026-09-04-hyperliquid-report-ui-design.md` (including Portfolio analysis).

## Implementation record — 2026-09-05

The read-only prototype is implemented. The detailed checklist below is retained
as the original plan; unchecked compound test items are not a claim of exhaustive
coverage. Work was kept inline and consolidated into a single UI/API commit
instead of the intermediate commit sequence, following the user's cost preference.

- API, generated TypeScript contracts, four dashboard tabs and portfolio analysis are present.
- Added safe artifact boundaries, bounded queries, explicit N/A policies and UTC date filtering.
- Correctness review identified four issues; all were fixed and the bounded re-review found none remaining.
- Verification: 444 package/app tests passed (8 opt-in tests deselected), 4 frontend unit tests passed, TypeScript/Vite build and generated-contract check passed, and 4 real-API browser tests passed.
- Desktop and mobile screenshots were visually inspected; no document-width overflow at 390px.
- Full-root collection remains blocked by four pre-existing missing `arblab.perps` imports. The new app does not change those modules.
- The existing user demo is available through the local loopback server on port 8010; artifacts are unchanged.
- Deliberate prototype limits: synthetic demo only, no live paper collector, no authenticated deployment, no strategy-performance claims, and no closed-trade metrics inferred from fills.
- Non-blocking toolchain warnings: legacy dependency deprecations and a roughly 525 KB uncompressed chart vendor bundle.
- Startup, demo generation and checks are documented in `docs/hyperliquid-explorer.md`.

The feature branch/worktree is retained. No merge, push, deployment or deletion
of user reports is part of this handoff.

## File responsibilities

Under `apps/hyperliquid-explorer-api/`:

- `pyproject.toml`: workspace package `hyperliquid-explorer-api`, depends on arblab, FastAPI, Uvicorn and httpx for tests.
- `src/hyperliquid_explorer_api/models.py`: explicit Pydantic HTTP schemas; scenario identity, normalized run metadata, typed rows and metric metadata.
- `repository.py`: root containment, artifact allowlist, inventory and summary/config reads.
- `queries.py`: parameterized DuckDB filters, deterministic paging, minute-grid statistics and extrema-preserving chart sampling.
- `analytics.py`: display-ready metrics, availability and short-history handling; no browser calculations.
- `app.py`: GET routes, sanitized exceptions and optional static serving; `create_app(root=None, web_root=None)` allows isolated tests.
- `export_schema.py`: emit deterministic OpenAPI JSON to stdout for type generation.
- `tests/conftest.py`, `test_repository.py`, `test_api.py`, `test_analytics.py`: temporary report fixtures and endpoint/formula tests.

Under `apps/hyperliquid-explorer-web/`:

- `package.json`, `index.html`, `tsconfig.json`, `vite.config.ts`: existing workspace-compatible React app and loopback API proxy.
- `src/api.generated.ts`: generated, committed OpenAPI types (never handwritten).
- `src/api.ts`, `format.ts`: small typed client, null-aware formatting and scenario keys.
- `src/App.tsx`: run selection, four tabs and request-state orchestration.
- `src/Overview.tsx`: scenario/benchmark controls, headline cards, analytics groups and comparison table.
- `src/Charts.tsx`: Recharts equity/drawdown/exposure views; UTC tooltip and legend.
- `src/Records.tsx`: typed tables, filters and pagination for traders/cohorts/fills/funding.
- `src/DataView.tsx`: safe provenance/config display and artifact links.
- `src/styles.css`, `src/main.tsx`: responsive dashboard styling and entrypoint.
- `src/format.test.ts`, `tests/explorer.spec.ts`, `playwright.config.ts`: unit and real-backend browser checks.

Root changes: register Python app/test path in `pyproject.toml`; add named scripts in `package.json`; update both lockfiles, README and `docs/hyperliquid-explorer.md`. Do not replace existing default commands.

## Task 1: Dependencies and failing contract tests

- [x] Add app manifests and uv workspace registration; use existing React/Vite major versions, openapi-typescript 7, Vitest 2 and Recharts 2. Python dependency ranges: `fastapi>=0.115,<1`, `uvicorn>=0.30,<1`, `httpx>=0.27,<1`; lock resolved versions.
- [x] Synchronize uv workspace and pnpm lockfile. Request escalation for sandbox network failures, not external data access.
- [x] Add a real temporary fixture by invoking existing prototype fixture/pipeline/report APIs; no checked-in generated reports. Add malformed and empty report variants by changing fixture files only.
- [x] Write repository/API tests before implementation. Baseline failing assertion: `create_app(tmp_path)` exposes `GET /api/runs` and rejects `POST /api/runs` with 405.
- [ ] Run `uv run pytest apps/hyperliquid-explorer-api/tests -q` and record expected missing implementation failures.

## Task 2: Safe repository and typed API schemas

- [ ] Test real discovery, newest-first order, invalid matching directories listed unavailable, missing optional artifacts, canonical numeric strings, mode/synthetic labels and missing metadata fallbacks from equity. A missing reports root returns an empty inventory, not an exception.
- [ ] Test unknown runs, `..`, encoded separators, forbidden artifact names and escaping symlinks. Requests must not alter file hashes.
- [x] Implement root-resolved path containment plus strict basename pattern, then fixed artifact allowlist. Use a report-specific exception carrying safe public detail; never echo paths/tracebacks.
- [x] Implement Pydantic models for run summaries and scenario metrics, not arbitrary untyped dictionaries. Retain raw config/provenance through safe JSON values while stripping path/credential-like fields. Derive starting equity/period from scenario equity when the minimal synthetic config lacks them.
- [ ] Run focused tests, then commit the repository/model boundary.

## Task 3: Bounded query endpoints

- [ ] Test exact strategy/control identity and null cash latency, invalid scenario 404, malformed filters 422, page-size cap 200, case-insensitive wallet search, decision/scope/coin/reason filters and stable ordering.
- [x] Add parameterized DuckDB scans with `hive_partitioning=false`, `WHERE` values bound, and whitelisted columns. Paging does not materialize full ledgers. Empty minimal-schema prototype Parquet files produce typed empty responses; absent files produce explicit unavailable responses.
- [x] Equity query computes running peak and drawdown in SQL over the full scenario before sampling. Preserve first/last and per-bucket extrema of equity/drawdown/gross/net exposure with a hard 2,000-point cap (allocate enough buckets for the union of extrema). Return original count, time bounds and downsampling status. Do not compute risk statistics from sampled rows.
- [x] Expose health, inventory, detail, equity, traders, cohorts, fills, funding and allowlisted downloads. Return explicit control-funding unavailable status. Ensure malformed source data yields generic structured errors.
- [ ] Run all API tests and commit.

## Task 4: Portfolio analytics

- [x] Write golden tests for stored-summary mapping plus Python-derived Calmar, current drawdown, starting equity, net PnL, fees/funding dollars, full-grid mean/max leverage and benchmark total-return difference.
- [ ] Test missing/nonfinite values, no orders versus 0% fill, zero drawdown, zero variance, non-positive equity, duplicate/missing minutes, synthetic/under-one-day suppression, one-to-29-day warnings, sample count and unchanged raw artifacts.
- [x] Implement `Metric(value, unit, reason, source, start, end, samples)` and grouped `Analytics` response. Every finite existing statistic comes from stored summary; validate full minute grid before curve-derived statistics. Leverage means use all full-resolution rows via SQL aggregates. Reuse prototype formulas for verification, not duplicate return-risk calculations in TypeScript.
- [ ] Match benchmark period and starting capital presentation explicitly. A mismatched/unavailable benchmark yields a warning and unavailable comparison, not an implicit rescaled or truncated series.
- [x] Add `GET /runs/{id}/analytics` with optional benchmark scenario triple. Return server policy for annualized metrics and descriptions/tooltips.
- [ ] Run golden tests and prototype regression tests, then commit.

## Task 5: Generated types and frontend states

- [x] Export OpenAPI from `create_app` without reading data or requiring a running server. Generate `src/api.generated.ts` with openapi-typescript and add drift-check command using a temporary output and byte comparison.
- [x] Add failing Vitest tests for null/duration/currency/ratio formatting, stable scenario keys including cash, unavailable-last sorting and short-history display.
- [x] Add failing Playwright assertions for the app title, synthetic warning, four tabs and API-backed run selection before frontend implementation.
- [x] Implement typed fetch calls, abort/stale-response protection, run selection/refresh and tab state. Avoid stale selected-run content while a different run loads. Use semantic accessible controls and visible recoverable errors/empty states.
- [ ] Run unit tests and TypeScript checks.

## Task 6: Dashboard views

- [x] Build Overview with five headline cards, grouped expandable analytics, scenario/latency/benchmark selectors, selectable/sortable comparison columns and chart legends/tooltips. Comparison values use the same server availability policy as headline cards, not raw unsuppressed annualized values.
- [x] Build Traders with wallet/date/scope filters, selected/excluded state, score and exclusion reasons, wallet-copy control and cohort history.
- [x] Build Execution with strategy/control filters, asset/reason inputs, fills/funding pagination and native residual quantities/entries from the selected scenario.
- [x] Build Data with normalized source/config, evidence state, warnings and artifact downloads. Render report strings as text; never raw HTML.
- [x] Add compact dark responsive CSS, focus states and accessible explanatory N/A values. No fake live ticker, chart data, order buttons or metric calculations in the browser.
- [ ] Run build/unit tests, then commit frontend.

## Task 7: Real browser verification and startup

- [x] Add a documented demo-fixture generation command reusing existing Python APIs without network calls; generate into a caller-specified new directory. Browser tests use their own temporary root and API port, never delete existing report directories.
- [ ] Run Playwright against actual Python backend with generated synthetic artifacts: select run/scenario/latency, verify all tabs/metrics/N/A/fill row, filters/paging, and absence of console/page errors. Use route fault injection only for unavailable network/loading states; normal smoke uses real endpoints.
- [ ] Exercise empty root/malformed run states and desktop/mobile widths. Take a screenshot, inspect it, fix material layout/contrast/overflow defects and rerun checks.
- [x] Serve compiled frontend from FastAPI with loopback binding; unknown `/api` remains JSON 404 and static routing cannot shadow GET API routes. Static resources cannot escape configured build directory. Missing build leaves API healthy.
- [ ] Add commands/docs for dev two-process setup, generated-type checks, Python tests, web tests, production build and local single-origin launch. Start the local app against the existing user demo report root and verify the URL actually responds.

## Task 8: Final review and handoff

- [x] Use one bounded correctness review per requesting-code-review skill. Fix material issues with regression tests.
- [ ] Run `uv run pytest apps/hyperliquid-explorer-api/tests packages/arblab/tests/hyperliquid_copy -q`, package/app regression tests, frontend build/unit tests, generated-type drift and browser smoke. Full-root existing `arblab.perps` errors remain separately reported.
- [ ] Check git diff and stage only feature files; preserve user reports and existing apps. Commit, retain the feature worktree and leave the verified local server running.
- [ ] Give the user the actual browser URL, short feature summary and explicit synthetic/read-only limitations. No further design approval gate is needed: the user approved the full spec on 2026-09-05.
