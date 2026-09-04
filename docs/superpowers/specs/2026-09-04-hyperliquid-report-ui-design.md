# Hyperliquid read-only report explorer

## Approved direction

The user approved a read-only first version: React, TypeScript and Vite frontend,
Python FastAPI backend, displaying the existing Hyperliquid prototype reports.
Preserve the working strategy engine and Markdown report. This spec makes that
direction concrete; it does not authorize paid data, live collection or trading.

## Architecture and boundaries

Create two focused workspace applications:

- `apps/hyperliquid-explorer-web`: React/TypeScript/Vite browser application.
- `apps/hyperliquid-explorer-api`: installable Python FastAPI application.

The API reads existing JSON and Parquet artifacts under one operator-configured
`HYPERLIQUID_REPORTS_ROOT`. The web app never reads filesystem paths or runs Python.
Use Pydantic response models and generate committed TypeScript API types from the
API's OpenAPI schema. React consumes a small typed fetch client; no duplicated
business calculations, client-side strategy engine, database, job queue or SSR.

Use the existing pnpm and uv workspaces without replacing their current default
apps. Add explicitly named explorer development/build/test commands. Development
uses a Vite `/api` proxy; both processes bind to loopback by default. The API can
serve the built frontend for a single-origin local launch. Missing frontend build
must not break API routes. Local operation only: authentication and deployment on
a public interface are outside this version.

## Data contract

Discover only directories with the `hyperliquid_trader_ensemble_` prefix and a
`summary.json` carrying schema `hyperliquid_copy_report_v1`. Identify runs by their
directory basename, not caller-supplied paths. List runs newest directory-name
first; offer a refresh action and retain selection by run id. No background live
feed is implied.

Expose GET endpoints beneath `/api`:

- `/health`: availability and API schema version, without local paths/secrets.
- `/runs`: run id, title, mode/synthetic label, period, starting capital, scenario
  count and availability warnings. Invalid matching reports appear as unavailable
  entries rather than silently disappearing or breaking the entire index.
- `/runs/{id}`: normalized config, manifest provenance, report warnings, all
  strategy/control summaries and artifact availability.
- `/runs/{id}/equity`: scenario type/name/latency filters; timestamped equity,
  collateral, unrealized PnL, gross/net exposure and derived drawdown.
- `/runs/{id}/traders`: daily scores and selection/exclusion information, filtered
  by decision date, scope/asset and wallet text, with deterministic pagination.
- `/runs/{id}/cohorts`: daily membership/cutoff history with pagination.
- `/runs/{id}/fills`: execution rows filtered by strategy/control, latency, coin
  and reason, with deterministic pagination.
- `/runs/{id}/funding`: strategy funding rows with scenario/asset filters and
  pagination. Control funding is explicitly unavailable in the current artifacts.
- `/runs/{id}/artifacts/{name}`: download only the fixed set of existing report
  artifacts; expose the original Markdown report here as well.

Scenario identity is the tuple `(scenario_type, name, latency_seconds)`. Cash has
null latency, never a fabricated zero-second value. Global trader scope is null,
not a missing coin. Canonical JSON config/manifest numeric strings are converted
only through declared model fields; do not coerce arbitrary strings or wallet ids.
API datetimes are UTC ISO-8601. Non-finite values are rejected or represented as
null with a warning, never emitted as invalid JSON.

Paginated endpoints default to 50 rows and cap page size at 200. Equity requests
cap display points at 2,000, with ordered first/last and bucket extrema retained.
Compute authoritative drawdown from the full sorted equity series before display
downsampling. Return total row count and a downsampling notice. Use bounded
DuckDB Parquet queries and avoid sending whole fill ledgers to the browser.

## Interface

A compact research dashboard with a neutral dark palette, restrained colour for
positive/negative values, readable tables and responsive layout. Do not imitate
exchange order-entry screens or add decorative market tickers.

Persistent header: report selector, date range, refresh, and a prominent
`SYNTHETIC DEMO`, `SMOKE ONLY`, or `RESEARCH REPORT` badge. Display unresolved fee,
funding approximation and data-quality warnings before performance metrics.

Four tabs:

1. **Overview:** strategy and latency selectors, benchmark selector, final equity,
   total return, max drawdown, fees and residual exposure; interactive equity and
   drawdown comparisons; gross/net exposure chart; sortable scenario comparison
   table. The synthetic fixture is flat-price and its signals overlap—do not
   fabricate different curves. Annualized ratios for synthetic/under-one-day runs
   display an explanatory unavailable label, not misleading precision. Raw
   artifacts remain downloadable unchanged.
2. **Traders:** decision-date/scope filters, ranked wallets, scores, selected status
   and exclusion reasons; membership/cutoff history. Wallet copy action and
   monospace truncated display with the complete address accessible. Do not imply
   that a wallet is a verified person or that its historical score is future skill.
3. **Execution:** simulated-fill table with requested/filled quantities, price,
   request/book times, fee and cancellation reason; funding table; native residual
   quantity/average-entry breakdown. Label these simulated, never actual trades.
4. **Data:** source/configuration, evidence status, warnings, available artifacts
   and downloads, including the original Markdown report.

Charts have legends, UTC time axes, currency labels and hover details. Provide
accessible table equivalents for essential metrics. Keyboard focus, associated
form labels, contrast, loading skeletons, empty tables, unavailable artifacts and
recoverable API errors are part of the first version, not follow-up work.

## Read-only and filesystem safety

No POST/PUT/PATCH/DELETE endpoints, subprocess launching, wallet connections or
remote market calls. Resolve both run and artifact paths and reject traversal,
encoded separators and symlinks escaping the configured root. Downloads use an
allowlist; report-controlled strings are escaped, never inserted as HTML. Return
404 for unknown runs/artifacts, 422 for invalid filters and a structured generic
error for malformed data. Do not return tracebacks or absolute filesystem paths.
The UI does not compute or alter strategy scores, signals or execution results.

## Verification and demonstration

Python tests use temporary report fixtures and FastAPI TestClient. Cover discovery,
numeric/date normalization, exact scenario filters including cash/null latency,
pagination, full-series drawdown, bounded chart output, missing/empty artifacts,
malformed reports, traversal and symlink containment. Assert unsupported mutation
methods fail and report files remain unchanged after requests.

Frontend checks: generated-type drift check, strict TypeScript build, unit tests
for filters/formatters, and Playwright browser smoke against the actual API. The
browser test selects the synthetic run, changes scenario/latency, visits all tabs,
checks the warning and a fill row, and verifies loading/error/empty states without
browser console errors. Check desktop and narrow-screen layouts.

The initial demo uses the already generated report at
`/home/lshoge/code/cryptoLab/reports/hyperliquid_trader_ensemble_demo_20260904T222830153668Z`.
Tests generate their own equivalent fixture; do not commit generated market/report
artifacts or depend on that absolute path in application code. Document exact
startup commands and leave the local demo reachable for the user after testing.

## Acceptance and exclusions

Done means the user can open a local browser URL and explore the current synthetic
report through Python-backed endpoints, with verified charts/tables, useful error
states, generated types and tested read-only boundaries. Existing prototype tests
must remain passing; existing unrelated `arblab.perps` collection failures are
reported separately.

Not included: backtest launching, parameter editing, paid downloads, paper/live
trading, new strategy calculations, multi-user auth, cloud hosting, or claims of
profitability. These require separate approval and must not leak into UI controls.
