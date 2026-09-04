# CryptoLab Polyglot Monorepo Design

## Purpose

Reorganize CryptoLab into a clear polyglot monorepo while preserving its four independently runnable frontends, shared Python domain code, historical reports, research, and notebooks. The migration must preserve current behavior and the user's existing uncommitted work.

## Goals

- Give every runnable application an explicit home and dependency boundary.
- Package shared Python code using a conventional `src` layout.
- Use `uv` for Python environments and workspace dependencies.
- Use `pnpm` workspaces for TypeScript dependencies.
- Provide consistent root commands for setup, tests, builds, and local development.
- Make runtime file paths independent of the caller's current working directory.
- Separate the old exchange-arbitrage code from the active Kamino and backtesting system.

## Non-goals

- Combining the four frontends into one application.
- Rewriting Streamlit applications in React or changing their user interfaces.
- Refactoring strategy behavior, backtest mechanics, or on-chain calculations.
- Reformatting or relocating the historical `reports/`, `research/`, or `notebooks/` trees merely for symmetry.
- Introducing remote build caching, containers, deployment infrastructure, or a publishing workflow.

## Tooling Decision

Use a deliberately lightweight combination:

- `uv` manages the Python workspace, lockfile, application dependencies, and development dependencies.
- `pnpm` manages the TypeScript workspace and lockfile.
- `just` exposes memorable repository-level commands and delegates to `uv` and `pnpm`.

This is preferred over Moonrepo because the repository does not yet need a separate task graph and cache service. It is preferred over Turborepo because most of the repository is Python and Turborepo would orchestrate that side only through custom commands. The workspace can adopt a larger task runner later without changing package boundaries.

## Target Structure

```text
cryptoLab/
├── apps/
│   ├── kamino-simulator/
│   │   ├── pyproject.toml
│   │   ├── src/kamino_simulator/app.py
│   │   └── tests/
│   ├── strategy-backtester/
│   │   ├── pyproject.toml
│   │   ├── src/strategy_backtester/app.py
│   │   └── tests/
│   ├── report-explorer/
│   │   ├── pyproject.toml
│   │   ├── src/report_explorer/app.py
│   │   └── tests/
│   └── strategy-system-card/
│       ├── package.json
│       ├── src/
│       ├── index.html
│       ├── vite.config.ts
│       └── tsconfig*.json
├── packages/
│   └── arblab/
│       ├── pyproject.toml
│       ├── src/arblab/
│       └── tests/
├── data/
│   ├── fixtures/
│   └── generated/
├── tools/
├── tests/integration/
├── legacy/exchange-arbitrage/
├── docs/
├── notebooks/
├── reports/
├── research/
├── pyproject.toml
├── uv.lock
├── pnpm-workspace.yaml
├── package.json
├── pnpm-lock.yaml
├── justfile
└── .env.example
```

Python import packages use underscores even though their containing application directories use hyphens. Each Streamlit entry point is a real importable module, which permits import smoke tests without depending on ambiguous root scripts.

## Component Boundaries

### Shared Python package

`packages/arblab` owns reusable domain and infrastructure code:

- Kamino account, risk, recovery, and on-chain integration
- Backtest engine, models, data access, metrics, optimization, and report helpers
- Reusable strategy implementations
- Shared repository-path and configuration helpers

It must not import a Streamlit application. Applications depend inward on `arblab`; the library never depends outward on applications.

### Python frontends

The existing `kamino_app.py`, `backtest_app.py`, and `strategy_report_app.py` become the `kamino-simulator`, `strategy-backtester`, and `report-explorer` applications respectively. Each application declares only its direct runtime dependencies and its workspace dependency on `arblab`. UI-specific helpers and tests belong with the application; reusable calculation or report transformation logic remains in `arblab`.

### TypeScript frontend

The current root Vite project becomes `apps/strategy-system-card`. It owns its Vite and TypeScript configuration, browser-facing source, assets, tests, and package scripts. The root Node package is private and contains only workspace-wide scripts or metadata; it is not itself an application.

The existing Python generator remains in `tools/` because it is a repository maintenance task. Its output path is explicit and targets the TypeScript application's data directory. Its Python dependencies are declared in a root `tools` dependency group in the workspace `pyproject.toml`, and it runs through that group rather than maintaining a separate ad hoc environment.

### Legacy code

The old `main.py` exchange-arbitrage runner and its tightly coupled `arblab/account.py`, `arblab/arb_lab.py`, and `arblab/utils.py` modules move to `legacy/exchange-arbitrage`. Repository usage confirms these modules form a separate subsystem. Legacy code is excluded from normal imports, tests, and workspace commands. A README records why it is isolated and that it may execute exchange operations; the migration does not repair or run it.

### Data and durable artifacts

Stable, checked-in inputs such as the Kamino IDL and sample payload move to `data/fixtures`. Generated intermediate inputs may use `data/generated` where they are not naturally owned by an application. Long-lived `reports`, `research`, and `notebooks` remain at the repository root so existing historical content does not undergo a noisy migration.

## Dependency Management

The root `pyproject.toml` defines the `uv` workspace, shared development dependency groups, and common tool configuration. The `arblab` package and three Python apps are workspace members. Each member owns its runtime dependency declarations, and application references to `arblab` resolve as workspace dependencies.

The root `pnpm-workspace.yaml` includes `apps/*`. Only Node-bearing packages participate. The checked-in `pnpm-lock.yaml` replaces `package-lock.json` after a verified install and build.

Requirements files and `setup.py` are removed only after equivalent metadata exists in `pyproject.toml` and clean installation is verified. The root README becomes the canonical command reference.

## Runtime Paths and Configuration

Applications must not rely on being launched from the repository root. A shared `arblab` path helper locates stable repository resources or accepts explicitly supplied roots. Report discovery, price-cache access, IDL loading, samples, and generated output paths use that facility instead of unresolved relative paths.

The repository retains one root `.env` convention. `.env.example` documents supported non-secret variables. Required configuration is validated at its point of use and produces a readable error. Importing an application must not initiate network or exchange activity.

## Commands

The root `justfile` is the human-facing command layer. At minimum it provides:

```text
just install
just test
just typecheck
just build
just dev-kamino
just dev-backtester
just dev-reports
just dev-system-card
```

- `install` synchronizes the Python workspace and installs the pnpm workspace.
- `test` runs the default offline Python and TypeScript test suites. Separate recipes document and run opt-in browser, RPC/on-chain, and market-data test categories.
- `typecheck` runs configured Python static checks, if retained or added during migration, plus TypeScript type checking.
- `build` builds the TypeScript application and any meaningful package build checks.
- Each `dev-*` command launches exactly one frontend.

Direct `uv` and `pnpm` commands remain documented so contributors are not forced to use `just`.

## Test Ownership

Tests move with the behavior they verify:

- `packages/arblab/tests` contains domain, strategy, backtest-engine, data, and reusable report-helper tests.
- Each Python app's `tests` directory contains tests of its entry point or UI-owned helpers.
- `apps/strategy-system-card/tests` contains its TypeScript and browser tests.
- `tests/integration` is reserved for tests that genuinely cross package or application boundaries.

Default tests must remain offline and deterministic. Existing functional, scenario, backtest, or on-chain markers remain available. Dedicated `just test-browser`, `just test-onchain`, and `just test-market-data` recipes (limited to categories actually present after test classification) provide explicit entry points for tests needing browsers, RPC services, or market data. Test discovery is configured centrally and verified after every move.

## Migration Sequence

1. Record the baseline status and test/build results without changing user-owned work.
2. Add root `uv`, `pnpm`, and `just` workspace configuration.
3. Move `arblab` into `packages/arblab/src/arblab`, add package metadata, and update test discovery/imports.
4. Introduce shared path/configuration handling and move stable fixtures.
5. Move each Streamlit frontend independently, updating its imports, owned tests, and launch command before proceeding to the next.
6. Move the React/Vite frontend and update the generator's target path.
7. Isolate the exchange-arbitrage runner and its coupled modules under `legacy`.
8. Update documentation and ignore rules, then remove superseded root configuration only after replacements pass verification.
9. Search for stale old paths and run the full verification suite.

Moves should preserve Git history where practical. Existing compatibility wrappers are not retained by default because they would leave two apparent entry-point conventions. If an external consumer is discovered during migration, a narrow deprecated wrapper may be added and documented instead of silently breaking it.

## Error Handling and Migration Safety

- Capture `git status` before migration and avoid overwriting or reformatting unrelated modified files.
- Treat existing tracked and untracked reports and research as user data.
- Never execute the legacy arbitrage runner during validation.
- Make path/configuration failures identify the missing resource or variable and expected resolution.
- Stop a migration step if tests reveal a behavioral change; fix that boundary before moving the next application.
- Do not delete old lockfiles or metadata until their replacements install and verify successfully.

## Acceptance Criteria

The reorganization is complete when:

- All four frontends are visibly separate and independently runnable through documented commands.
- Python dependencies install from a committed `uv.lock` and TypeScript dependencies install from a committed `pnpm-lock.yaml`.
- Python imports resolve from the new package layout without relying on the repository root being on `sys.path` accidentally.
- The Python test suite passes from the repository root.
- The React application type-checks and builds from the repository root.
- Each Streamlit entry point passes an import or startup smoke check without unintended network calls.
- Runtime resources resolve correctly when commands are launched from outside the repository root.
- No active code or documentation retains unintended references to removed root entry points or old fixture paths.
- Historical reports, research, notebooks, and all pre-existing uncommitted changes remain intact.
- The legacy arbitrage code is isolated and excluded from normal workspace operations.
