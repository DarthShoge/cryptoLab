# Hyperliquid read-only explorer

A React/TypeScript dashboard backed by Python/FastAPI. It reads existing local
`hyperliquid_copy_report_v1` artifacts; it does not download market data, submit
jobs, connect wallets, or place orders. Keep it bound to loopback: there is no
authentication, and artifact downloads expose the contents of the selected report.

## Start the prototype

Run from this checkout's root (the feature worktree, if using one):

```bash
uv sync --all-packages --all-groups --locked
pnpm install --frozen-lockfile
pnpm build:hyperliquid
HYPERLIQUID_REPORTS_ROOT="$PWD/reports" \
HYPERLIQUID_WEB_ROOT="$PWD/apps/hyperliquid-explorer-web/dist" \
uv run --package hyperliquid-explorer-api uvicorn hyperliquid_explorer_api.app:app \
  --host 127.0.0.1 --port 8010 --ws none
```

Open **http://127.0.0.1:8010**. Set `HYPERLIQUID_REPORTS_ROOT` to the directory
containing your `hyperliquid_trader_ensemble_*` folders, not an individual run.
The API defaults to the workspace reports directory. `--ws none` is intentional:
this application is HTTP-only and does not need the workspace's legacy WebSocket
dependencies. No API keys are required. If pnpm is absent, replace `pnpm` with
`npx --yes pnpm@10.30.3`.

To generate a **fabricated, three-minute BTC integration demo** without network
access, choose a new output directory (existing directories are never replaced):

```bash
uv run --all-packages python tools/generate_hyperliquid_explorer_demo.py \
  --output reports/hyperliquid_trader_ensemble_local_demo
```

This developer utility reuses the existing strategy test fixture, simulator and
report writer. Its wallets, prices and results are synthetic, not a strategy
backtest on observed market data. Real report creation is documented in
[the offline prototype guide](hyperliquid-trader-ensemble.md).

## What is in the UI

- **Overview:** scenario, execution delay and benchmark selectors; final equity,
  return, Sharpe, Sortino and maximum drawdown cards; equity, underwater and
  gross/net exposure charts; sortable scenario comparison with optional columns.
- **Portfolio analysis:** expand the panel below the headline cards for Calmar,
  annualized return/volatility, drawdown duration and recovery state, dollar
  fees/funding, turnover, fill ratio, stale requests, leverage and residual
  exposure. Tooltips explain definitions; the panel shows the period and sample
  count alongside selected and benchmark values.
- **Traders:** dated wallet rankings, score components, exclusions, cohort
  membership, wallet search, decision-date and score-scope filters.
- **Execution:** simulated fills and funding, asset/outcome filters, pagination
  and residual native positions. Empty ledgers and missing ledgers are distinct.
- **Data:** configuration, provenance, reconciliation evidence, warnings and
  allowlisted raw-artifact downloads. JSON metadata shown in the UI omits
  path/credential-like fields; raw downloads intentionally remain unmodified.

## Metric and data policy

The backend owns all portfolio calculations. Stored summary metrics are reused;
descriptive additions (Calmar, current drawdown, mean leverage, dollar costs and
benchmark return difference) are computed in Python/SQL. They are not recomputed
from chart samples. Sharpe/Sortino/CAGR/annualized volatility/Calmar are unavailable
for synthetic data or histories shorter than one day. Real histories under 30
days receive a short-sample warning. N/A is never silently converted to zero.

Returns use net simple UTC minute equity observations, zero risk-free/target
return, sample standard deviation and 365 × 1440 annualization. Sortino uses
downside RMS over all minute returns. Fee and funding drag are relative to initial
capital; negative funding cost is a credit. Benchmark excess return is a
percentage-point difference, not alpha. Benchmark windows must match; dollar
equity charts are not rebased, so compare starting capital in the analysis panel.

Missing/nonfinite required curve values, non-positive equity or missing/duplicate
minutes prevent curve-derived analytics and chart rendering. Stored summaries
remain identified as stored evidence, not independently recomputed verification.
Chart responses are capped at 2,000 points and preserve endpoints and bucket
extrema. Ledgers default to 50 rows per API request (25 in the UI), capped at 200.
Malformed reports are listed as unavailable without hiding valid reports.

The demo cannot establish trader skill or profitability. Funding uses the
prototype's historical-mid approximation. Win rate and profit factor are not
fabricated from fill rows: fills are not completed round trips.

## Development and checks

For hot reload, start the API command above, then run `pnpm dev:hyperliquid` in
another terminal. Open http://127.0.0.1:5174; Vite proxies `/api` to port 8010.
The API can run without a frontend build. Its schema is at `/api/openapi.json`;
interactive GET-route documentation is at `/api/docs`.

```bash
uv run --all-packages pytest apps/hyperliquid-explorer-api/tests packages/arblab/tests/hyperliquid_copy -q
pnpm test:hyperliquid
pnpm build:hyperliquid
pnpm types:hyperliquid
pnpm --filter @cryptolab/hyperliquid-explorer exec playwright install chromium
pnpm test:hyperliquid-browser
```

Browser tests build on the compiled frontend (run the build first), generate
their own temporary synthetic report and start a real API on port 8011. They
check all tabs, metric availability, filtering, loading/error recovery, empty
states, and desktop/mobile layouts. Screenshots and failure traces live under
`apps/hyperliquid-explorer-web/test-results/` (ignored). Temporary fixture roots
are kept in the OS temporary directory, never in your reports folder.

After changing Pydantic contracts, regenerate and commit TypeScript types:

```bash
pnpm --filter @cryptolab/hyperliquid-explorer types
pnpm types:hyperliquid
```

`types` exports OpenAPI directly from Python without a running server. The check
fails if the committed generated file differs. Do not edit it by hand. Existing
workspace default dev/build scripts continue to target the strategy system card.
