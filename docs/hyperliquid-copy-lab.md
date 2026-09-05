# Hyperliquid copy-strategy lab

A self-contained React/TypeScript frontend and Python/FastAPI backend for one
question: which historical trader-selection and copying rules are worth testing?
The application runs local development simulations. It never connects a wallet,
places exchange orders, or downloads data. Other strategy apps remain independent.

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
