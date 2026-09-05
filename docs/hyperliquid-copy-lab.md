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
