# Hyperliquid trader ensemble: offline reference implementation

## Status and safety boundary

This is an offline research prototype, not a trading bot. No exchange signing key
is needed, and the runner contains no order-submission API. No paid archive data
has been downloaded and no real-market performance result has been produced.

The checked-in smoke configuration is integration-only. It cannot win development
selection or enter validation/locked test. The research configuration deliberately
starts with unresolved leader fee semantics and refuses to run without matching,
dated reconciliation evidence.

The original implementation plan is **not fully complete**. In particular, this
reference implementation is not yet qualified for an all-wallet, 90-day archive.
The next sections distinguish implemented code from unverified operational work.

## Reused dependencies

- `hyperliquid-data==0.1.0`: public cost estimate and market-data CLI; canonical L2
  and funding formats. Its reduced `FillRow` omits `startPosition`, fees and other
  fields needed here, so only the full-fidelity fill parser/downloader is custom.
- `hyperliquid-python-sdk==0.24.0`: pinned for later read-only source reconciliation
  and prospective collection; not used to submit orders.
- `duckdb==1.5.5`: local Parquet filtering. PyArrow, boto3 and LZ4 are supplied by
  the pinned data package dependency graph.

References: [hyperliquid-data](https://github.com/bond-labs-dev/hyperliquid-data),
[official Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk).
The adapters are tested against the installed package's types and public outputs,
not imported private helper functions.

## Implemented behavior

The pipeline reconstructs signed positions from full fills, excludes left-censored
episodes from episode statistics, and ranks all observed candidate addresses with
strictly prior-window data. It does not use today's leaderboard to select a past
universe. Daily scores use five equally weighted percentile metrics. Default
eligibility is 30 active days, 20 completed episodes, $100,000 closing notional and
15-minute median episode duration over a 90-day window.

Three signals run at 1, 5, 15 and 60 second follower latency: equal direction,
score-weighted direction, and score-weighted trimmed conviction. The same signal
artifact feeds all four latency cases. Unknown wallet state remains in the cohort
weight denominator. Wallets require five known members and 60% known weight.

Execution walks available asks/bids and cancels any residual depth. The first book
at or after the executable timestamp is usable only within two seconds. Marks
must be at-or-before the valuation timestamp and at most 60 seconds old. Funding
settles before fills at an equal timestamp. Follower fees use the declared 4.5-bps
model assumption, not a claim about the user's actual fee tier.

Controls are cash plus four latency cases each of BTC perpetual buy-and-hold,
equal-universe perpetual buy-and-hold, all-eligible direction consensus, and a
deterministically score-shuffled selected cohort. Buy-and-hold controls open once;
partial entry depth is not synthesized or retried.

### Corrections and clarifications to the initial plan

1. Conviction is `position_qty * current_mid / trailing_notional_scale`.
   Dividing native quantity directly by a dollar scale mixes incompatible units.
2. Perpetual accounting is `equity = collateral + unrealized_pnl`, with realized
   PnL settled to collateral on closes. Opening a perpetual is not a spot purchase.
3. The helper package uses `btc_perp/date=YYYYMMDD` market directories and
   `BTC-PERP` funding instruments. Adapter tests lock down both conventions.
4. Funding uses settlement mid as an explicit approximation to the venue's
   settlement reference. Reports retain that warning; exact settlement accounting
   requires the correct historical reference-price series.
5. Exposure limits constrain requested targets at decision time. Price moves,
   execution delay and fees can produce marked leverage drift between decisions.
   Maximum observed leverage is reported; this is not a liquidation engine or a
   guarantee that instantaneous marked leverage always stays below 1x.
6. The smoke quality checker may diagnose up to 5% missing funding, but the
   simulator refuses to calculate through any missing settlement. There is no
   zero-rate imputation.

## Offline checks

From this worktree after dependency synchronization:

```bash
uv sync --locked --all-packages --all-groups
uv run pytest packages/arblab/tests/hyperliquid_copy -q
uv run python tools/download_hyperliquid_copy_data.py --help
uv run python tools/run_hyperliquid_trader_ensemble.py --help
```

Tests include synthetic archive formats, source identity, missing-state handling,
strict ranking cutoffs, fee semantics, flips, funding order, book walking,
canonical hashing, trial gates and an end-to-end 12-strategy/17-control report.
They use no remote credentials or paid data. Reports from pytest stay in pytest's
temporary directories.

## Costs and dry runs

Fill/L2 archives are requester-pays S3. AWS credentials resolve through boto3's
standard chain; never place credentials in config files, commands or reports.
Even the cost estimator makes billable LIST requests. Dry runs make no network
calls and write no cache files.

```bash
uv run python tools/download_hyperliquid_copy_data.py pull-market \
  --start 2026-08-01 --end 2026-08-07 \
  --cache-root .hyperliquid_cache --dry-run

uv run python tools/download_hyperliquid_copy_data.py estimate \
  --dataset fills --start 2026-08-01 --end 2026-08-07 --dry-run
```

After the operator explicitly approves estimator requests, omit `--dry-run` for
both `--dataset fills` and `--dataset l2book`. Inspect bytes, request counts,
sampled dates and pricing assumptions; the estimator is not a bill guarantee.
Only after separately approving the data bill should an operator run:

```bash
uv run python tools/download_hyperliquid_copy_data.py pull-fills \
  --start 2026-08-01 --end 2026-08-07 \
  --cache-root .hyperliquid_cache --accept-estimated-cost

uv run python tools/download_hyperliquid_copy_data.py pull-market \
  --start 2026-08-01 --end 2026-08-07 \
  --cache-root .hyperliquid_cache --accept-estimated-cost
```

Download dates are inclusive. Fill downloads verify all 24 objects before writing
an atomic Parquet partition and manifest, retain configured-asset wallet fills,
and reject parse failures. Reusing a partition requires matching coins/checksum.
L2 acquisition delegates to the upstream CLI, which stores its configured top 20
levels. Funding acquisition starts at the requested time and may fetch later rows;
the local reader bounds the consumed interval. No candle fallback is implemented.

## Historical run

The run CLI uses an **exclusive** dataset end date. Thus the seven-day pull above
maps to `--start 2026-08-01 --end 2026-08-08`. The smoke config reserves two days
of warmup and uses the remaining five for integration testing:

```bash
uv run python tools/run_hyperliquid_trader_ensemble.py run \
  --config configs/hyperliquid_trader_ensemble_smoke.json \
  --cache-root .hyperliquid_cache --study-id integration-001 \
  --split development --start 2026-08-01 --end 2026-08-08 --dry-run
```

For research, the same date range always describes the full study dataset.
After warmup, complete UTC days split 60% development, 20% validation and the
remainder locked test, with at least five simulation days. Selection uses net
five-second trimmed-conviction Sharpe, then return, then config hash. Validation
must have positive return and Sharpe and drawdown below 30%. The append-only
registry prevents repeated completed validation/locked tests; it is an operational
process guard, not tamper-proof security or a complete research preregistration
system. A failed trial requires a fresh study identifier; it is not overwritten.

`--unlock-test` must match the config hash and is rejected outside locked test.
`select-development` never runs market data or changes a winning configuration.
Registry defaults to
`.hyperliquid_cache/hyperliquid_copy/trials/trial_registry.jsonl` and supports an
explicit `--registry` override.

Each successful run creates a unique `reports/hyperliquid_trader_ensemble_*`
directory. It contains config, manifests, trial, reconciliation, scores, cohort
history, signals, simulated fills, equity, funding ledger, controls, summary and
report. Existing directories are never overwritten. `summary.json` preserves
native residual positions, undefined-risk-metric warnings, artifact hashes and
the shared signal checksum. A failed data gate does not produce a success report.

## Resource and evidence limits: do not skip these

The fill downloader streams batches, but the reference runner currently loads its
bounded normalized fill/book window into memory. `--max-rows` defaults to one
million total Parquet rows and fails before row loading; it never silently samples
wallets. Seven days of all-wallet fills and L2 may substantially exceed this.
Raising that ceiling is **not** a production-scale solution. Profile partition
sizes and implement day-partitioned/indexed replay and bounded market lookups
before approving a large research pull. The upstream L2 CLI also assembles a
coin/day partition in memory. Memory and download costs must be sized together.

Reconciliation functions compare supplied provider fill identities, terminal
positions and at least three simple two-fill episodes to distinguish gross from
fee-inclusive PnL. Their tests use synthetic independent inputs. There is not yet
an automated Info/Dwellir evidence-capture CLI. An arbitrary past clearinghouse
snapshot cannot be substituted with today's Info response. Historical provider
coverage, exact snapshot cutoffs and fee semantics must be verified with dated
evidence before any research-eligible use. The checked-in research config remains
intentionally blocked until that work is done.

### Remaining implementation-plan work

- Qualify bounded on-disk execution/ranking against realistic archive volumes.
- Add the read-only reconciliation capture command, provider-scope checks and
  reproducible raw evidence fixtures from real data.
- Complete the full report diagnostics (cohort turnover/concentration statistics,
  explicit largest-gain/loss forensics and latency-cost attribution). Current
  reports provide the underlying artifacts but not every planned analysis.
- Extend the registry to the complete declared-hypothesis/trial-count protocol.
- Run the approved seven-day remote smoke and inspect source gaps/fees/positions.
- Only then implement the prospective recorder, durable journal and live paper
  trading stages, followed by the planned soak test. No real trading is in scope.

These are explicit unfinished items, not claims implied by passing synthetic tests.
