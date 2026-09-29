# Weekly strategy diagnostics delivery

## Access

In the existing Hyperliquid copy-strategy lab, open **Saved backtests**, select
the completed annual weekly experiment `b32e5f1ed7db44008a927c6d965e7c3f`, then
choose **Diagnostics**. The new read-only endpoint is
`GET /api/lab/experiments/{id}/diagnostics`. The frontend build has been updated;
an already-running API process needs to load the new route before serving it.
No existing API/coordinator/worker was restarted by this change.

Direct review link (also survives refresh):
`http://127.0.0.1:8010/?experiment=b32e5f1ed7db44008a927c6d965e7c3f&tab=diagnostics`.
This opens the saved result without mounting the builder or issuing its POST
preflight validation, so it works on the read-only review server. The root page
still opens the builder; new runs and edits remain disabled on that server.

The tab includes the frozen system card, BTC/cash growth comparison, full-grid
drawdowns, exposure chart, monthly returns, weekly histogram and accessible data
tables, stored risk metrics, complete-week statistics, block-bootstrap mean
intervals, accounting and execution checks, worst-week exposure/context, and
dated cohort composition linking to the existing trader drilldown.

## Evidence and interpretation

- Source run: September 1, 2025 to September 1, 2026, $10,000 initial equity.
- Final equity: $8,368.01; return -16.32%; max drawdown 21.33%.
- BTC perpetual control return: -32.47%; cash control: 0%.
- 52 complete UTC Monday weeks and one partial day; 21 positive weeks (40.38%).
- Worst week: -5.59%; mean weekly return: -0.3540%.
- 95% four-week circular block-bootstrap interval for mean weekly return:
  [-0.7353%, -0.0537%]. Mean weekly excess-vs-BTC interval:
  [-1.4724%, +1.9654%]. Descriptive intervals, not forward return forecasts;
  they do not incorporate model risk or selection/multiple testing.
- Fees $60.45; net funding cost $45.68. All five internal arithmetic checks pass.
- Original `research_eligible=false` and `smoke_only_unreconciled` remain unchanged.
  Proxy marks/funding, limited account state and unequal asset cohort start dates
  prevent presenting this as independently validated live performance.

## Verification

42 focused backend tests passed (statistics, API, artifact export, existing
analytics and saved-backtest API). 14 frontend unit tests passed. Five Chromium
tests passed: diagnostics/navigation/mobile, loading/error handling, actual annual
payload rendering, existing save/clone/compare/drilldown, and mobile builder.
TypeScript checking and production build passed. Existing chart bundle-size and
Starlette/AnyIO deprecation warnings remain; they do not fail the checks.

An independent DuckDB calculation matched final equity, return, max drawdown,
complete-week count, mean, win rate and worst week. Seven original source artifact
SHA256 values still match the saved manifest. No original report was rewritten.
Spec and code-quality reviews passed after adding explicit malformed-artifact
handling and full-resolution worst-week exposure.

Generated evidence (not source files):

- `reports/hyperliquid_weekly_diagnostics_20260926_v2/diagnostics.json`
- `reports/hyperliquid_weekly_diagnostics_20260926_v2/experiment.json`
- `reports/hyperliquid_weekly_diagnostics_20260926_v2/report.json`
- `reports/hyperliquid_weekly_diagnostics_20260926_v2/weekly-diagnostics-desktop.png`
- `reports/hyperliquid_weekly_diagnostics_20260926_v2/weekly-diagnostics-mobile.png`
- `reports/hyperliquid_weekly_diagnostics_20260926_v2/weekly-return-profiles.png`

The earlier non-v2 export is preserved rather than overwritten. To export another
saved run without a coordinator, use the helper with a **new** output directory:

```bash
.venv/bin/python tools/export_hyperliquid_diagnostics.py \
  --lab-root .hyperliquid_lab \
  --experiment b32e5f1ed7db44008a927c6d965e7c3f \
  --output reports/hyperliquid_weekly_diagnostics_NEW
```

For repeatable real-artifact UI acceptance, set `HL_DIAGNOSTICS_EVIDENCE` to the
absolute export directory when running `npm run test:browser -- tests/diagnostics.spec.ts`.
The browser renders that snapshot using a temporary test lab, never a second
coordinator against the live annual-run database.

## Change boundaries

New diagnostics models, pure statistics, artifact service and accounting modules;
one GET route; three React components; a result-tab entry; generated API types;
focused tests and an export helper. No simulation-core files, frozen configuration,
source cache, running daily process, or existing experiment records changed.
Changes remain in the existing feature worktree; no merge or deployment performed.
