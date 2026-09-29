# Final archive readiness and research analysis — 2026-09-22

## Conclusion

Acquisition and prefix qualification finished successfully for all 67 batches.
The real annual strategy has **not been run or published**. Existing saved annual
weekly/daily experiments use synthetic BTC data; the only saved real experiments
cover August 4–6, 2026 with a two-day lookback. Neither establishes annual alpha.
Earlier progress estimates described archive processing, not remaining integration
and simulation time. No reliable end-to-end completion estimate is established.

## Verified evidence

Worktree: `.worktrees/hyperliquid-trader-ensemble`.
Job: `.hyperliquid_cache/annual_job_20250901_20260901_300gib`.
Final process session 3898 exited 0. Final report:
`batches/0066/qualified/qualification_6kv0n3kc/manifest.json`, SHA-256
`8672ce51736b76157592ff2b9ed2253a74af043a681b0006ba62987d88389368`.

- All 268 stage manifest files match their ledger hashes; all four stages contain
  67 entries covering indices 0–66.
- Final qualification reports 420,282,058 physical canonical rows, including
  duplicates and boundary spill. This is not a count of unique trades.
- Source partitions span June 2, 2025 through September 2, 2026 exclusive.
- Qualified compact files total 37,326,058,747 bytes (37.33 GB).
- Lifetime downloads including 83,604,778 retry bytes total 290,379,853,495 bytes.
  Approved cap: 322,122,547,200 bytes (300 GiB); remaining: 31,742,693,705 bytes.
  This cap measures cumulative downloaded bytes, not disk usage or the AWS bill.
- Final-batch cleanup ledger records 48 deleted raw/normalized files totaling
  1,179,995,443 bytes. Recovering discarded raw payloads requires external retained
  copies or redownload; compact history remains the reusable source.

The full observed-history scan (session 71475, exit 0) checked the final pinned
corpus, including file content checks before and after scanning. Market bundles
were independently reconstructed from retained sources (session 89005, exit 0).
Funding bundle SHA: `9b3d54065db4c73c00ba16839487055bce37ff9d09146ff34a233078f2607f25`.
Price bundle SHA: `1ceffd67fd921ce21c409a8616579eda656803f310b18d83568c5526df1181f2`.
No network requests or strategy modifications were needed for these checks.

## Market history and interpretation

| Market | Physical rows | First observed fill UTC | Funding coverage starts UTC | Earliest Monday allowed by 90-day history |
| --- | ---: | --- | --- | --- |
| BTC | 372,996,680 | 2025-06-02 00:00:00.086 | 2025-06-02 00:00 | 2025-09-01 |
| xyz:TSLA | 13,046,994 | 2025-11-13 14:31:56.848 | 2025-11-13 15:00 | 2026-02-16 |
| xyz:GOLD | 13,344,600 | 2025-12-22 14:42:19.649 | 2025-12-22 15:00 | 2026-03-23 |
| xyz:SP500 | 20,893,784 | 2026-03-18 13:26:16.574 | 2026-03-18 14:00 | 2026-06-22 |

Monday dates are derived lower bounds using the current hourly observed-history
rule and midnight UTC Monday decisions. They are not verified cohort membership:
trader eligibility, valid prices, and execution constraints still apply. They do
not establish listing dates or history before the retained observations.

Cross-class proxy price coverage starts February 3, 2026, before these admission
dates; BTC prices and funding span the full required interval. Both bundles end
September 2, 2026 exclusive. Proxy pricing does not reproduce native execution.

Consequences for analysis:

- The requested September 2025–September 2026 comparison has an expanding market
  universe. Before February 16 it can only copy BTC traders within this scope.
- Report returns and contributions separately around each admission date. An
  annual aggregate cannot demonstrate a year of cross-class diversification.
- These fill records do not establish complete wallet equity or deposits. Rank
  metrics must retain their actual definitions; scoped PnL efficiency must not be
  presented as wallet return on invested capital.
- The scope is BTC, TSLA, GOLD and SP500, not every Hyperliquid instrument. General
  market rankings on this dataset must disclose that restricted candidate scope.

## Confirmed unfinished integration

1. `registration_native_history.qualify_observed_history` currently requires
   funding coverage to start no later than the observed activity hour. All three
   cross-class markets violate that condition by one coverage hour. The previously
   approved separation of activity-history and funding-coverage starts is still
   unimplemented. Preserve early fills and independently bound funding coverage;
   do not shift activity history or invent zero funding.
2. The existing shared cache ledger still has a 17,179,869,184-byte ceiling
   (16 GiB), identity `real-first-decision-candidates-v2`, 495 retained allocations
   totaling 5,286,350,121 bytes, and no allocations in other states. The approved
   64 GiB transition has a design but is unapplied. Preserve its prior receipt and
   pinned dependencies. A larger ceiling is not proof that the annual run fits.
3. No real annual registered dataset or saved annual result exists in the current
   lab registry. Annual publication, capacity validation, simulations, accounting
   reconciliation, and rendered UI acceptance remain outstanding.

## Execution and analysis order

Complete the approved funding-boundary integration and cache transition with
focused regression checks. Validate final-source capacity, then publish the full
annual dataset with its 90-day warmup and historically expanding four-market scope.
Run/save the weekly trader-selection and weekly-position baseline, followed by
the otherwise matched daily comparison. Do not change ranking logic to improve
observed performance or substitute a shorter/BTC-only real run for acceptance.

For each result reconcile initial equity plus trading PnL minus fees/slippage plus
signed funding to terminal equity, including residual positions. Compare with the
same-window BTC benchmark on return, Sharpe, Sortino, maximum drawdown, exposure,
turnover and costs. Inspect the largest drawdowns and contributor concentrations.
Show historical trader membership, exclusions, cohort retention and market entry
dates in both the report and UI.

Only after a reconciled baseline, test explicitly recorded top-N/top-percentile,
ranking-weight and cost sensitivities. Treat multiple variants as exploratory
development tests; do not call the best retrospectively selected variant
out-of-sample evidence. A later forward paper-trading period remains the cleanest
next check of whether trader-selection skill persists.

This audit followed the crypto-strategy-report skill: establish provenance and
actual saved experiments before interpreting returns. No annual performance,
Sharpe, Sortino, drawdown or benchmark victory is claimed.
