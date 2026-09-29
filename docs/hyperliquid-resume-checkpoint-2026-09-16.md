# Annual backtest resume checkpoint — 2026-09-16

## Verified stopping point

- Worktree: `/home/lshoge/code/cryptoLab/.worktrees/hyperliquid-trader-ensemble`.
- Archive job: `.hyperliquid_cache/annual_job_20250901_20260901_300gib`.
- 52 of 67 batches completed. Batch index51 exited0 in session77346.
- Latest qualified manifest: `batches/0051/qualified/qualification_13pm8r8s/manifest.json`.
- SHA256: `1e865698e422c362633d00b2af87bc8b0d1c133cbf8b6f1cccb57ee9336fd9f5`.
- All32validation buckets completed,331481194physical source rows checked.
- Raw, normalized, compact and qualified manifest hashes matched the job ledger.
- Batch51 cleanup deleted336raw/import payloads totaling5832487557bytes;
  compact history remains. Raw recovery requires retained external copies or
  redownload; lifetime download spending is not refunded.
- Download reservations:216970268072bytes; approved ceiling322122547200bytes.
- Batch53 launch was interrupted at its tool approval. Subsequent read-only
  process check found no archive worker; SQLite contained no batch52artifacts and
  reservations were unchanged. Recheck process/state before any new launch.

## Immediate next action

The user questioned the cost of repeated high-effort polling. Recommend a cheaper
coding model at Low effort for established archive operations. Do not repeat an
AI turn every50seconds. A completion/failure-reporting runner is a recommendation,
not an implemented or approved new automation. Preserve the existing download
cap and fail-closed workflow; never blindly retry an interrupted command.

Once resuming approved acquisition, recheck for an existing worker and launch only
the next step using `tools/run_hyperliquid_archive_job.py step`, the job root above,
`--accept-approved-download`, and the existing reader credentials file. Never
print credentials. Network execution still requires the applicable tool approval.

## Pending design and implementation

The user approved both a64GiB ceiling in the SAME derived cache and separating
activity-history starts from verified funding-coverage starts. Neither change is
implemented or applied. Effective derived-cache ceiling remains16GiB.

The cache spec at
`docs/superpowers/specs/2026-09-16-hyperliquid-cache-64gib-design.md` passed a
read-only review without blockers. Written-spec user review was requested and is
still pending; automatic goal continuations are not a review response. Preserve
the prior8→16GiB receipt and its four pinned modules. No authority to delete
successful derived-cache objects or enable real feature expiry/ranking staging.

See `docs/hyperliquid-annual-capacity-check-2026-09-16.md` and
`docs/hyperliquid-native-start-qualification-2026-09-09.md` for measured evidence.
Do not mistake the64GiB ceiling for proof of annual capacity.

## Full goal still outstanding

Finish the entire approved archive; qualify final native/activity/funding and
market coverage; validate capacity; publish the annual dataset; save annual weekly
and matched daily comparisons; independently reconcile accounting; verify the
rendered dataset, historical trader cohorts and comparison UI. Evaluation remains
2025-09-01 through2026-09-01exclusive with90-day lookback and the expanding
BTC/xyz:TSLA/xyz:GOLD/xyz:SP500 market scope. No BTC-only, sampled, shortened or
relaxed-eligibility substitute. No annual result has been produced.

Worktree contains substantial uncommitted work. Preserve it; do not broadly
commit, reset or clean. The goal remains active and incomplete.
