# Optimized annual worker restart

The user authorized completing the optimization and restarting the annual job,
then leaving it running without ongoing agent monitoring.

## Implementation

`LabJobs` now accepts an explicit verification policy, defaulting to `full`.
The annual runner accepts `--verification-policy guarded-session-v1`. The
selected policy is frozen in each experiment's provenance and in the runner's
checkpoint; resuming a checkpoint with a different policy is rejected.

In the owned worker, a guarded verification session replaces the existing
checksum function at the `arblab.hyperliquid_copy` import boundaries. It covers
cache audits, publication validation and source-file verification. Already-loaded
aliases and later imports use the same session; all aliases are restored on exit,
including exceptions. The API coordinator continues full verification.

The session contains at most 4,096 digests, never persists them, and rejects reuse
across processes. It checks the file's open-descriptor and pathname identity,
size, mode, link count, mtime and ctime. Recently changed files are double-checked
and remain uncached for one second to avoid timestamp-tick ambiguity. Changed
files are rehashed and still compared against their expected stored digests.
This uses normal local filesystem mutation tracking, not protection against
metadata-invisible storage corruption or privileged timestamp manipulation.

Original numerical algorithms, cache budgets, schemas, data and pinned core
source files are unchanged. This avoids invalidating qualified source and cache
expansion receipts. Experiment source fingerprints now additionally record the
actual hashes of `lab_worker.py`, `lab_verification_session.py` and
`lab_file_hash_session.py`, so the runtime policy implementation is auditable.
Worker logs record policy activation and final hit/miss/bytes-read counts.

## Verification

The production hash implementation passed the same eight file-change tests as
the prototype. The guarded qualified-worker fixture passed weekly, daily and
preview execution, saving and cache reuse with a 64 GiB cache, rolling features
and bounded ranking staging. Context tests cover restoration, late imports,
full-mode behavior, invalid policy rejection and source provenance coverage.

The wider API regression was rerun outside the sandbox after its TestClient
stalled inside the sandbox: 12 tests passed in 33.31 seconds, with two dependency
deprecation warnings. This includes both full and guarded qualified-worker runs.
The eight production hash tests passed again in 6.33 seconds. The stalled test
process was stopped separately from the annual worker.

## Operation

The old worker (`2422494`, experiment `0cfd5f923b274bd5b86cd0591a54d7f4`) did not
exit after SIGINT. SIGTERM stopped it and the coordinator recorded the expected
failed status at 2026-09-25T08:01:25Z. Its completed feature history through
2025-12-14 and anchor at 2025-12-15 remain intact. The stopped cache had zero
pending allocations, so no quarantine or cache deletion was needed.

The optimized replacement uses the same dataset and frozen weekly/daily configs,
with new experiment identifiers and a separate runner checkpoint in
`reports/hyperliquid_annual_acceptance_20260925_optimized_v7/`. Its `launch.json`
records the exact command and PID. It reuses durable cache history; partial
simulation/scratch output may be recomputed. No additional data was downloaded.

The original audit benchmark showed a roughly 169-fold reduction in repeated
audit time. This is not an annual-runtime promise: initial verification, actual
fill processing, ranking and simulation still take time.

## Confirmed launch

The independent runner started at 2026-09-25T08:02:27Z with PID `3002142`.
Experiment `b32e5f1ed7db44008a927c6d965e7c3f` was confirmed `running`, with
`guarded-session-v1` in its saved provenance and all three runtime implementation
hashes present. Its stdout explicitly confirms `verification_session_started`
with that policy. A checkpoint-policy change rejection check also passed.
No agent monitoring loop remains active after startup confirmation.
