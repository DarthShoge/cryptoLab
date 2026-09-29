# Annual feature writer recovery — 24 September 2026

The failed weekly experiment `94567c9afdee42eea059ce65f71c0c36` completed
feature history through 2025-10-09 and six weekly anchors. Its 2025-10-10
observation output exhausted a 6,334,027-byte reservation while the checkpoint
stream still reserved 128 MiB. The checkpoint's actual file was 1,075,081 bytes.
The failed observation file lacked a Parquet footer. The 475-byte gap between
its file size and reservation was remaining capacity, not a measured overrun.

`FeatureOutput` now transfers unused pending capacity between the observation
and checkpoint streams, atomically updating the existing allocation ledger.
Every write, including footer writes, checks its current limit. Capacity freed
by settlement can also be reused after checking the shared cache budget and
free disk space. The 256 MiB daily limit, individual shard limit, artifact limit,
and 64 GiB cache limit are unchanged. No download is involved.

The regression reproduces the old footer failure, verifies byte-identical
Parquet output against the unconstrained writer, covers capacity being borrowed
back by checkpoints, and verifies that actual daily exhaustion still fails.
Local worker stderr now retains the exception traceback.

The exact source versions listed in `feature_writer_compatibility.py` share the
previous feature/metric cache identity: this fix changes reservation accounting,
not rows, their ordering, schemas, or shard boundaries. Any unrecognized source
version invalidates that compatibility. Actual source hashes, including the
compatibility declaration, remain recorded separately in experiment provenance.
All 94 retained weekly feature-day engines matched this declaration. This does
not assert compatibility of unrelated older cache versions.

Validation: the feature writer, publication, builder, resume and policy suite
passed 89 tests; the subsequent writer, compatibility, metric producer, saved
staged ranking, resume and worker-error suite passed 98 tests. Final writer,
compatibility, recovery and logging checks passed 33 tests. Recovery tests cover
published-output rejection and restoring both files and ledger on failure.

The nine exact abandoned allocations were verified unreferenced by publications
and moved under both the coordinator lock and cache lease to
`reports/hyperliquid_feature_write_recovery_20260924_94567c9a/`.
Its `plan.json` preserves original ledger rows and file hashes; `complete.json`
records the successful transaction. All nine files (158,418,437 bytes) remain
recoverable in that folder. Completed feature days, rankings, anchors, source
data and the failed experiment record are preserved.

The retry uses the same dataset `real_annual_20250901_20260901_staged_v5` and frozen
weekly/daily configs. It gets new experiment identifiers and a new runner
checkpoint under `reports/hyperliquid_annual_acceptance_20260924_staged_v6/`.
`launch.json` records the background PID and command; `runner.log` records the
runner's output. The normal coordinator saves results or a failed status, with
worker tracebacks under `.hyperliquid_lab/worker_logs/`.

The user explicitly requested launch without ongoing agent monitoring. The
background runner handles completion and the weekly-to-daily sequence itself;
there is no inference polling loop. Full annual success remains unverified until
that runner finishes. A genuine resource-cap exhaustion still stops the run.
