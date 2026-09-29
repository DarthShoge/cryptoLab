# Hyperliquid validation performance investigation

The running annual worker and its application source were left unchanged. The
prototype and benchmark live only in `tools/performance/`; they do not activate
themselves in a backtest, change any cache receipt, or patch the live interpreter.

## Findings

The host process check showed the worker had accumulated 3h37m CPU in 4h47m
elapsed, using about 168 MB resident memory. One read-only 10.012-second sample
observed 10,801,033,422 bytes returned by read calls, 4,631,879,680 bytes charged
as physical reads, and zero write bytes. Sampled open files included many
historical canonical source partitions. This is consistent with substantial
validation I/O, but it is not a stack profile or a measured whole-run percentage.

Three concrete repeat-read paths are present:

1. `CacheResources.reserve()` and `settle()` call `_audit()`, which hashes every
   retained artifact. The earlier inventory was approximately 15.8 GB and 1,539
   files, so even a small new output can trigger a large scan.
2. `PublishedArtifacts._records()` rehashes artifacts on publication lookup;
   `FeatureDay.verify()` repeatedly invokes lookup. Reusing a published result
   still repeats its content reads.
3. `QualifiedWindow.verify()` calls `QualifiedFile.verify()` on source files,
   rehashing the selected interval and reopening Parquet metadata. The existing
   `QualifiedSourceSession` supports verify-once semantics, but these paths still
   construct independent views and repeat checks.

Increasing DuckDB's current one-thread setting is a separate possibility. It
does not remove repeated Python hashing and can increase memory/spill pressure;
it is not the first optimization recommended here.

## Tested prototype

`FileHashSession` keeps a bounded, process-local LRU of SHA-256 results. A hit
requires matching device, inode, mode, link count, size, mtime and ctime. Each call
opens the file without following a final symlink and checks both its open handle
and path identity before returning. Missing, replaced or edited files cannot
reuse an old digest under normal local filesystem timestamp semantics.

Initial tests exposed rapid writes sharing a filesystem clock tick. The guarded
prototype consequently does not cache files written within the last second and
double-reads those recent files to check consistency. Every new session starts
with a full read. This is not protection against privileged timestamp manipulation
or metadata-invisible storage corruption; production must define the same local
immutability assumption already used by `QualifiedSourceSession` and retain full
verification at session boundaries.

Eight direct tests passed, covering unchanged/fresh sessions, same-size edits
with restored mtime, atomic replacement, symlink substitution, deletion, bounded
eviction, mutation during hashing, and recent-write handling. The existing 27
cache-resource tests also passed when this prototype replaced the hash function
in their separate test process. That substitution does not touch application
files or the running worker.

Two further integration fixtures passed with the prototype injected into cache,
publication and source hashing: three-day features matched the complete reference,
including dormant state, and contiguous building resumed under a fresh cache
lease. In total, 37 test cases passed across these checks. The live experiment's
179 recorded application source hashes still exactly matched the files on disk;
its status remained `running` after the investigation.

## Benchmark result

The benchmark copied six completed Parquet files from the preserved failed-write
quarantine into a disposable cache. It did not open or lock the active cache.
The copies totalled 127,883,701 bytes. It ran the real `CacheResources.audit()`
before and after replacing its hash function in the benchmark process only.
All returned accounting totals were identical.

| Measurement | Seconds |
| --- | ---: |
| Five ordinary audits | 0.372564 |
| Initial full audit with the prototype | 0.074661 |
| Five subsequent guarded audits | 0.002207 |

Repeated audits were approximately **169 times faster** on this sample. Ordinary
repeated auditing hashed 639,418,505 bytes; the session hashed 127,883,701 bytes
once and served 30 subsequent hits. This is a small warm-file audit benchmark,
not a 169-fold backtest speedup or an updated annual ETA. The real cache has more
metadata entries, and fill replay, ranking and simulation remain necessary.

Use `reports/hyperliquid_validation_benchmark_guarded_20260924.json`. The earlier
unguarded benchmark is superseded because its implementation failed mutation
tests. Reproduce with the existing environment:

```sh
.venv/bin/python -m pytest tools/performance/test_hyperliquid_hash_session.py -q
.venv/bin/python tools/performance/benchmark_hyperliquid_validation.py \
  --quarantine reports/hyperliquid_feature_write_recovery_20260924_94567c9a \
  --output /tmp/hyperliquid-validation-benchmark-new.json
```

## Recommended integration

Use one explicitly owned verification session across cache auditing, immutable
publication reads and qualified source checks. Preserve all accounting, schema,
membership, ordering and changed-file checks; only avoid rereading bytes already
verified in that session. Keep the cache bounded and discard it on a new worker
or verification session. Add a full-verification mode for independent audits.

Integration needs a deliberate code-fingerprint compatibility transition: the
existing cache expansion receipts pin `derived_cache_resources.py` and
`derived_publication.py`, and source qualifications pin their verification code.
Editing those files underneath the current worker could invalidate its run.
Do the integration and a representative weekly-window throughput comparison in
an isolated snapshot, then restart from saved progress when applying the change.
An already-running Python worker will not gain this optimization automatically.

The optimization was subsequently integrated as an explicit worker policy and
launched on 25 September. See `hyperliquid-optimized-restart-2026-09-25.md` for
the implementation, provenance binding, tests and confirmed experiment ID.
