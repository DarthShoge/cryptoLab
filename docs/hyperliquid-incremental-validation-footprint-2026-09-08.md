# Incremental native-identity validation: local sizing evidence

## Purpose and status

At the start of this investigation the registered loader validated the full corpus with
`ProxyActivity`, limited to 8 GiB. Rolling queries work after that validation but
do not remove this initial gate. Per-chunk validation alone is insufficient:
conflicting native fill identities and inconsistent trade counterparties can
occur in different chunks, including outside the current lookback.

An incremental derived identity index is a candidate replacement for that gate,
not a replacement for retained canonical Parquet or causal position checkpoints.
SQLite can retain native-identity/economic fingerprints while full canonical
events remain in Parquet. Qualification must still validate every row, preserve
full-reader semantics, and bind results to source hashes and engine versions.

## Local experiment

No network calls or source edits. The first 100,000 rows of the existing compact
week were streamed through DuckDB in 2,048-row batches, with 256 MB DuckDB memory,
256 MB spill and an 8 MiB SQLite page cache. Two SQLite WITHOUT ROWID tables held
32-byte identity hashes and 32-byte economic hashes: one for wallet fill identity,
one for `(coin, tid)` counterparties. This deliberately measures storage only.

- Input rows: 100,000.
- Distinct fill identities: 100,000.
- Distinct trade identities: 50,000.
- SQLite file: 11,894,784 bytes, 118.94784 bytes per sampled input fill.
- Elapsed: 0.925 seconds; observed process peak RSS: 202,404 KiB.
- Artifact: `.hyperliquid_cache/identity_footprint_0pt5ptui/footprint_only.sqlite3`;
  adjacent `measurement.json` records the result.

The sample used `INSERT OR IGNORE` and raw SQL JSON hashes: **it is not a
conflict verifier and must never be consumed as qualification evidence**. In
particular, production fingerprinting needs tested normalization for numbers,
signed zero, timestamps and nullable economics so it matches the existing full
reader's equality rules. Hashes must not hide economic conflicts.

The retained week's 4,875,344 rows scaled to 457 days would be about 318 million
rows; the sample's storage rate implies roughly 35 GiB of derived index storage.
This is a coarse extrapolation, not a bound or a runtime estimate. Native activity,
duplicate ratios, page utilization and random-write behavior vary with history.
Compact Parquet, raw staging, SQLite rollback journal and query spill are additional.

## Implementation implications

### Chosen lower-storage implementation

After inspecting the existing validator, deterministic trade-group partitioning
avoids building the fingerprint index at all. `sharded_validation.qualified_seed`
scans `hash(coin, tid) % 32` groups one at a time into bounded temporary Parquet.
All counterparties and native-identity duplicates stay together. Hash collisions
only co-locate unrelated groups; exact comparison still uses the existing SQL.
The source union is typed globally before filtering, preserving mixed numeric
types and signed-zero behavior instead of inventing a fingerprint codec.

Each group is checked with `ProxyActivity.validate_registered_scope` and native
availability validation. Strictly pre-cutoff wallet/market seeds are merged using
the original reverse native ordering. Only after all groups pass, raw source row
counts agree, and all frozen hashes still match does the context yield its seed.
The caller must bind that evidence into durable dataset/checkpoint publication.

Bounds: 64 GiB/5000 source files, a 2 GiB compressed group, 512 MiB/2 million seed
rows, one DuckDB connection at a time (256 MB memory/2 GiB spill), and at least
group + two seeds + spill + 64 MiB free scratch. Capped writes include Parquet
footers. Skew or exhaustion rejects, without relaxing those bounds. The cost is
32 sequential source scans; annual runtime is not yet measured.

This supersedes the fingerprint-index proposal below. The first helper restarts
validation after interruption; it does not publish partial qualification. It only
removes owned temporary group/seed files, not canonical history. Subsequent loader
integration now permits explicit `validation_mode: sharded_v1` datasets up to
64 GiB for scheduled v2 runs, with engine-bound reusable checkpoints. Missing mode
retains the legacy 8 GiB gate. Acquisition and rolling working-set limits remain
separate constraints.

Real-week verification subsequently processed all 4,875,344 rows in 240.21 seconds,
with observed process peak RSS 674,152 KiB. Its 16,925 canonical seed rows exactly
matched a higher-memory full-reader reference (zero differences in either SQL
`EXCEPT ALL` direction). The default full-reader 256 MB setting failed on this
week during duplicate aggregation; the diagnostic reference used 1024 MB and
observed 1,519,324 KiB RSS. Production memory settings remain unchanged. These are
single-run observations, with some concurrent work, not annual runtime guarantees.
Proof artifacts are under `.hyperliquid_cache/sharded_validation_proof_vbmtgml2/`.
Fourteen focused tests and the 397-test core suite passed.

### Requirements if an identity index is revisited

Use a disposable, versioned qualification index with an explicit disk ceiling and
bounded transactions. A single week-long SQLite write transaction can create a
large rollback journal; do not assume the table-file limit bounds peak disk.
Commit small insertion batches and publish partition qualification only after all
rows pass and source hashes remain unchanged. Restart must replay an unfinished
partition idempotently, while a conflicting record fails the qualification job.

Required correctness tests: distant-chunk economic conflicts; counterparties;
identical duplicates; numeric/null/time equality; source mutation; crash midway
through a partition; exhausted storage; engine changes; and full-reader equivalence.
Published qualification needs complete source membership, not just individual
hashes. It must not turn a filtered prefix or first returned funding timestamp
into complete historical coverage.

Do not remove the existing 8 GiB guard until this replacement is implemented,
measured and wired into initial checkpoint construction. The real annual source
budget (300 GiB requested) remains unapproved; this local experiment spends none.
