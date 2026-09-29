# Annual archive trade-identity validation incident

## Observed state

Archive worker37471 exited1 with `inconsistent market trade counterparties`.
The preserved job has24qualified batches and25raw/normalized/compact batches.
Index24 covers2025-11-17–2025-11-24 and is not qualified. No restart, new download,
counterparty-check bypass, old-report rewrite or ledger reset has been performed.
The original lifetime ledger records103,839,183,349bytes against300GiB.

The latest accepted prefix is
`batches/0023/qualified/qualification_xm2pg1ck/manifest.json`, SHA256
`0d6496363480fa5fc9d13fa60e032cfec9d331b28403984fe4aeecddfc59731f`.
The failed extension's compact manifest is
`batches/0024/compact/compact_dgy9j3nt/manifest.json`, SHA256
`e8581bbac57b4719d34614922a2bcd8a6b0672bb5a187b307bdd6f1838f4b56a`.
Both paths are relative to the existing annual job root. Its failed-batch raw and
normalized payloads remain intact; previously qualified canonical history remains.

## Confirmed root cause

The failing query in `proxy_activity.py` groups all records by `(coin,tid)` and
rejects more than one distinct `(exchange_time,px,sz)` tuple. Its market-volume
query also groups only bytid within each coin. Wallet deduplication uses
`(user,coin,tid,oid,side)` without a time discriminator.

The official [WebSocket trade schema](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions)
describes tid as a50-bit hash of order IDs and recommends time as part of global
trade identity. The completed cross-prefix diagnostic confirms this is the cause
of the observed failure, rather than inconsistent economics within these trades.

The failed extension alone was scanned in32deterministic bounded partitions:
no internal `(coin,tid)` counterpart-economics conflict was found. Source hashes
were checked before/after. Report:
`.hyperliquid_cache/counterparty_probe_1789022913383129842/summary.json`.
A pinned cross-prefix join completed as session36173, exit0, in796.642seconds.
Report: `.hyperliquid_cache/cross_time_probe_1789023061950390457/summary.json`.
It found BTC tid840478295334001 in two distinct trades:

| Exchange time (UTC) | Price | Size | Block number |
| --- | ---: | ---: | ---: |
| 2025-11-05 15:33:04.054 | 103308.0 | 0.14631 | 786787181 |
| 2025-11-23 14:44:03.404 | 86751.0 | 0.024 | 805984548 |

Each timestamp has matching buy/sell counterpart economics; within-time conflicts
are zero for this pair. Source hashes were verified before/after. The diagnostic
stopped at its first findings and is not complete corrected-engine qualification.
Do not restart this terminal diagnostic or the failed acquisition worker.

The November23 record was also verified directly in retained raw
`batches/0024/raw/proxy_archive_ju5twff1/fills_0158.lz4`, SHA256
`04a45652b65183f947f5799f46e8f7e0049efbc0b635c20d842ad291cad2a20b`.
Zero-based source line32510, event indices58/59, contains raw fill
`time=1763909043404` and envelope `block_time=2025-11-23T14:44:03.404656912`.
Both raw sides match the canonical time, tid, order IDs, price and size.
The raw hash was checked before/after the bounded read. November5 raw staging
was previously disposed after qualification; it was not downloaded again.

## Timestamp discriminator evidence

The parser stores raw fill `time` milliseconds as `exchange_time`. Envelope
`block_time` is separate, may have sub-millisecond precision and is absent in the
legacy pre-block archive. A correction must prove which representation supplies
the cross-schema trade discriminator. Blindly grouping by nullable block_time or
coalescing differently precise timestamps would change overlap deduplication.
Use the raw fill's millisecond timestamp consistently across both archive schemas.

A bounded sample read4096rows each from canonical June3legacy, July28block and
November17block files. All June rows lack envelope block_time. All8192block rows
have `floor(block_time to milliseconds)==exchange_time`; observed envelope offsets
are2–999microseconds (November maximum997). This supports using the raw fill's
millisecond time consistently across schemas, not unnormalized envelope timestamps.
The subsequent full audit completed as session83953, exit0, in79.113seconds.
Report: `.hyperliquid_cache/timestamp_audit_1789023769089446991/summary.json`.
It scanned all175canonical files from the first25batches in4096-row Arrow batches:

- 131,853,494 rows;93,785,698 with envelope block_time.
- Zero null or non-millisecond exchange timestamps.
- Zero block-format rows missing envelope time.
- Zero differences between exchange_time and envelope time floored to milliseconds.
- Source hashes verified before/after; peak decoded batch254,976bytes.

This establishes the discriminator for the retained corpus, not blanket future
source validity. The corrected validator must enforce this correspondence for
new data and continue rejecting ambiguous same-timestamp economic conflicts.

## Required correction scope and regressions

Independent review identified these coupled requirements:

- Correct market-trade validation, wallet-fill deduplication and volume counting
  together. Genuine same-trade conflicting economics must continue to fail.
- Preserve distinct trades sharing tid across time, while counting both wallet
  counterparties only once in market volume.
- Preserve exact legacy/block duplicate collapse, canonical provenance ordering,
  per-wallet episode inputs and strict causal cutoffs.
- Review projected schemas and derived cache identities. Day projections retain
  exchange_time but currently omit envelope block_time; do not silently reinterpret
  an existing projection if new identity requires additional provenance.
- Existing hash(coin,tid) sharding and tid-range overlap bounds remain conservative
  under a stronger identity. They need not change just to fix correctness.

## Frozen-engine transition constraint

The original job's metadata is immutable and hashed. `ArchiveJob` compares its
stored engine with current code; qualification reports, checkpoints and derived
day publications independently pin relevant engine versions. Editing the failing
query and then rewriting stored hashes is not an acceptable resumption procedure.

Remediation needs an explicit reviewed successor/version transition that preserves
the old reports, old caches, original source/canonical bytes and lifetime spending
ledger. Reuse already acquired bytes; no repeat acquisition is justified by a
local validation failure. Requalification under the corrected identity must
precede publication/cleanup of the failed batch. The proposed transition is in
`docs/superpowers/specs/2026-09-10-hyperliquid-trade-identity-transition-design.md`.
It is a design, not an applied migration: no engine, ledger, report or cache was
modified by these diagnostics.

## Approved correction implementation checkpoint

The user approved the transition design on September10. Preparation was then
implemented test-first and independently reviewed before touching frozen modules.
The original job was preserved under its existing lock, without activation,
cleanup, ledger writes or network requests. Session14558 completed successfully
in134.777seconds, followed by a successful reopened verification.

Preparation: `engine_transition_v2/manifest.json` within the original job root,
SHA256 `46b001b8a5f713a0e5549ca51f2ec822a776bc1307477d37fee242b671230ef8`.
It preserves23old-engine source files (152,305bytes),610retained-file references,
all existing stage/cleanup records and the original103,839,183,349byte spending
history. This is an audit snapshot, not corrected semantic qualification.

Only after preservation succeeded, `proxy_activity.py` was updated to include
millisecond exchange time in wallet/trade identity and volume grouping, and to
validate fill/envelope timestamp agreement before deduplication. The qualification
engine version is now2. Independent review found no material correctness issues.
Twenty-four focused identity/activity tests pass, including legacy overlap,
genuine economics conflicts, causal seed queries and sharded/day projection paths.

The initial broad run exposed a synthetic archive fixture missing block_time;
after fixing that fixture,752tests passed with one remaining spill-fixture
envelope mismatch. That fixture now shifts its envelope and fill time together,
preserving the original boundary-spill assertion. Final full core verification
passed753tests with22existing deprecation warnings in241.18seconds (session48435,
exit0). API verification passed45tests with2existing deprecation warnings
in92.27seconds (session19924, exit0). The earlier failing runs remain diagnostic
history; their failures are not counted as successful runs.

The original production job now correctly rejects ordinary opening because its
stored old engine differs from current code. A post-change real snapshot
verification also passed (session26842, exit0), confirming all preserved evidence
and unchanged spending, followed by the expected exact engine-mismatch rejection.
Activation and the baseline-backed qualification bridge are now implemented and
independently reviewed. The broader core run passed770tests in367.81seconds
(session45739), before the final one-line pending-file fsync correction.
After that correction, all18transition tests passed in122.36seconds
(session91980). All45API tests passed in96.27seconds (session36094).
The real first24batch baseline activation completed successfully as session4629,
exit0, in7067.621seconds. All32partitions passed, covering121,698,646rows.
Baseline: `engine_transition_v2/baseline/qualification_z2vnnjz1/manifest.json`,
SHA256 `2bde8d1c08020161d6d171d6f4381551718273af150d92c8239e016f19e2c027`.
Activation: `engine_transition_v2/activation.json`, SHA256
`7dd1047e21a40e764177e8f86a877173f3b31292b7f267ae7023460c82ad8316`.
Readonly database checks confirm24qualified and25compact batches, with unchanged
103,839,183,349reserved bytes against322,122,547,200cap bytes. Offline recovery
of index24 completed as session70687, exit0, in7712.701seconds. All32partitions
passed, covering131,853,494physical rows. New qualification:
`batches/0024/qualified/qualification_8myvjhkb/manifest.json`, SHA256
`7e75076853a7b30872fd795fa4eccea7348e06c2cc597c479b28c3a44c27e88e`.
The existing bridge verified preserved old state, reports and cleanup evidence.
Normal owned staging cleanup removed336payloads/5,956,874,372bytes, leaving
canonical history and manifests intact. These raw payloads require an external
cache or redownload to reconstruct; lifetime spending is not refunded. Exactly25
batches are qualified and reservations remain103,839,183,349bytes.

The approved remaining acquisition worker started as session72925, using the
existing IAM reader, original67batch scope and300GiB lifetime ledger. It checks
the exact25batch boundary and spending before source use, stops on any exception,
and has no automatic retry. Completion/new downloads are not established by its
start message alone. Track this specific live handle and never duplicate it.

The annual research goal remains unchanged. None of these component results
qualifies the failed source extension or completes upstream all-wallet feature
generation and annual saved comparisons.
