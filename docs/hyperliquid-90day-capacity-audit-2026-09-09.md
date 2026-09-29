# Real 90-day query capacity audit

## Outcome: the annual runner is not yet ready at real scale

Latest market-query checkpoint (September11): bounded observed-market and native
volume helpers are implemented, independently reviewed and tested. They query
qualified Parquet under the shared cache lease,256MB SQL memory and2GiB spill,
returning only50market IDs or one scalar. A concrete fractional-volume test
showed that removing native dedup before SUM changed market ties and cutoff
eligibility despite algebraically equal totals. Restoring the native-order
dedup CTE fixed all four cohort mismatches; no rounding/tolerance was added to
selection. Fresh post-format market/QualifiedWindow/ProxyActivity/ProxySelection
regression passed72tests in103.57seconds, including300006physical fills for
100002unique trades, multiple classes, timestamp-reused IDs, threshold-adjacent
selection and source/scratch failure checks. No real annual volume-capacity or
scheduled integration acceptance is implied. Plan/evidence:
`docs/superpowers/plans/2026-09-11-hyperliquid-qualified-market-queries.md`.

Latest native-position checkpoint (September11): a new selected-cohort query
reads qualified event days backwards under the shared cache lease,256MB SQL
memory and2GiB spill reservation. It retains only the current day's metadata and
a deduplicated file registry, deduplicates native identities before latest-fill
selection, preserves known zero/short positions and searches dormant positions
back to the authorized source origin. It does not open the full-prefix reader.
The32native tests plus existing QualifiedDay/ProxyActivity tests passed together:
53tests in99.08seconds after formatting. Fixtures cover100002same-day fills,
94days of history, source/engine/request mutation, lazy scans, shared accounting,
scratch replacement and overlapping-source metadata lifetime. Independent review
clarified that the reference must be clipped to the same authorized interval;
physical pre-origin spill is not extra coverage. Actual annual registration has
a later padded start, which still requires explicit adapter reconciliation.
No actual selected-cohort query, pipeline routing or annual result is claimed.
Plan: `docs/superpowers/plans/2026-09-11-hyperliquid-qualified-native-positions.md`.

Latest metric-builder checkpoint (September11): complete candidate-to-metric
publication and disk-scoring integration are implemented and independently
reviewed. Focused tests verify exact reference values/exclusions/ranks/weights,
dormant membership, cache reuse and failure accounting. Core regression passed
853tests with22existing warnings (716.73s); six fresh targeted tests cover the
subsequent lazy-scan correction described below.

The first real full-ranking probe (7582) was OOM-killed before partition planning
finished. The cause was eager materialization by DuckDB's parameterized relation
view API, despite a256MB query-buffer setting; the kernel recorded11159084KiB
anonymous RSS. A small execution-plan reproduction and failing regression exposed
the COLUMN_DATA_SCAN. The corrected literal TEMP VIEW preserves a lazy Parquet
scan with safely escaped validated inputs. Corrected probe77381 stopped before
scanning because an added4GiB virtual-address ceiling prevented ICU loading.
Initialization diagnostic19186 also aborted under that ceiling: virtual size
was nearly4GiB while resident use was only148MiB. The guard needs correcting;
neither attempt proves real lazy-scan capacity, and neither is still running.
Initialization44083 subsequently succeeded without that unsuitable virtual cap
(about143MiB resident use). Current real probe70326 instead uses an external
2GiB resident-memory watchdog, tested on over/under-limit children. It is live in
partition planning; no completed real ranking is claimed yet.
No metric publication was created by the failed probe. Its3GiB scratch and512MiB
output reservations remain charged in the existing8GiB cache, with no deletions
or download-cap changes. A further512MiB failed-output obligation remains from
77381; old failed-work obligations total4GiB, not a budget refund. The live
70326attempt additionally reserves3GiB scratch and512MiB output: read-only ledger
inspection shows7.5GiB pending overall,28,913,155retained bytes and32MiB fixed
metadata, all within the existing8GiB envelope. Do not open a second build lease
or treat old unfinished work as reusable/free space.
Detailed evidence and handles:
`docs/superpowers/plans/2026-09-11-hyperliquid-all-wallet-metric-producer.md`.

Latest candidate-index checkpoint (September11): a reusable daily
first-observation index and complete causal candidate-history publication are
implemented and independently reviewed. They read qualified source columns
directly, so candidate enumeration does not retain another event-level corpus.
The complete index starts at the exact qualified source origin, preserving
dormant wallets; no rolling-lookback origin substitution is accepted.

The32focused tests passed in42.03seconds, including100,001synthetic wallets
through source import, qualification and final index publication, interrupted
reads/writes, source changes, strict decision cutoffs and shared resource bounds.
Review fixes pin the query context/daily chain and bound day-count allocation
before constructing metadata. Full-core regression passed803tests with22existing
warnings in499.23seconds (session79026, exit0). Real candidate-index measurement
completed as session3576, exit0: all108,185candidate wallets, exactly matching
the preceding-source-day audit. Build/reopen took100.333seconds, with289,252KiB
peak process RSS and60,106,110total cache bytes including32MiB fixed metadata.
The91daily indices plus final candidate list retain26,551,678payload bytes;
no pending reservations remain. The final list is2,361,477bytes. Ordering,
uniqueness, final hash and unchanged publication/charges on reopening passed.
Evidence: `.hyperliquid_cache/candidate_history_real_914u8kwn.summary.json`.
This is candidate-membership acceptance, not metric/scoring or annual-run
acceptance. RSS covers only this probe; offline archive recovery ran concurrently.
Ordered wallet metrics, reusable features and scheduled ranking integration are
still incomplete, independently of archive acquisition progress below.

The first planned decision, September 1, 2025 00:00 UTC, already exceeds two
ranking guards, and opening its minimal lookback prefix fails under the current
query-memory limit. The completed synthetic integration tests did not exercise
this volume. Acquiring the remaining source alone will not produce a runnable
annual comparison; a bounded query/ranking scalability fix is required.

The audit used the qualified source prefix through September 29 exclusive,
report `batches/0016/qualified/qualification_j0kpv3h8/manifest.json`, SHA-256
`d114c2e304572a37d1f96f3d4c678407a1e81c73bb34b5937e1e97e794cb6467`.
Exact event-time filtering selected BTC activity in June 3–September 1 exclusive.

| Measurement | Observed | Current guard |
| --- | ---: | ---: |
| Candidate wallets, including dormant prefix wallets | 108,185 | 100,000 |
| Wallets active in the 90-day lookback | 107,825 | — |
| Physical lookback rows | 58,085,358 | — |
| Largest wallet's distinct native fill identities | 6,662,648 | 100,000 fills per wallet |
| Wallets above 100,000 physical lookback rows | 73 | — |
| Candidate lookback files, excluding seed | 90 /5,033,916,475 bytes | 8 GiB query input |

The largest wallet's physical count and distinct `(coin,tid,oid,side)` count were
equal; duplicate source records do not explain its oversized history. Candidate
counting included the preceding source day to include dormant candidates. The
bounded count audit took 24.725 seconds, with one DuckDB thread, 256 MB memory
and 2,048 MB temporary-spill cap.

A separate call to the existing `ProxyActivity` constructor used 91 prefix files
(5,086,228,004 bytes) and its default resource limits, with the exact query window.
It failed before ranking: `OutOfMemoryException`, unable to allocate another
256 KiB at 244.1 MiB used. The probe completed in 63.583 seconds. It used a minimal
prefix; the actual scheduled reader additionally needs decision-hour data and
checkpoint seed handling. This failure is not evidence that increasing one guard
would make the complete scheduled path work.

Both probes checked selected source hashes/sizes before and after execution and
rechecked the pinned report/engine. Owned query temporaries were cleaned on exit.
No downloads, raw deletion, source edits, saved scenarios or trading occurred in
these probes. Machine-readable measurements:
`.hyperliquid_cache/annual_capacity_audit_20260909/summary.json`.

## Approved direction and reviewed specification

The user approved the disk-backed ranking direction. The detailed specification
is `docs/superpowers/specs/2026-09-09-hyperliquid-disk-ranking-design.md`.
Independent review approved it after adding a shared resource lease, transactional
pre-write reservations and crash recovery that keeps unfinished staging charged
against the budget. The user subsequently approved the written specification.
The calculation foundation below is implemented independently of the runner.

The recommended direction is a disposable disk-backed ranking cache with bounded
candidate output, chronological streaming computations, and exact episode/median
handling. Reusable daily/episode features could avoid replaying tens of millions
of fills at every weekly or daily decision, but must first prove equivalence to
the current lookback boundary, censored/flip episode, fee, ordering and percentile
semantics. Preserve all canonical fills and wallets, including excluded candidates;
do not sample, truncate, tune the scoring weights or replace the annual scope.

Alternatives are streaming complete fill windows at every decision (simpler
equivalence story but much more repeated work), or merely increasing RAM/list
limits (does not establish a bounded annual footprint and is not recommended).
The design must also audit the annual ranking artifact's 50-million-row /4-GiB
ceilings, the 100,000-row sink-call limit, daily comparison output, query sorting,
checkpoint advancement, and final dataset loading—not just the two ranking guards.

Keep the live archive engine and lifetime AWS budget unchanged. Any new execution
path must be isolated from the frozen acquisition modules. The specification
retains existing resource envelopes as starting bounds and requires measured
footprint disclosure before any necessary increase. No runner change has been
made to the scheduled runner by this audit. The long-dated end-to-end goal remains unfinished.

## Streaming wallet calculation foundation

New `wallet_metric_spool.py` and `streaming_wallet_metrics.py` compute the base
reference metrics from an already qualified/deduplicated ordered single-wallet
iterator. Completed episode observations are spooled rather than accumulated;
daily and active-episode state are bounded. Ordered builtin sums preserve reference
floating arithmetic; exact disk medians use the central one/two sorted values.
This is not yet the all-candidate cache, scoring or scheduled reader integration.

Independent review found two resource gaps, both fixed with RED/GREEN regressions:
the Parquet writer now has an independent 4,096-row-group metadata limit, and disk
median queries require their complete 2-GiB spill allowance plus 64-MiB free-space
reserve before starting. Other limits remain 4,096 buffered scalar observations,
512-MiB spool, one query thread, and 256-MB DuckDB memory allocation. Invalid or
non-finite intermediate observations fail closed, never yield a partial result.
The primitive does not enforce the future shared cache lease itself.

The focused suite passed 31 tests before final formatting, including fee modes,
flips/left censoring, sparse days, exact floating cancellation, even/odd medians,
invalid streams, cleanup and a generated 100,002-fill wallet. Independent focused
review reported no remaining blockers. After formatting, the full core suite passed
**584 tests in 233.82 seconds**, with 22 existing exchange-calendars/NumPy deprecation
warnings. `git diff --check` also passed. No API/UI integration changed in this unit.

Two read-only probes used the pinned qualification report above and verified all
selected source file sizes/hashes before and after, plus the frozen archive engine.

| Probe | Fills | Elapsed calculation/query | Numeric spool | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| Tractable first-day subset, exact reference equality | 10,000 | 1.167 s | 79,653 bytes | 284,616 KiB |
| Full oversized wallet, ordered native feed | 6,662,648 | 161.955 s | 52,635,942 bytes | 1,273,296 KiB |

Both used at most 4,096 scalar-buffer rows. The full window retained all native
identities for the audited wallet, with 90 active days and one complete episode;
its exclusion was `insufficient_episodes`. No complete-reference list was built
for the full wallet. Full reference equality is established only on the tractable
subset/fixtures, not by comparing two 6.66-million-row Python lists.

The full probe succeeded under the configured 256-MB DuckDB allocation and 2-GiB
spill cap, but **total process RSS was approximately 1.21 GiB**, not 256 MB.
Python/Arrow/native-library allocations must be included in machine sizing. A
mid-run disk observation was approximately 1.1 GiB, not an instrumented disk peak.
Source sorting resources closed before median queries; owned scratch was removed
on exit, leaving diagnostic reports only. Timings exclude before/after hashing.

Reports:
`.hyperliquid_cache/wallet_metrics_probe_subset_1788969771107824001/summary.json`
and `.hyperliquid_cache/wallet_metrics_probe_full_1788970023922558993/summary.json`.

Remaining gates: process-safe shared resource ledger; reusable causal derivations;
all 108,185 candidates without lists; exact disk scoring and report budgets;
positions/volume/conviction/checkpoint integration; final qualified annual source
and native funding boundaries; actual saved weekly/daily comparison and independent
accounting/UI acceptance. No strategy performance conclusion follows from these probes.

## Shared derived-cache resource foundation

`derived_cache_lease.py` and `derived_cache_resources.py` now provide the isolated
local resource contract: one nonblocking Linux process lease, a separate SQLite
catalog, and an aggregate 8-GiB maximum including a 32-MiB metadata allowance.
Unfinished allocations remain charged at their reserved maximum across crashes;
retained payloads remain charged at their finalized size/hash. Removing or
truncating the catalog cannot reinitialize an existing cache or erase obligations.
The catalog checks safe paths, metadata bounds, payload integrity and free-space
headroom; callers must still cap writers and query spill within their reservations.

38 focused tests pass without warnings. Independent review cleared the
sync-before-refund fix; failed directory sync leaves the reservation intact.
The frozen source-acquisition engine still verifies. The full core run passed
620 tests in 193.89 seconds with 22 existing calendar warnings, followed by a
clean 38-test focused rerun after the last two edge-case additions and formatting.
`git diff --check` passed. These helpers do not delete data, publish derived files,
change the acquisition budget or yet replace the scheduled reader.

## Daily projection feasibility and atomic publication

A read-only probe deduplicated three qualified BTC source days by native identity,
retained the authoritative order as a per-day ordinal, and wrote the 14 fields
needed for metric/native-query work plus that ordinal. It retained every row in
these particular days (physical and deduplicated counts matched). This is a
projection/storage experiment, not reusable episode-feature equivalence or a
qualified annual derived dataset.

| UTC day | Rows | Canonical input bytes | Projection bytes | Query/write seconds |
| --- | ---: | ---: | ---: | ---: |
| 2025-06-03 | 675,938 | 58,158,601 | 31,772,148 | 3.829 |
| 2025-08-31 | 315,916 | 28,312,419 | 15,163,991 | 0.984 |
| 2025-09-28 | 317,908 | 28,335,310 | 15,442,783 | 0.966 |

Maximum decoded Arrow batch was 606,208 bytes /4,096 rows. Cumulative process peak
RSS was 551,072 KiB, distinct from the 256-MB DuckDB allocation. The probe used the
same 2-GiB query-spill cap, reserved through the new resource ledger. Source hashes
were verified before/after; the frozen archive engine was rechecked. The probe
emitted a DuckDB `fetch_record_batch` deprecation warning; the implementation
builder should use the replacement Arrow reader API. Three days do not establish
an annual storage bound, cross-class coverage or a guaranteed 8-GiB fit.

Report: `.hyperliquid_cache/projection_capacity_1788973902894394688/summary.json`.
That diagnostic v1 root deliberately keeps its three scratch obligations charged;
its 62,378,922 bytes of projected payload are retained. It is not a production cache.

`derived_publication.py` implements content-keyed atomic catalog visibility over
immutable finalized `artifacts/` paths. A single SQLite commit publishes the full
set; no file rename is needed, and incomplete/unpublished outputs remain charged
and undiscoverable. Metadata shares the existing bounded database. The new schema
is v2: older diagnostic v1 roots are preserved and rejected, never silently
migrated/reset. The annual acquisition catalog is separate and unchanged.

Tests cover exact idempotent reuse, distinct source/config/type-sensitive keys,
pending/unsafe/duplicate rejection, corrupt descriptors/payloads, metadata limits,
failed fsync/insert, and publication surviving abrupt process exit. Independent
review's typed-input equality finding was reproduced and fixed using typed JSON
comparison of the full descriptor. 59 focused resource/publication tests passed;
the formatted full core suite passed **643 tests in 187.84 seconds**, with 22
existing calendar/NumPy deprecation warnings. `git diff --check` passed.

A separate real-file reuse probe copied the three measured projections into a v2
diagnostic cache, published/reopened them, and verified exact repeated reuse with
unchanged accounting: 62,378,922 retained bytes +33,554,432 metadata allowance,
zero pending bytes. Both diagnostic copies remain on disk (124,757,844 payload
bytes combined); this is not a claim that only one copy exists. No remote request
or source deletion was used for either probe. Report:
`.hyperliquid_cache/publication_reuse_1788974542756889163/summary.json`.

## Qualified production day builder

`qualified_day.py`, `day_projection.py` and `derived_day_builder.py` now select
all overlapping qualified source files, preserve native identity/order and numeric
types, split oversized days deterministically, and publish complete days through
the shared resource ledger. Source and engine pins are reverified on both builds
and cache hits. Failed outputs stay charged and unpublished. Only newly owned,
empty query scratch is removed after handles close; old caches are preserved.

The independently reviewed implementation passed 20 focused tests. After scoped
formatting, the full core suite passed **663 tests in 251.09 seconds**, with 22
existing calendar/NumPy deprecation warnings.

Real June 3, 2025 acceptance retained 675,938 rows in four partitions totaling
31,917,099 bytes. Build time was 3.651 seconds; overall acceptance took 5.262
seconds with process peak RSS 488,368 KiB. Production queries used 256-MB DuckDB
allocation and a shared-ledger-reserved 2-GiB spill ceiling; RSS is not that
allocation limit. A bounded 10,000-fill wallet comparison matched reference
metrics and exclusions exactly. That separate diagnostic reference query set
256-MB allocation but did not explicitly set its own spill ceiling. Reopening
and reusing the production publication added no charge: zero pending bytes,
31,917,099 retained bytes and 33,554,432 metadata allowance. Source pins and the
frozen acquisition engine verified.

Report: `.hyperliquid_cache/qualified_window_1788980574597207367/first.json`.
The full June 3–September 1 exclusive 90-day projection check completed in the
same cache, reusing the first day. Its 358 output files retain **58,085,358 rows**,
matching the earlier physical-row capacity audit for this window, totaling
**2,761,887,362 bytes (2.57 GiB)**. Accounting after close is zero pending bytes
and 2,795,441,794 total bytes including the 32-MiB metadata allowance, inside the
existing 8-GiB envelope. Total elapsed time was 1,024.799 seconds; peak process
RSS was 569,348 KiB. Reopen/reuse added no charge and the frozen acquisition engine
verified. No additional source downloads were used. Full report:
`.hyperliquid_cache/qualified_window_1788980574597207367/window.json`.
This passes full-window projection/storage acceptance, not the all-wallet metric,
ranking, reusable-feature or scheduled-reader gates.

The next scoring subsystem now has an independently reviewed plan. Its initial
`disk_metric_rows.py` writer streams candidate metrics into capped Parquet without
an all-candidate list. It rejects lossy/nonfinite numeric conversions, bounds
records/batches/row groups and leaves failed writes charged and unpublished.
20 tests passed after RED/GREEN development, including 100,001 generated records;
the formatted writer/resource/publication group passed 68 tests in 2.59 seconds.
The subsequent full core suite passed **683 tests in 230.34 seconds**, with the
same 22 existing warnings. A sandboxed API regression stalled after eight tests
and was terminated (exit143); it is not a passing result. The isolated API test
file then passed outside the sandbox:7 tests in5.87seconds,2 existing deprecation
warnings. The full outside-sandbox API rerun passed: **45 tests in90.62seconds**,
with the same2 deprecation warnings.
Independent review found no actionable issues in this primitive. Duplicate-user
validation, causal universe provenance, ordered-config publication identity,
percentile scoring and integration remain pending. This writer returns only a
settled unpublished token and must not be treated as a qualified ranking result.

## Disk score and cohort query primitives

`disk_score_query.py` and `disk_cohort_query.py` implement eligible-only exact tied
percentiles, ordered Python score accumulation, disk ordering, cohort sizing,
selected weights and streaming RANKING_SCHEMA output. Independent review found no
actionable blockers in these primitives.26 tests passed after RED/GREEN development,
including exact reference comparisons and100,001 generated candidate records.
One test-fixture isolation error was corrected without weakening the cache guard;
an oversized integer weight now produces an explicit validation error rather than
OverflowError. That formatted full core regression passed709tests in252.02seconds
with22 existing warnings.

A synthetic capacity probe ran both stages sequentially under one shared lease
with pre-reserved2-GiB scratch and512-MiB capped outputs. It retained all100,001
ranking records (99,900 eligible), selected five, and reopened the diagnostic
publication without extra charges. All three payloads remain:1,499,690 bytes total,
zero pending,35,054,122 bytes including metadata allowance. Wall time3.263seconds,
peak process RSS261,348KiB. Highly repetitive synthetic values compress well;
these bytes/time are not a real annual sizing estimate or a performance result.
Report: `.hyperliquid_cache/disk_scoring_probe_1788991572992396808/summary.json`.

## Shared-budget scoring orchestration

`disk_cohort_scoring.py` now wraps the two passes in one held resource lease,
reserves spill/output capacity before use, and publishes immutable scores and
cohorts separately. It requires a published metric input with matching context
and a verifier returning the currently verified upstream provenance. Source pins,
ordered scoring terms, fee interpretation, eligibility parameters, scope and
derivation engine are checked before/after work and reuse. Cadence alone does not
duplicate identical decisions; cohort-only changes reuse scores.

`disk_score_result.py` reconstructs counts and at most250selected records via
bounded Parquet scans, with no new SQL sort on final cache hits. Returned selections
are fresh copies of immutable internal JSON; complete ranking batches verify file
identity before/after. Failures retain pending/settled obligations; only verified
new empty scratch is removed.15 wrapper tests passed after RED/GREEN development,
including new-lease reopening, corruption, source mutation, small shared budget,
empty results, reordered terms and effective v2 xyz:GOLD configuration. Independent
review found no actionable blockers. The formatted full core suite passed
**724tests in210.62seconds**, with22 existing warnings.

A synthetic100,001-candidate probe used the production wrapper, retained all rows,
selected five from99,900eligible records, and reopened without new charges.
All three payloads total1,499,690bytes, zero pending; total accounting35,054,122bytes
includes32MiB metadata. Wall time3.761seconds; peak process RSS401,052KiB. The final
ranking hash matches the earlier query-only probe. Report:
`.hyperliquid_cache/scoring_wrapper_probe_1789022765381590976/summary.json`.
Both synthetic diagnostic copies are retained; neither is an annual storage or
performance estimate. Frozen acquisition engine verified at probe completion.

Upstream all-wallet causal metric generation, reusable episode/daily features,
native-query integration and actual saved annual comparisons remain incomplete.
The wrapper's verifier contract does not establish those facts on its own.

## Archive acquisition stop (September10)

The original archive worker session37471 terminated with exit1:
`ValueError: inconsistent market trade counterparties`. No automatic retry or
new download was started. The catalog has24qualified batches (through index23,
source end2025-11-17),25raw/normalized/compact batches; batch25/index24 covers
2025-11-17–2025-11-24 and remains unqualified. Its original payloads are retained.
Lifetime reservations total103,839,183,349bytes against the unchanged300GiB cap.

The failing legacy check groups by `(coin,tid)` across time. Hyperliquid's official
[WebSocket trade schema](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions)
states that tid is a50-bit hash and a globally unique trade key also includes block
time. The new batch has no internal `(coin,tid)` economic conflict across all32
bounded diagnostic partitions. Source pins verified before/after; report:
`.hyperliquid_cache/counterparty_probe_1789022913383129842/summary.json`.
The cross-prefix diagnostic completed (session36173, exit0), confirming BTC
tid840478295334001 appears on both November5 and November23 with different trade
economics but consistent counterparties at each timestamp. A bounded raw read
also confirms the November23 record. The full timestamp audit (session83953,
exit0) scanned131,853,494rows: every fill time is millisecond-aligned, every
block-format row has envelope time, and all93,785,698envelope timestamps agree
with fill time when floored to milliseconds. Both diagnostics verified source
pins before/after; neither is corrected-engine requalification.
See `docs/hyperliquid-trade-identity-incident-2026-09-10.md` for exact evidence.
At diagnosis, no frozen code, ledger or qualification certificate was modified
to bypass the failure. Following explicit approval, the old engine and evidence
were preserved in `engine_transition_v2/manifest.json` (SHA256
`46b001b8a5f713a0e5549ca51f2ec822a776bc1307477d37fee242b671230ef8`).
The timestamp-aware correction is now implemented and independently reviewed:
753core and45API tests pass. Post-change verification confirms preserved evidence,
unchanged spending and expected rejection of the old job by the changed engine.
The activation bridge is now implemented and independently reviewed. A broader
core run passed770tests before the final one-line durability correction; all18
transition tests and45API tests passed afterward. Real offline baseline
activation completed as session4629, exit0:121,698,646rows across32partitions,
7067.621seconds including final verification. The immutable activation and
baseline pins are recorded in the incident report. Offline failed-batch recovery
completed as session70687, exit0:131,853,494rows,25qualified batches and unchanged
103,839,183,349reserved bytes. It disposed336job-owned staging payloads totalling
5,956,874,372bytes, preserving canonical history and manifests. Remaining
acquisition worker72925 has started under the unchanged300GiB lifetime cap;
its start is not yet evidence of another completed batch. Do not duplicate it.

Next: causal rolling iteration and reusable episode/daily features, disk all-wallet
scoring, scheduled integration and actual annual weekly/daily saved comparison.
Raw projections alone do not satisfy those requirements.

### September11 continuation checkpoint

Read-only inspection of the live job catalog now shows26qualified batches and
27raw/normalized/compact batches out of67. Original session72925 is still live;
batch26(zero-based) is qualifying. Lifetime reserved bytes are111,360,642,207 of
322,122,547,200; this ledger is not an AWS invoice. Cumulative approved staging
disposal is8,738files/125,503,624,723bytes, preserving canonical data. The latest
completed batch added336disposed files/4,576,678,155bytes. No download refund or
retry was introduced.

Native-query regression session7056 passed128tests in254.42seconds, including
hourly exposure, positions, market queries and reference activity/conviction.
Shared directory-descriptor guards and final post-source caller-pin checks passed
their mutation regressions and final independent review. These modules are not
yet routed through the annual scheduled runner.

Original real metric probe70326 remains live at partition planning with no
completed scoring result. A prior direct process sample showed2h55m44s elapsed,
2h48m22s CPU and404,628KiB RSS for child1858733. Code inspection identifies a
full filtered Parquet count/min/max query for each radix-prefix planning node.
This is evidence of a serious planning cost, not completed annual capacity proof.
An isolated generated-data experiment compared the exact partition sequence
against one bounded SQL user-count aggregation followed by prefix queries over
those counts: session95278 produced identical286leaves/305prefix queries for
1million rows, reference1.358207s versus0.123018s including aggregation(11.04x).
This is not a measured full-backtest speedup. The separate counted planner now
has16passing tests and independent review; post-format new/existing planner
regressions passed36tests in35.22seconds(session59109). It is not yet integrated.

Subsequently original probe70326 emitted plan_complete at11260.456seconds:
1786partitions,1453nonempty,58085358physical rows. It remains live, now proceeding
through metrics; no complete scored cohort has been emitted. Frozen live-engine
files and original workers remain untouched. Acquisition72925 has reached3/32
validation buckets for batch26(zero-based),13684312rows. Fresh acquisition-engine
hash still matches12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

### Disk cohort selection/report integration checkpoint

SelectionState.advance now accepts ScoredCohort with an installed RankingSink.
The new selection_disk_evidence adapter streams complete verified batches of
at most4096rows, preserving excluded/unselected evidence and retaining at most
250selected rows in Python. It checks source decision/scope, selected metadata
and complete counts before committing the current trader cohort. Legacy list
selection, pooled/per-asset transitions and market exits retain their behavior.

Post-format session33986 passed58tests in66.17seconds, including100001candidate
streaming, real SelectionState rotation parity, corruption/interruptions and
existing scheduled/weekly/report/scoring regressions. Independent narrow review
found no material issues. The live acquisition hash remains unchanged.

Empty ScoredCohort currently carries no verifiable decision/scope. The adapter
explicitly rejects it, rather than assigning requested context to unbound data.
After live probe70326 is terminal, the scoring result/wrapper must gain verified
immutable context on both publication and reuse; proper empty-cohort acceptance
and wrong-context rejection are mandatory before the annual scheduled route.
Bounded preview and the scheduled native/candidate reader remain unimplemented.
Current live metric probe has written40partitions at11937.635seconds; source
worker72925 has reached5/32qualification buckets of the27thbatch. No annual
score/result or full reusable-feature acceptance is claimed.

### Selected native-position batching checkpoint

The actual proxy `_signals_at` call site now uses `selected_positions`: at most
250distinct selected users, one `positions(users,coin,decision)` call per nonempty
market for a batch-capable reader, and legacy scalar calls only when no batch
capability exists. Failed batches are never retried as scalar queries. Exact
user-key coverage and finite numeric/None values are required; missing rows are
not silently interpreted as unknown/flat. Cohort input order is preserved.

Initial session60383 had14expected missing-helper/old-call-site failures.
Session97915 passed26tests; extended session50156 passed16focused tests including
actual qualified native queries versus full-reference signals/contributions for
direction and conviction weighting. Post-format session25949 passed39tests in
13.95seconds across the new helper, proxy pipeline, weekly strategy and scheduled
activity. Independent read-only review found no material issues. The acquisition
engine hash remains unchanged, and git diff --check passed.

This prepares the follower consumer; existing ScheduledActivity does not yet
expose the batch capability. The future qualified scheduled facade must delegate
it to native_positions under its shared lease and verified source context.
No claim of real annual query speedup is made from these fixture tests.

Original real metric probe70326 has now emitted metrics_progress at12630.074s:
5000wallets,2478951fills,71partitions,whale_fills0. It subsequently reached80written
partitions at12803.557s. The6,662,648fill whale and complete108185candidate scoring
are not yet completed. Original acquisition72925 reached7/32validation buckets
for batch26(zero-based),31916852rows, elapsed15881.124s. Both original handles
remain live; no worker restart, frozen-engine change or budget refund occurred.

### Qualified scheduled composition checkpoint

New QualifiedScheduledActivity composes the existing bounded native helpers and
complete disk candidate/scoring producer under a borrowed exclusive cache lease.
prepare establishes a monotone whole-hour cutoff without opening full-prefix
activity or checkpoints. Native positions are batched; lagged volume and hourly
conviction queries cannot exceed their prepared decision. Effective strategy
settings, fee semantics, source pin and helper engines are checked on each call.
Only successfully verified preparation commits a new cutoff; mid-query cutoff
mutation and reentrancy reject. Closing the facade does not close the caller's
lease or dispose of cache artifacts.

Post-format session44044 passed88tests in108.15seconds, including actual qualified
native/reference comparisons, disk ranking and cache reopen reuse, and actual
selection/report-sink/signal integration without a full-prefix constructor.
Two lifecycle regressions were observed RED before correction (56366cutoff
mutation,39180failed prepare commit); targeted20656passed13tests in20.49seconds.
Final independent review found no remaining material issues. The acquisition
engine hash is unchanged and diff check passed.

This class is not yet routed through annual dataset registration/API. Its exact
source-origin gate rejects the actual annual registered interior's later start,
pending explicit candidate/seed boundary reconciliation. Empty ScoredCohort
context and bounded preview also remain required. The underlying metric producer
still uses full lookback replay, not completed reusable daily/episode features.
No actual annual backtest is claimed from this composition test.

Original70326 is still live at140written partitions,13960.867seconds; its latest
metric count remains10000wallets/4366738fills, with the whale not yet processed.
Original72925 reached11/32qualification buckets for the27thbatch,50142620rows,
17262.858seconds. Neither worker was restarted; no paid-data or cache budget changed.

### Reusable metric semantics: measured counterexamples

Before implementing reusable daily/episode features, isolated diagnostic57771
completed successfully against current Python3.12.3 and current proxy validation.
Script `/tmp/hl_reuse_boundary_diagnostic.py`; retained fixture directory
`/tmp/hl_reuse_boundary_xf8w__lq`. No live corpus/cache or worker was modified.

1. A three-fill history (open before lookback, another observed open from flat at
   the boundary, then close) passes ProxyActivity registered-scope validation.
   Proxy validation checks each fill's position delta, not continuity between
   separate fills. Filtering complete global episodes by opened_at>=lookback
   start yields0episodes. Rebuilding the window yields1complete episode,
   PnL2.0,2fills,60minutes. Do not silently impose strict position continuity to
   make the cache shortcut appear valid; the current proxy mode permits this
   input. Boundary evidence/replay must preserve the reference's local reset.
2. Ordered fill PnLs [1e8,1e-6,1e-6,-1e8] total0.000002; summing two cached daily
   totals gives0.000001996755599975586. Day-level totals are not a lossless
   replacement for fill-order total PnL, even with Python3.12 compensated sum.
3. Notionals [1e12,0.0001,0.0001,0.0001] total1000000000000.0002; summing two daily
   totals gives1000000000000.0004. With the latter as the minimum-notional
   threshold, the reference excludes the wallet but daily regrouping admits it.
   This changes eligibility, not merely a tolerated display decimal.

Durable characterization tests are in test_reusable_metric_semantics.py; their
initial run67588 passed3tests in0.56seconds. These tests constrain future cache
work, not a claim that a reusable feature cache has been implemented.
After exact-file formatting, session43977 passed19tests in2.82seconds across
those characterization tests and the existing streaming-wallet metric suite.

Next design direction under the approved disk-ranking spec: preserve the ordered
per-fill numeric observations required by reference totals, and cache episode/day
evidence only with explicitly verified boundary behavior. A compact numeric
projection can avoid repeated raw fee/position decoding, while reusable contained
episodes can avoid rebuilding all episode state. The first intersecting episode
per wallet/coin requires local reconstruction when the observed state demands it;
pooled daily PnL must retain native inter-coin order. Do not replace either by
prefix subtraction or regrouped floating totals without equivalence proof.

Before adopting that layout, measure its complete annual retained/scratch/file
metadata footprint within the existing shared8GiB cache and256MB/2GiB query
limits. Preserve deterministic native ordering, fee semantics, all candidates,
and publication/reuse verification. No new resource allowance, sampled ranking,
weaker eligibility rule or annual-performance claim is authorized by this note.
Latest original-handle polls:70326 reached200written partitions at14957.569s;
72925 reached13/32qualification buckets of batch26(zero-based),59263190rows at
18038.067s. Both remain live; the full ranking and annual dataset are incomplete.

### Verified empty scored-cohort context

Implemented bound_scoring_context.py without editing the live scoring engine.
The binder reads bounded indexed publication descriptors and verifies the chain
cohort_rankings -> candidate_scores -> candidate_metrics through existing artifact
lookup. It checks requested decision, scope, metric settings, selection, engine
and final artifact identity, then rechecks dependencies and request before return.
This binds an internal producer result, not independent source qualification.
QualifiedScheduledActivity now binds its directly produced result; the selection
adapter accepts a bound empty cohort and records cash/previous exits, while still
rejecting unbound empty artifacts or mismatched decision/scope.

Session46325 observed17missing-module RED failures. Session15016 passed61tests
in80.92seconds across binding, qualified scheduled activity and selection evidence.
Independent read-only review approved with no material findings. Exact-file
formatting completed; post-format regression session79438 passed76tests in91.28s,
including the existing disk scoring suite. The frozen
acquisition engine hash remains12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541;
git diff --check passed. No source/cache budget or worker restart occurred.

Original metric session70326 reached15000wallets/7687259fills,257partitions at
15933.345seconds, whale_fills0. Original acquisition72925 reached16/32validation
buckets for batch26(zero-based),72940894rows at19185.78seconds. Both remain live.
The annual source boundary contract, bounded previews, reusable metric performance,
complete acquisition and actual saved weekly/daily annual results remain required.

### Disk-backed scheduled previews

Implemented disk_preview.py with a filtering RankingSink: selection consumes and
verifies all historical scoring batches but only the requested decision/scope is
written. Complete candidate/exclusion evidence is retained; market/trader cohort
history remains available without accumulating prior ranking rows. Actual empty
cohorts remain actual, and hypothetical previews do not mutate selected membership.
The scheduled API worker now creates a scratch RankingFile and uses existing
publication/copy/pagination. Legacy non-scheduled previews remain unchanged.

Session32910 observed21missing-module RED failures. First implementation3608
passed20tests; the large test's requested INDEX market was exiting and correctly
returned0rows. Corrected its fixture target to active BTC, with65296 passing2tests
in14.87s, including100001excluded candidates retained in<=4096row sink calls and
corrupt discarded historical source rejection. API61300 observed2expected old-list
route failures;30877 passed2actual/midweek publication+pagination tests in3.68s.
Additional34922 passed2requested-source/output interruption tests in3.88s.
Independent review approved with no material findings. Exact-file formatting
completed; combined post-format regression19440 is still running. Acquisition
engine identity is unchanged and git diff --check passes.

Original metric70326 reached20000wallets/9433304fills,269partitions at16412.442s
and280written partitions at16580.874s; whale_fills remains0. Acquisition72925
reached17/32validation buckets for batch26(zero-based),77504632rows at19518.52s.
Both original handles are live. No source restart or budget change was made.
This closes preview artifact plumbing, not the outstanding annual dataset reader
registration/source-boundary integration or the reusable ranking performance gate.

### Preview regression completion and registration seed evidence

Combined sandbox regression19440 stopped advancing at the first existing HTTP
integration test after the core suites. Read-only process inspection confirmed the
test process remained live; it was explicitly interrupted (terminal exit130), not
silently replaced on an observation timeout. Separate post-format core84189 passed
55tests in95.23s. API9876, run outside the restricted sandbox with a300s limit,
passed all14tests in80.72s, including synthetic annual weekly/daily save/clone/
comparison and both new scheduled preview publication/pagination cases. Its45s
diagnostic stack dump showed normal annual completion polling; it then finished
successfully. Two existing dependency deprecation warnings remain. No production
worker was interrupted. These are fixture integrations, not real annual results.

New test_registration_seed_semantics.py characterizes the source-boundary issue
using actual publish_interior and ProxyActivity. One wallet opens before retained
coverage and is then dormant; another trades in the retained interval. Both full
and cropped files validate in their respective declared scopes. Full source has
two candidates and a known open dormant position; the interior has one candidate
and returns unknown for the dormant position. The dormant source row is explicitly
ineligible with no_activity_in_lookback, and the other wallet's ranking row is
identical. Therefore retaining seed context must not extend metric lookbacks or
drop dormant candidates. Initial characterization passed1test in0.18s; post-format
session71276 passed11tests in0.32s with the existing partition projection suite.

The registration path currently projects fills to metric warmup coverage and
opens registered_activity/ScheduledActivity, not QualifiedScheduledActivity.
A verified source-origin seed contract and final-path qualified-reader/cache
integration remain necessary; do not simply relax the facade's coverage gate or
label an interior projection as preserving the all-observed-source universe.

Latest original-handle polls:70326 has25000wallets/11618543fills (whale_fills0)
and420written partitions at18938.306seconds. Acquisition72925 has25/32validation
buckets for the27thbatch,113984820source rows at22112.786seconds. Both remain live.

### Explicit qualified source-seed policy

Reviewed qualified-registration plan is in
docs/superpowers/plans/2026-09-11-hyperliquid-qualified-registration.md.
It keeps the original qualified archive as a visible durable dependency and lends
an explicitly supplied existing cache, not a newly allocated second8GiB envelope.
Review corrected a proposed publish_interior(S,E) shortcut: source-date bounds
are not event bounds, and re-filtering would change canonical pins. Registration
must use unfiltered bounded retain-all copying/hardlinks with spill regressions.

First unit implemented: QualifiedScheduledActivity accepts only the explicit
qualified_source_seed_v1 policy for padded interior coverage. The original source
origin is retained for candidate/position seed queries, while finish is registered
coverage end and metric warmup must fit registered coverage. Coverage/policy join
the frozen request context. Default source-origin equality is unchanged. This is
not yet the registered loader or a new source-completeness claim.

New test_qualified_seed_policy.py uses an actual qualified4day archive with one
wallet observed only on day1. It compares disk candidate membership and native
positions with the full-source reference, preserves the dormant exclusion, and
checks policy/padding/warmup/end/mutation rejection. Session2596 observed10expected
unexpected-keyword RED failures and1existing default guard pass. Session76081
passed39new/existing facade tests in59.17s. Independent review approved this unit.
Exact-file formatting completed; post-format session7799 passed56facade/binding
tests in69.83s. The acquisition engine hash remains unchanged and diff check passed.

Original full-scale probe70326 emitted whale_complete at20629.324seconds:
all6662648distinct fills, numeric spool52635942bytes. Its ordered whale artifact
was504504472bytes, within the512MiB writer envelope. Later progress reached
35000wallets/23521729fills at21362.292seconds and560partitions at21399.309seconds.
This proves oversized-wallet traversal completed; complete108185candidate scoring,
final RSS/accounting and annual throughput acceptance are still pending.

Acquisition72925 completed batch26(zero-based),27of67qualified batches, with
145882896physical prefix rows and report
batches/0026/qualified/qualification_h5a1vt_x/manifest.json,
sha256ea9328d08b1e6ffd572827a421deffd9c189c297bd510e2faf6d6a96b46b37b9.
Lifetime reserved bytes remain111360642207. Approved cleanup cumulative totals
are9074files/130287015288bytes; the newly completed batch adds336files/4783390565bytes.
Canonical content remains; original raw recovery requires another copy or paid
redownload, without budget refund. Original handle status must be polled before
assuming another batch or completion; never launch a duplicate from this log.

## Qualified registration integration continuation (2026-09-11)

Implemented explicit qualified_v1 source/copy binding verification and an owned
existing-cache loader. Cache references require the initialized identity and the
existing 8GiB envelope; no implicit cache creation. Initial verifier tests passed
15 (session16708), cache reference tests passed7 (15011), and owned loader tests
passed5 (98917). Dataset routing/rename/full-source reference and matched
weekly/daily reuse tests passed2 (37604,21.76s). These are fixture checks, not real
annual acceptance or throughput evidence.

Review exposed stale on-disk manifest metadata at close. Actual file-edit tests
failed as intended before load and before close (95209,2failures). Fresh structural
manifest checks were added; loader/dataset integration then passed9 in115.29s
(59838). A second injected edit during the final full source scan also reproduced
the missing final recheck (1084); a post-scan fresh check is now under regression.
Lease release remains unconditional after a verification failure.

Added retain_full_source, preserving original canonical bytes including boundary
spill, plus optional cache_reference on register_annual_dataset. None preserves
the legacy sharded path. Qualified publication retains original archive paths as
declared durable dependencies and borrows the explicit cache for stage validation.
No original source, shared cache or spend ledger was removed/recreated. Seven new
retention tests first failed for the missing function (78788), then17new/legacy
partition tests passed (82523). Publication test fixture initialization errors
were corrected before observing the actual missing-keyword RED (55391). The new
publication/rename/reopen/disk-backtest fixture passed1 in8.44s (3737).

Review found fallback copying could exceed pinned bytes if a source grew. The
injected growth regression reproduced1119bytes written against1089pinned (1084).
Copy fallback now caps reads/writes at the pinned allocation and checks one extra
byte for overflow; final content rechecks remain. Combined regression86964 is
running; do not infer its result from this entry. Exact-file formatting occurred
before these two final review fixes; post-fix formatting/regressions remain.

Acquisition engine hash verified unchanged at
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

## Historical fallback lifecycle acceptance — 2026-09-15

Postformat receipt/scheduled suite48763 completed61passed in144.58s. Extended
historical test34581 completed2passed in26.08s: both scopes, independent reference
cache, native positions, fresh-lease weekly/daily receipt reuse with both producers
forbidden. Initial mixed BTC/GOLD active-market test93279 passed1 in8.20s.

Read-only integration review found no production correctness blockers but asked
for forward advancement beyond the existing cutoff, independent market reference
caches, pending-recovery/rejection and capacity acceptance. Updated the forward
test to advance one hour beyond the prior cutoff, and moved market references
into separate temporary caches. Added test_historical_retirement_order.py:
22921 passed2 in10.38s for prepared/detached recovery completing before raw
producer invocation;81239 passed1 in5.44s for exhaustion preserving saved evidence.

Combined60950 completed18passed/1failed in84.43s. Failure was fixture setup:
the corrupt-anchor case attempted a successful resources.audit after deliberately
corrupting a retained payload. Replaced that assertion with direct allocation and
publication ledger snapshots around the expected rejection. Production guards
were unchanged. Three test files formatted; combined20test suite62603 running.
Narrow final review requested. Annual resource sufficiency and actual saved annual
weekly/daily runs remain unproven; these are temporary-fixture checks only.

Final formatted lifecycle suite62603 completed20passed in101.53s, including the
corrected corrupt-anchor rejection and all recovery/capacity/market/continuation
cases. Narrow read-only Task3 review approved this acceptance coverage, without
claiming annual capacity or authorizing real expiry. Registration/annual-publication/
owned-activity/API-worker regression59397 is running. No production changes were
made during this turn; test coverage and documentation were expanded. Archive81924
is still confirmed live, latest output23/32buckets on batch0049.

## Retained ranking intermediate footprint — 2026-09-15

Read-only SQLite transaction against the existing shared research cache found
495 retained allocations,5,286,350,121bytes and no pending allocations. With the
32MiB metadata allowance, the approved16GiB cap leaves11,859,964,631bytes.
This is a ledger/Parquet-footer observation, not a complete integrity audit.

Distinct artifact bytes by publication kind (shared artifacts would be counted
once per kind, so categories must not generally be summed):

| Kind | Artifacts | Bytes |
| --- | ---: | ---: |
| qualified_features_day | 385 | 5,186,454,160 |
| cohort_rankings | 5 | 25,699,876 |
| candidate_scores | 5 | 25,139,498 |
| candidate_day | 91 | 24,190,201 |
| candidate_metrics | 5 | 19,997,131 |
| candidate_history | 3 | 4,868,152 |
| approved_cache_expansion | 1 | 1,103 |

Three large existing variants each contain108,185candidate rows, with identical
measured file sizes: metrics6,445,897bytes (59.582bytes/row), scores8,145,258bytes
(75.290bytes/row), final rankings8,323,840bytes (76.941bytes/row). The two small
variants contain6,421rows and compress differently. These are observed historical
files, not an annual compression guarantee or a fresh computation.

Re-read annual_row_bound_20260915/result.json: daily BTC candidate rows lower
bound114,133,907; weekly16,521,928. The old result's existing_run_row_limit and
daily_lower_bound_exceeds_limit fields describe the former limit, not the current
approved250m ceiling. Applying the large-file observed bytes/row illustratively
to the daily row bound gives metrics6.800GB, scores8.593GB, rankings8.782GB,
combined24.175GB before features/checkpoints/journals/scratch. This byte estimate
is NOT a proven lower bound: annual compression, later entrants and other markets
are not measured. It is strong reason not to assume retaining all intermediates
fits16GiB. Removing only metrics/scores would still leave the growing standalone
ranking payloads plus rolling features and old cache contents to account for.

Next capacity design decision: consider exact receipt-backed retirement of
recomputable metric/score intermediates, preserving full saved ranking payloads,
candidate history and canonical sources. Existing standalone receipt loading
already avoids intermediate dependencies; existing producer rebinding does not.
Any cleanup must therefore verify consumer ownership and crash recovery rather
than simply delete files after scoring. Completed retirement receipts must not
silently allow logical-key resurrection on a later new hypothesis. No cleanup
implementation, real deletion or additional cache allowance is authorized by
this measurement. Annual final-source capacity and real-target authority remain
open. Archive81924 latest progress24/32buckets,239,701,270source rows;49/67batches
remain confirmed complete. Registration/API suite59397 is still live.

## Historical integration regression complete; next staging design — 2026-09-15

59397 terminal0:37passed in659.10s across qualified annual publication,
registration, owned registered activity and API worker. Together with final61-test
receipt/scheduled and20-test lifecycle results, the historical fallback's reviewed
fixture regression gate is complete. This is not annual backtest completion.
All four pinned cache module hashes were rechecked unchanged.

Storage design review rejected implementing published intermediate retirement
without additional generation/authority/crash/metadata-reservation machinery.
The superseded proposal is retained as decision history. Selected simpler direction:
unpublished, pending, invocation-owned metric/score staging, followed by full
standalone ranking publication and successful owned temporary cleanup. Existing
ScoreQuery/CohortQuery kernels accept paths. New explicit provenance verification
must replace intermediate-publication binding; no validation bypass is permitted.

The scoped design is
docs/superpowers/specs/2026-09-15-hyperliquid-temporary-ranking-staging-design.md.
It leaves default producers unchanged, accounts for simultaneous reservations,
preserves failed obligations and excludes automatic restart deletion. Narrow spec
review is pending; no staging implementation or real cleanup has been performed.
Original archive81924 remains live on batch0049, last output24/32buckets.

Temporary-staging spec review approved the revised unpublished approach. Reviewer
required explicit handling of crashes between reservation and manifest creation.
Implementation plan now uses conservative fixed upfront reservations:3GiB scratch,
three512MiB files (metrics/scores/final ranking),64KiB manifest, totaling4.5GiB+64KiB
above existing data. This is reserved capacity, not measured disk usage and not an
increase to individual stage limits. Candidate preparation precedes this envelope;
all existing pending work blocks new production, even with missing manifests.
Exact saved reads remain nondeleting. No automatic recovery is introduced.

Plan `docs/superpowers/plans/2026-09-15-hyperliquid-temporary-ranking-staging.md`
is under narrow review. No staging code exists yet. The conservative reservation
may itself constrain annual admission; it is not a capacity success claim.
Archive engine rechecked unchanged;81924 now25/32buckets,249,689,220source rows.
Only49/67batches confirmed complete. git diff --check passed.

Plan review identified creation-time ownership as essential: existing query writers
open/close outputs internally, so inspecting a path after return could adopt a
replacement. Revised plan requires writer-created FD capture before callbacks,
held identity checks, pre-capture replacement regressions and narrowly scoped
optional writer hooks with fingerprint consequences acknowledged. The all-pending
admission gate explicitly also blocks unfinished feature retirement until separately
authorized feature recovery is performed. Spec and plan updated consistently.
Final narrow plan review approved fixture implementation. No staging code or real
deletion has been performed; Task1 pending-invocation ownership is next.

## Temporary staging ownership implementation started — 2026-09-15

Implemented bounded ranking_staging_manifest codec and initial RankingStagingOwner
admission/FD verification in new modules only. The owner preflights the complete
manifest envelope, rejects existing pending obligations, reserves the fixed4.5GiB
plus64KiB, writes an immutable manifest, and captures actual created FDs before
writer callbacks. All temporary obligations remain pending. Closing releases file
descriptors only, never files or accounting obligations. Cleanup and final-ranking
publication are intentionally not implemented yet.

TDD:6735 terminal1,22 missing-codec failures; codec22green. Owner missing API
test run9failed, then combined91249 completed31passed in0.32s.75139 reproduced
oversized complete-manifest rejection only after allocation; added complete-envelope
preflight using fixed-length identifiers before reservations. Combined32passed;
after formatting four files, final32passed in0.34s. Cases cover malformed/oversized
manifest, invalid roles/bounds/paths, duplicate ownership, interrupted reservations,
context/manifest mutation and output replacement before/after FD capture.

Initial creation/verification code is under narrow read-only review before adding
cleanup. It is not wired into production, does not claim a complete staging
lifecycle, and has not run against real data. Archive81924 remains live; no cap,
existing cache file or acquisition-engine dependency changed.

Narrow ownership review found two verification gaps.99918 reproduced2failures:
late engine callback could mutate immutable manifest after its read, and pending
allocation rows with non-NULL bytes/hash were accepted. Moved engine callbacks
before manifest authentication, kept later FD checks callback-free, re-read bounded
manifest bytes last, and require pending state with NULL bytes/hash. Combined34
passed; final postformat34passed in0.39s. Narrow rereview requested. Cleanup still
not implemented or enabled. Archive81924 now26/32buckets,259,681,880source rows;
49/67batches remain confirmed complete.

## Pending metric writer and creation hooks — 2026-09-15

Continued with the independently testable pending-writer primitive while successful
cleanup remains unimplemented. New ranking_staging_rows reuses existing schema,
record, batch and capped-output validation. No temporary settle/publication/refund.
74217 RED10 missing-writer failures, then83430 GREEN44 staging tests in0.85s.

Added optional on_created(fd) hooks immediately after exclusive file creation and
before ParquetWriter in ScoreQuery/CohortQuery. Default callers and algorithms
are unchanged.69205 RED6 missing-hook failures;24983 GREEN32 hook/query tests in
3.81s. Archive engine does not include these query modules and was checked unchanged.
Scoring/receipt fingerprints do change, as intended. After formatting,68413 passed
91combined staging/query/published-scoring tests in6.05s.

Read-only review requested content pinning before final callbacks. A valid Parquet
same-inode overwrite during final owner.verify reproduced DIDNOTRAISE. Added
ranking_staging_artifact: bounded hashing from the actual completed writer FD with
full physical identity capture; writer now returns that immutable pin and checks
it after final owner verification.19519 passed11writer tests in0.49s. Final formatted
combined92-test run11480 passed in6.09s; narrow pin-boundary rereview approved.

No staging producer/orchestrator, successful cleanup or policy wiring exists yet.
Nothing ran against real cache data.81924 latest27/32qualification buckets,
269,668,162source rows;49/67batches confirmed complete.

Original72925 remains live: batch27 compact published and qualification reached
7of32buckets at28308.094s. Original70326 remains live:50000wallets/30436067fills
at24782.719s,760partitions at24914.321s. Neither job has reached annual completion.

Remaining integration includes API qualified-mode scheduled preflight, full
publication failure/busy-cache regressions and broader core/API checks. The API's
old min(fills,100000) ranking estimate is not a valid all-wallet capacity proof:
the actual first-decision candidate set already exceeds100000. Measure complete
per-scope/decision candidate output before annual daily acceptance; do not silently
raise the50million ranking-row cap or shrink the approved market/time scope.

Follow-up verification: session86964 passed27 loader/full-retention/legacy-partition/
qualified-publication tests in93.80s, including both injected review regressions.
API qualified-mode preflight regression47385 reproduced incorrectly ready=True for
an hourly config; scheduled-mode gate now covers qualified_v1 and sharded_v1.
Session30754 passed the regression in0.90s. Exact-file formatting completed after
the fixes. Acquisition engine hash remains unchanged and git diff --check passed.
Broad post-format registration/core/preflight regression53427 is live; preserve
that handle and do not edit production modules during its fixture publications,
which pin all package module bytes. Independent focused follow-up review requested.

## Qualified worker and publication failure acceptance (2026-09-11)

Previous goal turn changed implementation and verified the focused review fixes;
it was progress, not an idle status turn. Independent follow-up review confirmed
both findings resolved, with no remaining findings in that narrow scope.
Broad post-format regression53427 completed62 tests in347.21s (exit0).

New test_qualified_worker.py publishes a completed fixture archive through the
qualified annual registration path, then uses DatasetCatalog, typed submissions,
LabJobs, the worker and publication to save weekly/daily backtests and a cohort
preview. The old full-memory ProxyActivity constructor is forbidden. It verifies
ranking pagination, retained report evidence, scratch disposal, unchanged source
records and identical shared-cache accounting after the repeated decision. The
API comparison query returns two25point series with weekly/daily configuration.
This one-day fixture proves routing/publication/reuse, not a year of distinct
decisions or real annual strategy performance. Initial worker13112 passed1 in14.02s.

New qualified publication failure regressions cover a held shared-cache lease and
an injected late validation failure: no target or temporary registration directory
survives; source records/cache accounting remain unchanged and leases can reopen.
Session76237 passed2 in18.82s. After test formatting and comparison assertions,
combined worker/publication57043 passed4 in37.02s. No production code change was
needed for these additional acceptance checks. git diff --check passed.

Full API regression82861 is now live outside the restricted sandbox with an
explicit300s timeout and45s traceback diagnostics; HTTP TestClient is known to
hang inside the sandbox. It includes old proxy workflows and the new qualified
worker/preflight tests. Preserve its handle; do not infer completion from this log.
Original acquisition72925 and metric probe70326 remain live on their original
handles; no restart, retry, source deletion, ledger reset or new budget occurred.

Full API regression82861 completed16tests in85.09s with2 dependency deprecation
warnings (Starlette/httpx and AnyIO BlockingPortal). The45s diagnostic trace showed
the annual comparison fixture waiting on its worker, not a terminal test timeout;
the original suite continued and exited0. No duplicate runner was launched.
Updated the operator-facing lab guide to describe qualified source retention,
explicit existing cache references, durable archive dependencies and current
limits without claiming actual annual source/performance acceptance.

## Bounded replay scan reduction (2026-09-11)

Original probe70326 and acquisition72925 were polled and remain live. The current
probe engine is deliberately unchanged. Inspection confirmed source SQL queries
must close before per-wallet disk medians; a cursor spanning wallet calculations
would violate the sequential query envelope. Therefore the scoped optimisation
uses coalesced bounded temporary replay intervals, not concurrent source/median SQL.

Plan docs/superpowers/plans/2026-09-11-hyperliquid-coalesced-replay.md was reviewed
and approved. The exact counted radix planner remains unchanged. New independent
wallet_replay_groups.coalesce_plan validates1..5000complete contiguous descriptors,
then greedily combines small adjacent leaves up to the existing requested row
bound (maximum250000), retaining indivisible oversized-wallet leaves. Full address
coverage, physical count and address ordering remain unchanged; only internal
replay boundaries change. Empty coverage remains represented. It is NOT connected
to the live engine; integration and full metric/dedup equivalence tests must wait
until the original probe is confirmed terminal.

New32test suite observed missing-module RED50891. GREEN79076 passed48new/count-
planner tests in2.77s. Independent focused code review approved Task1 without
material findings. Exact-file formatting completed. Postformat10879 is running
the new/count-planner/old ordered replay suites; preserve its handle.

Generated address-only diagnostic /tmp/hl_coalesced_scan_diagnostic.py initially
showed264→98nonempty ranges,528→196source queries,2.40x on1million physical rows.
After replacing diagnostic grouping with the actual new helper, session35801
confirmed identical row stream hash
2b534ab2d312a70407d6abd2fbe11d7edfe7d142ebff3702a18f38117b9b4a47,
3.023763s→1.294821s (2.335x), same range/query counts. This synthetic address scan
does not prove economic dedup, metric equivalence or annual throughput. No real
source/cache read or mutation was involved; diagnostic fixtures are in /tmp.

Latest original70326 output:860written partitions at26371.241s (no terminal event).
Original72925: batch27 qualification11of32buckets,52071270source rows,29553.931s.
No live worker restart, resource increase, source disposal or reservation refund.
Reusable exact features and full annual acceptance gates remain outstanding.

Postformat10879 completed68tests in39.07s (new grouping/count planner/existing
ordered replay). git diff --check passed and acquisition engine hash is unchanged:
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

## Reusable metric boundary characterization (2026-09-11)

The previous goal turn implemented and verified an independent scan-reduction
primitive; it was progress. Both original job handles were revalidated live.
Frozen metric/acquisition production modules remain untouched.

Extended test_reusable_metric_semantics.py to investigate exact episode reuse,
not to install a cache. For each wallet/coin, replay only the window's prefix
through its first close or flip. At that boundary the post-event state agrees
with global episode construction; later complete episodes can be taken from the
global sequence provided their completion is strictly after that boundary and
strictly before the decision. A coin with no boundary inside the lookback has no
complete local episode and must not use a later global closure.

The characterization uses1200interleaved BTC/ETH fills, native-valid individual
positions with inter-event resets, both flip directions, fees, two fee semantics,
five overlapping one-day windows, and an optional ETH closure after all windows.
It compares the spliced observations against independent build_episodes(window)
and full streamed metrics/exclusions/counts/notional/volume. Ordered per-fill
numeric triples are retained; daily subtotal regrouping remains prohibited by
the earlier threshold/rounding counterexamples. Global derivation still runs only
inside this test: no persistent cache, eviction or production reuse is claimed.

Multiple closures can share one timestamp. A dedicated assertion shows a valid
cached suffix episode closing at the same timestamp as the local first boundary;
timestamp-only filtering would drop it. Durable cache design therefore needs an
exact stable completion-order identity, plus user/coin and source/engine binding.
Test-local enumeration ordinals are characterization aids, not yet an approved
cross-publication identity or durable cache schema.

Session17673 passed11initial characterization tests. Session79000 passed29 after
adding exact full-metric comparison. Session10181 passed39combined characterization/
streaming tests after adding the future-closure cases. Exact-file formatting
completed; broader postformat metric/spool regression is running. Focused semantic
review requested before using these observations to design a persistent cache.
These fixtures do not prove arbitrary global suffix correctness, incremental
checkpointing, capacity or annual throughput by themselves.

Original70326 reached55000wallets/33360831fills at26769.221s,881written partitions.
Original72925 reached12of32qualification buckets for batch27,56805426source rows,
elapsed29915.958s. Neither original handle emitted a terminal event.

Postformat93590 passed51characterization/streaming/spool tests in2.21s. Diff checks
and frozen acquisition engine hash check passed. Independent semantic review
supports boundary replay plus global suffix, subject to explicit canonical
dedup/order and fee-semantics binding, inclusion of the boundary fill/fee split,
strict decision cutoff, and bounded full-prefix streaming when no boundary occurs.
Before cache implementation add near-zero tolerance, empty windows, cutoff-equal
closures, multiple wallets and cancellation-sensitive episode-sum regressions.
For pooled metrics, retain wallet-wide cross-asset completion/numeric order:
per-wallet/asset ordinals alone cannot reconstruct the reference sum sequence.
No pending test process from this section remains; original70326/72925 stay live.

## Reuse edge cases and arithmetic contract (2026-09-11)

Refactored the test-only splice model into check_reuse so edge cases exercise the
same characterization, without adding production caching. Existing23cases passed
after extraction (79467). Added near-zero positions below/at/above1e-9, a closing
fill exactly at the exclusive decision cutoff, an empty lookback, and separate
wallets trading the same asset. Both fee conventions remain covered. All33reuse
characterizations passed45551 in1.50s; postformat metric/spool regression is live.

The cancellation case is intentionally asymmetric: episode.pnl uses incremental
`+=` and produces0.0 for1e16,1,-1e16, while Python3.12 `sum` of those fill values
produces1.0. Cached episode totals must preserve their original incremental
accumulator bits; per-fill window totals must retain ordered observations and the
reference compensated sum. Neither a common SQL sum nor a blanket compensated
episode sum is a reference-equivalent replacement. A cutoff-equal cached closure
must not turn the currently incomplete local episode into a complete episode.

The test model now explicitly accepts only one wallet; the multi-wallet case
compares separate wallet states with the independent builder's (user,coin) episode
keys. A persistent builder must isolate wallet state and preserve wallet-wide
cross-asset order. These remain characterization results, not persistence,
checkpoint-resume, capacity or end-to-end annual throughput evidence.

Latest live probe70326:65000wallets/37279256fills at28135.11s,960partitions at
28366.103s. Acquisition72925:17of32qualification buckets for batch27,
80483864physical rows at31386.302s. Original handles still live, engines unchanged.

Postformat1363 passed61reuse/streaming/spool tests in2.20s. Next implementation
unit documented in docs/superpowers/plans/2026-09-11-hyperliquid-episode-state.md:
a pure bounded lossless active-episode codec for chronological derivation. It
explicitly binds wallet, midnight cutoff and resolved fee semantics, including
empty state, and requires exact resumability/float-bit tests. No persistent
publisher, source-completeness assertion, deletion or budget expansion is included.
Plan review requested before implementation; the cache as a whole remains unbuilt.

Subsequent original70326 output reached70000wallets/38850004fills at28749.427s,
980partitions at28851.55s. Original72925 reached19of32qualification buckets for
batch27,89952406physical rows at32014.961s. Both original handles remain live.

Episode-state plan review found a genuine compatibility issue with the proposed
float-only accumulator validation: _episode can preserve an integer peak_notional.
New focused regression95020 passed1 in0.14s, showing a native typed fill with
px=2**53+1 produces an exact integer peak that float conversion would lose.
Corrected plan preserves exact int/float types, finite floats, bounded integers,
and requires future Parquet encoding to preserve that distinction explicitly.
Independent plan re-review approved; no material blockers remain for codec Task1.
No codec implementation or frozen reference change has occurred yet. Exact test
formatting completed; combined postformat metric regression is running.

Postformat33759 completed62metric characterization/streaming/spool tests in2.17s.
Diff and frozen acquisition hash checks passed. No test/format handle remains
live from this unit; original70326/72925 remain the long-running work handles.

### 2026-09-11: lossless active-episode codec verified

Implemented the independent episode_state.py encoder/decoder with strict wallet,
midnight cutoff and resolved fee-policy binding, including empty checkpoints.
It retains exact integer/float accumulators and signed zero, bounds shallow
structures before serialization, and restores fresh independent episode objects.
No source-consumption certificate, persistent publisher or cache allocation is
implied. No frozen acquisition or running metric engine module was edited.

Initial missing-module RED60213 had47 failures; initial implementation passed47.
Midnight resume tests cover both fee conventions, empty state, position resets,
flips, interleaved assets and cancellation-sensitive incremental episode PnL.
Additional extreme UTC conversion tests produced RED57102:2 failed/48 passed,
both because datetime.astimezone raised OverflowError instead of the codec's
validation ValueError. The minimal local timestamp wrapper now translates that
overflow, without changing shared contracts or reference arithmetic.

GREEN40680 passed112 codec/reusable-metric/streaming/spool tests in2.28s.
Independent read-only review approved with no material findings. Exact-file
formatting completed; postformat37282 passed112 tests in2.24s. git diff --check
passed and the acquisition engine remains
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

Original acquisition72925 was confirmed live at22of32 qualification buckets for
batch27,104160714 physical prefix rows. Original scoring70326 was confirmed live
at1040 written partitions,29706.766s; latest wallet progress remains70000 of
108185 candidates. Neither job has been restarted or declared terminal.
Persistent feature publication, rolling retention, real throughput acceptance
and the actual annual four-market weekly/daily comparison remain outstanding.

### 2026-09-12: first-decision real probe terminal; day-feature core initial tests

Original70326 returned exit0. Its complete event reports108185 wallets,
58085358fills,6662648whale fills and1453 written nonempty partitions. It published
ranking key e2d5e59d39ffc36ff0b25104c1405e616dbf2c2c850b514340be669da8a441da,
artifact artifacts/de54fd21cb04460bbbc0e2eeaf17cdaa.parquet,8323840bytes,
SHA408f4ce50b71e4cefed22af7b4ba5ec7325170e83311a0d1bc678350411f3fbd.
Result iteration/count assertions and same-input reopen/accounting assertions
passed. Final accounting: reserved4294967296bytes, retained51828150bytes,
metadata33554432bytes, total4380349878bytes. The old4GiB pending obligations remain
charged; this terminal result does not authorize refund or deletion of them.

External RSS guard exited0, observed peak1158135808bytes below2147483648 limit,
limit_triggered=false, elapsed36465.701s (about10.13hours). Internal ru_maxrss
1139496KiB. This proves the first full candidate computation fits its memory
envelope, not acceptable annual throughput or a research return. The running
metric-engine freeze can now lift for the reviewed counted/coalesced planner
integration; acquisition engine remains frozen. Do not rerun original probe.

Acquisition72925 remains live. Latest output is zero-based batch33 (34th batch),
raw/normalized/compact published and prefix qualification3of32buckets,
17159824physical rows at90581.063s. Earlier explicit batch27 completion recorded
28of67qualified and114527645473lifetime reserved bytes; subsequent batches have
advanced, so that spending figure is not current. Continued approved cleanup
removes only qualified job-owned raw/normalized payloads, retaining canonical
data; raw recovery requires another existing copy or redownload, no ledger refund.

New wallet-day feature plan received independent read-only approval. Added
wallet_day_features.py and test_wallet_day_features.py after RED29053:14 expected
missing-module failures. GREEN64635:14passed1.68s. Exact-file formatting completed;
combined postformat59385:126passed4.44s (dayfeatures,codec,reuse,streaming,spool).
Includes three-day two-market resume under both fee modes, original numeric
values/full native ordering, immutable observations,100001fill lockstep streaming,
source/sink failure propagation and unchanged input checkpoint.
Remaining plan validation cases and focused code review are not yet complete;
Task1 is not declared complete. No persistent feature publisher or reader exists
yet. The frozen acquisition engine hash remains12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

### 2026-09-13: counted/coalesced planner integrated; measured comparison launched

After original70326 terminal success, implemented the reviewed planner Task2 in
ordered_wallet_partitions.py: aggregate physical per-wallet counts once, obtain
the exact bounded radix plan, coalesce adjacent bounded ranges, then construct
the existing AddressPartition descriptors. Query/source checks and cached-bound
rejection remain; both helper files are hashed in ordered-reader context and
candidate_metric_producer._engine. Removed now-unused local _bounds helper.
No larger row, scratch, SQL-memory, writer or cache allowance was introduced.

New test_replay_planner_integration.py observed RED54101:3 failures for old ranges
and missing helper pins. Full GREEN48569:103passed71.32s across new integration,
ordered replay, candidate producer, counted planner, coalescer and disk scoring.
Independent read-only review approved without material findings. Exact-file
formatting followed; postformat83794:103passed72.40s. Acquisition engine hash
checked unchanged and git diff --check passed. Existing tests cover full native
dedup/order, cross-market scope, >100000fill whale, complete metrics/cohorts,
source mutation, failure/accounting and weekly reuse.

Launched exactly one optimised full-case probe57079 with the original pinned
source/config/cache identity and external2147483648byte RSS guard. It additionally
streams and compares every field of all108185 ranking rows, using typed _encode,
against original artifact de54fd21cb04460bbbc0e2eeaf17cdaa.parquet (SHA above).
No first-decision result, timing improvement or annual throughput is claimed yet.
Do not duplicate this handle or edit its producer/replay/scoring engine modules
while it is live. Prepared script is stored as hl_coalesced_metric_probe_source
and wrapper hl_coalesced_guard; launch result hl_coalesced_probe_launch.
The old pending4GiB allocations were not removed or refunded. Read-only catalog
inspection showed96retained payloads totalling51828150bytes,2pending payloads and
1pending scratch allocation before launch. Active lease rechecks total budget.

Original acquisition72925 confirmed live:40of67completed at zero-based batch39,
lifetime reserved158531007106bytes at that completion. Its qualified report is
batches/0039/qualified/qualification_kdtj6i81/manifest.json,
SHA6a22731170055786f264f872f4b41790bd3f6dba44d37b84b850569d631a1068.
Batch40 raw/normalized/compact published;7of32qualification buckets,52831830
physical prefix rows at183497.367s. Cleanup cumulative13442files/189160854461bytes,
qualified job-owned raw/intermediate payloads only; canonical history retained.
Raw recovery needs another copy/redownload, and lifetime spending is not refunded.

### 2026-09-13: wallet-day arithmetic verification complete

Completed the remaining Task1 day-feature tests: context/day/callback rejection
before consumption, UTC/min/max overflow,50market cap including restored active
state,400day dormant episode continuity, fee-token asymmetry, signed zero, and
cancellation-sensitive1e16,+1,-1e16 across midnight with both fee modes and four
near-zero closure thresholds. These are reference characterization tests of the
existing transducer; no additional production arithmetic was changed.

Focused98502:37passed1.76s; independent read-only review approved Task1 with no
material findings. Exact test formatting completed. Combined postformat63639:
149passed4.20s across wallet-day,episode codec,reuse,streaming and numeric spool.
git diff --check passed; acquisition hash53597 remains
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.
This completes the bounded arithmetic core, not persistent reusable history.
Accounted multi-wallet observation/checkpoint publication, retirement and strict
cutoff feature-reader integration remain required before annual acceptance.

Optimised original57079 confirmed live: plan_started70.643s,plan_complete82.732s,
240groups(all nonempty),58085358physical rows:12.089s measured planning interval.
Latest45000wallets/28230314fills at4139.158s,106written groups. Whale6662648fills
completed3093.963s with unchanged504504472byte ordered artifact and52635942byte
numeric spool. Original reader had1453nonempty replay groups. Full golden ranking
comparison, final peak RSS and cache accounting remain pending; do not restart.

### 2026-09-13: persistent record format verified; optimised probe terminal

Feature-publication implementation plan written and reviewed. Review identified
that completed episodes and closing fills share native keys; corrected the
storage contract to episode-before-fill with exact same-coin pairing validation
across batch/shard boundaries and final dangling-pair rejection. Re-review approved.

Task1 feature_records.py and tests implemented after RED45722:25missing-import
failures. GREEN15566:25passed0.19s. Real Parquet roundtrips preserve exact native
keys, large integer values, floats/signed zero and checkpoint context. Malformed
and oversized payloads/key fields/counters reject. Independent code review approved;
exact formatting completed and postformat38381 passed174combined record/day/codec/
reuse/stream/spool tests4.04s. Diff/acquisition engine checks passed. Accounted
writer, cross-record order validator and qualified-day publisher remain unbuilt.

Optimised57079 is now terminal, exit1 due solely to diagnostic comparison code:
_encode(actual ranking row) rejects datetime values. The producer had already
processed108185wallets/58085358fills/6662648whale fills across240groups and published
its ranking. Failure occurred8855.595s; external guard8856.517s, peak1358553088bytes,
RSSlimit2147483648,limit_triggered=false; internal ru_maxrss1341848KiB.
Do not repeat this expensive computation to fix the diagnostic.

Read-only catalog inspection found new cohort publication
28614a33669e4ba610eb95cd2a1e7db909770fff1f583ce7bddb8d4740e2d57b,
artifacts/a453074a1816498290f5da034355682c.parquet,8323840bytes, declared
SHA408f4ce50b71e4cefed22af7b4ba5ec7325170e83311a0d1bc678350411f3fbd,
identical to the original ranking's declared digest. Independent actual-file
hash/row/reopen/accounting verification88282 is live; it forbids source planning
and must reuse the already published result. Do not claim the digest comparison
verified until that handle completes. Script stored as
hl_coalesced_verification_source; no duplicate source replay was launched.

### 2026-09-13: optimised result verified; accounted feature writer

Verification88282 exited0 in173.425s without any source replay (planning explicitly
forbidden). It reverified actual old/new ranking file hashes,108185candidate rows,
42447eligible and5selected wallets. All ranking bytes are identical. Reopening
left accounting unchanged:4294967296reserved,74743145retained,33554432metadata,
4403264873total bytes. Both scoring handles are terminal; no rerun is needed.
Source acquisition72925 remains active and its engine stays frozen.

Implemented feature_writer.py and reusable constant-space feature_order.py under
the approved publication plan. Two128MiB output reservations precede payload
writes; buffers, decoded batches, shard rows/bytes and total artifacts are bounded.
Episode/fill pairing carries across output shards. Finish returns settled tokens
only; no source completeness or catalog publication is implied. Exceptions poison
the writer and retain partial/pending charges. No payload deletion was introduced.

InitialRED74913:13missing-module failures; GREEN7368:13passed0.55s. Additional
poisoning/inode replacement/shared-budget/buffer/shard tests30408:18passed0.62s.
Review identified original descriptor closure before settlement as an inode-reuse
race. RED389204 reproduced3finalization failures. Fixed by retaining the original
descriptor through flush, pre/post-settlement identity checks and token acceptance,
closing it in finally. GREEN78170:21passed0.71s. Focused re-review approved.

Exact formatting completed. Postformat87374 passed195feature/record/day/codec/reuse/
stream/spool tests4.60s;60743 passed59cache-publication/resources/lease tests1.91s.
Diff and fresh acquisition hash checks passed. The next required unit is verified
feature publication opening plus the qualified multi-wallet contiguous-day builder;
retirement, strict-cutoff metric integration and annual acceptance remain pending.

Original72925 confirmed41of67qualified at batch40, lifetime163205614011bytes at
that completion; report batches/0040/qualified/qualification_hbiup378/manifest.json,
SHA4b67475956240172dd89da5f05b79f12efaa4aa8902aaacc0190e4660383dcd2.
Batch41raw/normalized/compact published;13of32qualification buckets,
101703818physical prefix rows at201224.257s. Approved cleanup cumulative13778files,
194925803641bytes, canonical preserved; raw recovery needs another copy/redownload.

### 2026-09-13: immutable feature publication reopening verified

Added feature_publication.py and focused tests for expected-context catalog
lookup, artifact hashes and explicit schema/shard limits; bounded observation and
checkpoint iteration, strict day/scope/native pairing across files, checkpoint
wallet order and before/after verification including early iterator closure.
The qualified builder must still prove source consumption and supply expected
inputs; this opener does not independently certify raw-source completeness.

InitialRED4686:10missing-module failures; GREEN36278:10passed12.94s. Added frozen
handle requirement to prevent accidental shard/context rebinding: RED23903 failed,
then GREEN59789 passed11tests14.21s. Review found engine verification occurred
before lengthy catalog/payload lookup but not afterward. RED91885 reproduced both
direct-verify and iterator-finalization gaps; confirmed after user interruption,
without restarting any background worker. Added post-lookup context/engine check.
GREEN69132 passed13tests15.53s; focused re-review confirmed the fix.

Added actual dormant400day checkpoint state across files (large integer/signed
zero preserved) and reversed-checkpoint-shard rejection. Exact formatting followed.
Postformat3784 passed269combined publication/writer/record/day/codec/reuse/metric/
cache/lease tests21.37s. Diff and fresh acquisition engine hash checks passed.
Task3 qualified multi-wallet source-to-day building remains unimplemented;
publication opening is verified, not the complete persistent-history pipeline.

Original acquisition72925 remains live at41completed batches, zero-based batch41
qualification20of32buckets,156480266physical prefix rows at204594.003s. No worker
restart, cap change or additional cleanup action was introduced by this unit.

### 2026-09-13: qualified source-to-day publication implemented

Added feature_day_builder.py and builder/query-directory code pins to the feature
engine. It validates the report's source origin and exact previous-day context,
merges ordered current fills with the entire prior active-state stream, retains
dormant open states and checks independently counted input/output fills. All
source groups and previous checkpoints must exhaust. ThreeGiB scratch plus two
128MiB output reservations precede sorting. Only fully consumed FD-pinned ordered
temporary files and verified empty owned directories are removed; no retained
cache or unknown pending allocation is deleted/refunded. Complete output tokens
publish together after source/previous/engine checks, with verification on reuse.

RED10427:5missing-module failures; GREEN87043:5passed6.18s, including3consecutive
qualified days versus reference, dormant empty third day, no-sort reuse, missing
days and interrupted/changed source. Extended5142 had11passed/1failed because the
missing-fill fault reached the earlier episode-pairing guard, not its targeted
count check. Adjusted that injection to suppress episode callbacks too; no
production relaxation. Independent read-only builder review approved without
material findings. Combined89567 passed281tests52.19s. Additional99777 passed2
crypto/commodity whole/subset scope fixtures2.72s. Final builder-only postformat
verification59252 was running at this audit checkpoint; record its terminal
result before claiming final formatting verified. Acquisition hash/diff checks
passed. Real day-feature capacity, >100000fill composed-builder proof, rolling
retirement and strict-cutoff metric integration remain mandatory pending work.

Original72925 confirmed42of67completed; batch41qualification report
batches/0041/qualified/qualification_twh75ijt/manifest.json,
SHAf8fe389d06de77f0edd665739aa730ab1aeaf68b6492c0a822722218bd459275,
lifetime167987890990bytes at that completion. Zero-based batch42compact published,
qualification31of32buckets,251789484physical prefix rows at224964.548s. Cleanup
cumulative14114files/200886381047bytes, canonical retained; raw recovery requires
another copy/redownload with no lifetime refund. The original handle remains live.

Post-checkpoint verification: original59252 terminated successfully with14builder
tests passed in21.73s. No production changes followed that formatter/test run.

### 2026-09-13: composed whale and fresh-lease feature acceptance

Added full qualified100002-fill wallet acceptance and contiguous resume after
closing/reopening CacheLease/CacheResources. Both use real production paths;
no implementation changes were needed. Initial15879 passed16builder tests53.88s,
postformat19778 passed16tests51.22s. git diff --check passed and frozen acquisition
engine remains12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

Started original92473: real2025-06-02 BTC feature publication in existing
candidate_history_real_914u8kwn, identity real-first-decision-candidates-v2.
One day only, existing baseline report SHA
2bde8d1c08020161d6d171d6f4381551718273af150d92c8239e016f19e2c027,
gross_excludes_fee, external2GiB RSS guard, unchanged8GiB cache cap.
Initial accounting reserved4294967296, retained74743145, metadata33554432,
total4403264873bytes. Old pending obligations remain charged. No new AWS request,
new cache or retained-artifact cleanup. Probe is still running at this checkpoint;
preserve/poll92473 and do not edit its engine dependencies meanwhile. Store keys
hl_real_feature_day_probe_source and hl_real_feature_day_guard retain diagnostics.
Publication/observations/checkpoint counts, retained delta and reopen checks are
pending; this is capacity evidence only, not an annual research result.

Acquisition original72925 has published zero-based batch42 qualification at
batches/0042/qualified/qualification_qnvstpsz/manifest.json after32/32buckets and
259914588physical prefix rows. Cleanup intent committed14450targets; no terminal
batch completion/cleanup result observed yet. Preserve original72925.

Real one-day probe92473 subsequently terminated exit0. Publication key
2e33b2f284c740116585abe3b285644a705fb27d14f24253ccf730e162d7cb66,
624032fills/16723episodes/3188active-wallet checkpoints,51967703retained bytes
added. Build77.119s, full iteration/reopen103.838s, external guard107.395s with
454672384peak RSS bytes (2GiB not triggered). Accounting reserved4294967296,
retained126710848, metadata33554432, total4455232576bytes; reopen unchanged and
no pending refund. Four published artifacts remain retained in the original cache.
This one day does not prove90-day footprint or annual throughput.

Acquisition72925 now confirms43/67batch completions, lifetime172855664537bytes,
batch42reportSHA1cc28ba7a40db26abd2bb403a39cf0770177385ae83b39d62be5baefc5bc59ea.
Owned cleanup cumulative14450files/207053521326bytes, canonical retained;
raw recovery requires another copy/redownload and lifetime is not refunded.

### 2026-09-14: feature-window arithmetic reducer

New feature_wallet_metrics.py consumes one already-qualified wallet/window stream,
replays the first flat/flip boundary inclusively per coin, then uses subsequent
persisted global episodes. Ordered per-fill numeric values feed unchanged
MetricSpool/_finish; no daily-total regrouping or approximate medians. Complete
episode/fill pairing, strict native order/window and bounded50-market state are
checked. This module is not a qualified range reader or integrated ranking path.

Plan read-only review corrected the side field required by closing_size, then
approved. RED61259:12missing-module cases;GREEN49382:12cases. Expanded93424:
20passed5.35s, including Parquet roundtrip/differential offsets/both fee modes,
both flip directions, cancellation-scale inputs, malformed and interrupted streams,
empty metrics and lazy100002-fill whale with only1boundary replay call. Focused
code review approved without material findings. Exact-file formatting followed by
87554:144combined tests passed7.86s. Frozen acquisition hash unchanged and
git diff --check passed. No production source-cache dependencies were modified;
the verified real day remains reusable under its existing feature engine.

Original72925 remains live. Zero-based batch43 raw manifest published at
batches/0043/raw/proxy_archive_9sagufal/manifest.json,168objects/4293092301expected
bytes. Completed count remains43/67 until a new terminal batch result is observed.
No additional new probe process remains live after92473's verified completion.

### 2026-09-14: exact qualified multi-day feature binding

Reducer acceptance extended without production changes: invalid configuration,
fee policy,51stmarket and tiny boundary/threshold cases. Initial21307 passed34,
postformat91492 passed34tests5.15s.

New feature_window.py binds a complete immutable calendar-day chain to one fully
qualified enclosing source, projecting exact per-day membership from pinned
metadata. It permits exact intraday query bounds but does not perform queries;
future row filtering remains mandatory in the next ordered reader. Missing,
reversed/duplicate days, fee/scope mismatch and changed source/features reject.

Plan reviewed approved. RED62632:4missing-module tests;GREEN22229:4passed7.24s.
Mutation-during-final-check RED42803 failed2expected rejections; before/after
streamed stat digest of canonical/catalog/feature pins fixed both. GREEN49848:
6passed10.52s. Combined9350 passed85tests95.19s before the final review fix.
Review found qualification/feature code mutation after earlier checks: RED56876
reproducedboth; final engine comparisons fixed them. GREEN41315 passed8tests14.50s,
focused re-review approved. Final formatting touched only test formatting;
postformat focused recheck pending at this checkpoint. Diff whitespace passed.

Source acquisition engine verified unchanged.72925 remains live; current
zero-based batch43download148/168objects,3929604059batch-reserved bytes at
226970.056s. Do not confuse this batch counter with the lifetime ledger or call
batch44completed. Actual saved annual four-market weekly/daily runs, ordered
feature queries/ranking integration, retirement and90daycapacity remain pending.

Terminal postformat1358 passed8feature-window tests13.10s. Acquisition engine
freshly verified unchanged afterward. Only original acquisition72925 remains live.

### 2026-09-14: ordered feature query and real metric-stream capacity

Implemented ordered_feature_partitions.py: qualified feature window, pending
combined3GiBscratch, one256MB/2GBspill SQL query, complete counted/coalesced wallet
partitions, no feature deduplication, one512MiB-capped ordered artifact at a time,
closed SQL during metric reduction and bounded decoded readers. Existing episode/
fill pairs survive full native ordering and strict intraday predicates. Caller
owns cleanup/publication; this primitive cannot certify an incomplete annual run.

RED93852 missingmodule3;GREEN89517 passed3. Expanded RED83960 exposed omitted
feature-window code dependency; fixed,32659 passed8. Code review identified
runtime identity checks preceding expensive final verification; RED25169
reproduced both code and spill mutation. Before/after runtime checks with pinned
directory descriptors fixed them,84527 passed10. Re-review resolved findings.
Exact-file formatting followed by47981 passed120combined tests66.56s, including
raw-fill-vs-feature metric comparison, count/pair preservation, overflow retaining
charges, output exclusivity and early-close mutation rejection. Frozen acquisition
hash unchanged and git diff --check passed.

Original real query probe19187 TERMINAL exit0. Existing2025-06-02BTC publication
2e33b2f284c740116585abe3b285644a705fb27d14f24253ccf730e162d7cb66,
640755observations=624032fills+16723episodes. All4partitions consumed into exact
metric reducer with SQL closed:6421trading wallets,maxwallet82513fills.
Planner1.802s cumulative, finalverification48.557s, externalguard48.835s;
peakRSS496562176bytes<2GiB, largest ordered artifact16331342bytes.
Accounting while scratch reserved:7516192768pending+126710848retained+
33554432metadata=7676458048bytes. After exact-owned completed scratch cleanup,
accounting returned byte-for-byte to4294967296pending/126710848retained/
33554432metadata/4455232576total. Old obligations remain; no retained data deleted,
no new cache or AWSrequest. Diagnostic sources in store
hl_real_feature_query_probe_source and hl_real_feature_query_guard.

Single day is not90daycapacity. Its largest wallet is below100000fills, so composed
larger-whale query acceptance still needs explicit coverage. Candidate metric
publication/ranking reuse, chronological rolling features/retirement, daily
artifact budget and actual annual saved weekly/daily/UI/accounting remain open.

Acquisition72925 remains live,43/67completed. Zero-based43compact manifest
batches/0043/compact/compact_987rggur/manifest.json; current prefix qualification
6/32buckets,50459616physical prefix rows at230669.234s. No other probe/test remains
live at this checkpoint. Canonical/acquisition engine unchanged.

### 2026-09-14: feature-backed complete candidate metric/rank publication

New feature_candidate_merge.py and feature_metric_producer.py integrate complete
candidate history, ordered feature groups, exact metric reducer and existing
candidate_metrics/scoring artifact formats. Feature provenance has a distinct
identity while cadence-only Monday changes reuse it. Dormant candidates remain;
one candidate/group lookahead, no complete-universe Python list. Pending3GiBscratch
and512MiBoutput are reserved before sort, under the same existing8GiBcache.
FD-pinned exact-owned scratch cleanup must finish before final verification and
atomic publication; cleanup failure leaves no visible metric publication.

Plan review corrected cleanup ordering then approved. Merge RED17911(6), helper
midnight fixture corrected, GREEN45163(6). Producer RED23919(4), GREEN50975(4).
Expanded17534 passed16tests34.35s including100001lazy dormant candidates,
raw-vs-feature ranking equality, weekly reuse, source/config/engine/candidate
mutation, output overflow and cleanup failure. Review tail-mutation RED83565
reproduced4cases; final engine/window/candidate-stat guards fixed them,3132 passed
13producer tests24s, re-review approved. Combined83969 passed62tests83.79s.

Further dormant provenance test RED17688 reproduced canonical source outside the
lookback changing during final candidate hash. Added bounded streamed source-stat
digest from the pinned full report, supplementary to existing candidate-history
content verification.91870 passed14producer tests27.30s; focused review approved.
Postformat42296 passed21merge/producer tests42.43s. Acquisition frozen hash freshly
unchanged; diff whitespace passed. No existing feature-derivation dependency was
edited, preserving the real day publication.

Original77144 launched real one-day feature ranking/raw replay equivalence and
weekly reuse under2GiBRSSguard. Existing candidate_history_real_914u8kwn and
baseline qualification only, source2025-06-02→2025-06-03, one-day lookback,
original five-metric weights/top5/eligibility settings. This is capacity/equivalence
diagnostic, not annual strategy performance. Initial accounting4294967296pending,
126710848retained,33554432metadata,4455232576totalbytes. No newAWSrequest/cache/cap.
Preserve/poll77144; keep all its engine dependencies frozen until terminal result.
Store hl_real_feature_ranking_probe_source / hl_real_feature_ranking_guard.
Real publication/equivalence/reuse result is pending at this checkpoint.

Original72925 still live:43/67completed, zero-based43qualification8/32buckets,
67274646physical prefix rows at231715.178s. No other test/formatter live.
Annual four-market weekly/daily saved runs/UI, chronological reuse/retirement,
90daycapacity and all-candidate annual output budgets remain unfinished.

### 2026-09-14: real ranking integration failure, bounded handoff fix

Original77144 is TERMINAL exit1, NOT a successful ranking probe. At13.187s/
344727552peakRSS, it stopped with 'One owned ordered artifact at a time required'.
No candidate metric/ranking was published by this probe. Small fixtures previously
fit one partition; the real four-partition run exposed groupby read-ahead starting
the next sort before the previous wallet MetricSpool had closed. The storage guard
correctly refused overlapping directory state. No cap increase or guard relaxation.

Systematic reproduction75091 with max_partition_rows=4 failed identically. Added
feature_metric_stream.py: a globally checked one-lookahead candidate cursor feeds
each complete address interval independently; empty physical intervals still emit
dormant candidates. Each partition's reduction and FD-pinned artifact disposal
finish before another query starts. Removed old concatenating source helper from
producer and engine-bound the new stream.2781 passed15producer tests27.78s; narrow
review approved. Null-candidate sentinel regressionaae287 failed then distinctEOF
fixed it;57506 passed targeted test. Formatted exact files; combined64247 running
at this checkpoint, with multi-partition/raw metric equality added.

Read-only lease audit after failed probe: pending4831838208,retained126856046,
metadata33554432,total4992248686bytes. Exactly one new pending failed payload:
token81700ebfd33a48a5b40f52c709e81b51,
artifacts/462c49875aa34543941468338edb0a4b.parquet,actual1159bytes,
reserved536870912bytes. Three older obligations remain unchanged, including
scratch/21b0f11bac8b4f42995184a052e3aecd and the older two payloads.
Async user approval requested to remove ONLY this new failed unpublished payload
and release ONLY its reservation. No response/approval or deletion yet. Do not
silently refund, adopt, retry, create a new cache or raise the8GiBcap. Standard
retry working reservation would otherwise exceed the remaining cache budget.

Acquisition advanced: read-only job.sqlite3 proves45contiguous qualified batches
(0..44), all15122cleanup rows deleted,total219017730380bytes (canonical preserved;
raw recovery requires another copy/redownload, no lifetime refund). Batch44report
batches/0044/qualified/qualification_xbmi_z7q/manifest.json,
SHA676e7ff0f4c1bec518c02db2f9df9013891e55dbf09d5c8c5350997fd7e2c6e9.
Budget.sqlite3 lifetime184982223557bytes against322122547200cap at audit. Original
72925 remains live; zero-based45raw manifest proxy_archive_ho3o1_e2,168objects,
4704989390expectedbytes; latest observed download82/168objects at261594.279s.
Only72925 and test64247 live; all real ranking results remain unverified.

Postformat64247 subsequently terminated0:33merge/producer/query tests passed70.40s.
No test or real ranking process remains live. Frozen acquisition hash and diff
whitespace freshly passed. Awaiting explicit failed-output cleanup approval before
real ranking retry; the full annual goal remains active and unfinished.

### 2026-09-14: mixed-market whale and fresh-lease acceptance checkpoint

The composed acceptance fixture imports and qualifies 100002 fills for one wallet
across BTC and xyz:GOLD, builds actual persisted feature days, verifies the full
fill count and nonempty episode output, and compares every ranking row against
raw replay for combined and per-market scopes. It uses the real V2 crypto/commodity
configuration, not the legacy BTC/ETH/SOL-only configuration. A strict intraday
cutoff excludes later observations from the same persisted shards; the oversized
wallet remains indivisible despite max_partition_rows=4.

Fresh-lease acceptance closes the original cache lease, reopens saved feature days
and ranking under a new lease, forbids another partition plan, and checks identical
ranking and cache accounting. These are fixture correctness/reuse checks, not real
90-day or annual capacity evidence. No production changes at this checkpoint.

The prior test handle45084 was no longer available after context recovery, so its
terminal result is not claimed. Fresh verification91314 terminated0: both
test_feature_query_acceptance.py and test_feature_metric_producer.py passed all17
tests in85.34s. git diff --check passed. Acquisition engine hash freshly remained
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

Read-only ledger verification now confirms47 contiguous qualified batches0..46;
original acquisition72925 is still processing zero-based47 (48th batch). Latest
qualified report is batches/0046/qualified/qualification_350livjr/manifest.json,
SHA2c5e474cadcf032fc54e402d2e115b8fb74312bd52357378fc3a2c0081541c30.
Lifetime reserved acquisition bytes195505603159 remain below322122547200cap.

Read-only revalidation confirms the failed-probe token81700ebfd33a48a5b40f52c709e81b51
still holds512MiB, its exact unpublished artifact is1159bytes, and no publication
descriptor references it. Specific cleanup/retry approval was requested again;
none received at this checkpoint. Nothing deleted, no reservation released, no
real ranking retry launched. Chronological orchestration/retirement, scheduled
integration, capacity gates and actual annual saved results remain unfinished.

### 2026-09-14: chronological feature-history controller

Implemented feature_history.py under the approved chronological-derivation design.
FeatureHistory binds one source/lease/scope/fee context, builds missing calendar
days sequentially from origin using the existing feature builder, and returns an
exact-cutoff FeatureWindow containing only the requested lookback. It keeps at
most733day descriptors, not decoded wallet histories. Same-decision requests and
fresh-lease restarts reuse published days without sort/new allocation. Invalid
requests fail before building; failed advancement does not commit the cursor.
Completed day publications and unfinished obligations remain untouched.

Plan reviewed approved. Initial1231 RED3missing-module tests;65937 GREEN3passed.
Expanded failure tests exposed validation/reentrancy/context gaps;64568 passed21.
Review and independent late-hash tests60333/25704 reproduced caller pin/coins and
resource replacement escaping final verification. _check_context now checks the
current engine/lease/caller/resource binding before and after the report hash.
66487 passed24tests34.63s; re-review confirmed resolution. Exact two files formatted.
Postformat56055 completed:64history/day/window/producer tests passed147.53s.
Frozen acquisition hash remains
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541;
diff whitespace checks passed. User-facing docs distinguish this internal progress
from a runnable annual result.

No existing source/acquisition/feature-derivation dependencies changed. No real
probe, cleanup, budget increase or new acquisition launched. Original72925 remains
live:47qualified batches, zero-based47prefix validation observed at19/32buckets,
180747534physical prefix rows,307317.52elapsed seconds. Read-only ledger remains
195505603159lifetime bytes under322122547200cap.

Specific failed-output cleanup approval is still outstanding. Scheduled rank
routing through FeatureHistory/build_and_score_features is the next integration;
automatic ownership-safe retirement additionally needs durable resume anchors
so reopening does not rebuild deliberately retired history. Both, measured90day
feature capacity, annual all-candidate output budgets, completed67batch source
qualification and actual weekly/daily saved comparison/UI/accounting remain open.

### 2026-09-14: scheduled reader uses chronological feature rankings

QualifiedScheduledActivity now lazily borrows FeatureHistory under the same cache
lease and calls build_and_score_features for rankings. Fixed full-source coins
produce reusable days; each effective per-asset/pooled query keeps its own market
scope and strict cutoff. Candidate history still begins at the qualified source
origin, including pre-coverage dormant wallets. Native positions, market volume,
hourly exposure, seed policy and source bounds are unchanged. No raw fallback on
feature errors, no extra cache/cap, no automatic retirement/cleanup.

Plan review approved.28033 RED3tests detected the old route/absent features.
24702 passed31new/existing scheduled tests45.78s after routing. New guard tests
first had a pytest reserved parameter-name setup error (21663), corrected without
production changes;41898 then reproduced4post-history context gaps. Facade and
effective-config checks were moved before metric production.15410 passed57
scheduled/seed/registered-reader tests172.80s.2687 passed mixed BTC/gold active
scope expansion/contraction, exact raw ranking equality and shared-day publication.

74578 passed2API tests14.10s: real fixture registration through saved weekly/daily,
cohort preview, unchanged shared-cache accounting, matching ranking evidence,
25point equity reports and comparison query. These are small fixture results,
not the user's actual annual saved comparison or rendered UI acceptance.

Review found that a prepared-cutoff change during history building could reach
the producer before the existing outer guard rejected it.89919 RED reproduced
this; added _decision(decision) immediately after history/facade verification.
25532 passed all5post-history mutation cases15.08s. Re-review confirmed closure.
Exact three files formatted. Combined postformat64424 is still running at this
checkpoint; no final result claimed yet. Frozen acquisition engine freshly matches
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

Original72925 remains live; zero-based47qualification observed20/32buckets,
190264542physical prefix rows at308038.03elapsed seconds. No new real probe or
acquisition launched. Failed-output cleanup approval remains outstanding. Next
runtime gate is ownership-safe retirement/restart, followed by measured90day and
annual footprint,67batch qualification and actual annual saved/UI/accounting work.

Postformat64424 terminated0:61combined scheduled-feature/scheduled-reader/seed/
registered-reader/API worker/preflight tests passed176.44s. Diff whitespace passed.
No test or ranking probe remains live; acquisition72925 remains live. This closes
the fixture-level scheduled feature route and saved-evidence integration gate,
not rolling retirement, annual capacity or the actual annual backtest objective.

### 2026-09-14: read-only retirement prerequisite and consolidated cache audit

Before designing deletion of useful rolling history, audited the existing real
cache without mutation. Acquired its exclusive lease, verified CacheResources
identity real-first-decision-candidates-v2 and full accounting, checked every
publication's artifact tokens, then released the lease without cleanup. No live
cache owner blocked the lease. No pending token below has a publication reference.

| Pending token | Exact relative target | Reservation bytes | Actual file bytes |
| --- | --- | ---: | ---: |
| ab02b6ab235940d3a5ccf214d8f452d0 | scratch/21b0f11bac8b4f42995184a052e3aecd | 3221225472 | 0 |
| 31f2a8f64fca435689dada10cbde9723 | artifacts/7f4068260a0f433ea59cbfdb0b385491.parquet | 536870912 | 0 |
| ce54475d37eb4393b027bafeaf1ce060 | artifacts/6f2656e56d664083a37e53ac0cebc90e.parquet | 536870912 | 1159 |
| 81700ebfd33a48a5b40f52c709e81b51 | artifacts/462c49875aa34543941468338edb0a4b.parquet | 536870912 | 1159 |

Four pending obligations reserve4831838208bytes (4.5GiB), with three payload files
totalling2318bytes and no files in the scratch allocation. Retained payloads:
104files/126856046bytes. Metadata allocation33554432bytes. Accounted total is
4992248686bytes, leaving3597685906bytes under8GiB. The standard3GiBscratch+
512MiBmetric-output reservation requires3758096384bytes and does not fit.
After explicit safe disposal/release of all four obligations, remaining accounted
capacity would be8429524114bytes. This releases local cache reservations only;
it changes neither the8GiBcap nor the300GiBAWS lifetime ledger and is not proof
that90days or the annual run will fit.

Published retained products are91candidate_day files/24190201bytes,
3candidate_history files/4868152bytes,2candidate_metrics files/12891794bytes,
2candidate_scores files/16290516bytes,2cohort_rankings files/16647680bytes and
1qualified_features_day with4files/51967703bytes. Thus indiscriminate retirement
would discard valid evidence while leaving the actual abandoned reservations.

Read-only architecture check: FeatureHistory currently resumes by reopening the
origin chain, and feature metric reuse re-verifies its FeatureWindow. Removing
older day publications/payloads without a durable resume anchor and a verified
reuse path for saved rankings would break restart or Monday reuse. Do not add
an unlink loop or weaken those guards. Keep retirement separate from this exact
failed-staging recovery. Real one-day ranking equivalence/capacity remains the
next measurement prerequisite before choosing a complete retirement policy.

Requesting consolidated explicit approval for ONLY these four abandoned targets
and their reservations, replacing the earlier narrower one-payload cleanup request.
No cleanup, retirement, refund, real ranking retry or new acquisition occurred.
All saved/canonical data remains untouched. Revalidate ownership, identity, sizes,
publication references and current lease again before any approved disposal.

### 2026-09-14: validation paused for explicit recovery authority

Revalidated unchanged four pending obligations4831838208bytes and retained
126856046bytes. Available3597685906bytes still cannot cover the standard
3758096384byte ranking working reservation. The missing cleanup authority has
persisted across more than three consecutive goal turns; automatic continuations
are not approval. Safe prerequisite implementation and fixture/API integration
are now verified, and the next real equivalence/capacity probe cannot proceed
within the unchanged cache contract. Do not bypass this with a new cache, smaller
working reservation, removal of saved evidence or a cap increase.

Marking the interactive backtest-validation goal blocked pending the consolidated
four-target cleanup decision. This is not completion and does not cancel the
separately authorized acquisition process. Original72925 was polled live now:
47/67qualified, zero-based47prefix qualification21/32buckets,
199781158physical prefix rows at308736.122elapsed seconds. Its existing approved
controller/cap continue to govern it. Resume after explicit approval, revalidate
the exact owned targets and dispose only those authorized objects, then run the
preserved real ranking probe and remaining full-scope capacity/retirement gates.

### 2026-09-15: approved four-target recovery completed; real ranking retry

User explicitly approved the consolidated four-target cleanup. Original cleanup
67844 terminated0 after acquiring the exact existing cache lease, checking all
four pending rows/expected sizes and absence of publication references, holding
open file/directory inode pins, and removing only the approved three partial
payloads plus the empty scratch/spill tree. Released only their four reservations
using release_missing after exact paths were absent. No recursive broad deletion.
No other pending or retained records were removed or changed.

Removed2318actual payload bytes and released4831838208reserved bytes. Before:
4831838208reserved/126856046retained/33554432metadata/4992248686total. After:
0reserved/126856046retained/33554432metadata/160410478total. Complete retained
allocation rows and publication contents matched before/after; publication digest
5efb9662faf41408076ce5329f735b222c92fd88097d6f3c12c5b2a4fe286628.
Only failed temporary output was removed; canonical data and successful saved
products remain. This is not an AWS lifetime refund or cache cap increase.

Preserved guarded real one-day ranking probe relaunched as88994, using the same
June2,2025 feature publication/baseline pin, all6421candidate wallets, original
five-metric/top5 thresholds, strict one-day window and2GiBRSSguard. It will compare
every ranking row against raw replay and verify weekly reuse without a sort or
accounting change. Initial accounting is the post-cleanup160410478total above.
No result claimed yet; keep producer/query/feature/scoring dependencies frozen
while88994 is live. Guard/source preserved in existing session stores.

Original72925 remains live and reached48/67qualified batches. Latest completed
report batches/0047/qualified/qualification_z850ayu1/manifest.json,
SHA2ab482a24020bacd988caeebce10991ebe6c540da928b4843ccaf188e7cd3bec.
Zero-based48compact is batches/0048/compact/compact_1jk65_el/manifest.json;
prefix qualification observed5/32buckets,48874502physical prefix rows at
320852.719elapsed seconds. Acquisition's own approved job cleanup preserved
canonical products; original raw recovery requires another copy or redownload.
No new acquisition process was launched. Full annual goal remains unfinished.

Real retry88994 subsequently TERMINATED0. Feature ranking published at51.997s:
6421candidates,965eligible,5selected. Every streamed ranking row matched raw replay
at73.894s, and both364178byte Parquet outputs have identical SHA
c2c91b7fefe3cc086c0e1ca84fa4c2d6f0d0c80f01d2845410a5327a11d22720.
Feature ranking key664b7961b1e8b0d1720734ffd51670ad2a0a735a29555fc5d7ed5d383d5b8b85,
artifact artifacts/d0ed1437da484eedbddcdf1b12590e56.parquet. Raw ranking key
63c4196782c4b31993744be3c8d2db7b0d26fad6211242758d29b6c47f3ad5b8,
artifact artifacts/33c3dddfd6004cdebd2b05690957594c.parquet.

Weekly cadence reuse verified at76.827s with feature sorting forbidden and
identical resource accounting. Final0reserved/128947566retained/33554432metadata/
162501998totalbytes. Guard terminal0 at77.236s, peakRSS562880512bytes;2GiBlimit
not triggered. Original real multi-partition failure is now resolved on actual
data, not only fixture tests. Cleanup permission blocker is resolved; no further
cleanup authority is assumed. The next gate is the full90day feature-history
capacity/throughput case, before any annual-readiness claim. Annual retirement,
output budgets, acquisition completion and actual saved comparison remain open.

### 2026-09-15: full-source 90-day feature capacity probe launched

Original guarded process60063 is live; do not relaunch on observation timeout.
One-off diagnostic script /tmp/hyperliquid-feature90-probe-U2Worr/probe.py and
session stores hl_real_feature90_probe_source_20260915 /
hl_real_feature90_guard_20260915 preserve the exact invocation. No production
code changed during probe preparation. Narrow read-only review approved it.

Uses the same immutable baseline report SHA
2bde8d1c08020161d6d171d6f4381551718273af150d92c8239e016f19e2c027,
whose verified source coverage is June2 through November17,2025 exclusive.
Feature scope is BTC/xyz:GOLD/xyz:SP500/xyz:TSLA, matching the scheduled reader's
full source scope. Builds/reopens91calendar days from June2 through August31;
the exact90day ranking window is June3 through September1 exclusive. BTC ranking
uses unchanged original five metrics, thresholds and top5 selection. This first
decision capacity diagnostic is not a BTC-only replacement for the annual study.

Reference is already-verified raw90 ranking publication
28614a33669e4ba610eb95cd2a1e7db909770fff1f583ce7bddb8d4740e2d57b,
artifact artifacts/a453074a1816498290f5da034355682c.parquet,
8323840bytes, SHA408f4ce50b71e4cefed22af7b4ba5ec7325170e83311a0d1bc678350411f3fbd.
No raw90 replay. Preflight verified source/context/selection/publication pins
and108185reference rows. Diagnostic-only corrections canonicalized JSON tuple/list
context representation and used the same ranking_batches reader for both streams
so unused null percentile fields have identical representation. Numeric row
comparison remains exact. Expected108185candidates/42447eligible/5selected;
weekly cadence-only reuse must perform no feature sort or accounting change.

Initial accounting0reserved/128947566retained/33554432metadata/162501998total.
Existing8GiBshared cache, one-thread256MiBDuckDB,2GiBspill and external2GiBRSS
guard remain unchanged. Guard also samples apparent scratch bytes every2seconds.
No retry, eviction, failed-output deletion or resource-envelope expansion.
Freeze feature/query/producer/scoring dependencies until process termination.
First emitted day is June2; no capacity or equivalence result claimed yet.
Original acquisition72925 remains live:48/67qualified and batch48 qualification
7/32buckets,68426680physical prefix rows at322120.175elapsed seconds.

First two days completed in original60063: June2 at61.05s (day60.18s),
publication73c0c21608ac50960c547cf48c4e5e47de066aab50c57cfe3f139b84c42971c2;
June3 at127.653s (day66.461s), publication
8054a76239f544f701ccd33c5100a84287846f5f6d479ea05b86f4e6aadbe3ab.
After two:0reserved/238004547retained/33554432metadata/271558979totalbytes.
June4 started at127.658s; probe remains live. These are progress measurements,
not extrapolated annual capacity or final90day acceptance.

Read-only retention inspection while the probe runs confirms the following design
constraints (no implementation or deletion yet):

- FeatureHistory currently resumes from origin. FeatureDay verification does not
  recursively open its predecessor; a verified retained contiguous window plus
  one complete boundary day could therefore support an explicit resume anchor
  without changing day derivation or throwing away dormant episode checkpoints.
  An anchor still needs strict current source/engine binding, payload verification,
  monotonic advancement and crash tests; this is not implemented behavior.
- Existing feature ranking reuse reopens the feature window and candidate/metric
  dependencies before reading the final ranking. Saved Monday reuse after old-day
  retirement therefore needs a separate verified final-result receipt, or those
  dependencies must remain retained. Blind expiry by calendar date is unsafe.
- CacheResources accepts exactly three tables and audits every retained payload.
  A safe retirement journal must preserve accounting across unlink/DB crash points;
  deleting retained files and later adjusting the ledger is not a supported path.
  Preserve the current derivation engine during60063; shared resource/publication
  edits also change existing derivation identities and invalidate reuse.
- API preflight still estimates ranking rows using min(fills,100000). This is not
  an all-wallet bound for the108185-candidate first decision. Exact qualified
  decision/scope counts or a defensible bound are needed before annual acceptance;
  preserve the50million-row/4GiB saved-run envelope pending measurement/approval.

Next: monitor original60063 to terminal evidence, then make a measured retention
and total-footprint decision. Do not launch another90day/raw replay or modify its
dependencies while it is live. Acquisition72925 is independently still live.

### 2026-09-15: host restart, durable progress and scoped offline recovery

The host/session restarted. Handles60063 and72925 now return Unknown process id.
An approved host-level read-only ps inspection showed host processes only minutes
old and no Python probe/acquisition process. Neither original job is claimed live.
Session stores and /tmp probe/formatter cache were also lost. Do not infer terminal
success or a specific exception from missing stdout; no terminal RSS/scratch guard
measurement survived in the available state.

Read-only SQLite inspection proves86 full-scope feature-day publications, June2
through August26,2025; the older BTC-only June2 publication also remains. Thus
5calendar days remain for the planned91day construction, and the90day feature
ranking/equivalence/reuse checks have NOT completed. Catalog totals:
5070546282retained bytes,33554432metadata,3355443200pending reservations,
8459543914combined obligations against8589934592limit. Payload hashes have not
been re-audited in this read-only metadata inspection.

Interrupted, exactly identified obligations (not deleted or refunded):
- pending d20e41e620264a9ab259746b4acfe3d8, scratch/9c82c8f5c561498f9efda1ff7bbd2727,
  3221225472reserved; contains ba88ed6689184a25b213b39c55cce536.parquet,
  27400689actual bytes. No spill child was present on inspection.
- pending dae46fc47e8d4b0ba10d65cef1d8641e,
  artifacts/4438a2c9e11042daa1d0be8dabab6abc.parquet,134217728reserved,
  1307127actual bytes.
- retained but not referenced by any publication: ad174903d6814b36b70b9f92dbaf655a,
  artifacts/d82b46cbced446b684de54995fc90eaa.parquet,20593164bytes;
  cf1fbf9aa03048599f42c2942913aa7d,
  artifacts/edbd852c2cb948a4a495caede4e3d425.parquet,21576697bytes.

Existing four-target cleanup approval applied only to the earlier completed
cleanup, not these new objects. Preserve all of them pending a scoped recovery
decision. Only130390678bytes of reservation headroom remain, less than a further
128MiB writer reservation. This is a measured envelope constraint, not proof of
the missing process's exact termination cause. Do not simply relaunch the probe
or expand limits. Successful old publications must remain reusable and accounted.

Acquisition ledger still has48qualified batches and49compact/raw/normalized
batches. Lifetime download reservations are200829760562bytes under the original
322122547200cap. The295900377630bytes in objects are the full frozen inventory,
NOT additional spending and must not be added to reservations. Batch48 has no
qualified stage/report. Its raw and compact products are already committed.

After verifying the host had no old worker, resumed ONLY the existing batch's
qualification via the standard OFFLINE command:
`.venv/bin/python -u tools/run_hyperliquid_archive_job.py step --root .hyperliquid_cache/annual_job_20250901_20260901_300gib`
New process59982, no credentials or accept-approved-download flag, no network
source, no new download request. This is one recovery step, not an automatically
restarted acquisition loop. Its ordinary previously approved cleanup policy and
original engine/cap remain in force. Record its terminal result before another
step. Freeze its dependencies while live.

Resume-anchor work proceeded only in new module/test files; prior34tests passed
57.22s. Review found compressed oversized descriptors were bounded only after
Arrow decoding. Regression10887 failed at iter_batches before rejection; added
pre-decode row-group row/uncompressed-byte limits. Regression3468 passed1.46s.
Two new files formatted using the approved pinned formatter. Postformat anchor/
history/day/window suite76786 and a replacement narrow read-only reviewer are
running. No retirement, integrated anchor resume or annual backtest claim yet.

Post-restart cache integrity audit69736 TERMINATED0 under the exclusive cache
lease, verifying every retained payload hash and accounting. Totals matched the
read-only metadata inspection above. Four interrupted payload SHA pins, in the
same order as the exact targets above:
c43fc4ee62b21ca8236a04e43f1ae327dd16d238b07ca0029106ecb7a7433bfb,
9a665ae629592485d17368e32f3d23071c91e84834240fcc88b3f9d447033c55,
69ef3f275669c4ed0f6bc69d0cdf9d8d707605c4c8e384eebb2cba85055840c2,
1e7fd994fa82eda2c6918ecf244cb2512ead667e62d31f7235a7ba2ffec24e8e.
Combined actual interrupted payload70877677bytes (67.594MiB). A further128MiB
writer reservation would exceed the8GiBcap by3827050bytes. No cleanup or cap
change was performed; the exact cause of original process termination remains
unavailable, but simply continuing its writer cannot fit the existing envelope.

Anchor regression76786 completed83passed149.06s. Final review identified lease
expiration during last engine hashing;90286 independently reproduced both public
verification and publisher-final verification failures. Final lease checks added;
92061 passed2tests3.57s. Focused rereview found no remaining material blockers.
Final postformat22254 TERMINATED0:85anchor/history/day/window tests passed142.51s.
Frozen acquisition engine still12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541;
diff whitespace passed. Offline recovery59982 remains live on direct polling,
with no emitted qualification progress yet. Do not launch a duplicate step.

Proposed recovery decision for user approval, NOT authorization or implementation:
raise only the shared local derived-cache disk envelope from8GiB to16GiB, while
keeping2GiB probe RSS guard,256MiB DuckDB/one thread,2GiB spill and300GiB AWS lifetime
cap unchanged; and remove only the four interrupted payloads listed above plus
their exactly owned scratch container/reservations. Preserve all86complete
full-scope days, the old BTC-only day, reference rankings, canonical source and
saved studies. A versioned budget transition must preserve verified old feature
reuse: changing shared resource code changes derivation engine hashes, so blindly
editing a constant and rebuilding/ignoring old pins is not acceptable.16GiB is
proposed working headroom, not a proven upper bound for every annual scenario.
No additional AWS requests are implied by this local-disk proposal. Anchor
integration and ownership-safe retirement still need implementation and tests.

### 2026-09-15: explicit anchor integrated; cap/cleanup approval still pending

Continued safe fixture-based integration without applying the requested16GiB cap
or touching interrupted real outputs. FeatureHistory now accepts keyword-only
anchor_inputs, opens the verified retained interval under its current lease,
preserves canonical source origin, indexes lookbacks relative to retained _base,
and derives only missing days. It rejects earlier-than-retained requests rather
than shortening their lookback. Same-decision/intraday requests retain exact
cutoffs, even when later days exist physically. Existing no-anchor behavior stays.
Full retained descriptors, including future cached days, count toward the limit.

Plan docs/superpowers/plans/2026-09-15-hyperliquid-history-anchor-integration.md
reviewed and executed inline.32322 RED: unsupported keyword;14085 GREEN:
1passed1.88s. Expanded69869 passed38tests48.27s. Review found initial source
hashing preceded anchor snapshot, so a changed caller dictionary could be adopted.
24353 reproduced substitution of a different valid anchor; snapshot moved before
expensive constructor checks.57278 passed1test1.95s; rereview cleared the finding.
Final postformat56391 TERMINATED0:129history/resume/anchor/scheduled/seed-policy/
API tests passed214.35s. No implementation delegation or shared-engine edits.

Feature derivation engine remains
eccdd99cf22c312ba73746fa0ab4834694af0e7ef0f69974f576ade45be13f3f;
acquisition engine remains
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.
Thus this integration does not invalidate completed real feature publications or
the live offline acquisition engine. History/downstream facade identity changes
normally; no old facade result is silently adopted. No cache budget transition,
real anchor publication, automatic retirement or saved-ranking shortcut occurred.

Offline recovery59982 is confirmed live on direct polling; latest emitted
qualification progress1/32buckets,9778498source rows. It remains one offline step,
not an acquisition loop.48/67batches remain durably qualified until its eventual
commit. Keep its handle/dependencies; do not restart on observation timeout.
Approval for the proposed16GiB shared-cache envelope and four exact interrupted
payload removals remains outstanding. This goal continuation is not approval.
Annual full-source acquisition, real90day ranking, capacity/retirement, saved
weekly/daily studies and independent accounting/UI acceptance remain unfinished.

Approval boundary rechecked after consecutive automatic continuations: no user
approval for the16GiB cap/67.594MiB interrupted-output cleanup has arrived. The
initial request, completed safe integration work and subsequent verified waits
did not grant that authority. Current read-only marker still8589934592bytes;
both pending reservations above remain unchanged. Safe recovery checks and the
independent anchor integration have completed; do not continue the real probe,
delete its outputs or silently change its envelope without the requested decision.

Pausing automatic goal continuation for this persistent authority blocker, not
because qualification is slow. Offline process59982 is still confirmed live and
continues independently; latest emitted progress2/32buckets,19552164source rows.
Do not cancel or duplicate it when resuming. Its existing approved offline scope
does not grant cache cleanup/budget authority. On user approval, recheck its handle
and exact cache targets, then execute a reviewed budget transition/recovery plan
that preserves completed source/feature/ranking evidence and lifetime accounting.

### Explicit approval received: exact cleanup and 16 GiB envelope

The user has now explicitly approved both the proposed 8-to-16 GiB shared-cache
transition and removal of the four exact interrupted payloads (70,877,677 bytes)
listed above. The earlier approval blocker is resolved. This does not raise the
AWS lifetime cap or RAM/spill limits. Neither operation is implied complete by
approval itself.

Read-only preflight71581 terminated0 and confirmed the exact allocation records,
all four file sizes/hashes, absence of publication references, and accounting
8,459,543,914 total bytes. The reviewed one-off cleanup journals the two retained
unpublished outputs as equally charged pending obligations before any unlink,
uses pinned directories/files, and releases only missing exact targets. Review
identified a same-size mutation gap; frozen inode/size/mtime/ctime/link-count
checks now surround hashing and precede unlink. No recursive cleanup is used.
Execution and final accounting will be recorded separately. Offline59982 remains
live, now3/32 qualification buckets and29,324,450 source rows; no new AWS requests.

Revised preflight93948 terminated0; narrow rereview found no remaining concrete
blockers. Approved cleanup66093 terminated0: exactly70,877,677 bytes removed from
the four approved interrupted outputs. These partial files are not recoverable
from this cache (their work can be recomputed from preserved canonical inputs).
All completed publications and every unrelated allocation row remained unchanged.
Final full integrity audit: reserved0, retained5,028,376,421,
metadata33,554,432, total5,061,930,853 bytes. The 86 full-scope feature days and
existing raw/feature ranking evidence are preserved. Cache marker still8 GiB;
the separately approved16 GiB transition has not yet been applied.

The scoped implementation plan
`docs/superpowers/plans/2026-09-15-hyperliquid-approved-cache-expansion.md`
received narrow read-only review: Approved, no concrete blockers. It specifies
receipt-backed16 GiB opening and bounded, explicitly pinned DB/marker crash
recovery without editing legacy feature/acquisition engine dependencies. No new
policy/migration production code has been written or applied yet. User approval
already covers this work; no repeated approval gate is needed. Latest direct
poll of59982 confirms it remains live:4/32 buckets,39,097,388 source rows.

### Approved 16 GiB transition implemented and verified

New `derived_cache_policy.py` and `derived_cache_expansion.py` implement the
receipt-backed envelope without editing legacy resource/feature/acquisition
engine dependencies. Tests55043 RED (missing modules),94036 GREEN2. Interruption
tests58238 RED exposed missing recovery fsync and late caller-input substitution;
74941 GREEN22 after correction. Policy67382 RED exposed caller mutation during
open; correction and broader21184 passed82 tests. Preparation35373 RED exposed
directory replacement and policy changes during publication; both corrected.
Independent review demonstrated a hot SQLite rollback journal cannot be recovered
through the initial read-only connection. Subprocess53119 reproduced the exact
readonly error; bounded native SQLite rollback bootstrap fixed it.69902 passed39
policy/expansion tests. Final postformat20264 TERMINATED0:187 policy/expansion/
resource/publication/feature-day/history/resume/anchor/window tests passed159.75s.
Narrow rereview approved the correction, no remaining concrete blockers.

Real dry-run24304 terminated0 with the exact expected accounting and86 full-scope
days. Real application55055 TERMINATED0 in63.562s, including a fresh lease and
reopening every87 old feature publication with original keys. Every original
allocation and publication row is preserved; exactly one1,103-byte immutable
receipt payload/publication was added. Staging obligation released only after
durable metadata replacement and complete audit.

Final accounting: limit17,179,869,184 bytes; reserved0; retained5,028,377,524;
metadata33,554,432; total5,061,931,956. The cap is not preallocated disk or RAM,
nor proof of the eventual annual footprint. AWS300 GiB, one-thread256 MB DuckDB,
2 GiB spill and2 GiB process RSS guard remain unchanged.

Durable receipt inputs, before-state snapshot, events and result are in
`.hyperliquid_cache/cache_expansion_20260915/`. Open the migrated cache through
`open_expanded_cache(lease, identity, receipt_inputs)`; legacy8 GiB opening fails
closed. Do not rerun its one-off `apply.py --execute` or edit pinned policy modules
under the live migrated cache without an explicit future policy transition.
The feature engine remains eccdd99cf22c312ba73746fa0ab4834694af0e7ef0f69974f576ade45be13f3f;
acquisition remains12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.
Next is completing five missing warmup feature days and the exact90-day ranking
comparison/reuse check; annual saved-study/accounting/UI gates remain unfinished.

Resumed probe preflight7095 TERMINATED0 confirmed86 full-scope retained days,
the exact original metric/selection context, and108185/42447/5 reference counts.
The reference artifact remains8,323,840 bytes with SHA408f4ce50b71e4cefed22af7b4ba5ec7325170e83311a0d1bc678350411f3fbd.
Narrow read-only diagnostic review approved the invocation and one-attempt guard.

Original resumed guard56663 is confirmed live by direct polling. It owns the
single probe worker and its lease; do not launch duplicates or edit its feature/
producer/scoring/resource-policy dependencies. Durable scripts/logs live under
`.hyperliquid_cache/feature90_resume_20260915/`. The worker passed run preflight
and began August27,2025 (the first of five missing days). It will derive full
BTC/GOLD/SP500/TSLA features through August31, then compare every BTC90 ranking
row to the already-verified reference and check weekly cadence reuse without
feature sorting/accounting change. No raw90 replay, downloads or automatic cleanup.
The guard samples RSS/high-water memory and apparent scratch bytes every2seconds,
enforces2 GiB RSS, and persists terminal output. No equivalence result yet.

Offline qualification59982 is separately confirmed live, latest24/32buckets and
234,637,134 source rows. Its durable qualification count remains48/67 until commit.
The new receipt-backed cache still needs explicit scheduled API loader/registration
integration. Ownership-safe rolling retirement and saved-ranking receipts, actual
annual all-candidate output capacity, full source qualification and real saved
weekly/daily studies/accounting/UI acceptance remain outstanding.

### Expanded-cache registration integration and completed warmup derivation

Reviewed scoped plan: `docs/superpowers/plans/2026-09-15-hyperliquid-expanded-cache-registration.md`.
Implemented optional `expansion_receipt` in the exact cache-reference schema;
legacy path/identity stays8 GiB-only. Reference precheck is bounded, typed,
deep-copied, marker/nonce/root/engine-bound and read-only. The owned scheduled
loader uses the existing receipt-backed factory under its exclusive lease on
open and close; no automatic migration, creation, fallback or refund. Pinned
policy/core/feature/acquisition modules remain unchanged.

91513 RED demonstrated unsupported expanded reference;90750 GREEN load/rank/
close/reopen reuse in11.46s.39212 TERMINATED0:17 tests passed176.61s, covering
expanded safety plus genuine annual registration into saved weekly/daily/preview
worker fixtures under both8 and16 GiB policies. Ranking/portfolio artifacts,
cohort pagination, comparison series and unchanged shared-cache accounting/source
job records were asserted. This is short-fixture integration evidence, not real
annual performance or rendered-UI acceptance.

Review found the new closing receipt audit followed the last caller/engine check.
22302 RED reproduced both late caller-receipt and loader-engine substitutions.
Final caller/engine checks moved after receipt/disk verification;54394 GREEN2 in
22.07s. Narrow rereview approved, no remaining blockers. Final postformat broad
registration/manifest/owned-loader/API suite53124 is confirmed live; no terminal
outcome yet. Do not change source files while its registration engine snapshots
are being tested, or mistake the earlier17-test result for this full regression.

Guard56663 remains directly confirmed live. All five missing full-scope days
completed: August27 keyf8fdcb3c15eec4636f45510f6d0ef0cf25e7ba061ee1b20c0eac29164fef1f75;
August28 key158cfe5a323ef263cdf68b76de3e95f9a91cc669c82b1104447e00cdc869b0db;
August30 key8d003b2dc13e59fb3166910685ce7a12f05349fbd0e83f32b3d156e97c996960;
August31 keya78a7845c74bf19023690b0e4a1f1413c11203f189d8e0d506f4aa936de11f54.
August29's exact key is in the durable events log (do not infer it here).
Full warmup now91days June2–August31. Ranking phase started with reserved0,
retained5,263,435,126, metadata33,554,432, total5,296,989,558 bytes. Latest guard
observation840.89s: peak RSS1,076,404,224, apparent scratch78,348,288 bytes; no
guard trigger or ranking-equivalence result yet. Source qualification59982 also
remains live, latest25/32buckets and244,413,902 physical prefix rows.48/67 durable
qualified batches until its next commit. No acquisition requests this turn.

Final expanded-cache integration suite53124 TERMINATED0:83 tests passed471.48s,
including legacy/expanded annual registration, owned readers, manifests, saved
weekly/daily/preview workers and API preflight. This supersedes its earlier live
status. Narrow review approved the final closing-context check. No changes to
the migrated cache's four pinned policy/core files or feature/acquisition engines.
The missing warmup entry above is August29 publication
4ef403e8dc32afef3d304608e0d4f7ba026602969551d286ef689cf05d264b6a.

### Measured annual ranking-row lower bound: current50-million cap cannot fit

One read-only diagnostic86542 TERMINATED0 in162.182s. Durable script, source pin,
daily entrant series, logs and result:
`.hyperliquid_cache/annual_row_bound_20260915/`. It verified all336 canonical file
hashes/schema/row counts before and after the query, plus unchanged stat identities
and qualification report/engine. Source June2,2025–May4,2026 exclusive:
304,421,078 rows /26,747,138,462 bytes, qualified47 report SHA
2ab482a24020bacd988caeebce10991ebe6c540da928b4843ccaf188e7cd3bec.

The DuckDB query used one thread,256 MB memory setting, UTC and0-byte allowed
disk spill. No source/cache mutation, ranking replay, new download or acquisition
request occurred. Measured process peak RSS895,168,512 bytes (not a claim that
total process memory was256 MB).

Candidate membership is every BTC wallet observed since source origin strictly
before each midnight UTC decision, including dormant and ineligible wallets.
The first decision exactly matched the independently verified108,185 candidates.
By May4 there are416,639 BTC candidates. Holding that number constant after the
known cutoff (no future new traders), and excluding all three other copied assets,
gives these conservative contributions for September1,2025–September1,2026:

- Daily365 decisions: **114,133,907 ranking rows minimum**.
- Weekly53 decisions: **16,521,928 ranking rows minimum**.

Independent read-only review recomputed the saved entrant series and confirmed
strict-cutoff semantics against CandidateHistory. This bound applies to the
planned BTC-inclusive daily comparison, not arbitrary configurations omitting
BTC. It is a capacity diagnostic, not a BTC-only annual strategy substitute.
The current50,000,000-row per-run cap is therefore insufficient by at least2.283x.
The `min(fills,100000)` preflight estimate is not an all-wallet bound and remains
to be replaced with complete causal capacity evidence.

This does not prove encoded bytes or a final annual row upper bound. Proposed
separate user approval: raise per-saved-run ranking ceilings to250 million rows
and16 GiB (currently50 million/4 GiB), preserving streaming batches, query/RSS
bounds,16 GiB shared derived-cache cap and300 GiB AWS lifetime cap. These would
be hard limits, not a promise of sufficiency or predicted consumption; report
publication copies can temporarily require up to twice the ranking-file cap.
No such per-run increase is implemented or authorized yet. Preserve the full
weekly/daily, expanding four-market scope; do not sample or drop the daily study.

At the latest direct checks, original guard56663 remained live at1442.401s,
peak RSS1,076,404,224 and sampled scratch106,332,160 bytes, with ranking still
running; offline qualification59982 remained live at26/32buckets,
254,195,528 source rows. Neither was restarted. Row-cap approval is a new boundary,
not grounds to declare these live computations stopped or the goal complete.

### Per-run ceiling approval received; engine-pinned transition deferred

The user's subsequent `approved` authorizes the proposed250-million-row/16 GiB
per-saved-run ceilings. AWS300 GiB lifetime and shared-cache16 GiB stay unchanged.
These are ceilings, not forecasts or proof of final annual fit. Publication can
temporarily hold two ranking-file copies (up to32 GiB per run at the new ceiling).

Scoped plan: `docs/superpowers/plans/2026-09-15-hyperliquid-approved-report-caps.md`.
Production limits are not yet changed: `disk_cohort_scoring._engine()` directly
fingerprints `ranking_artifact.py`, and guard56663 is still running. Changing it
now would invalidate the live verification. Latest direct guard observation:
1984.02seconds, peak RSS1,076,404,224bytes, scratch106,332,160bytes, no terminal
result. Offline qualification59982 also remains live. Neither job was restarted;
no downloads, policy changes, or source deletions occurred in this checkpoint.

Cap regression tests prepared while production engine remains unchanged:
6276 TERMINATED1, two expected RED failures (old default50m vs approved250m;
explicit approved limits rejected),23passed in0.21s. Constructor validation tests
cover above-cap rows/bytes, booleans, zero, negatives and non-integer values;
existing small-file bounds, corruption, no-overwrite and disk-space tests remain.
No huge allocation was used. `test_ranking_artifact.py` intentionally remains RED
pending production transition, not a completed cap change. Guard56663 directly
remained live at2285.181s; later durable guard2345.529s showed peak RSS1,076,404,224
and aggregate scratch2,868,836,680bytes. The guard enforces2GiB RSS; aggregate
scratch includes output artifacts and is not the individual2GiB SQL spill limit.
Offline qualification59982 remains live, latest28/32buckets/273,748,456rows.

### Rolling retirement integration constraint (inspection only)

Current `CacheResources._connect` permits exactly metadata/allocations/publications
tables; `_audit` permits only pending/retained allocations and rejects missing
retained files. `release_missing` only releases pending allocations. A future
retirement transaction must therefore preserve this schema and use a bounded,
accounted journal artifact, or explicitly design a separate policy migration;
adding a retirement table/state or unlinking a retained file first is invalid.
Publication descriptors may share artifact tokens: deleting a publication alone
does not prove exclusive ownership of its payload. All surviving references need
checking before any conversion to pending and exact-file disposal. Existing
FeatureAnchor/FeatureHistory support restart from retained complete days, but do
not publish anchors automatically or retire anything. Saved-ranking receipts and
their verified independence from expired feature artifacts are still absent.
This is an implementation constraint, not a reviewed retirement design or an
authorization to delete real files. No pinned production modules were changed.

### Verified source-session prerequisite implemented

Reviewed scoped plan: `docs/superpowers/plans/2026-09-15-hyperliquid-saved-feature-rankings.md`.
Task1 adds standalone `qualified_source_session.py` and its tests. Immutable
process-local session reuses QualifiedDay's1..5000-file/64GiB/50-market corpus
validation, fully hashes canonical files on construction, then guards unchanged
regular single-link file identities, report hash, source-engine and caller pin.
Defensive-copy inputs bind full canonical membership and source/code context;
fresh sessions repeat full checksums. No saved receipt bypasses startup checks.

4503 RED12 missing-module failures;33346 GREEN12 in11.91s. Narrow read-only review
approved Task1. Final postformat49885 TERMINATED0:37 source-session/day/window
tests passed37.29s. Task2 saved-ranking capture/load is still unimplemented; this
does not yet enable reuse after expiry or authorize real-cache deletion.

Directly checked feature engine remains eccdd99cf22c312ba73746fa0ab4834694af0e7ef0f69974f576ade45be13f3f
and acquisition engine12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.
Guard56663 still live at3007.793s, RSS peak1,076,404,224bytes, aggregate scratch
2,868,836,680bytes. Qualification59982 still live, latest29/32buckets and
283,514,522source rows. Neither restarted. Approved report caps still pending
the live scoring-fingerprint gate; their two RED tests remain intentionally open.

### Saved-ranking receipts implemented and verified independently of intermediates

`saved_feature_rankings.py` now supplies explicit capture/load. Capture calls the
real feature producer, reuses existing bound-context publication graph checks,
checks feature source/engine, and shares the original retained ranking token.
Receipt inputs bind the complete qualified source-session evidence, exact metric
and selection context, original ranking key and producer/scorer/receipt engine.
Load verifies the receipt and complete ranking payload, returning BoundScoredCohort
without querying old feature/candidate/metric/score publications. Changed source,
query, engine or payload rejects; no fallback replay or arbitrary result callback.

New-lease fixture test removes intermediate publication records only (all payload
allocations remain intact), forbids producer/query replay, and verifies exact rows,
selected cohort and unchanged retained/reserved bytes. Recapture is idempotent and
cadence-only changes reuse the same receipt. Wrong producer-artifact identity and
different source-pin/engine adoption are rejected. This proves read independence,
not a production file-disposal protocol or annual throughput.

TDD evidence:77321 RED missingmodule;82153 initial9GREEN19.61s after correcting a
test config direction map.61381 RED4 late-final-lookup mutation cases;58145 GREEN13.
29218 RED late receipt-engine change during source verification;86928 GREEN5.
Narrow review found ranking bytes could change after final lookup and source bytes
could change during receipt-engine hashing.14254 reproduced both (RED2/4pass).
Ranking path identity now spans the final reads and checks. Source session exposes
verify_identity as an adjunct final file/caller guard, not a replacement for full
verify.2857 GREEN31 in53.50s; rereview approved both fixes.

Final postformat33379 TERMINATED0:89related tests passed115.76s. Earlier broader
76140 had61pass and a test-only bad manifest filename; fixed before final run.
Feature/acquisition engine hashes remain their original pins. Latest direct guard
56663 live3729.214s, peak RSS1,076,404,224bytes, aggregate scratch2,868,836,680bytes.
Offline59982 live, latest30/32buckets and293,292,052source rows. No jobs restarted,
AWS requests issued, real payloads deleted or receipt-pinned policy modules edited.
Per-run cap production change still waits for the live scoring gate; two cap tests
remain intentionally RED. Scheduled receipt integration, bounded retirement,
complete capacity evidence/acquisition and real annual weekly/daily results remain.

### Exact receipt discovery and next acquired-source step

Reviewed plan: `docs/superpowers/plans/2026-09-15-hyperliquid-scheduled-ranking-receipts.md`.
Task1 adds saved_ranking_lookup.py and SavedFeatureRankings.find. Catalogue row and
descriptor/input bounds remain existing constants. One metadata row is processed
at a time and at most one exact matching receipt retained. All bounded descriptor
digests/canonical schema/normalized kind+inputs/publication keys are authenticated
before classification; no SQL JSON filter can hide a corrupt saved kind or parse
oversized JSON first. Nonmatching payloads are not read. Duplicate matches fail
closed. Matching load verifies full ranking bytes; final source/engine/query/lease,
catalogue and ranking-file checks span the expensive work on both hit/miss paths.

54017 RED missingfind;38723 GREEN6.86821 RED late ranking mutation after load,
fixed with checksum and identity snapshot;97331 GREEN17. Review found premature
kind classification;19910 RED2 reproduced hidden-corruption miss and early JSON
parsing.35769 GREEN19 after bounded authenticated scan; rereview approved.
Final postformat86896 TERMINATED0:50 lookup/receipt/source-session tests passed
90.07s. Task2 scheduled facade integration has not started; no physical retirement.

Original offline archive step59982 TERMINATED0.49/67source batches are now durably
qualified. New report0048 qualification_9cop38oa/manifest.json SHA
a6a053e8e5f4f1fbf4fd3d150b418cb05e286e443162cdd85c9488792f6cbe78.
SourceJune2,2025–May11,2026 exclusive,343canonicalfiles,312,843,638rows,
27,496,374,463bytes. This is qualified source coverage, not annual research eligibility.
Read-only cleanup ledger confirmed currentbatch48 deleted336owned stagingfiles /
6,463,844,950bytes. The terminal summary's16,466files/242,161,343,380bytes is cumulative
cleanup across the prefix, not this turn's deletion. Canonical history retained;
raw reconstruction needs an external retained copy or redownload, without refund.

Prelaunch read-only checks: next zero-based49 May11–18,2026 has168objects /
5,289,695,521bytes; no stage records or already-reserved objects.49records exist in
each raw/normalized/compact/qualified stage. Lifetime8233reservations /
200,829,760,562bytes under322,122,547,200cap. Started the next single explicitly
approved acquisition step via IAM reader (no credential output), handle81924 now
confirmed live. No automatic retries/loop or cap increase; no duplicate59982 step.
Guard56663 remains live (last direct4630.695s, RSS1,076,404,224bytes, aggregate
scratch2,868,836,680bytes). Per-run cap production changes still deferred.

### Terminal feature validation and approved report-cap implementation

Guard56663 subsequently TERMINATED0 after6399.241seconds, without a guard trigger.
Durable feature90_resume_20260915/events.jsonl confirms all108,185 ranking rows
exactly match the raw reference (42,447eligible,5selected), followed by successful
cadence-only weekly reuse without sorting. Ranking SHA256
408f4ce50b71e4cefed22af7b4ba5ec7325170e83311a0d1bc678350411f3fbd.
All91 full-scope warmup days are retained. Peak RSS1,076,404,224bytes; aggregate
scratch2,868,836,680bytes includes outputs, not just the separately capped SQL spill.
Final cache accounting:0reserved,5,286,350,121retained,33,554,432metadata bytes.
This validates one real90-day decision and reuse, not the annual strategy result.

After verifying the real-cache lease was available and preserving old publications,
changed only MAX_RANKING_ROWS to250_000_000 and MAX_RANKING_BYTES to16*1024**3
in ranking_artifact.py.28739 first confirmed two expected RED old-cap failures
and23passing tests.19461 then TERMINATED0:42 ranking-artifact, disk-cohort-scoring
and API qualified-worker tests passed19.55s. Narrow read-only review approved.
Existing score fingerprints remain intact; new scoring uses the normal new engine.
AWS300GiB, shared derived cache16GiB, per-decision and query/RSS limits unchanged.
Per-run publication may temporarily need two ranking copies, up to32GiB per run.
Full causal annual capacity evidence and safe physical retirement remain outstanding.

Acquisition81924 remains live and has emitted its batch0049 raw manifest:
raw/proxy_archive_mvufcl9j/manifest.json,168objects,5,289,695,521expected bytes.
This is manifest creation, not evidence that this batch is fully downloaded or qualified.

### Scheduled receipt integration verification in progress

Implemented lazy source-session/receipt lookup before FeatureHistory; misses capture
through the real producer, hits return verified bound rankings without intermediates.
58060 RED replay regression became98240 GREEN. Added final checksum/source checks
and identity guards through facade operation exit after reviewer feedback reproduced
by16366 (2failed,4passed).50975 then passed18 scheduled receipt/history tests.
The first broad regression47629 finished59passed/3failed. One failure coincided with
formatting engine-fingerprinted source while tests ran; subsequent runs use stable
formatted source. Two genuine API failures exposed registration's intentional
canonical hardlinks.77754 reproduced initial linked-source rejection (2failed,
12passed). Narrow reviewed compatibility change accepts initial canonical hardlinks
only, fully hashes contents and pins their topology; alias writes/new links reject.
Owned ranking/cache/report files still require one link. Final expanded suite65877
is running; no green integration claim yet. No physical retirement or real annual run.
Acquisition81924 was directly confirmed live, latest2/168downloaded objects;
its engine fingerprint remains unchanged after integration edits.

Expanded65877 completed99passed/1failed in263.10s. Scheduled/API-worker and
canonical-hardlink tests passed; the remaining failure was catalogue mutation
during a receipt miss, relying only on file metadata identity. Isolated86734
passed, then48431 deterministically reproduced the gap by holding catalogue stat
identity constant (1failed). Discovery now also observes SQLite data_version
on the same open connection across the read-only operation; no transaction,
schema/policy edit or full payload scan added. Physical identity checking remains.
Postformat lookup24397 and a fresh full expanded regression are running.

24397 subsequently TERMINATED0:19lookup tests passed36.19s; narrow review approved.
Full expanded rerun5982 remains live; no source edits during this verification.

### Scheduled integration terminal verification

5982 TERMINATED0:100passed169.90s on stable postformat source. Suite covers source
sessions, receipt capture/load/discovery, scheduled receipt/history/facade behavior,
and saved weekly/daily API runs plus preview under legacy8GiB and approved16GiB
cache policies. Narrow review approved final guard, canonical-hardlink and
SQLite data-version changes. This completes scheduled receipt integration, not
safe physical retirement, full annual capacity preflight or the real annual run.

Read-only next-step inspection confirmed lab_proxy_datasets.inspect still clamps
candidate estimates using min(fills,100000), and scales ranking estimates with
follower decision_times. Actual SelectionState.advance distinguishes trader
schedule, market entry/change and exit; complete capacity evidence must model
these rather than treating a higher report cap as proof. CandidateHistory already
defines full source-origin distinct membership with strict first_observed<decision;
lookback/eligibility truncation is not an acceptable capacity shortcut.

### Complete candidate-capacity implementation started

Reviewed scoped plan2026-09-15-hyperliquid-candidate-capacity.md covers a complete
source-origin daily per-market/pooled count artifact, registration-time export,
and conservative actual selection-tick preflight. No new caps/downloads/strategy
changes. Task1 producer/reader now exists;72727 RED missingmodule then46842 GREEN7.
After correcting cross-market fixture provenance to use real download/import,
76836 passed22 capacity/source-session tests20.91s. Reuse does not rerun the query;
counts include dormant candidates and pooled overlap without duplicate wallets.

Task1 remains incomplete following narrow review: staged validation must precede
publication (48709 RED zero-entrant output left a publication), final caller/lease
and source guards must follow engine hashing, string decoding needs preallocation
bounds, and pooled/per-market cumulative consistency must be checked. No real
capacity artifact has been built; registration/preflight are unchanged. The next
action is these four reviewed integrity fixes, not treating partial tests as annual
capacity evidence. Acquisition81924 remains live with unchanged archive engine;
last direct observation91/168objects downloaded in batch0049.

### Candidate-capacity Task1 integrity fixes verified

Late engine-hash regression95440 RED2/1passed confirmed caller/lease gaps.
17590 RED5/10passed also covered missing pooled counts and predecode enforcement.
Extracted candidate_capacity_records.py validates staged output before publication,
guards compressed string decoding with uncompressed metadata bounds and retained
Arrow dictionaries, and enforces max(per-market cumulative)<=pooled<=sum.
Context guards now recheck caller/source identity/lease after engine hashing.
99506 passed15;74491 passed33 after additional malformed-output cases.

12860 reproduced staged file replacement leaving a publication;20344 reproduced
actual decoded rows bypassing footer/max-row checks. Added staged physical identity
guards across validation/settlement/final context, actual decoded row bounds before
accumulation and final footer equality. Final stable postformat63109 TERMINATED0:
36passed33.98s; narrow reviewer approved. This verifies the count producer/reader,
not complete annual row capacity or API integration. Registration export is next.
Acquisition81924 remains live at96/168objects of batch0049, with archive engine
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541 unchanged.

### Registration retains candidate-capacity evidence

Task2 implemented registration_capacity.py, annual qualified export hook and
optional ProxyDatasetManifest validation. Qualified registration builds count
evidence under its existing cache policy after canonical hardlinks are established;
retained descriptor/payload bind full source membership, engine and original
publication identity. Reopening requires neither cache nor raw aggregation. The
API estimate has not yet changed. Legacy missing evidence remains explicitNone.

79689 RED missing metadata;98398 GREEN1 real fixture registration/reopen/payload
corruption test.29730 passed31 integration tests87.74s. Review found final code/
provenance/export-descriptor verification gaps, reproduced by53527 RED2 and85770
RED4. Fixed with frozen reader/source/provenance code bindings, final provenance
physical identities and exported descriptor hash/identity checks. Final postformat
6033 suite is live; no green final Task2 claim yet. No real capacity export or annual
dataset registration has been performed. Acquisition81924 remains live at146/168
objects in batch0049, original archive engine unchanged.

## Candidate capacity/preflight terminal verification — 2026-09-15

Task2 final6033 terminated0:60passed300.31s and narrow review approved. Task3 now
removes both100000 clamps, sums complete source-origin candidate upper bounds over
actual selection events, fails closed for missing/out-of-range qualified evidence,
and labels the bound and byte/cache limitations in the readiness UI. Retained
counts are indexed once per preflight with initial/final provenance and caller
guards; no raw query on catalogue/inspection paths.

Review regressions:43175 RED uncovered interval accepted;86930 RED2 missing batch
operation;15696 RED2 omitted schedule-engine pin and initial caller mutation.
All corrected and final narrow rereview approved.25457 interim23passed102.36s.
Final stable74489:49passed149.42s;81310:17passed74.62s;62165:14frontend tests plus
production build and schema consistency check exit0. HTTP tests needed escalation
after sandbox TestClient/AnyIO startup hangs (77601/32468/6125 interrupted130;
not reported as passes). Escalated suite completed; two dependency deprecation
warnings and the existing frontend charts-chunk warning remain.

No real annual capacity artifact or annual saved run yet. No cache retirement,
new acquisition, spending increase or source-engine change in this increment.
81924 is still live on batch0049 qualification, most recently5/32buckets;49/67
batches are confirmed complete. Archive engine verified unchanged:
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

## Owned-retirement inventory implementation — 2026-09-15

Reviewed plan: docs/superpowers/plans/2026-09-15-hyperliquid-owned-cache-retirement.md.
This implements fixture-tested infrastructure, not authority to delete successful
real cache objects or automatically expire feature history. The plan preserves the
pinned cache schema/policy. An immutable journal and pre-unlink atomic detachment
remain to be implemented; no post-disposal completion metadata will be required.

Read-only inventory authenticates all bounded publication descriptors and their
allocation references, hashes target payloads, identifies exclusive/shared tokens,
and rejects protected evidence/policy kinds and caller-protected keys. Held
catalogue/marker/namespace descriptors plus data_version and final caller/code/
lease checks prevent stale ownership results after replacement or mutation.

75581 RED9;83872 GREEN9;98170 GREEN16. Review boundary regressions e98cf1 RED2
reproduced SQLite replacement and unbounded protected-key traversal. Fixed both.
Final stable postformat43063 exit0:93tests passed3.21s; narrow rereview approved.
No real inventory, journal, detachment or deletion was performed. Four pinned
cache-policy module hashes remain unchanged.81924 is directly confirmed live,
most recently7/32qualification buckets in batch0049; confirmed complete remains
49/67batches. Annual saved runs and retirement policy are not complete.

## Owned retirement transaction and real-cache footprint — 2026-09-15

Journal preparation, atomic detach-to-pending and exact-file recovery are now
implemented in cache_retirement_journal.py/cache_retirement.py. No real cache
object was detached or deleted.7416 RED5;66525 GREEN5.39428 RED3 reproduced caller
mutation and disappeared intent boundaries, fixed before disposal. e228b0 RED
forged shared ownership was corrected by transactional ownership recomputation.
78513 GREEN18 includes10000publication capacity, SQLite FULL rollback, expanded
policy recovery and file substitution guards. c8a7ed RED original output FD gap
fixed;35133 GREEN26 includes actual physical fixture retirement and unchanged
weekly/daily saved results without replay. be6c8a RED2 strict schema/reason checks
fixed. Final stable52902 exit0:159passed102.16s; narrow rereview approved.

The finish transaction preserves pending obligations on crashes after unlink or
before commit. Verified absence plus committed intent proves completion without
new metadata allocation. Shared artifacts, policy receipts and saved evidence are
protected. No automatic retention selection or real deletion authority is implied.
All four pinned cache-policy code hashes and acquisition engine remain unchanged.

Read-only footprint42701 held the existing real cache lease, read SQLite in ro mode,
checked data_version and payload sizes, and classified feature Parquet schemas.
It did not hash/revalidate all payload contents or run a retirement inventory.
495retained allocations total5286350121bytes. By publication kind:
approved_cache_expansion1103bytes/1file; candidate_day24190201/91;
candidate_history4868152/3; candidate_metrics19997131/5;
candidate_scores25139498/5; cohort_rankings25699876/5;
qualified_features_day5186454160/385. Feature observations5083390548bytes/293files;
checkpoints103063612bytes/92files. Observations are about96% of retained cache
bytes, so expiring complete old feature days matters much more than checkpoints
or score intermediates. These92day publications include prior probe variants,
not a completed annual store. Current full-source pin reuse remains unproven.

81924 directly confirmed live at8/32qualification buckets in batch0049.49/67
batches remain confirmed complete. Next: semantic rolling-anchor/retention policy,
real-target review and cumulative annual capacity; actual annual registration,
weekly/daily saved runs, accounting reconciliation and UI inspection still pending.

## Rolling retention plan and owner binding — 2026-09-15

Reviewed plan: docs/superpowers/plans/2026-09-15-hyperliquid-rolling-feature-retention.md.
It composes existing history, verified anchors, saved ranking receipts and owned
retirement. Recovery must precede any new publication. Review added a guard that
rejects unresolved matching-context operations owned by another anchor, even if
a compatible anchor is available. No implicit cross-owner adoption or stale
prepared-baseline replacement is allowed. Plan rereview approved.

First Task 1 increment implemented: optional strict 64-hex owner binding in both
retirement receipt inputs and authenticated journal content. Existing unowned
calls retain their prior shape. Journal-owner changes/removal, malformed owners
and owner/body mismatch reject before detachment. Discovery, semantic recovery,
rolling controller and registered opt-in integration remain unimplemented.

RED87493:10failed/1passed for absent owner keyword. GREEN21215:52passed4.64s.
Final stable postformat83173 exit0:134passed22.58s, including cache resources,
publication/expansion and physical-retirement saved-ranking reuse regressions.
Narrow code review approved. git diff --check passed. All four pinned cache-policy
module hashes and acquisition engine remain unchanged. No real cache intent,
detachment, deletion, new download or spending change in this increment.

81924 is still live: batch0049 qualification most recently11/32buckets,
109854548source rows processed. This progress is not a terminal qualification
result;49/67batches remain confirmed complete. Parent annual acceptance remains
open, and the next implementation step is bounded owner-scoped recovery.

## Bounded anchor-owned recovery — 2026-09-15

Implemented rolling_retirement_inventory.py, rolling_retirement_recovery.py and
rolling_anchor_guard.py. This is explicit recovery of existing intents, not new
target selection or automatic expiry. Bounded catalogue discovery authenticates
outer descriptors, filters owner-anchor metadata before opening relevant journal
payloads, rejects foreign unresolved owners and duplicate unresolved targets, and
classifies prepared/detached/complete operations. It validates the original target
against the anchor's exact source/origin/scope/fee semantics and pre-boundary day.
Prepared baselines are never replaced. Complete operations are verified and skipped.

The protection snapshot precedes inventory and covers all canonical source paths,
the report, anchor and retained feature publications. Canonical hardlinks retain
their initial topology; owned payloads remain single-link. Optional transaction
validation performs expensive context checks before final protected-file checks,
followed by cheap held catalogue/marker/journal/namespace checks before deletion.
No new metadata is allocated during recovery and no failed action is auto-retried.

40794 RED9 missing module;57873 GREEN9.8804 RED5/1pass reproduced validation
occurring too late;99789 GREEN15 after pre-disposal protection.83863 RED3/1pass
reproduced canonical hardlink, unrelated journal and late transaction-context
issues.20103 RED1 reproduced the second anchor-context hash boundary; fixed by
ordering hashes before final identity checks.24970 GREEN29.60135 RED2 exposed
adoption of an old canonical-file mutation after inventory; snapshot moved before
inventory,37222 GREEN31.64f01c RED1 reproduced catalogue replacement inside
validation; cheap held checks now repeat after validation.90228 GREEN55 in60.30s.
Final narrow rereview approved Task1. Final stable postformat95636 exit0:
181passed104.44s, including cache policy, saved-ranking physical reuse and
anchored feature-history regressions. git diff --check passed.

All four pinned cache-policy module hashes and acquisition engine were verified
unchanged. No real cache intent, detachment or deletion was performed. No new
archive step or spending change was made.81924 remains live on batch0049,
most recently13/32qualification buckets and129832088source rows processed.
49/67batches remain durably confirmed complete. Rolling controller/registered
opt-in, actual annual registration, capacity, saved weekly/daily runs, independent
accounting reconciliation and rendered UI acceptance remain pending.

## Rolling feature controller — 2026-09-15

Implemented explicit RollingFeatureHistory and bounded rolling_feature_catalogue.
The default FeatureHistory and existing registered datasets remain non-retiring.
The controller selects exact source/origin/full-scope/fee/engine anchors, recovers
owned unfinished work before new publication, builds the complete active window,
publishes and reopens its replacement anchor, and verifies identical window inputs
before retiring obsolete derived days one at a time. It preserves canonical data,
future days, other fee contexts, saved evidence and all active checkpoint days.
Failed operations do not advance the cursor; fresh leases recover the owned work.

Caller inputs are frozen before first hashing. A verified canonical source session
and protected anchor identities span discovery, disposal and final cursor commit.
Catalogue helpers bound metadata, day inputs and target file snapshots. Selected
anchor corruption never silently falls back; ambiguous old-day variants reject.
Final validation callbacks stay inside the held catalogue observer, and physical
target/anchor snapshots remain checked afterward.

Task2 development evidence:76180 RED7 missing modules;13800 GREEN7 after timestamp
parsing correction.91387 and38388 reproduced caller/guard ordering faults;
87338 GREEN12 after fixes.62621 GREEN23 broadened ownership, resource, intraday,
scope and lease coverage.14211 RED4 and49495 RED1 exposed standalone catalogue
validation and snapshot bounds;13047 GREEN28.7576 postformat179passed228.17s.
Final review found a trailing callback outside catalogue observation: the initial
test omitted SQLite commit and was corrected;90513 then genuinely reproduced a
miss returning after committed catalogue mutation (1 failed,1 passed). Removing
that callback passed both cases.4751 also reproduced missing immediate replacement
anchor reopening; reopening and comparing exact window inputs fixed it.
20139 GREEN31 in66.69s; narrow rereview approved. Final stable postformat combined
suite91911 exit0:182passed237.99s. Task2 verified; git diff --check passed.

All work above uses temporary fixtures. No successful real cache object was
expired, no existing dataset policy enabled and no new download step launched.
Pinned cache module hashes and acquisition engine remain unchanged. Archive81924
is still qualifying batch0049, last observed17/32buckets,169780762source rows;
49/67batches remain confirmed complete. Task3 registered/scheduled opt-in and all
actual annual capacity, saved-run, accounting and UI acceptance gates remain open.

Integration caveat: chronological retention and exact saved-ranking reuse do not
yet establish arbitrary backward-in-time cache misses. Old anchors can reference
legitimately expired days, and completed-retirement safeguards reject republished
targets. The matched weekly/daily integration must test this explicitly before
claiming historical misses can rebuild; do not weaken corruption/owner checks or
assume all daily and weekly causal market universes share receipt keys.

## Explicit scheduled/registered rolling policy — 2026-09-15

Added optional feature_history_policy="rolling_feature_anchor_v1" to annual
registration, qualified metadata validation, the owned loader and scheduled
reader. Missing/null policy preserves the non-retiring reader; unknown values
reject, and rolling policy requires qualified registration and an explicit shared
cache. No existing manifest or CLI default changed. Exact saved-ranking lookup
still precedes lazy feature construction, including under the rolling policy.

Temporary integration fixtures advance multiple decisions and per-asset/pooled
scopes, capture all ranking rows, physically retire obsolete days, and reopen
weekly/daily matching requests under a fresh lease with feature construction and
sorting forbidden. Full rows, selected cohorts, score bindings and positions are
compared, not just counts. This verifies exact receipt reuse, not arbitrary
historical cache misses or an annual run.

Per-window controller validation now carries the scheduled policy/config,
effective ranking config, decision and registered disk/in-memory manifest context
to pre-detach and pre-unlink guards. It clears after success/failure. Expensive
registered engine checks precede final manifest identity/context checks; the
facade repeats cheap frozen input checks after nested validation. Physical source
and anchor checks still follow these callbacks. Real policy revocation cannot be
treated as permission to finish deleting first and reject at reader close.

19824 RED9 and63174 RED2 established missing integration;54783 GREEN9.
39967 GREEN3 covered annual publication/rename and owned-loader propagation.
Annual fixture assertions predating capacity export were corrected to account
for the exact retained capacity publication, including on later validation
failure; no refund or production behaviour was added.10308 reproduced the stale
failure-case assertion (1failed/3passed).
71957 reproduced three pre-unlink context failures;67154 reproduced registered
policy revocation not rejecting the rank.84533 GREEN21 in132.06s after propagation.
Initial first-callback mutation tests42385 passed because a subsequent guard
caught the mutation; calibrated FINAL-callback tests24016 genuinely reproduced
three file-deletion failures.51197 GREEN4 in45.21s after final-ordering fixes.
Narrow rereview approved the boundary changes.

Postformat API worker16239 exit0:4passed48.70s, covering both original8GiB and
approved16GiB cache policies, each with/without rolling opt-in. Saved weekly/daily
results, cohort preview, comparison curves and unchanged ranking reuse passed.
Combined backend regression60757 subsequently completed:113passed915.78s.
This verifies the opt-in wiring, not the open historical-miss clause. Pinned cache module hashes and
acquisition engine are unchanged; git diff --check passed.

No real cache retirement, manifest enablement or additional download step was
performed.81924 remains live on batch0049, last observed19/32qualification
buckets and189756398source rows;49/67batches remain confirmed complete. Earlier
cache misses still need a bounded, authenticated reconstruction lifecycle before
real enablement. All final annual capacity, saved-run, independent accounting and
rendered UI acceptance gates remain open.

## Historical miss route plan and initial regressions — 2026-09-15

Reviewed and approved scoped plan:
`docs/superpowers/plans/2026-09-15-hyperliquid-historical-ranking-misses.md`.
Selected reuse of the existing bounded raw candidate-metric scorer for requests
before the verified retained feature interval. Exact saved-ranking lookup remains
first. The plan requires owner-scoped pending recovery before either producer
writes, complete raw QualifiedWindow plus source-origin candidate provenance, one
exact-query saved receipt across both routes, and continued forward feature use
after a historical miss. It never catches corruption as a fallback or republishes
expired feature keys. No additional downloads or cache expansion are implied.

Alternatives considered were retaining all feature days (conflicts with rolling
footprint) and a new feature-generation/checkpoint rehydration format (larger
lifecycle change). The raw fallback trades CPU time and accounted intermediate
storage for avoiding that migration. Annual capacity/runtime remain unproven.

Initial route tests99558 terminal1:9 expected failures for the missing
rolling_history_route module. They exercise cold/nonallocating classification,
same/forward/earlier calendar boundaries, intraday endpoints, corrupt latest
anchor rejection and prepared/detached owner recovery. No production module was
written while fingerprint-sensitive integration60757 was live. After its terminal
113passed result, the initial route implementation was added;53919 is running.
No real retirement or additional acquisition step was performed.81924 remains
live, most recently20/32qualification buckets and199749674source rows on batch0049;
49/67batches remain confirmed complete.

## Historical producer routing implementation — 2026-09-15

Added rolling_history_route.py. It freezes the verified source/full-scope/fee/
interval context, selects the latest anchor independently of the requested
interval, and resolves that anchor's owned retirement before allowing a producer.
Foreign pending owners and corrupt selected anchors remain errors. Earlier first
or calendar-end boundaries return the raw route; cold/chronological requests
return features. Classification itself does not build metrics or feature days.

53919 GREEN9 in32.87s after initial implementation.99990 RED1 exposed a late
anchor mutation after recovery; an AnchorGuard now spans recovery through final
classification checks.10749 GREEN15 in87.36s included foreign-owner and invalid
scope/fee/interval validation.94142 RED1 reproduced a final callback changing the
cold catalogue after its observer closed; the redundant callback was removed.
Final narrow Task1 rereview approved. Stable postformat combined33444 is running.

Task2 raw-receipt tests86930 terminal1:3 expected failures because capture_raw is
not yet implemented. They require per-asset/pooled raw rankings to reopen through
normal lookup without either producer, and require an existing exact feature
receipt to be reused without duplicate raw capture. The historical fallback is
not wired into scheduled ranking yet; no end-to-end historical-miss success is
claimed. Existing opt-in integration60757 passed113 tests before this increment.

No real cache objects expired and no new download step was started. Acquisition
engine remains12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.
git diff --check passed.81924 remains live on batch0049, most recently21/32buckets
and209739954source rows.49/67batches remain confirmed complete.

## Raw ranking capture verification — 2026-09-15

Routing/catalogue/recovery suite33444 completed:59passed in404.04s. Initial
raw-capture plus saved-feature suite60514 completed:22passed in223.10s.
Raw capture authenticates exact lookback membership, original candidate-history
origin/scope, producer engine and bounds, and shares route-independent receipts.
Scheduled historical-miss integration is still outstanding.

The additional provenance fixtures initially omitted the scorer's required
verify_source callback; corrected the fixtures without changing the scorer.
58811 then completed with5passed and1expected regression failure in31.23s:
an existing receipt's artifact could change during the redundant verifier after
the discovery guard closed. Removed that trailing verification and explicitly
reject a vanished second lookup. Suite45426 is running against this correction.
This is temporary-fixture testing, not authority to expire real cache objects.

Archive81924 remains live:22/32qualification buckets,219729810source rows in
batch0049.49/67batches remain confirmed complete. No acquisition retry or new
download step was started; caps and real cache contents are unchanged.

Narrow Task2 read-only review found no correctness blockers, requesting explicit
raw-first/feature-second reuse and zero-eligible cohort tests. Added both via the
per-asset/pooled fresh-lease test;6145 is running. Scheduled integration test25574
reproduced the intended missing integration: a new earlier hypothesis attempted
RollingFeatureHistory construction after actual fixture feature retirement.
The first attempt97391 had a fixture metric-direction mismatch, corrected before
25574. This is a RED integration test, not an implemented historical workflow.

45426 completed28passed in224.38s, including the late-artifact regression.
6145 completed4passed in20.96s for fresh-lease raw/feature reuse with both scopes
and zero-eligible variants. Scheduled routing is now wired after exact receipt
lookup and before any feature construction; explicit rolling policy invokes
prepare_history_route with the existing frozen-rank callback and raw misses call
capture_raw. No arbitrary error fallback was added. Scheduled suite43708 is
running. Broader historical lifecycle/multi-market/capacity acceptance remains
open; no annual end-to-end result is claimed.

Latest archive81924 progress:23/32buckets,229717216source rows. Still49/67batches
confirmed complete. git diff --check passed; no real retirement or new download.

43708 completed12passed in40.09s, including the first historical new-hypothesis
regression (full rows/selection/binding and unchanged feature/anchor descriptors
and retired paths). Five changed source/test files were formatted after all
fingerprint-sensitive tests terminated. A combined postformat receipt/scheduled
feature/rolling regression run is now pending. Archive engine rechecked unchanged:
12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

## Pending scoring orchestration — 2026-09-15

New ranking_staging_score composes the existing score/cohort kernels over pending
outputs, captures writer-FD content pins before verification callbacks and returns
full score/ranking pins plus a detached summary. It freezes effective configuration,
metric/selection context, source-verifier provenance and engine. All outputs remain
pending; no source completeness, publication or cleanup is inferred by this helper.

76789 RED10/2pass: missing scorer API plus write-only hook FDs could not support
pread-based content pinning. Changed both shared output opens from xb to x+b;
exclusive creation/default computation remain unchanged.20373 GREEN12 tests in
1.13s, covering full rows/summary, empty/per-asset/pooled and changed inputs.

91815 RED1/8pass exposed final-owner callback config mutation. Added two more
RED regressions for empty provenance and final-artifact callback config mutation.
Reordered owner verification before engine/context checks, added final cheap
configuration comparison after artifact hashes, and reject empty provenance before
output.53882 GREEN17 in1.39s. Final formatted combined59753 GREEN103 in7.04s,
including staging ownership/writer/hooks and existing score/cohort/publication
tests. Narrow rereview approved the fixes. Causal source verification and successful
cleanup/receipt policy integration remain open; Task1 cleanup is not yet complete.

Archive engine rechecked unchanged;81924 remains live at28/32buckets with
279,656,782source rows in batch0049.49/67batches remain confirmed complete.
No real cache deletion, extra acquisition or cap change. git diff --check passed.

## Successful temporary staging cleanup — 2026-09-15

Implemented ranking_staging_cleanup.finish_staging for the live invocation only.
It authenticates the exact saved query, final token/bytes/checksum, all pending
temporary allocations, immutable manifest, held physical ownership and receipt
before disposal. Metrics, scores, empty scratch and finally the manifest are
removed with directory fsyncs. Pending ledger releases share one transaction;
an interrupted unlink/fsync/delete/commit rolls all refunds back, leaving missing
temporary files charged. The owner is closed and cannot be retried implicitly.
The retained full ranking and its receipt are never deleted.

Initial missing-module RED:3failed, then3passed in0.48s. Adversarial tests reproduced
late namespace/source changes after hashing:2failed/13passed; final cheap namespace
checks and moving source verification after hashes resolved both (15passed1.71s).
Review approved the revised ordering and bounded one-entry scratch check. Added
all8sync boundaries and SQLite delete/commit denial tests. The latter initially
needed fixture corrections for the existing contextmanager and wrapped SQLite
error API; no production ledger changes were needed.

Final combined88765:128passed in10.07s, covering25cleanup tests plus existing
staging owner/manifest/writer/scorer/hooks and disk score/cohort suites. Cleanup
tests use temporary fixture caches only. This closes the initial successful-cleanup
primitive, not causal source provenance or annual capacity acceptance. Source-bound
raw/feature streams, standalone binding and explicit registered policy remain next.

Archive81924 remains live; latest observed progress29/32buckets and289634078source
rows for batch0049 (batch50of67).49completed batches remain the confirmed baseline.
No additional acquisition, real-cache cleanup, cap increase or annual result claim.

## Source-bound temporary metric production — 2026-09-15

Added ranking_staging_sources and ranking_staging_producer. Preparation builds
the existing complete source-origin candidate history before staging admission,
captures exact daily/history receipts and freezes full configuration, source pin,
lookback/feature membership, engine and physical identities. Subsequent verification
is read-only: it uses source/window verification and exact publication lookups,
never the older builder-backed verification callbacks. Missing candidate receipts
fail rather than rebuilding during a cleanup transaction.

The producer composes unchanged raw/feature partition and metric kernels inside
the invocation's reserved scratch. Candidate streams on both routes, raw ordered
fills and the complete output stream must all be exhausted. Spill removal checks
held ownership; failures keep charged files. Final source verification is followed
by cheap manifest/namespace/output identity guards, not another hashing callback.

13866 RED8 missing-module failures;29781 GREEN8 in15.37s for full metric rows,
pooled/per-asset and dormant candidates.56395 RED4/10pass plus80343 RED2 reproduced
partial writer consumption, late scratch substitution, replaced prepared binding
and late output mutation.38258 GREEN16 in29.16s.22247 reproduced incomplete input
consumption; its4 score-comparison failures were test representation mismatches
(the existing reader strips null percentiles). Comparing both complete stored
Parquet outputs fixed the test boundary without changing scoring arithmetic.
68986 GREEN21 in43.37s includes full raw/feature rankings with eligible and
zero-eligible cohorts.

Review identified a final catalogue callback gap.50362 RED1 proved a candidate
receipt could disappear during final engine hashing. The verifier now holds
database/marker identities and a data-version observer through all callbacks,
with final lease checks.73866 GREEN3 in4.87s covers this regression and missing
daily/history receipts with allocation forbidden. Narrow rereview approved both
the catalogue guard and explicit input consumption tracking.

Combined post-format regression91382 completed102passed in88.01s, covering the
new producer and existing raw/feature producers, staging cleanup/scoring/writer.
Standalone saved provenance/binding and scheduled policy
integration remain unimplemented. No real source expiry or staged run enabled.
Archive81924 is still live, latest31/32buckets and309605360source rows for batch0049.
The archive engine remains12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

## Standalone staged ranking receipts — 2026-09-16

Added an explicit bounded_ranking_staging_v1 SavedFeatureRankings policy and
capture_staged API; the default still uses published intermediate producers.
Staged-policy objects reject legacy capture methods. Exact saved lookup precedes
pending-obligation admission and source preparation; hits still honor external
validation and do not clean up failed work.

New binding/provenance/capture/read modules retain only the full ranking plus
bounded metadata publications. Provenance includes exact query, source and causal
candidate context, ordered effective scoring configuration, producer/resource
bounds and engines, consumed metric/score checksums/bytes/counts, final ranking
pin and independently checked counts/selected-row digest. No temporary path,
allocation token or invocation ID is embedded in publication inputs. Complete
descriptor size and causal source/query validation precede staging admission.
Loading validates exact raw/feature window membership using the already-verified
source session's immutable metadata, without reopening retired feature files.

96390 RED5 missing-API failures; initial implementation exposed typed tuple/list
context comparison, diagnosed from persisted fixture metadata (canonical bytes
matched).5507 GREEN5 in12.12s covered raw/features/pooled full-row equivalence,
successful cleanup and fresh-lease no-producer reuse.49925 RED3/12pass demonstrated
rehashed window/candidate/producer-engine forgeries; persisted candidate context
and route-specific checks produced99365 GREEN15 in31.92s. Crash tests cover final
settlement, provenance publication, saved publication and precleanup interruption:
charges remain, new computation blocks, and published saved receipts stay readable.

15141 RED3 exposed final external artifact/receipt mutation and a noncallable
validator accepted too late.66522 GREEN3 in5.92s after pinning retained artifact
and catalogue through callbacks and rejecting invalid callbacks before work.
37200 RED2 and93140 RED1 covered direct-load late catalogue mutation and exact-hit
validation bypass; a reusable staged catalogue guard and guarded hit validation
fixed these.46756 GREEN21 in51.44s. Expanded84328 GREEN33 in79.30s adds feature
forgeries, changed effective config, boolean resource bounds and explicit policy
rejection of permanent-intermediate capture.

Final narrow review caught consumed feature/candidate mutation after external
validation.98621 RED3 plus48070 targeted RED reproduced removal before rejection
and wrong prepared scope stranding reservations. PreparedStagingSource now exposes
a cheap verify_identity guard for frozen binding/config/caller, prepared data and
named engine-code physical identities; capture invokes it after external callbacks.
Preflight validates complete causal metadata before admission.14738 GREEN3 in6.29s;
narrow rereview approved. New helper dependencies intentionally change the staged
source fingerprint. No real staged policy is enabled.

Combined postformat78481 (staged/legacy saved receipts, lookup, raw receipts,
staged producer/cleanup/scorer) completed:143passed in220.27s, exit0.
Scheduled/registered/API policy wiring, real capacity acceptance
and annual saved runs remain open.

Archive81924 completed successfully:50/67batches, coverage2025-06-02 through
2026-05-18 exclusive,319594956qualified source rows. All4 batch0049 manifest hashes
matched the SQLite records. Qualification SHA:
3e87c397c58710fd2262db34e6c1626b32b295909f57a06a2573fb310e2e7331.
Approved batch cleanup deleted336raw/import files totaling6217224763bytes;
cumulative16802files/248378568143bytes, all recorded deleted. Canonical compact
payloads remain; raw recovery requires redownload unless an external copy exists.
Lifetime reservations206119456083bytes/8401keys were verified against the unchanged
322122547200byte cap before advancing. Next batch0050 covers2026-05-18→2026-05-25,
168objects/5765371425bytes, projected cumulative211884827508bytes. One approved
step started in session39205 (no loop/retry); latest observed31/168downloads.
Archive engine reverified unchanged. No derived-cache deletion or cap increase.

## Explicit scheduled and registered staging — 2026-09-16

Task5 now wires ranking_staging_policy independently of feature_history_policy
through annual registration, proxy manifest/qualified registration validation,
owned loading and scheduled activity. Both metadata and live facade contexts freeze
the policy; absent policy keeps legacy published producers. Scheduled engine pins
include staged binding and policy dependencies. Exact saved lookup precedes the
blanket pending gate, which runs before historical-route recovery/feature preparation.
Both feature and historical raw staged captures receive validate_rank through the
independent receipt checks and every successful cleanup boundary. No real manifest
or pinned cache-resource module changed.

94669 RED6 reproduced missing scheduled policy API.42310/32722 exposed two fixture
mistakes (invalid reservation purpose, weekly effective config passed to a daily
facade); after correction20371 GREEN6 in7.94s. Legacy scheduled tests in32722
passed33.61781 RED showed missing annual policy API. Lightweight registration
tests reproduced ignored/unknown policies; their extra positional placeholder was
corrected after93131, then GREEN7 in0.23s.93131 also passed all6 annual publication
fixtures, including all four staging/feature-policy combinations and rename/reopen.

28908 GREEN3 in48.94s covers staged historical raw misses after actual fixture
feature retirement, independent complete raw reference, pooled/per-asset scopes,
native positions, forward history and fresh-lease weekly/daily exact reuse.
18176 API GREEN4 in50.86s covers staged saved weekly/daily/preview comparison,
both8/16GiB envelopes and both feature policies.81847 GREEN14 in68.13s includes
calibrated final-cleanup-engine mutation of scheduled policy/effective config/
decision and registered disk/in-memory/facade policy; both temporary Parquets
remain when rejected. It also covers distinct full BTC/GOLD/pooled historical
receipts.77864 GREEN1 in7.18s verifies three-decision retained ranking growth,
exact conservative admission, no permanent metric/score publications and no
successful pending allocations or temporary files.

Initial narrow review found no wiring blocker and requested cleanup-boundary
regressions, now added. Rereview approved the scoped Task5 coverage, conditional
on terminal postformat evidence; no annual-capacity or real-expiry approval.
All thirteen affected files formatted
only after fixture jobs terminated. Postformat combined backend/API regression
70280 completed exit0:93passed in501.56s. Task5 fixture integration is verified;
this does not establish annual capacity. Annual capacity,
real policy enablement, all67 acquisition batches and saved annual runs remain open.
Archive39205 remains live, latest141/168downloads for batch0050; no retry launched.
Read-only SQLite observation confirms real derived cache unchanged:495retained
allocations/5286350121bytes, no pending. This is not a full payload-integrity audit.
Archive engine hash remains12b5ab96733c29479e6067b214ec5416fb8923261773851bd374171392aab541.

## Post-staging real-capacity gate — 2026-09-16

Read-only ledger inspection confirms495 retained allocations/5286350121bytes and
no pending. Under17179869184bytes shared cap,33554432metadata and4831903744fixed
staging admission, only7028060887bytes remain for ALL new retained objects. This
includes new full-source-pin features/candidates, ranking receipts and anchors;
the old5.286GB is preserved. No replacement cache or deletion is assumed.

Independently reopened the existing real full-ranking Parquet footer:108185rows,
8323840bytes; SHA408f4ce50b71e4cefed22af7b4ba5ec7325170e83311a0d1bc678350411f3fbd
matches the recorded reference. Its76.94079585894532bytes/row applied to the earlier
114133907daily-BTC known-prefix row lower bound gives8781553639illustrative bytes,
1753492752more than the remaining admission headroom before any new feature or
candidate object. Fitting that BTC row count alone would require at most
61.57732677108828bytes/row if no other new retained data existed. This is a capacity
warning, NOT an encoded-byte lower bound or proof of annual impossibility.
Compression may change. Weekly Monday receipts may share daily outputs; do not
double-count without schedule/query identity evidence.

Do not enable real staged/rolling production based on fixture success. A read-only
all-four-market candidate-growth diagnostic63320 completed exit0 against the fixed
batch0049 qualified pin (350files/319594956rows, cutoff2026-05-18). It hashes all
inputs before/after, uses1DuckDB thread/256MB/no spill, and writes no cache payloads.
It reports counts per market under an all-markets-every-decision scenario; actual
expanding-market/native qualification must still be applied before strategy bounds.
This scan uses only local data, no AWS requests. Archive39205 downloaded168/168
objects and published batch0050raw, then began import; it is not yet qualified.

63320 terminal evidence:241.675seconds, peak RSS928759808bytes; all before/after
file hashes/identities and qualification/engine checks passed. Candidates at known
cutoff:BTC430600,GOLD51008,SP50034363,TSLA31436. Daily all-market scenario136488871
rows; weekly19765963rows. These are NOT final expanding-market strategy bounds.
Full table/method/capacity caveats recorded in
`docs/hyperliquid-annual-capacity-check-2026-09-16.md`. Requested user approval for
up to64GiB in the same local cache; no response/authority yet. AWS300GiB unchanged.
Additional direct dataset/qualified-registration/annual-registration regression
4096 completed exit0:56passed in186.50s. Archive39205 last observed140/168imported,
5414210rows. The local-cache cap decision remains pending; no expansion performed.
