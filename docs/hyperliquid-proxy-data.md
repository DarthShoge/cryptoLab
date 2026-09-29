# Hourly research proxy acquisition

## Real-scale runner blockers — September 9

The first actual 90-day BTC lookback exceeds the runner's existing candidate and
per-wallet history bounds: 108,185 candidates and 6,662,648 distinct fills for the
largest wallet, versus respective 100,000 limits. Opening the minimal prefix also
fails under the 256 MB query-memory cap. These findings supersede any implication
that passing synthetic publication/API tests establishes real annual readiness.
See [the measured capacity audit](hyperliquid-90day-capacity-audit-2026-09-09.md).
The acquisition worker continues unchanged; a bounded query/ranking design fix is
needed before actual annual execution, without sampling wallets or shrinking scope.

## Annual publication integration — September 9

`annual_registration.register_annual_dataset` now connects the completed-job gate,
verified market bundles, observed-history evidence and interior canonical projection
to normal immutable dataset publication. It validates both weekly and matched daily
price coverage, closes and removes staging-path checkpoint caches, rechecks sources
and engine identity, then publishes the final directory. It performs no downloads.

The final-path integration test rebuilds the checkpoint cache, loads 96 retained
synthetic physical activity rows and executes a 25-point hourly equity series.
This is a short synthetic integration proof, **not an annual strategy result**.
Numeric default-configuration settings retain JSON numeric types. Exact original
price, funding and qualification manifests are retained byte-for-byte; all four
provenance sidecars are pinned and checked on dataset construction/reverification.
Corruption, missing provenance, mismatched identities and overwriting an existing
target reject. An injected late coverage failure leaves no published target or
owned staging directory and leaves the original job records unchanged.

Forty-eight focused registration, native-evidence and dataset tests passed in
50.54 seconds; narrow review found no remaining important issue. After formatting,
the full core suite passed **556 tests in 214.89s** (22 existing calendar/NumPy
deprecation warnings), and the API suite passed **45 tests in 102.46s** (two
dependency deprecation warnings). The initial combined sandboxed run stalled at
HTTP integration and was interrupted; the bounded API rerun outside the sandbox
completed. Its 45-second diagnostic stack dump occurred during the synthetic
annual comparison, which subsequently passed; it was not an assertion failure.
The live archive's frozen engine identity was separately verified unchanged.

Archive checkpoint: **17 of 67 batches qualified**, covering source days June 2
through September 29, 2025 exclusive: 119 canonical files, 71,724,744 physical
rows, 6,246,179,279 compact bytes. The latest report is
`batches/0016/qualified/qualification_j0kpv3h8/manifest.json`, SHA-256
`d114c2e304572a37d1f96f3d4c678407a1e81c73bb34b5937e1e97e794cb6467`.
Its bytes, persisted stage records and frozen engine identity were independently
checked after the live worker published it; this did not repeat the full corpus
validation. Lifetime reservations at that checkpoint were 67,762,666,277 bytes
(about 63.1 GiB of the approved 300 GiB cap). Cumulative job-owned staging cleanup:
5,714 files /77,037,932,982 bytes, including 336 files /5,525,656,352 bytes from
the latest batch. Raw reconstruction requires redownload; the lifetime ledger is
not refunded. No additional download worker or retry was started. Earlier detailed
notes below are historical checkpoints. Final real native-boundary qualification,
publication, saved annual weekly/daily runs and independent reconciliation remain
unfinished until the full archive is qualified.

## Observed-history evidence gate — September 9

The completed-source observed-history qualification and loader-side evidence
verification are implemented and reviewed. See
`docs/hyperliquid-native-start-qualification-2026-09-09.md` for the exact policy.
This combines complete scoped source coverage, validated event bounds and verified
funding; it does not infer listing dates. The loader now rejects altered evidence
bytes, per-market dates, scope, or missing sidecars for datasets claiming the policy.
Thirty-eight related tests pass; legacy datasets remain compatible.

The real annual archive is still partial and cannot yet pass the gate. Final
dataset publication, post-rename cache rebuilding and saved annual runs remain
unfinished; helper tests are not evidence of that end state.
## Registration market-evidence verification — September 9

`registration_market_inputs.verified_market_bundle` now checks each pinned bundle
and normalized output, reconstructs the bundle offline from its retained original
acquisitions in an owned temporary directory, compares the complete reconstructed
manifest, and rechecks the original pins. Rewriting a source-completeness flag and
recalculating the bundle hash does not bypass original coverage evidence. Temporary
reconstruction outputs are removed; retained source inputs remain untouched.

Both real unified bundles passed this reconstruction check: 14,034 price bars and
28,084 funding events. Seven related bundle-verification/completed-job tests pass
in 7.87 seconds, and narrow review found no important issues. Native-history
qualification and final dataset publication are still separate unfinished steps.
## Completed-job registration gate — September 9

`annual_registration.completed_job_inputs` now rejects any job without every
planned batch's qualified stage. For a complete job it verifies the final pinned
report, canonical content and retained raw-source provenance, exact source-key
membership, the padded interior interval, and unchanged job metadata/artifact
identities. It returns source inputs only, with native availability explicitly
unqualified. It does not perform downloads, retries, cleanup or dataset publication.

Two synthetic-job regressions passed, covering rejection of an already-qualified
but incomplete prefix and acceptance of a completed corpus after raw cleanup,
plus padding and report-corruption rejection. Narrow review found no issues.
An actual check against the live annual job correctly returned
`Archive job unfinished: every planned batch must be qualified`.

Full core verification after these changes: **535 tests passed in 160.63 seconds**,
with 22 existing exchange-calendar/NumPy deprecation warnings.

The remaining annual-registration work is native/funding boundary qualification,
bundle revalidation, final dataset publication/cache handling and actual saved
weekly/daily runs; the new source gate is not that completed end state.

## Annual publication preparation — September 9

Unified bundles now provide the single price/funding inputs for annual registration,
under `.hyperliquid_cache/annual_unified_market_inputs_20260909/`:

- `market_bundle_62df406284c5454bbc72db7f85316148`: 14,034 price bars,
  473,495 Parquet bytes, eight original acquisition manifests.
- `market_bundle_f88746ab919e465b96fde3f501d72499`: 28,084 funding events,
  649,285 Parquet bytes, fourteen original acquisition manifests.

All four markets and their distinct retained intervals remain explicit; cross-class
price coverage still begins February 3, 2026 and is not an annual-eligibility claim.

`registration_partitions.publish_interior` implements bounded event-time projection
into unpublished dataset staging. Wholly interior files use hardlinks (or bounded
copy fallback), boundary files use streamed exact filtering; every canonical column
and duplicate is preserved. Original source identity is rechecked. Output limits
cover Parquet footer writes through the existing capped stream. Ten focused tests
cover exact boundaries/out-of-order spill, empty partitions, preservation, copy
fallback, corruption, overwrite refusal, timestamp precision and byte caps.

Real projection proof: `.hyperliquid_cache/annual_interior_proof__dfu4ur6/`, selecting
June 2, 2025 12:00 UTC–June 3 exclusive from the first canonical daily file.
Independent full Arrow-table equality checked all columns: 624,032 source rows to
352,376 retained rows, 29,621,966 bytes, source hash unchanged. Output SHA-256:
`d71de73afcdec51d547ffc23f4bb54fb76b633fea76dc2c99714aa1ba511afc9`.
This is an isolated projection proof, not a registered annual dataset.

The remaining registration plan is
`docs/superpowers/plans/2026-09-09-hyperliquid-annual-registration.md`. Its final-job,
native-history, bundle, loader and saved-run integration tasks are unfinished.
Review identified a staging-path cache hazard: scheduled validation caches must be
closed and moved outside the staged dataset before atomic publication; reopening
the final dataset path must rebuild and verify its own cache.

## Annual price-validation integration — September 9

Run validation now uses the same explicit native-history warmup lower bound as
market selection for non-BTC markets. Previously it demanded prices throughout
the market's warmup even though no follower position could yet be opened. BTC
still requires full-period benchmark prices. Missing native-history evidence does
not waive earlier coverage, and shorter lookbacks require their own earlier prices.
The check remains conservative: all scheduled sessions after the lower bound are
required, not just sessions traded by a realized winning cohort.

Deferred session markets additionally require a real prior completed close, fresh
through the first hourly accounting tick under the configured stale-price limit.
A new XNYS regression reproduces the 13:30 opening / 14:00 funding / 14:30 first
bar-close edge case and verifies use of the prior Friday close and stale-seed
rejection. No synthetic price or funding fill is introduced. Review confirmed the
fix; 33 related availability, weekly, annual and pipeline tests passed in 10.73s
(six existing calendar/NumPy deprecation warnings).

An offline diagnostic using the retained real-price bundles passes the annual
September 1, 2025–September 1, 2026 price check for both 90-day weekly and daily
configurations; a 30-day lookback correctly rejects insufficient earlier coverage.
**This diagnostic conditionally supplied funding-source boundaries as native
starts; it does not prove those boundaries.** Final native-history qualification,
registration and actual saved annual comparison remain required. The live archive
engine's frozen hash identity was verified unchanged after this pipeline-only fix.

## Offline market bundles — September 9 checkpoint

The new `proxy_market_bundle.bundle_market_inputs` combines existing acquisitions
without network calls or modifying the running archive engine. It verifies input
hashes, bounds, conflicts, calendar coverage and settlement timestamps before
publishing a uniquely named, durable bundle. Incomplete original source flags and
gap evidence remain pinned; bundles claim neither native availability nor research
eligibility. Review findings about pre-expansion date bounds and timestamp
precision were reproduced and fixed. All 24 focused tests pass (three existing
exchange-calendar/NumPy deprecation warnings). The full core suite subsequently
passed: **514 tests, 16 exchange-calendar/NumPy deprecation warnings, 141.93s**.

Four real bundles are under `.hyperliquid_cache/annual_market_bundles_20260909/`:

| Bundle suffix | Coverage | Rows | Parquet bytes |
| --- | --- | ---: | ---: |
| `fcbf000563ce4efeb663b2fc39701e0e` | BTC prices, June 2, 2025–September 2, 2026 | 10,968 | 381,758 |
| `482fd216394e431ea89344d3d84a6e60` | BTC funding, same interval | 10,968 | 333,533 |
| `34b53d3adff24cf08a5e053f875eeacf` | Cross-class funding, each retained source interval | 17,116 | 401,368 |
| `430715c4bd394a6294c04584ac091157` | TSLA/GLD/SP500 proxy prices, February 3–September 2, 2026 | 3,066 | 71,582 |

Directory names prepend `market_bundle_`; all ends are exclusive. Independent
offline comparison checked every output row against its retained source projection
and verified manifest/output hashes and counts. Cross-class price counts are
1,022 per market. Funding counts are TSLA 7,017, GOLD 6,081 and SP500 4,018, starting
November 13, 2025 15:00 UTC, December 22, 2025 15:00 UTC and March 18, 2026 14:00 UTC
respectively. These are input intervals, not independently established listings.

**Annual registration remains gated:** native-history/eligibility evidence must
show that both weekly and matched daily followers require no prices before the
retained February 3 interval. January gaps were not filled or erased. There is no
saved real annual comparison yet. Archive checkpoint during this work: 14 qualified
batches, batch 14 compact and validating, 57,871,679,814 lifetime reserved download
bytes. SQLite and the live worker supersede this snapshot.

## Approved annual acquisition underway

The user approved a **300 GiB (322,122,547,200 byte) lifetime new-download cap**,
then confirmed proceeding after the approximate US$30-before-tax cost explanation.
Earlier pending-approval notes below are historical. The live job is
`.hyperliquid_cache/annual_job_20250901_20260901_300gib/`; its `APPROVAL.md` records
scope, and SQLite freezes inventory, parser, markets and spending. No automatic
retry or spending-ledger reset is authorized.

Batch 0 (June 2–8, 2025) completed acquisition, compaction, qualification and
journaled disposal. It reserved 2,865,419,167 download bytes and retained 3,942,424
canonical rows in seven daily files totaling 338,839,457 bytes. All 336 job-owned
raw/intermediate payloads (3,363,174,912 bytes) were removed after validation;
manifests and compact evidence remain. Raw reconstruction requires redownload,
which is not automatically authorized. The old approved shared cache is untouched.

A sequential worker is advancing the remaining 66 batches and exits on its first
error. Read job artifacts and spending SQLite for current progress; this note is
not a substitute for polling the live worker. The annual evaluation and saved
weekly-versus-daily comparison have not yet been produced.

Subsequent checkpoint: batches 0–1 are qualified through June 16, 2025 exclusive,
retaining **8,832,552 rows / 759,595,842 bytes** in 14 daily files. Lifetime transfer
reservations are 6,497,269,403 bytes. Cleanup totals are 672 job-owned staging files
and 7,614,154,033 bytes; reconstruction of removed raw bytes requires separately
authorized redownload. Batch 1's report is pinned at
`batches/0001/qualified/qualification_n6w0jgt7/manifest.json`, SHA-256
`cc877012d6cbcc61097eb7a6be750afafd2d68911d9b0aeebd06a3698f64a0b5`.

**Real pruning observation:** the second prefix validated all 14 files and skipped
zero old files. Observed daily BTC trade-ID intervals span almost the entire same
numeric range, so interval pruning has not reduced the candidate corpus here.
Correctness remains checked, but annual validation cost must not be estimated as
constant work per new batch. The running frozen engine was not modified.

Two BTC price segments are complete (June 2–September 3 and September 3–December 5,
2025 exclusive), each with 2,232 expected hourly bars and no missing starts, under
`.hyperliquid_cache/annual_btc_price_segments_kevaewyy/`. The BTC price worker is
continuing. The earlier mixed-proxy worker stopped on a preserved TSLA early-close
session anomaly; see `.hyperliquid_cache/annual_price_segments_81ci20fb/YAHOO_SESSION_FAILURE.md`.
This is a price-input subset, not a crypto-only replacement strategy.

### Remaining annual publication integration

Cross-class price inputs now exist under
`.hyperliquid_cache/cross_class_prices_2026_7vo5k549/`. The January–April segment
is explicitly incomplete (TSLA/GOLD/SP500 gaps on January 30–February 2), while
the two later segments are complete. Independent hash, OHLC/overlap, policy and
XNYS-calendar checks establish **1,022 bars per market with exact coverage from
February 3 through September 2, 2026 exclusive**. Earlier gaps remain recorded and
unfilled. Use of this later interval in the annual 90-day-lookback run is conditional
on native-history/eligibility evidence proving that both weekly and matched daily
followers need no earlier prices. It does not authorize earlier/shorter-lookback
comparisons or make the incomplete source manifest complete.

Latest inspected archive checkpoint: **14 qualified batches**, source coverage
through September 8, 2025 exclusive, **62,208,546 rows / 5,393,892,487 compact bytes**.
Lifetime reservations were 54,934,859,783 bytes at inspection (including the next
batch's in-flight requests). Cleanup records 4,706 removed job-owned payloads /
61,430,181,194 bytes. Shared cache and all canonical/provenance artifacts remain.
The live worker is advancing; SQLite is authoritative for subsequent progress.

Additional native funding inputs are verified under
`.hyperliquid_cache/annual_cross_class_funding_5ie2r4nu/`: TSLA 7,017 hours from
November 13, 2025 15:00 UTC, GOLD 6,081 hours from December 22, 2025 15:00 UTC,
and SP500 4,018 hours from March 18, 2026 14:00 UTC, all through September 2, 2026
exclusive. Raw/output hashes and exact combined hour sequences were checked;
there are no missing or duplicate hours in the requested intervals. These are
funding-coverage facts, not independent native listing/activity-coverage proofs.

BTC market-input acquisition has now finished: five price segments under
`.hyperliquid_cache/annual_btc_price_segments_kevaewyy/` and five native funding
segments under `.hyperliquid_cache/annual_btc_funding_segments_cggvsav0/`, each
covering June 2, 2025 to September 2, 2026 exclusive. Independent offline checks
verified source/output hashes and exact combined timelines: **10,968 price bars
and 10,968 funding hours, no missing or duplicate hours**. Actual funding settlement
times remain distinct from their hourly coverage buckets. Each directory contains
`VERIFICATION.md`. These complete the BTC inputs, not the other market inputs,
annual native-fill acquisition, registration or saved comparison runs.

Read-only inspection while acquisition is active confirms the following gaps:

- `proxy_download.download_prices` and `proxy_funding_download.download_funding`
  both retain their 93-day acquisition bounds. Annual inputs need bounded segment
  orchestration and exact overlap/coverage reconciliation, not an unbounded request.
- `proxy_registration.register_dataset` accepts one normalized activity manifest,
  at most 169 files/8 GiB, and loads the legacy full validator. It cannot publish the
  new annual compact-prefix job as-is. The example registration CLI also still fixes
  its dates and default strategy to the old August 2026 short-window demonstration.
- The dataset loader already supports `validation_mode=sharded_v1`, 64 GiB/5,000
  files, scheduled v2 configs and explicit native-history evidence. Its dataset
  paths must remain inside the registered directory; canonical sharing/publication
  must respect that boundary, preserve immutable inputs and enforce disk limits.
- Annual native-history records must cover every selected market with explicit
  evidence and whole-hour boundaries. First observed fills, proxy availability and
  funding response starts are not interchangeable listing/coverage proofs.
- Existing proxy corporate-action policy is raw prices with no detected actions;
  annual segment composition must check the whole retained interval rather than
  inherit a short segment's action-free status. Funding must have exact required
  hourly coverage for the qualified native intervals; missing rates are not zero.

These are publication/acquisition integration requirements, not evidence that an
annual research dataset is already complete. No frozen acquisition engine files
were changed during this inspection.

## Restart-safe owned staging cleanup (2026-09-08)

New jobs use SQLite schema v2 and automatically dispose of qualified job-owned
raw `.lz4` payloads and normalized `.parquet` intermediates before acquiring another
batch. All raw/normalized manifests, original key/ETag/size/hash provenance, compact
files, qualification reports, catalog and spending state remain. Job-owned copies
of cache objects may be removed; the original external cache is never a target.
Old job databases/engine snapshots are not migrated or rewritten.

The deletion journal freezes the complete exact target list and qualification pin
before any unlink. Every remaining payload is checked first. Paths must stay under
the recorded job staging directory; symlinks, missing unjournaled files, changed
content and journal mismatch fail closed. Each unlink is followed by directory
fsync before its completed status is committed. A restart can finish an interrupted
unlink without another GET, while a recreated completed target is left untouched
and reported as an error. No recursive deletion is used.

Job results now expose `cleanup.policy`, cumulative `deleted_files`/`deleted_bytes`,
and `raw_payloads_retained`. Immutable qualification reports still grant no disposal
authority themselves; the job's validated scope and durable journal implement the
approved cleanup policy. Research eligibility remains false. Deleted raw payloads
can require redownload to reconstruct, and spending reservations are never refunded.

An offline synthetic two-batch smoke job removed **96 staging files / 451,162 bytes**,
retained two compact partitions, eight manifests and all 48 external-cache payloads,
and reopened at `all_batches_qualified` with zero network requests/reservations.
SQLite integrity was `ok`; its journal records all 96 files deleted, with no raw or
normalized payloads remaining inside the job. Evidence:
`.hyperliquid_cache/cleanup_job_proof_nvnt_0th/measurement.json`. Runtime was 6.97s,
peak RSS 268,176 KiB, with concurrent tests; this tiny synthetic measurement is not
an annual footprint estimate. No real archive or existing proof-job data was deleted.

Verification: **490 core tests passed**, with 13 existing dependency warnings.
The 12 cleanup cases cover cross-batch reuse, original-cache preservation, interrupted
intent/unlink recovery, missing/changed/symlink targets, compact corruption, recreated
deleted paths and damaged journal membership/pins/status. Narrow safety review found
no important concrete gaps. This completes the staging-disposal infrastructure, not
annual acquisition, coverage qualification or the saved annual strategy comparison.

## Durable job qualification gate (2026-09-08)

New acquisition jobs now finish each batch through four persisted stages: raw,
normalized, compact, and qualified. `step` returns `prefix_qualified`, including
the compact manifest and a hash-pinned `qualification_report`; a completed job
returns `all_batches_qualified`. Qualification remains canonical/source validation,
not a research result: `research_eligible` and `raw_disposal_authorized` are false.

SQLite records the report only after publication and directory fsync. Each report
must match the frozen compact prefix, raw provenance, markets, source interval,
previous report pin and qualification engine. Restart verifies prior reports and
retained canonical content before another network request. A compact-but-unqualified
batch is finished locally before advancing. If report publication survived but its
SQLite pin did not, recovery recomputes validation locally and requires the exact
same report hash before adoption; a newly computed hash of an untrusted orphan
alone is insufficient. Changed orphan bounds, missing reports and corrupt content
fail closed. The original orphan is preserved on mismatch.

The 23 focused acquisition-job tests pass, including interruption at all four
publication boundaries, orphan-bounds tampering, no repeated GET during recovery,
qualification failure/retry, prior-report corruption, lifetime budget preservation,
and the offline CLI. No real acquisition or data cleanup was performed for this
integration. Existing proof-job engine snapshots are not migrated or rewritten;
their original results below remain historical evidence. Owned raw/intermediate
disposal journaling is still the next implementation stage.

Full verification: **478 core tests passed**, with 13 existing dependency warnings.
Read-only review's orphan-adoption finding was reproduced by a failing test and
resolved by exact local recomputation; follow-up review confirmed the fix.

## Incremental canonical prefix qualification (2026-09-08)

`prefix_qualification.qualify_prefix` publishes a hash-pinned validation report
for a contiguous prefix of boundary-spill-preserving daily compact batches.
It preserves complete frozen source-key membership and original raw-object
key/ETag/size/hash provenance, canonical content/schema identities, and exact
per-file event-time and per-market trade-ID bounds. Raw manifests are rehashed;
raw payloads are not re-read, and the report explicitly records that distinction.

An extension rehashes all retained canonical files, requires the same Arrow schema
and exact prior manifest prefix, then validates every new file together with older
files whose measured per-market trade-ID intervals overlap. Both native conflict
keys contain market and trade ID, so disjoint intervals cannot share a conflict
key. This does not assume that IDs increase with time or that overlaps occur only
on adjacent days. Overlapping candidates use the existing exact sharded SQL checks;
no probabilistic economic fingerprints or wallet sampling are substituted.

Prior report reuse requires the caller's pinned `{path, sha256}` identity and the
same engine version. Input/report mutation fails closed before atomic publication.
The corpus remains capped at 64 GiB/5,000 files, reports at 16 MiB, with existing
256 MB DuckDB and 2 GiB spill limits. Rehashing the old corpus still costs sequential
I/O; the shortcut avoids repeating expensive conflict validation for disjoint files.

Verification: 473 core tests passed with 13 existing dependency warnings. Fourteen
focused cases cover distant duplicates/conflicts, per-market inclusive range
overlap, strict extension, schema changes, incomplete source keys, corruption of
skipped old files, and manifest/report mutation. Narrow read-only review found no
important correctness gaps. This helper is not yet wired into controller cleanup.
Its report explicitly sets `research_eligible=false` and
`raw_disposal_authorized=false`: source-partition membership is not proof of
exchange completeness, native availability, or proxy/funding coverage. No paid
requests, raw deletion, or production dataset changes were made for this stage.

The retained real Aug 1–7 week passed all 32 groups: **4,875,344 rows**, seven
canonical files totaling 462,486,347 bytes, and provenance for 168 raw objects.
Fresh-process runtime was 240.26 seconds and peak RSS 672,840 KiB (about 657 MiB);
core tests ran concurrently during the early part. The report is 58,119 bytes.
Evidence: `.hyperliquid_cache/prefix_qualification_proof_qq44k7xm/measurement.json`
and its pinned `qualification_oddxdm_2/manifest.json`. This is a full first-prefix
measurement, not a measured annual incremental speedup. All original data remains.

## Bounded large-object transport (2026-09-08)

New raw manifests now freeze a **384 MiB compressed-object ceiling**, covering the
largest inventoried object (359,334,134 bytes). Absent `max_object_bytes` continues
to mean the legacy 128 MiB ceiling on resume. The transfer still reads at most
1 MiB at a time, checks the frozen length/ETag/hash, reserves bytes before GET and
never retries automatically. The 6 GiB batch/lifetime budget controls and importer
limits are unchanged: **2 GiB decoded per object, 96 GiB decoded per batch**.

Fresh-process synthetic measurement transferred a 384 MiB generated object plus
23 tiny objects, then verified completed resume with no repeated request. Maximum
read was 1,048,576 bytes; peak RSS was 85,120 KiB (about 83 MiB), versus an 82,660 KiB
baseline. Runtime was 1.71 seconds on local generated data—not an AWS throughput
estimate. Evidence is
`.hyperliquid_cache/synthetic_transport_proof_ipxbfax3/measurement.json`.
Those raw fixture bytes are deliberately **not LZ4 or market data**.

The real read-only annual planner now reports no unsupported compressed objects:
10,969 objects, 67 batches, 168 verified reusable objects and 290,296,248,717 bytes
(270.36 GiB) of remaining transfer. `transfer_ready: true` means only this size
gate passes; it does not grant paid approval, decoder qualification, gap-free
event coverage or research eligibility. The 300 GiB download-cap approval remains
pending. No real source GETs were performed for this change.

Verification: 459 core tests passed with 13 existing dependency warnings, including
real-maximum and ceiling-sized generated transfers, rejection above limits, exact
hashes, one-time reservations and legacy resume behavior. Existing job engine
snapshots deliberately fail closed across source-code revisions; old proof job
metadata and spending state were not rewritten to bypass that check. Published
compact evidence remains intact. Global qualification and owned raw cleanup are
still separate outstanding stages.

## Persistent bounded acquisition controller (2026-09-08)

`tools/run_hyperliquid_archive_job.py create` freezes the validated inventory,
ingestion markets, verified cache identities, parser/engine hashes and lifetime
download cap in a job-owned SQLite database. `step` advances at most one
chronological batch through raw acquisition, boundary-spill-preserving
normalization, daily compaction and catalog publication. Both commands are offline
by default. Network execution requires `--accept-approved-download` and an explicit
IAM reader CSV; creating a job does not grant that approval. A cached-only job
can use `--max-download-bytes 0` and has no network budget database.

The controller verifies committed compact content before advancing, and recovers
a uniquely published stage after a crash before its SQLite record. It does not
retry requested-but-uncommitted transfers or silently repeat incomplete local
normalization/compaction. Ambiguous stages, changed engine/scope, corrupt content
and unsafe paths fail closed. A missing catalog can be rebuilt from verified
compact files; missing or empty lifetime spending history must never be rebuilt.
The spending-history check occurs before even remote HEAD requests.

Directory entries are synced before SQLite commits artifact references. Existing
seven-day/6 GiB raw and 8 GiB per-compact limits remain. Before another batch starts,
the job conservatively reserves a full 8 GiB within its 64 GiB canonical ceiling,
including any recovered uncommitted compact output. It also requires 22 GiB plus
64 MiB free space for raw, normalized and compact staging. This may stop a job
before the actual remaining canonical space is full; it never publishes an extra
batch first and checks the aggregate ceiling afterward.

Raw and intermediate files currently remain in the job directory. Consequently,
this controller does **not yet implement the final rolling disk footprint or raw
cleanup policy**. Its terminal acquisition status is `all_batches_compacted`,
with `qualification: pending_global_validation`, not research eligibility or a
completed annual backtest. Global semantic/event-coverage qualification must
precede any disposal. Known oversized source objects are rejected before creation.

Verification so far: 451 core tests passed (13 existing dependency warnings),
including 18 controller regressions for stage and mid-batch restart, cache-only
execution, CLI approval gating, budget loss, corruption, locking, paths, disk
capacity and directory durability. The separately acquired real annual dataset
and its saved weekly-versus-daily comparison remain outstanding.

Real local controller proof: `.hyperliquid_cache/archive_job_proof_AnunUjLw/`.
Its inventory is explicitly derived from the retained annual LIST inventory for
August 1–7 only; its four markets are BTC, xyz:GOLD, xyz:SP500 and xyz:TSLA.
With a zero-byte network cap it copied the 168 existing raw objects, re-normalized
all **4,875,344 rows**, and published **seven daily files / 462,486,347 bytes**.
Reopening and stepping returned `all_batches_compacted` with zero reservations,
without repeating acquisition or normalization. The read-only `reconcile.py`
compared every canonical field against the earlier daily compact baseline except
`ingested_at` (intentionally different for a fresh ingestion): **zero mismatched
rows**, with successful job/catalog SQLite integrity checks. Results are in
`reconciliation.json` beside the script. All original files, production datasets
and reports remain untouched. Batch progress calls local copies “downloaded” and
counts their staging bytes; the job's separate final `reserved_bytes: 0` is its
network reservation measure.

## Reproducible annual acquisition plan and cache reuse (2026-09-08)

`tools/plan_hyperliquid_archive_job.py` reads an existing LIST inventory and
explicitly supplied completed raw manifests. It makes no AWS requests, reads no
credentials, and writes no budget or job state. Example (from the feature worktree):

```sh
.venv/bin/python tools/plan_hyperliquid_archive_job.py \
  --inventory .hyperliquid_cache/inventories/annual_metadata_a8xxziuo/inventory.json \
  --cache-manifest .hyperliquid_cache/proxy_archives/proxy_archive_5_ynp7r_/manifest.json \
  --summary
```

Without `--summary`, JSON includes exact objects in each planned batch and cache
content identities. Source keys, totals, sizes and ETags are validated rather than
trusting audit summary fields. Batches contain complete chronological source days,
at most seven days and 6 GiB each. Oversized individual sources remain in scope
and are explicitly reported; `transfer_ready` is only the current object-size
gate, never approval, decoded-size qualification or event-time coverage.

The real offline run verified all **168 retained raw content hashes** against the
annual frozen identities: **5,604,128,913 reusable bytes**. It recomputed **67
batches**, maximum **6,377,542,749 bytes**, covering **10,969 objects** and
**295,900,377,630 total bytes**. Remaining transfer is **290,296,248,717 bytes**
(270.36 GiB), excluding separately authorized retries. The same six objects exceed
the downloader's 128 MiB per-object limit; the plan reports `transfer_ready: false`.
Inventory SHA-256 is
`57b5326fa5756ab5c2131e2a6dbcf6738d831679db571a2cdcd31b5020c46999`.

`VerifiedArchiveCache` / `CachedArchiveSource` integrate with the existing batch
downloader: cached bodies are verified before opening and checked again against
their frozen hash before successful EOF. Corruption fails without remote fallback.
Only missing objects reach a matching `BudgetedArchiveSource`; local copies do not
consume lifetime network reservations. Batch staging still includes their bytes,
so reuse avoids transfer, not temporary local copies. Frozen cache-manifest and
raw identities are available for the future controller's durable provenance.

This is not yet a durable annual stage controller, a cleanup implementation or an
annual registered dataset. All retained files remain; no paid GETs were issued.

## Source-day compact history (2026-09-08)

The local compactor now accepts `--partitioning source_day` (default remains
`source_file`). It publishes one canonical Parquet file per declared archive day,
including empty days, within the existing seven-day batch limit. Grouping uses
the archive source key, not exchange timestamp: boundary-spill events and duplicate
native identities are retained for subsequent global validation. Canonical schemas
must match exactly; projection comparisons and source rehashes precede publication.
A shared 8 GiB output cap is checked before writes, including Parquet footers.

Offline measurement of the retained August 1–7 week preserved all **4,875,344 rows**
while reducing **168 files to 7**. Input was 717,018,008 bytes; daily output is
462,486,347 bytes (35.50% smaller), essentially the same size as the earlier
source-file compact copy. Runtime was 23.59 seconds and measured peak RSS
352,940 KiB (about 345 MiB). This is one local measurement, not an annual guarantee
or a hard memory ceiling; a test suite was running concurrently.

Evidence: `.hyperliquid_cache/compact_history/compact_9qb6ghd5/manifest.json`.
The original raw, normalized and compact files remain untouched. No production
dataset or catalog was replaced, and no source coverage or native availability
was inferred. The annual controller, large-object handling and paid acquisition
remain separate outstanding steps.

## Boundary-safe multi-batch normalization (2026-09-08)

For annual assembly, `import_archive(..., retain_boundary_spill=True)` and the
import tool's `--retain-boundary-spill` preserve all parsed mapped fills, including
timestamps outside a source batch's dates. The normalized manifest explicitly
records `boundary_spill_retained` and its time policy. Source start/end identify
acquisition partitions, not complete exchange-time coverage. Final assembly must
filter the globally retained interval and deduplicate native identities; individual
batch boundaries must not discard those rows.

Default imports retain their previous half-open timestamp filtering. Existing
normalized files and reports are untouched. The retained August week's manifest
records zero outside-window rows, so this change is not evidence that the current
short-window dataset lost fills. The synthetic adjacent-batch regression preserves
96 fills where independent old-style filtering retains 92.

Source manifest identity is frozen before parsing, and all source hashes are
rechecked before normalized-manifest publication. Changed inputs fail without a
success manifest. All existing source, decoder, batch and output byte limits remain.
This mode neither downloads data nor grants event-time coverage or raw disposal.

## Registered bounded-validation mode (2026-09-08)

Operator-registered datasets may explicitly set `validation_mode: "sharded_v1"`.
This permits a total input corpus up to 64 GiB, with the existing 5000-file and
250,000-row-per-market-file ceilings. It requires scheduled proxy v2 configuration
(weekly or daily); API preflight explains incompatible hourly configuration, and
the loader refuses it before materializing data. Missing mode means `full_v1`,
which retains the old 8 GiB corpus ceiling and legacy behavior.

The initial loader validates every source partition through deterministic trade
groups, including empty partitions that influence global column types. Only the
fully verified causal seed is published, bound to the dataset and validation
engine hashes. Future invalid rows fail before checkpoint publication. Later
loads reuse that verified checkpoint rather than repeating full validation.

This does not raise rolling-query working-set limits or qualify source coverage by
itself. Large registered datasets still require completed acquisition, canonical
daily partition publication, native availability and proxy/funding qualification.
No production dataset was switched to this mode and no live server was restarted
as part of implementation. The requested annual download budget remains pending.

## Job-wide transfer reservations (2026-09-08)

`ArchiveBudget` freezes exact source key/ETag/size identities and a lifetime byte
limit in SQLite. `BudgetedArchiveSource` adapts the existing batch downloader:
HEAD must match that frozen identity, and full object bytes are committed in a
transaction before GET. Failure does not refund bytes. A second request for the
same key is refused across batches and restarts; concurrent reservations serialize.
The underlying AWS client must use `total_max_attempts=1` so SDK retries cannot
bypass the reservation. Completed local files are reused through batch resume.

This is implemented and tested infrastructure, not an approval grant or an active
annual job. No actual annual budget database has been created. Annual controller
wiring, verified import of the existing cache, oversized-object handling and
incremental full-history qualification remain. The 128 MiB downloader object guard
is unchanged. Byte reservations do not cap AWS request charges or the dollar bill.

## Restart-safe archive batches (2026-09-08)

`resume_archive(s3, manifest_path)` and the downloader's `--resume-manifest PATH`
option continue only previously unrequested objects. The command still requires
`--accept-approved-download`; this acknowledges the original batch approval, not
a new budget. The existing seven-day, 6 GiB batch and 128 MiB object limits remain.
No annual acquisition has been authorized or initiated by this change.

Before any GET, resume verifies frozen keys/dates, canonical filenames, object
sizes/ETags, byte accounting and every completed file's SHA256. It preserves
reservations and skips verified completed objects. A completed batch is verified
and returned without network access. New and resumed transfers share an exclusive
directory lock. Publication syncs both file content and directory renames.

An interrupted `requested` object, partial file, corrupted completed object or
unexpected output stops resume before GET. It is deliberately not retried: its
reservation remains spent and recovery needs separate authorization. No source
files are deleted. This per-batch primitive does not yet provide an annual job's
lifetime budget, incremental full-history validation, or raw-disposal controller.

## Expanding-universe eligibility (2026-09-08)

The user approved keeping a full-year evaluation with markets admitted only after
sufficient qualified native history. Registered manifests can now include
`native_history`, one record for every declared coin, with `instrument_id`,
whole-hour timezone-aware `available_from`, `evidence_sha256`, and `description`.
The start is qualified retained-history availability, not a proxy mapping date or
an inferred listing timestamp. Source qualification remains an operator obligation;
a hash alone does not prove it. No production dataset has been populated with
unverified start dates from the preliminary funding probe.

The selected strategy's full required lookback (trader ranking, conviction scale,
and lagged market volume) must fit after that start. The initial hypothesis uses
90 days. Dynamic universe history includes declared but not-yet-observed markets,
with `native_history_not_available` or `insufficient_native_history` exclusions.
Old datasets without these records retain their previous behavior.

The loader rejects malformed/incomplete qualification and fills contradicting the
declared native start, including before reusable checkpoint publication. Replay
requires every funding hour from native availability onward; earlier missing
funding is not filled with synthetic zeros. Preavailability signals/funding reject.
BTC buy-and-hold still requires availability throughout evaluation. Proxy-price
coverage does not need to precede native availability. A market starting after a
selected subwindow ends remains excluded rather than failing that earlier run.

Focused tests cover delayed weekly admission, dynamic future-market explanations,
metadata validation, hourly accounting without backdated funding, and an early
subwindow ending before a new market starts. Existing saved reports and registered
data are untouched. Actual native-history qualification and bounded bulk
acquisition remain outstanding. The older parser was already supported; the
within-hour July 27, 2025 source handoff has since been fixed and tested.

## Registered rolling loader enabled (2026-09-08)

Scheduled proxy v2 backtests and previews launched through the API now use
`ScheduledActivity`; legacy hourly v1 loading is unchanged. First load validates
the complete registered fill corpus and publishes an initial seed. Reuse keys bind
dataset identity, warmup cutoff, and validation code/dependency versions. Source
hashes remain checked on every load; malformed fills or corrupt cached seeds fail
closed. Registered manifests, source files and saved results are not rewritten.

Disposable cache location: `<dataset>/.activity_checkpoints`. SQLite stores indexed
metadata, Parquet stores seeds. Advanced checkpoints are reusable across runs with
the same frozen lineage and cutoff. Publication is serialized across writers and
enforces an **8 GiB aggregate cache cap**, counting partial files. Before writing,
it reserves the maximum 512 MiB seed plus 8 MiB metadata overhead and requires an
additional 64 MiB free-space margin. A capped stream rejects oversized seed writes
before they exceed 512 MiB, including footer writes. Decoded Arrow batches have a
64 MiB post-decode guard. Old engine-version caches and failed partials remain
counted; they are not automatically deleted. The cap is additional to compact
source history, query spill and saved ranking/report space, not total application
disk usage. External programs can still consume free space concurrently.

Verification: 341 core tests and all 43 API tests passed. The API's annual fixture
executes weekly and daily runs through the rolling loader, saves and compares them,
and checks shared initial-checkpoint reuse. Browser save/clone/compare checks passed
for annual, multiweek and legacy hourly fixtures. A measured cold daily annual
fixture took 36.27 seconds before the final capped-writer change; annual-only test
deadlines were increased from 25 to 90 seconds, not production resource limits.
The local app on port 8010 was refreshed with no active jobs and passed health.

**Still outstanding:** registered input validation retains the 8 GiB ceiling.
Incremental validation/acquisition and verified year-long real-data coverage are
not implemented by this bridge. Annual tests remain explicitly synthetic; no new
downloads, source deletion, or real annual performance claims were made here.

## Checkpoint-backed strategy reader (2026-09-08)

`ScheduledActivity` now lets the strategy pipeline consume one trailing activity
window at each decision. It advances existing SQLite/Parquet checkpoints, closes
the previous query reader before opening another, and retains compact source
history. Warmup includes trader ranking, conviction normalization and lagged market
volume. Both market selection and hypothetical trader preview prepare the window.
Failed window opens leave no stale reader and preserve the last successful
checkpoint for retry. Initial lineage validation still has the existing 8 GiB cap.

Conviction normalization now loads only past/current hourly samples and evicts
expired samples and deselected wallet/market pairs. It no longer materializes
samples through the future end of the backtest; this does not change its causal
scale calculation.

Multiweek regression fixtures compare complete pipeline outputs for weekly and
daily decisions across all three aggregation modes against the full-history
reader. These are infrastructure equivalence tests, not historical performance.

The API loader integration originally outstanding at this checkpoint is now
completed as described above. Bounded annual acquisition remains outstanding.
No new downloads or source deletion were performed for this stage, and no
year-long real-data result is claimed.

## Annual scheduled execution and saved comparison (2026-09-07)

Proxy v2 now supports at most 366 evaluation days when supplied a disk-ranking
output. The API worker supplies that sink in its owned scratch directory; report
and API publication copy the finalized, hash-verified file rather than iterating
all rows. Fixed metric/percentile schemas preserve metrics appearing after initially
inactive cohorts. Disk limits: 50 million rows, 4 GiB compressed/file, at most three
simultaneous ranking copies (12 GiB) during publication, with free-space checks and
a 64 MiB reserve. Byte checks occur after bounded writes; a failed oversized file
is not a successfully finalized artifact. Ranking batches are at most 4096 rows;
per-cohort ranking still has existing wallet/candidate bounds.

Legacy hourly/no-disk execution stays capped at 93 days. Annual simulation adds
explicit 250,000 asset-hour and signal-row guards, retaining bar/input/output
ceilings. Dataset calendar coverage permits up to 732 days for evaluation plus
warmup, but does not raise the 8 GiB / 5000-file / 250k market-row input limits.
Late previews discard old ranking lists while retaining historical cohort state;
this bounds retained scores, not total preview CPU time.

The scheduled report path also now exports proxy request and benchmark funding
ledgers; it previously dispatched those only for legacy proxy v1. Weekly prose
and proxy quantity/execution conventions are correct in newly generated reports.
Existing immutable reports are not rewritten.

Local app acceptance uses the explicitly synthetic dataset
`synthetic_annual_scheduled_20260907`: one fabricated BTC wallet, continuing daily
native fill cycles, 90-day ranking lookback, equal direction copying, 50% asset cap,
1 bps follower fee and 5 bps proxy slippage. Evaluation August 3, 2026 through
August 3, 2027 exclusive. Its copied prices are fabricated, not historical BTC.
Saved runs:

- Weekly: `af0e7b5795064abda4a0ae96f052a5e7`, 53 decisions/fills.
- Matched daily: `d06abe048a12402aa992e1f70a8d017d`, 365 decisions/fills.

Reports live under `.hyperliquid_lab/reports/hyperliquid_trader_ensemble_<id>`;
cohort/ranking drilldowns under `.hyperliquid_lab/results/<id>`. Both have 8,761
hourly points and 8,760 strategy/benchmark funding rows. API comparison retains
the full point count and downsampled display curves. An independent ledger replay
checked every hourly equity/cash point, funding quantity/rate/mark, execution price,
fee and terminal proxy position for strategy/BTC/cash in both runs: maximum observed
accounting discrepancy zero. This is accounting/infrastructure evidence only; no
performance or real annual coverage inference is justified by the fixture.

Verification: 313 core tests, 43 API tests, 13 frontend unit tests and all 13 browser
tests pass; production build and scoped code review pass. Existing dependency and
frontend chunk-size warnings remain. A browser/build concurrency mistake briefly
made the test frontend unavailable; the full browser rerun with a fixed build
passed. Old browser fixtures now explicitly choose their dataset.

Still required for the full goal: connect rolling checkpoints to execution and
bounded conviction queries; incremental cross-source validation and resumable
annual ingestion; native/proxy coverage and acquisition-budget audit/approval;
then actual long-period weekly/daily runs and reconciliation. No new source data
downloaded or source history deleted. Do not confuse this working annual calendar
path with evidence that a full native year fits the existing loader.

## Persistent causal checkpoints (2026-09-07)

`ActivityCheckpoints(root)` now publishes canonical seed Parquet and SQLite metadata
for a frozen compact-history input set. `build(catalog, manifest_ids, cutoff, ...)`
validates native-ID/economic conflicts across that entire bounded set, including
future rows, before retaining the latest strictly pre-cutoff event per wallet and
market. `open(id, start, end, ...)` checks seed and required replay-partition hashes;
expired partitions are no longer query inputs. `advance(id, cutoff, ...)` creates
an independent immutable checkpoint from prior seeds and intervening activity.
New source identities require a freshly validated lineage, not reuse by timestamp.

Replay rows are filtered at the checkpoint cutoff **before** unioning seeds. This
prevents an overlapping file's old duplicate provenance from overriding canonical
seed ordering. Exact native timestamps/order remain; seeds are not synthetic trades,
account balances, or ranking history. Queries before the retained cutoff reject.
The cache remains valid as a derived artifact even if old inputs are unavailable;
this is not a claim that those old files are still intact or that archive coverage
is complete. Rebuilding/changing the history still requires the retained inputs.

Publication fsyncs the immutable seed output before committing its hashed metadata
record. Unreferenced partial/orphan caches do not become reusable records. Engine
and dependency fingerprints invalidate incompatible caches. Inputs are never
deleted; this does not grant raw-cleanup authority.

Real retained-week check: initial cutoff August 2 produced 16,925 seeds in 2,294,880
bytes (about 2.2 MiB), taking 27.06 seconds including whole-source validation.
Full-reference versus cached August 2–4 queries produced exactly equal 30,963 BTC
ranking rows, five selected positions and observed markets. Volume differed by
0.000000954 USD on 2,871,290,635.83 USD (floating aggregate summation order, not a
bit-for-bit equality claim). Reference load/rank took 49.41 seconds versus cached
22.39 seconds; materialized rows were 4,875,344 versus 1,364,935. These are one local
`memory_mb=512` DuckDB-budget checks, not an annual runtime or process-RSS guarantee.

Advancing the persisted cutoff from August 2 to August 3 took 2.58 seconds and
produced 26,481 seeds in 3,621,330 bytes. A newly opened store reused that checkpoint
and reproduced the five previously checked August 4 positions exactly, retaining
860,319 active/seed rows. The advance/reopen process peaked at 846,884 KiB RSS
(about 827 MiB), illustrating that the DuckDB budget is not a whole-process cap.
Local metadata: `.hyperliquid_cache/compact_history/checkpoints/checkpoints.sqlite3`;
initial ID `b9fd2d3502c1404eb8e480dabd011737`, advanced ID
`965909111bd345ebab5e1a1dacb3fef9`. Caches bind the current engine fingerprint.

Fresh core suite: 298 passed, including 14 checkpoint tests; 13 existing dependency
warnings. Scoped read-only review found no important issues. Initial validation
still has the 8 GiB / 5000-file ceiling, and seed output is capped at two million
rows / 512 MiB (byte check after writing). Annual initial validation, bounded
annual simulation/artifact streaming, acquisition resumption and coverage/budget
approval remain required. The 93-day application cap has not been bypassed.

## Weekly strategy schedule (2026-09-07)

The versioned `hyperliquid_copy_lab_proxy_v2` configuration supports daily or
Monday-weekly trader/market selection and portfolio targets. Legacy proxy v1
continues hourly targeting. The API, saved/clone configuration, previews, output
estimates and TypeScript builder share this schedule. Hourly valuation and actual
native funding are unchanged; quantities do not continuously rebalance between
scheduled targets. Pending targets retain next-actual-open, delay, supersession
and maximum-wait semantics. No forced initial selection on a non-Monday start.

Core regressions cover changing leader positions midweek, held quantities, cash
before the first Monday, preview timing, and delayed/expired/superseded targets.
The multiweek synthetic API fixture exercises save → clone → daily comparison;
it is infrastructure test data, not strategy-performance evidence. The full core
suite passes 284 tests and API suite 42 tests. Frontend unit tests (13), schema
type check and production build pass; existing dependency/chunk-size warnings
remain. All 12 browser tests pass, including weekly save/clone/daily comparison and
the original hourly workflow. Scoped review found no important correctness issues.

The 93-day guards are intentionally unchanged. This does not yet wire compact
rolling history into the runner, persist incremental checkpoints, or qualify
one-year real source coverage. No new paid downloads or raw-data deletion.

## Windowed reader with causal seeds (2026-09-07)

`ProxyActivity(query_window=(start,end))` now materializes the active interval plus
the last strictly pre-start canonical event per wallet/market. Old events retain
their original timestamps; no fake new trades are inserted. This preserves dormant
positions, prior candidate membership and the last native market price for exposure
normalization. Ranking/volume lookbacks outside the retained window reject clearly;
reopen a wider window rather than silently return partial history.

`rolling_activity.open_rolling_activity` selects a retained historical prefix from
the SQLite catalog and verifies file identities before/after opening the reader.
It does not select only active-window files, which would lose dormant positions.
Full-prefix economic duplicate checks remain. Existing default ProxyActivity
behavior and input/file/memory/spill bounds remain unchanged.

Real local comparison: window August 2–4 exclusive, decision August 4 UTC, two-day
preset lookback, existing compact manifest and 512 MiB diagnostic DuckDB budget.
All 30,963 BTC ranking rows, selected positions, observations and unique BTC volume
match the full-prefix reference. Materialized rows: 1,701,086 reference versus
1,364,935 windowed, including 16,925 old seed records. Timings 23.59 versus 24.50
seconds: no speedup claim. No new data acquired or raw files removed.

This version rebuilds seeds from a bounded frozen prefix; it does **not** yet
persist/incrementally advance checkpoints, avoid old-file validation scans, or
support a year beyond existing bounds. It is not wired into dataset registration
or weekly UI/runtime yet. Catalog bounds are candidate-file bounds, not proof of
complete exchange coverage. Tests exercise dormant/future/boundary events, advancing
windows, stale prices, old conflicting duplicates and guarded input paths/budgets.
Fresh full core suite: 278 passed (nine windowed-reader tests), 13 existing
dependency warnings. git diff --check passed.

## Compact SQLite catalog (2026-09-07)

`compact_catalog.py` adds a transactional metadata index over immutable Parquet;
it does not load fills into SQLite or change `experiments.sqlite3`. This follows
the existing app's use of SQLite for local records. DuckDB remains the analytical
engine. SQLite's local application-file use is documented at
https://www.sqlite.org/whentouse.html.

Registered existing compact history into
`.hyperliquid_cache/compact_history/catalog.sqlite3`: 278,528 bytes (272 KiB),
168 partition records. Initial hash/time-bound validation and registration took
1.468 seconds. Manifest identity:
`23719b491556090c255b03460a2d003cce3d0d7dc3749a7f1525dd136c97f557`.
Reopening/re-registering revalidates source files and is idempotent. Real
registration also passed after strengthening canonical Arrow type checks.

The half-open August 3–4 UTC query selects 25 candidate files using actual event
timestamp bounds. Archive-hour labels are not substituted for event time. Consumers
must still verify file identities, apply exact row filters and native-ID deduplication.
Catalog lookup alone is **not** a rolling position reader: pre-window dormant
positions and prior-observed membership still require the planned causal checkpoints.
The index does not certify gap-free coverage or authorize raw deletion/downloads.

Fresh core suite: 269 passed, 13 existing dependency warnings, including 14 catalog
tests. Review found missing non-timestamp type checks; three failing tests reproduced
it before the fix. Empty partitions, multiple manifest selection, incomplete/corrupt
registration, half-open bounds and unrelated database protection are covered.
Weekly UI/runtime and annual acquisition remain unchanged and incomplete.

## Reusable compact-history proof (2026-09-07)

User approved weekly trader selection and weekly positions, with reusable compact
history and eventual job-owned raw cleanup. First local stage implemented in
`proxy_compact.py`, with CLI `tools/compact_hyperliquid_proxy_history.py`:

```bash
.venv/bin/python tools/compact_hyperliquid_proxy_history.py \
  --manifest .hyperliquid_cache/proxy_activity/proxy_activity_import_pcxc4x05/manifest.json \
  --output-root .hyperliquid_cache/compact_history
```

Do not repeat solely to recover results: retained output is
`.hyperliquid_cache/compact_history/compact_acsjsl0e/manifest.json`.

- 4,875,344 rows, all wallets in the four ingested markets retained.
- Normalized input 717,018,008 bytes; compact output 462,826,924 bytes
  (35.45% smaller, excluding small manifests). Raw download was 5,604,128,913
  bytes across all markets; do not confuse market filtering with compression.
- Compaction took 18.92 seconds; process high-water RSS 182,712 KiB (178.4 MiB).
  Arrow batch checks are post-decode guards, not a hard allocator memory limit.
- Every retained canonical value and type compared exactly against source; input
  identities checked before/after. Only optional raw_details_json is dropped.
  Existing fields for economics, wallet identity and deterministic order remain.
- 19 compactor tests; fresh full core suite 255 passed, 13 existing dependency
  warnings. Tests cover query equivalence, dormant/future positions, duplicates,
  empty partitions, schema failures, resource failures and mid-run source mutation.
- No downloads, raw deletions, dataset replacements or runtime-default changes.
  The proof format is not yet wired into registration/resumable ingestion.
- Full-week legacy-reader comparison exceeded its default 256 MiB limit during
  source duplicate validation. Diagnostic rerun uses 512 MiB explicitly, not a
  production limit change. Annual support still needs bounded rolling queries.
- That rerun passed: 30,963 BTC candidate ranking rows, five selected wallet
  positions, unique BTC volume, observed markets and deduplicated row count match
  exactly at August 4 00:00 UTC with the two-day preset lookback. Original/compact
  runs took 67.96/49.43 seconds respectively; these one-off timings are not a
  controlled speed benchmark. Independently rehashed all 168 compact partitions.

Code review found no blocking defects. Added safety tests and clarified that a
failure during final publication can leave a pending or visible manifest with
unconfirmed durability. This proof tool has no automatic restart or deletion
authority based on those artifacts; that belongs to the subsequent ingestion stage.

Next stages remain the weekly scheduler/config/UI, rolling partition catalog and
causal checkpoints, restart-safe ingestion/cleanup, then annual coverage and budget
audit. These results do not establish a year of available data or authorize paid
acquisition. Compact storage retains the whole acquired period, not only 90 days.

## Real application checkpoint (2026-09-07)

- Offline import completed: 4,875,344 records for BTC, xyz:TSLA, xyz:GOLD and
  xyz:SP500, with all mapped-market wallets retained. Source:
  `.hyperliquid_cache/proxy_activity/proxy_activity_import_pcxc4x05/manifest.json`.
- Registered dataset: `.hyperliquid_lab/datasets/real_cross_class_aug2026`, about
  490 MiB. Coverage is August 2 through August 7 exclusive; the source archive
  includes complete adjacent days. Native timestamps bracket both retained
  boundaries for every market. Prices/funding and raw source identities verified.
- Default hypothesis (chosen before portfolio results): August 4–6 exclusive,
  two-day trader lookback, top five per market, equal trader/asset weights,
  minimum one active day/two episodes/USD10k closing notional and gross volume,
  default equally weighted five profitability/copyability metrics, 60-minute
  updates,5bps proxy slippage,4.5bps fees,5s minimum delay,gross budget1.
  This is a short integration/research window, not a quarterly skill estimate.
- API was gracefully refreshed on port8010 with the same data directories after
  confirming no active jobs. First actual HTTP-submitted experiment:
  `af8f0ec48be5458797c6e3b493f56f98` completed, as did the cloned 25bps sensitivity
  `fbc95ab3678945a1b04b54ca48a485b0`. Preflight ready with 3,800,641 input rows.
- Baseline return -0.4720%, BTC benchmark +1.9341%; higher-slippage strategy
  -1.0583%. Both are saved and comparable in the UI. Independent cashflow/mark
  reconciliation passes for all 49 equity points in both runs and their controls;
  this is proxy-model accounting, not exact exchange execution reconciliation.
- Fresh core/API verification: 277 tests passed (15 dependency deprecation
  warnings). Existing saved runs were preserved. Full results and limitations:
  [real-data acceptance report](hyperliquid-real-proxy-acceptance-2026-09-07.md).

Earlier checkpoints below are retained as acquisition history.

The user approved the August 1–7 bulk native-fill download with a 6 GiB requested
object-byte ceiling and USD 1 estimated-cost budget. The guarded downloader freezes
all object sizes/ETags before GET, disables automatic retries, reserves each full
object before requesting it, and hashes/fsyncs local output. Acquisition completed:
`.hyperliquid_cache/proxy_archives/proxy_archive_5_ynp7r_/manifest.json`.
All 168 files were independently rehashed after the process exited successfully:
5,604,128,913 bytes (5.2193 GiB), within the 6 GiB ceiling. Manifest SHA-256:
`678637ea731e53bc45239f19d18aa4d59c8567d660410e2aaa65b05e06744b91`.
No retries or additional bulk requests were needed. Do not download this batch
again; normalization can use the verified local archive. Billing has not been
checked, so the planning estimate is not a confirmed AWS charge.

Offline importer: `tools/import_hyperliquid_proxy_archive.py --manifest <path>`.
It requires a completed, hash-verified full archive, preserves all wallets for the
declared markets and raw native fields, and bounds decoded lines, row batches,
batch bytes and disk output. Its success means complete source partitions, not
proven exchange-time completeness at the final edge. Registration must establish
boundary coverage (e.g. use interior dates with retained adjacent source days);
do not silently treat the final source-hour cutoff as a complete exchange-time
cutoff. Twelve downloader/importer tests pass; download review found no critical
cost/integrity issue. Import review prompted byte-bounded batches and explicit
source-partition-vs-exchange-time coverage labeling, both regression-tested.
The real archive has not yet been normalized or registered; no real saved strategy
result exists yet. The earlier checkpoints below predate bulk approval/acquisition.

Latest real-data checkpoint (2026-09-06): public prices and native funding for
2026-08-01 through 2026-08-08 exclusive have been acquired successfully:

- Prices: `.hyperliquid_cache/proxy_prices/proxy_prices_zl3q_a4r/manifest.json`.
  BTC has 168 hourly bars; TSLA, GLD and S&P 500 have 35 regular-session bars each.
  No scheduled bars missing; 30,430 raw source bytes.
- Funding: `.hyperliquid_cache/proxy_prices/proxy_funding_fjql18x7/manifest.json`.
  168 native events per mapped instrument, no missing hours; 62,891 raw bytes.
- Price acquisition now rejects in-period splits, dividends and capital-gain
  distributions, malformed event evidence and unsupported/rolling Yahoo types.
  The manifest records the price-policy qualification and its source limitations.
  This is not independently guaranteed corporate-action completeness: it records
  what the explicitly requested source event payload returned.
- 23 focused source/download tests and all 221 core tests pass. No bulk native
  fill download or complete real dataset registration has occurred yet.

Metadata-only S3 HEAD estimate on 2026-09-06, bucket `hl-mainnet-node-data`,
168 exact `node_fills_by_block/hourly/YYYYMMDD/H.lz4` objects:

| UTC day | Compressed bytes |
| --- | ---: |
| 2026-08-01 | 393,245,493 |
| 2026-08-02 | 534,488,928 |
| 2026-08-03 | 866,343,969 |
| 2026-08-04 | 964,387,774 |
| 2026-08-05 | 1,016,043,056 |
| 2026-08-06 | 933,092,785 |
| 2026-08-07 | 896,526,908 |
| Total | 5,604,128,913 |

This is 5.22 GiB; the largest object is 86,142,635 bytes. At the prior planning
assumption of USD 0.09/GiB the transfer component is about USD 0.47, excluding
requests/tax/account-specific pricing or free allowances. AWS bills requester-pays
requests and transfer to the reader account; this is not an enforced billing cap.
Separate bulk approval is required. Before GETs, the acquisition tool must freeze
fresh sizes/ETags and enforce a 6 GiB total ceiling and bounded object/decode/disk
limits. A metadata estimate is not approval and proves no fill coverage by itself.

This offline price-input tool supports the approved approximate copy-trader
research mode. It does not run a strategy or register a runnable lab dataset yet.
Hyperliquid funding and bounded trader-activity replay are still being integrated.

From this worktree, after `uv sync --all-packages --all-groups`:

```bash
.venv/bin/python tools/download_hyperliquid_proxy_prices.py \
  --mappings examples/hyperliquid_proxy_mappings.json \
  --start 2026-08-03 --end 2026-08-04 --with-funding
```

Dates are UTC, with exclusive end. Every run creates a fresh ignored directory
under `.hyperliquid_cache/proxy_prices`; existing runs are never overwritten.
No AWS keys or paid services are used. A failed acquisition may leave raw files
for diagnosis but does not publish a success manifest. The tool limits ranges
to 93 days, 50 mappings, 4 MiB per source response and 64 MiB retained raw data.

Outputs include raw responses, a source/hash manifest, frozen session windows
and `bars.parquet`. Binance daily archives are checked against their published
SHA-256 checksums. Missing expected bars are reported, not filled in. Future
coverage is rejected before downloading, so current live candles cannot become
completed historical evidence. A price-complete manifest is not a complete
backtest dataset. Price manifests explicitly record that they contain no funding;
`--with-funding` produces a separate native Hyperliquid funding artifact, preserving
actual settlement times and hourly coverage buckets. Missing hours reject funding
acquisition and leave a diagnostic coverage file rather than a success manifest.

Mappings are explicit, extensible JSON records. Their date windows are researcher
choices, not historical listing claims. Market eligibility must still be based
on Hyperliquid activity available before each strategy decision. Unmapped markets
are excluded visibly; the example is a four-class integration sample, not the
entire market universe.

Initial session support is 24/7 crypto and regular US equity sessions using pinned
exchange-calendar schedules, including holidays, early closes and DST. Extended
hours and futures-session calendars are not yet qualified. GLD therefore serves
as an explicit gold-return ETF proxy, with ETF tracking/expenses and regular-hour
limitations. ^GSPC is a price index, not a directly executable instrument.
Copy notional weights, never native Hyperliquid quantities across proxy units.
Binance USDT is treated as USD at parity and that assumption is retained.

Prices are raw provider OHLC; no additional adjustment or repair is applied by
this tool. Corporate-action and futures-roll effects still need qualification
before strategy acceptance. No historical profitability claim follows from
successfully downloading these prices.

Verified sample on 2026-09-06: one day, 2026-08-03, yielded 24 BTC bars and 7
bars each for TSLA, GLD and ^GSPC, with zero missing scheduled bars. Total source
download: 13,087 bytes. It is retained at
`.hyperliquid_cache/proxy_prices/proxy_prices_elw9emws/`; its manifest predates
the added acquisition-cutoff and scheduled-bar-count fields.

A subsequent run with native funding retained price evidence in
`.hyperliquid_cache/proxy_prices/proxy_prices_e9vd8nj7/` and funding evidence in
`.hyperliquid_cache/proxy_prices/proxy_funding_ysjg911b/`. All four markets had
24 actual funding settlements for August 3, no missing hours; 9,002 funding
response bytes. These prices and rates are not yet connected to a saved strategy.

The disk-backed trader query layer has also been checked against the previously
downloaded August 1 one-hour archive. Local artifact
`.hyperliquid_cache/source_probes/activity_9dts4f4j/` retains normalized fill
Parquet and a source-hashed diagnostic manifest: 20,838 fills across eight markets,
1,030 BTC candidate wallets, 159 eligible and five selected under deliberately
relaxed one-day-lookback integration settings. All five selected positions were
known strictly before the decision. This is not a full day of data, sufficient
ranking warmup, or a portfolio return result; the manifest explicitly marks it
incomplete for backtesting.

Queries deduplicate native wallet fill identities across archive overlaps and
count each market trade once rather than summing both counterparties. Rankings
share the existing scoring rules, retain dormant wallets as excluded candidates,
and use UTC daily bins independent of the host timezone. Input-byte, DuckDB
memory/spill, candidate-count and per-wallet-history bounds fail explicitly;
they never trigger wallet sampling or silent truncation. Source coverage must
still be established by acquisition manifests before registering a dataset.

The standalone hourly portfolio simulator now consumes normalized bars and native
funding directly. It sizes notional-weight targets at actual execution opens,
accounts for fees/slippage, preserves delayed pending targets through closures,
and uses completed closes for hourly valuation and approximate funding notional.
Closed-market and threshold-skipped holdings retain their exposure budget.
Source/session completeness still needs validation before calling this engine.

A local cross-class diagnostic used the retained August 3 price/funding artifacts
from 15:00 UTC through midnight with fixed equal weights: four fills, 36 funding
records and ten equity points; final cash plus unrealized PnL reconciled exactly.
It did not use trader-copy signals and was not saved through the application.
The core strategy runner is now connected to historical market and trader
selection, hourly copy signals, portfolio replay, BTC/cash benchmarks and
market/trader/contribution drilldown records. It supports top-N and top-percentage
selection, pooled/per-market ranking and the existing three aggregation methods.
Conviction normalization uses native quantity times the last native trade price,
not proxy share/ETF units; this is disclosed as an approximation to native marks.
Future-position/price tests verify earlier signals do not change.

The application now accepts `hyperliquid_copy_lab_proxy_v1` for registered hourly
datasets. The worker dispatches to the hourly runner, freezes settings and mapping
provenance, saves results and releases disk-backed query resources. Legacy v1/v2
configurations and saved runs are not migrated implicitly.

The builder selects execution mode from the dataset. Hourly mode locks target
updates to 60 minutes and exposes slippage, maximum mark age and execution wait.
It shows provider/ticker/unit/session mappings and the observed-only universe
limitation. Dataset presets are available for real as well as synthetic inputs.
Cloning preserves proxy settings; comparison warns when those settings differ.
Saved market drilldowns retain proxy tickers and availability basis. Portfolio
analytics use the declared hourly clock, not minute-grid assumptions; benchmark
funding and pending-request evidence can be downloaded. Cancellation and normal
shutdown remove job-owned scratch after the worker exits, without touching data.
Abrupt death of both coordinator and worker can still leave scratch requiring
operator cleanup; do not confuse scratch with registered datasets.

The standalone registered-dataset loader now accepts a distinct
`hyperliquid_lab_proxy_dataset_v1` manifest with `fills-*.parquet`, `bars.parquet`
and `funding.parquet`—never fabricated books or listing metadata. It verifies
frozen manifest/file identities, row counts, path containment, price-policy
qualification and declared source scope. Real registrations must declare complete
hourly archive keys and all-wallet coverage for their stated markets. These are
operator provenance assertions: the acquisition tool must establish them, and
checksums alone are not evidence of historical completeness.

Canonical fill checks run on disk, including native IDs, market/time scope,
finite values and position deltas. Both successful use and failed validation close
query resources. A registered synthetic fixture has loaded and run through this
boundary and the browser save/clone/compare flow; no real dataset has yet been
registered with the application. Bulk fill acquisition with an explicit cost/byte
cap and source/corporate-action qualification remains the next acceptance step.
