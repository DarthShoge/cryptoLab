# Native-start qualification checkpoint

## New verified boundary mismatch — 2026-09-16

Read-only native scan21774 completed exit0 in108.238seconds, peak RSS200101888bytes.
It verified all319594956physical rows in the batch0049 qualified prefix,
2025-06-02→2026-05-18exclusive. ReportSHA:
3e87c397c58710fd2262db34e6c1626b32b295909f57a06a2573fb310e2e7331.
This is not final annual qualification; later source partitions can contain
earlier events, so final publication must rescan the completed corpus.

| Market | First observed fill UTC | Current activity-hour rule | Retained funding coverage starts UTC |
| --- | --- | --- | --- |
| BTC | 2025-06-02 00:00:00.086 | 2025-06-02 00:00 | 2025-06-02 00:00 |
| xyz:TSLA | 2025-11-13 14:31:56.848 | 2025-11-13 14:00 | 2025-11-13 15:00 |
| xyz:GOLD | 2025-12-22 14:42:19.649 | 2025-12-22 14:00 | 2025-12-22 15:00 |
| xyz:SP500 | 2026-03-18 13:26:16.574 | 2026-03-18 13:00 | 2026-03-18 14:00 |

BTC's eventual registered activity start is clamped to the padded interior start;
the table shows source-observed hours, not final published starts.
Physical rows by market:BTC293524766,GOLD8514304,SP5008897460,TSLA8658426.

The existing qualify_observed_history inequality rejects funding_begin > activity
hour. This evidence violates that condition for all three cross-class markets.
The earlier conditional concern below is now observed, not hypothetical. Do not
move activity starts forward, discard early fills, or manufacture zero funding.

Both retained annual market bundles were independently reconstructed and matched
their pinned manifests in session86531, exit0. Prices manifestSHA:
1ceffd67fd921ce21c409a8616579eda656803f310b18d83568c5526df1181f2,
14034bars/473495bytes; funding manifestSHA:
9b3d54065db4c73c00ba16839487055bce37ff9d09146ff34a233078f2607f25,
28084settlements/649285bytes. Both end2026-09-02exclusive. Cross-class price
coverage begins2026-02-03, before the90-day-native-warmup admission dates; this is
not a claim of native history at the proxy price start. Rebuild scratch was owned
temporary verification output under/tmp and was removed by its context manager;
original price/funding/source artifacts were untouched. No downloads were made.

Design approved on2026-09-16: retain distinct observed activity and verified funding
coverage starts, with explicit evidence binding and fail-closed uncovered-exposure
checks. Legacy datasets keep existing semantics. Implementation remains pending; no new
policy, schema or simulator behavior is implemented by this diagnostic.

## Implemented qualification policy

The reviewed `complete_source_observed_history_v1` policy now combines a completed,
fully verified job with exact observed event bounds and reconstructed funding
coverage. It uses max(interior coverage start, first observed UTC hour) as a
conservative usable observed-history start. It rejects absent/post-interior markets,
BTC first observed after the interior start, and funding that fails to span the
declared native interval. This is explicitly not listing-date evidence and does
not claim trading history before observation. A scan alone or a partial prefix
cannot qualify. The real annual job is still partial, so this policy has not yet
qualified its markets.

`registration_native_history.qualify_observed_history` produces the evidence;
`native_history_evidence.verify_native_history_evidence` checks the persisted
sidecar during normal dataset loading. The check binds actual file bytes, market
scope, start dates, coverage dates, final source-report identity and funding bundle
identity to the registered metadata. Legacy datasets that do not claim the new
policy keep their prior semantics. Synthetic source/funding and sidecar tampering
tests pass: 38 related tests in 22.67 seconds. Review found no important issues;
the archive worker's frozen engine identity remains unchanged.

## Reusable observed-time scan

`native_history_scan.scan_qualified_history(report_pin)` now performs the bounded
exact scan needed by this audit. It streams only coin and UTC microsecond event
timestamps, verifies physical row counts including duplicates, and rechecks all
pinned files/manifests and engine identity before returning. Five focused tests
pass, including repeated/spilled events, corruption and explicit absent markets;
narrow review found no defects. It makes no native-availability claim.

The real scan of the report below completed in **14.408 seconds** and counted all
65,219,008 physical rows as BTC. Earliest BTC event:
`2025-06-02T00:00:00.086000+00:00`; latest:
`2025-09-14T23:59:58.559000+00:00`. All three cross-class records explicitly return
zero rows and null event bounds. This is still a partial source-prefix result,
not final annual coverage or a completed backtest.

## Verified evidence, not listing inference

Reverified the recorded archive-job qualification using
`archive_job_qualification.verify(..., recheck_content=True)` against the current
frozen engine, canonical files and retained raw-source manifests. No download,
retry, cleanup or source mutation was performed by this check.

- Job: `.hyperliquid_cache/annual_job_20250901_20260901_300gib`.
- Report: `batches/0014/qualified/qualification_4asofh9z/manifest.json`.
- SHA-256: `bd138222227ec6e9ed413242c70942bedccb6f53fa7f8419ee0451e87a2c7f54`.
- Complete source prefix: June 2–September 15, 2025 exclusive.
- Canonical corpus: 105 daily files; 65,219,008 rows; 5,663,297,154 bytes.
- Exact validated per-file coin/trade bounds contain BTC only. None contains
  `xyz:TSLA`, `xyz:GOLD` or `xyz:SP500`.

This establishes absence from this verified source prefix, not their later
listing dates or usable-history starts. The archive worker remains active beyond
this snapshot. Deleted raw payloads were not rehashed; their retained original
hash/provenance records were checked as such.

## Remaining boundary checks

Once the source prefix reaches each market's first activity, compute its actual
minimum exchange timestamp from the canonical rows, including retained boundary
spill. Do not use source partition dates as event timestamps. Bind results to the
exact verified corpus; final annual publication must recheck against the completed
corpus so a later source partition carrying an earlier event cannot be ignored.

Compare first activity with the independently retained native funding series.
**The first trading hour and first funding-settlement hour need not coincide.**
Current registered metadata supplies native-history starts to the simulator as
funding starts too. If the real boundary shows a difference, do not move activity
dates forward, drop early fills, or fabricate a zero funding settlement to satisfy
that contract. A separately evidenced funding-coverage start will be needed.
Keep the ranking warmup tied to native activity/history, not a convenient price
or funding cutoff. This is a conditional integration concern, not yet an observed
failure in these markets.

The [S&P DJI announcement](https://www.spglobal.com/spdji/en/index-announcements/article/sp-dow-jones-indices-licenses-sp-500-to-trade-xyz-for-perpetual-contracts-on-hyperliquid/)
corroborates a March 18, 2026 SP500 launch date but does not establish an exact
first-fill/funding hour. The September 9 targeted search did not supply comparable
primary timestamp evidence for TSLA or GOLD; no search-result inference was
promoted to registered metadata.

## Annual publication dependency

The legacy `proxy_registration.register_dataset` accepts one normalized import,
at most 169 activity files / 8 GiB, and requires each coin's observed events to
bracket the whole interval. That contract cannot publish this historically
expanding annual corpus. The annual path must consume the fully qualified job,
retain the all-wallet compact corpus and source evidence, enforce interior event
boundaries and native starts, combine the verified market bundles, and load using
the existing bounded `sharded_v1` scheduled-dataset contract. No annual dataset or
saved annual comparison has been published by this checkpoint.
