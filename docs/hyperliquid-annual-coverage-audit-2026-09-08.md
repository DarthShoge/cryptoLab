# Annual native-history feasibility audit — 2026-09-08

## Outcome

A year-long run cannot treat the current BTC / xyz:GOLD / xyz:SP500 / xyz:TSLA
universe as if all four native markets existed throughout the year. Public funding
responses begin at different dates; S&P's own announcement dates its market launch
to March 18, 2026. Price proxies cannot manufacture earlier Hyperliquid wallets,
positions, fills or funding.

Two honest research designs remain: a year with a historically expanding universe
(new markets remain excluded until their configured warmup is available), or a
shorter common-coverage cross-class run. A crypto-only year is a separate scope
choice, not an automatic fallback. The user has since approved the expanding
universe. No annual bulk transfer budget has been approved here.

## Exact candidate-year object inventory (subsequent approved metadata audit)

Candidate evaluation: September 1, 2025 through September 1, 2026 exclusive,
365 days starting Monday. The 90-day warmup starts June 3, 2025; one full source
padding day on each side gives June 2, 2025 through September 2, 2026 exclusive
(457 source days). This replaces the earlier September 8 candidate because its
right padding day is not yet complete at this audit's date.

Seventeen separately approved monthly-prefix ListObjectsV2 requests, MaxKeys=1000,
RequestPayer=requester, no pagination or retries, **zero GETs**. Every response was
untruncated. July 2025 queried both old and block prefixes. Exact expected keys use
the corrected within-hour handoff, including both hour-8 objects.

- Expected and listed objects: **10,969**, no missing expected keys.
- Total compressed bytes: **295,900,377,630 (275.5787 GiB)**.
- Existing August cache: **168 matching key/ETag/size identities**,
  **5,604,128,913 bytes**. Local complete archive hashes were verified in the
  preceding resume test; no redownload is needed if identities remain unchanged.
- Remaining transfer before separately authorized retries:
  **290,296,248,717 bytes (270.3594 GiB)**.
- Greedy chronological whole-day batches bounded by seven days and 6 GiB produce
  **67 batches**, largest **6,377,542,749 bytes**. Batch lengths are 1, 5, 6 or 7
  days. Thus a fixed seven-day acquisition cadence is not sufficient everywhere.
- Largest source day: August 21, 2026, **1,601,935,296 bytes**.

Six objects exceed the downloader's original 128 MiB object guard:

| Block source date/hour | Compressed bytes |
| --- | ---: |
| 20250822/14 | 156,299,698 |
| 20251010/21 | 359,334,134 |
| 20251010/22 | 216,563,602 |
| 20251010/23 | 144,731,951 |
| 20260206/0 | 139,233,731 |
| 20260822/5 | 197,971,337 |

Do not silently skip these busy hours or raise decoder/input limits blindly. The
current 2 GiB decoded-per-object guard also needs to be considered separately;
compressed metadata cannot establish decoded size or parser validity. The annual
controller now has byte-aware batches, persisted job-wide reservations, verified
reuse of the old cache, and measured streaming transport with a 384 MiB compressed
ceiling. All six sizes fit that transport gate; their actual decoded contents have
not been acquired or qualified. Legacy full-corpus validation retains its 8 GiB
limit; explicitly registered sharded mode supports a bounded 64 GiB corpus, with
separate rolling working-set limits. See [implementation and measurement status](hyperliquid-proxy-data.md).

Machine-readable audit, including all selected keys/ETags/sizes and daily totals:
`.hyperliquid_cache/inventories/annual_metadata_a8xxziuo/inventory.json`.
SHA256: `57b5326fa5756ab5c2131e2a6dbcf6738d831679db571a2cdcd31b5020c46999`.
Individual listing responses are retained alongside it. These are metadata
identities, not content hashes, gap-free event-time proof, a completed dataset,
or permission to download. No annual bulk allowance has been granted.

AWS bills successful Requester Pays requests and data transfer to the requester;
see [AWS Requester Pays documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html).
No dollar estimate or free-tier eligibility is established by this inventory.

## Public funding probe (earlier bounded sample)

Four bounded POSTs to `https://api.hyperliquid.xyz/info`, type `fundingHistory`,
startTime 1749427200000 (2025-06-09 UTC), endTime 1788739200000 (2026-09-07 UTC).
Each response contained 500 rows. These are first returned timestamps, not a
complete funding audit or independently verified listing dates.

| Market | First returned UTC timestamp | Last returned UTC timestamp | Response bytes |
| --- | --- | --- | ---: |
| BTC | 2025-06-09 00:00:00.057 | 2025-06-29 19:00:00.122 | 44,478 |
| xyz:TSLA | 2025-11-13 15:00:00.014 | 2025-12-04 10:00:00.081 | 46,760 |
| xyz:GOLD | 2025-12-22 15:00:00.012 | 2026-01-12 10:00:00.042 | 47,238 |
| xyz:SP500 | 2026-03-18 14:00:00.026 | 2026-04-08 09:00:00.047 | 48,298 |

If these starts are subsequently qualified as usable activity/funding availability,
90-day warmup followed by Monday selection would permit TSLA no earlier than
February 16, GOLD March 23, and SP500 June 22, 2026. These are planning bounds,
not admitted market metadata. Exact fills, continuous coverage, mapping identity,
session bars and corporate-action policy still require qualification.

Primary corroboration: [S&P DJI launch announcement](https://www.spglobal.com/spdji/en/index-announcements/article/sp-dow-jones-indices-licenses-sp-500-to-trade-xyz-for-perpetual-contracts-on-hyperliquid/).

## S3 metadata-only survey

Thirteen `ListObjectsV2` requests to `hl-mainnet-node-data`, RequestPayer=requester,
MaxKeys=1000; every response was untruncated. IAM reader credentials only. No
GetObject calls, downloaded fill objects, or source deletions. Request charges may
apply; this is not a claim of a free AWS operation.

| Date | Prefix below bucket | Objects | Compressed bytes |
| --- | --- | ---: | ---: |
| 2025-06-09 | node_fills/hourly | 24 | 450,773,948 |
| 2025-06-09 | node_fills_by_block/hourly | 0 | 0 |
| 2025-07-26 | node_fills/hourly | 24 | 421,282,559 |
| 2025-07-26 | node_fills_by_block/hourly | 0 | 0 |
| 2025-07-27 | node_fills/hourly | 9 | 133,057,253 |
| 2025-07-27 | node_fills_by_block/hourly | 16 | 358,812,351 |
| 2025-08-04 | node_fills_by_block/hourly | 24 | 517,845,053 |
| 2025-10-13 | node_fills_by_block/hourly | 24 | 757,887,772 |
| 2026-01-05 | node_fills_by_block/hourly | 24 | 512,737,289 |
| 2026-03-18 | node_fills_by_block/hourly | 24 | 806,736,555 |
| 2026-08-01 | node_fills_by_block/hourly | 24 | 393,245,493 |
| 2026-08-31 | node_fills_by_block/hourly | 24 | 1,011,665,825 |
| 2026-09-07 | node_fills_by_block/hourly | 24 | 882,999,053 |

Critical handoff detail: July 27 legacy objects cover hours 0–8 inclusive;
block-format objects cover 8–23 inclusive. Hour 8 exists in both formats. A simple
date-only switch loses hours 0–7, while concatenation needs semantic deduplication
and verified overlap. Presence and object names do not prove complete event-time
coverage. Both hour-8 files were subsequently sampled under a separate 2 MiB
approval, as recorded below.

A candidate 365-day evaluation September 8, 2025 through September 8, 2026 exclusive
needs June 10, 2025 warmup for a 90-day rule, plus source padding. Thus it crosses
the legacy-format interval. Correction after inspecting source records: the
existing parser already supports the legacy `[wallet, fill]` stream. The bug was
the date-only source-key switch, not the parser. Merely raising the seven-day
limit would still leave coverage incorrect.

## Bounded handoff content probe and regression fix

Two approved `Range: bytes=0-1048575` GETs, each preceded by HEAD and guarded by
IfMatch, HTTP 206 and ContentLength checks. No retries. Total downloaded content
2,097,152 bytes; neither full object was downloaded or persisted. LZ4 prefix
decoding was capped at 8 MiB per object; each yielded a 4 MiB decoded block.
Only complete JSONL lines were parsed, without a market filter.

| Hour-8 source | Full object bytes | Complete prefix lines | Parsed perp events | Earliest / latest sampled UTC fill |
| --- | ---: | ---: | ---: | --- |
| node_fills/hourly/20250727/8.lz4 | 12,726,384 | 10,996 | 9,582 | 08:00:00.018 / 08:03:28.235 |
| node_fills_by_block/hourly/20250727/8.lz4 | 1,517,603 | 877 | 9,942 | 08:50:10.273 / 08:52:53.323 |

Legacy sample SHA256 `c2a09285d1e66a6c2bb6fa188969c08ae69853f4494157c45c4e76d59866a87c`,
ETag `b21cdc16a4ec69d9c5b72c0c27ff148d-2`.
Block sample SHA256 `dba5d64791333c5bfcc1ed3f6a41ad6f8affe6ef24a56bb28d77e096a6b542d7`,
ETag `60d4c1277f76651b5f04666bbe8d7594`.
The 1,414 legacy and 462 block parsing issues were all `invalid perp instrument ID`
(the unfiltered probe also encounters spot identifiers). These partial samples
confirm envelope compatibility, not full-hour completeness or exact overlap.

Source enumeration now retains legacy hours 0–8 and block hours 8–23 on July 27.
A crossing seven-day batch consequently allows 169 partitions in compaction,
catalog registration and dataset publication; existing byte limits are unchanged.
Synthetic regressions cover both formats, missing either hour-8 object, native-ID
deduplication, padded registration and rejection above the 169-partition bound.

Verification: 356 core strategy tests passed (13 existing dependency warnings),
plus a scoped read-only code review with no correctness findings. No annual bulk
download or real annual run has been performed. The initial full-corpus 8 GiB
validation limit still needs a bounded incremental replacement before annual
ingestion; the handoff fix does not bypass it.

The ten sampled days (combining both handoff prefixes) range 0.366–0.942 GiB/day,
mean 0.582 GiB/day. Multiplying by 455 days gives 166.6–428.7 GiB, mean 264.7 GiB.
This is a sparse-sample extrapolation, not a bound, exact manifest or billing
quote. Include padding and actual historical volume before requesting approval.
Selecting fewer markets reduces retained output, not all-market source transfer.

## Existing helper reassessment

[`bond-labs-dev/hyperliquid-data`](https://github.com/bond-labs-dev/hyperliquid-data)
provides archive/funding/price acquisition helpers. Its inspected version declares
0.1.0, alpha, MIT. No package was installed and no upstream code was executed.

Its [prefix mapping source](https://raw.githubusercontent.com/bond-labs-dev/hyperliquid-data/main/src/hyperliquid_data/prefixes.py)
has a July 27, 2025 legacy handoff helper, but `dataset_location` explicitly routes
fills and trades only to the newer prefix; the legacy switch is used for
liquidations. Consequently it is not a drop-in annual all-wallet fill importer.
The date-only legacy helper also does not express the observed within-day overlap.
Reuse suitable public helpers after pinning/review, but do not replace the tested
canonical fill semantics or assume its README establishes our coverage.

[Hyperliquid's historical-data documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data)
confirms the separate old/new sources, requester-paid transfer and possible gaps.

## Next decision and implementation

The user approved the recommended expanding-universe design in the subsequent
reply “lets go with your recommendation then”. The design choice is resolved;
actual source qualification and a separately approved download budget are not.

Prefer the full-year expanding-universe design if retaining the annual objective:
mark each asset ineligible before qualified availability plus its lookback, and
preserve those exclusions in historical universe drilldown. That requires explicit
native funding/market lifetime handling, now implemented for explicitly qualified
`native_history` metadata. No real annual qualification is yet attached. Do not
turn pre-market missing funding into fabricated zero rates or stretch mappings
backward.

Remaining: freeze dates/markets; enumerate exact source identities;
implement bounded incremental validation and
resumable acquisition with lifetime spending; qualify price/funding/action policy;
request a specific bulk budget; only then acquire and save the real comparison.
