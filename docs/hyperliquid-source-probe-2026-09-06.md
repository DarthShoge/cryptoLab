# Hyperliquid real-data source probe — 2026-09-06

## Scope and authorization

The user approved archive metadata checks followed by a small sample download
with an estimated-cost ceiling of USD 1. Only IAM reader credentials were used;
no root credentials were read or used. This probe downloaded four objects,
bounded in advance to 64 MiB. No backtest was launched and no archive-derived
dataset was registered in the lab.

Raw evidence is retained, Git-ignored, at:

`.hyperliquid_cache/source_probes/20260801_00_e8758bip/`

Its `manifest.json` records source bucket/key, byte size, ETag and SHA-256 for
each object. These are downloaded source artifacts, not a complete runnable
dataset. No credentials are included in the manifest or this document.

## Acquisition

| Source | Object | Bytes |
| --- | --- | ---: |
| hl-mainnet-node-data | node_fills_by_block/hourly/20260801/0.lz4 | 23,438,314 |
| hyperliquid-archive | market_data/20260801/0/l2Book/BTC.lz4 | 203,703 |
| hyperliquid-archive | market_data/20260801/0/l2Book/ETH.lz4 | 202,229 |
| hyperliquid-archive | market_data/20260801/0/l2Book/SOL.lz4 | 139,567 |

Total: 23,983,813 bytes (22.87 MiB). Transfer estimate at the previously stated
USD 0.09/GiB assumption: USD 0.002010, excluding request charges, tax and any
allowances. This is not an AWS billing reconciliation. Each GET was conditional
on its HEAD ETag; byte lengths and SHA-256 were checked locally. No automatic
download retries were enabled.

## Observed fill format and qualification checks

- 50,658 block envelopes, 256,033 fill rows, 328 distinct raw market identifiers;
  113,369,242 uncompressed JSONL bytes. Counts include both sides and spot markets;
  they are not unique trades or a claim of 328 supported perpetual markets.
- Envelope fields: `local_time`, `block_time`, `block_number`, `events`.
- First fill: 2026-07-31T23:59:59.937Z; last fill:
  2026-08-01T00:59:59.665Z. File partition boundaries are not exact exchange-time
  boundaries; selection must filter actual event timestamps.
- All 17,838 BTC/ETH/SOL rows failed the current production parser with
  `timezone-aware timestamp required`. Archive block timestamps omit a timezone
  suffix and contain nanosecond precision; exchange timestamps are epoch ms.
- A diagnostic, in-memory UTC interpretation of block timestamps allowed all
  244,320 rows matching the current identifier rule to parse, across 273 market
  identifiers including 98 namespaced ones. No production parser or raw artifact
  was changed. UTC interpretation must be made an explicit source-adapter rule,
  not a relaxation of timestamp validation everywhere.
- With that diagnostic normalization, 226,455 successive wallet/instrument
  position checks had zero discrepancies above absolute tolerance 1e-8. This is
  one-hour internal continuity evidence, not independent account-state or PnL
  reconciliation.
- Examples of namespaced fills: `xyz:GOLD`, `xyz:SP500`, `xyz:TSLA`, `xyz:CL`.
  Their presence establishes fill coverage in this hour, not complete price,
  funding, contract metadata or execution-book coverage.
- Current identifier validation excludes digit-leading `0G` (256 fills) and
  `2Z` (222 fills). The archive metadata also contains a `0G` order-book object.
  Spot identifier `PURR/USDC` (225 fills) and `@...` instruments must not be
  admitted merely by widening the symbol regex: actual instrument classification
  remains necessary.
- Raw fills include `deployerFee` and `twapId` in addition to the previously
  supported fields. 124,603 regex-admitted rows have nonzero `deployerFee`.
  Preserve raw fields and verify whether this fee is included in `fee` before
  accounting for it; do not automatically add it and risk double counting.

## Order-book observations

All three files contain 671 valid distinct exchange-time snapshots, each with
20 bid/ask levels. Source shape is `time`, `ver_num`, and
`raw.channel == l2Book`, with `raw.data.coin`, `raw.data.time` (epoch ms) and
`raw.data.levels`. These normalize to the current `MarketData` book shape.

For each sampled asset:

- First snapshot: 2026-08-01T00:00:02.402Z.
- Last snapshot: 2026-08-01T00:59:59.520Z.
- Mean interval: 5.368833 seconds; minimum 2.736, maximum 5.597 seconds.
- 59 of 60 minute marks satisfy the existing 60-second mark-age limit.
  The midnight mark lacks a prior-hour book, which this isolated probe did not
  download. This is not evidence of an archive-wide midnight gap.
- Only 23 of 60 minute-spaced targets find a book within two seconds after a
  five-second execution delay. The present execution-book coverage gate would
  reject this sample; do not silently widen it or claim five-second execution
  using later snapshots.

Prior metadata survey: Aug 1, 4 and 7 each contained 24 fill objects and 4,224
L2 objects (176 identifiers, each with 24 hourly files). No namespaced L2
filenames were present under those sampled `market_data/<date>/` prefixes.
Hourly file presence alone does not prove usable within-file time coverage.

## Next implementation decisions / unresolved evidence

1. Source-specific timestamp normalization, digit-leading perpetual IDs, raw fee
   preservation and streaming/partitioned import need real-source regression tests.
2. Execution needs an explicit archive-snapshot policy: execute at the next
   available snapshot after the requested delay within a declared cap, record
   actual elapsed delay, and reject missing/stale evidence. This changes the
   model and requires approval; it is not a hidden tolerance fix.
3. Namespaced non-crypto execution books, dated contract/classification metadata,
   market-wide volume counting and fee/funding semantics still require
   qualification. Do not replace them with crypto evidence or today's metadata.
4. The single-hour fill count shows that the lab's one-million-input-row limit
   cannot support a seven-day all-market run unchanged. Do not raise that limit
   as a substitute for bounded on-disk processing or silently sample wallets.
5. No full-period download, strategy results, profitability claim or completion
   of the real-data objective is implied by this successful probe.

Reference: [Hyperliquid historical data documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data).

## Follow-up: execution approval and HIP-3 book source

The user approved next-available-snapshot execution with a declared wait cap
and actual elapsed delay recorded. That model is approved but not implemented
by this source investigation.

SonarX documents public CC0 top-20 L2 snapshots for HIP-3 markets, captured
every 20 blocks, under:

`s3://sonarx-hyperliquid-public/market_data/hip3/{market}/l2-summary-snapshots/{partition}/{height}.json.gz`

Its [HIP-3 announcement](https://www.sonarx.com/blog/sonarx-releases-hyperliquid-order-book-snapshots)
describes weekly loading with a two-day lag and gzipped JSON arrays containing
block height/time, market, bids and asks. Its
[standard-perp expansion](https://www.sonarx.com/blog/sonarx-adds-standard-perp-order-book-snapshots)
documents a sibling `market_data/perp/` prefix. Public/free licensing does not
remove requester-pays transfer charges.

A bounded anonymous listing of `market_data/hip3/` returned `AccessDenied`
with the explicit explanation that anonymous callers cannot access requester-pays
buckets. No SonarX objects were downloaded or paid authenticated requests made.
The reader policy previously supplied to the user covers only the two original
buckets. At that stage SonarX access, actual date coverage, symbol mapping and
cadence were unverified pending permission for this additional bucket.

Hydromancer Reservoir also documents HIP-3 fills, candles and position snapshots,
but its published dataset list does not establish historical execution-book
coverage. Candles must not be substituted for L2 execution evidence under the
approved model.

## Follow-up: authenticated SonarX sample verified

After the user confirmed the IAM policy extension, authenticated requester-pays
LIST, HEAD and GET succeeded using the same IAM reader. No root credentials
were used. The HIP-3 market-prefix listing was not truncated and included
commodity, equity and index identifiers.

Six objects were downloaded under the existing USD 1 sample authorization,
with a separate hard cap of 1 MiB for this probe. Actual compressed transfer
was 19,382 bytes. At the earlier USD 0.09/GiB transfer assumption that is under
USD 0.000002, excluding request charges, tax and allowances; this is not a bill.
Each GET used its HEAD ETag and verified byte length, with automatic retries
disabled. Source keys, ETags, byte lengths and SHA-256 hashes are retained in:

`.hyperliquid_cache/source_probes/sonarx_20260801_vv0wao2r/manifest.json`

For each of `xyz:GOLD`, `xyz:TSLA` and `xyz:SP500`, the sampled keys were:

`market_data/hip3/{market}/l2-summary-snapshots/1093450000/{height}.json.gz`

where file heights were `1093454000` and `1093455000`. The directory market
matched the JSON `market` field in every sampled row. Each file held 50 rows;
the file-height label is the final height, not the first row's height. Importers
must filter actual row heights/timestamps rather than assuming key boundaries.

For each market, the two files together contained:

- 100 distinct snapshots, from `2026-07-31T23:59:05.087831675` through
  `2026-08-01T00:01:24.436710916` (source strings omit timezone suffix).
- Exactly 20 blocks between successive snapshots.
- Mean interval 1.407564 seconds; minimum 1.314762, maximum 1.914632 seconds
  (timing statistics computed at Python datetime microsecond precision).
- 100/100 books with 20 bids and 20 asks, ordered prices, positive quantities
  and best bid strictly below best ask.

Across the shared snapshot heights, 60 appeared in the downloaded official fill
archive. All 60 block-time strings matched exactly, including nanoseconds. This
establishes local cross-source alignment; it does not independently establish
timezone semantics or guarantee full-period completeness.

The immediate non-crypto execution-book source gap is resolved for these sampled
markets and times. Historical instrument/settlement metadata, funding and fee
semantics, complete period coverage, general-universe volume construction and
bounded replay remain to be qualified or implemented. No full-period pull was
authorized by this probe, no lab dataset was registered and no backtest was run.

## Follow-up: funding confirmed; historical catalogue qualification blocked

Four read-only public `fundingHistory` requests succeeded for `BTC`,
`xyz:GOLD`, `xyz:TSLA` and `xyz:SP500`, with start `1785542400000` inclusive
and end `1785628799999` inclusive (2026-08-01 UTC). Each returned 24 records.
For all four, the first returned timestamp was `1785542400080` and the last
was `1785625200048`: actual settlement timestamps are not exact hour boundaries.
These diagnostic responses were inspected but not registered as dataset files.
The implementation must explicitly distinguish scheduled funding buckets from
actual settlement times and preserve raw source timestamps.

Source: [official perpetuals API documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals).
It documents HIP-3 funding history; `meta` and classification requests do not
document historical timestamp parameters. Current metadata must not be assigned
an invented historical `known_at` timestamp.

Authenticated, untruncated root listings found:

- `hl-mainnet-node-data`: explorer_blocks, misc_events_by_block, node_fills,
  node_fills_by_block, node_trades, replica_cmds.
- `hyperliquid-archive`: Testnet, asset_ctxs, market_data.
- `sonarx-hyperliquid-public`: market_data plus README, LICENSE and CHANGELOG.

No historical metadata/state-snapshot prefix was exposed in these root indexes.
This is not a claim that no provider possesses the data. Official node
[documentation](https://github.com/hyperliquid-dex/node/blob/main/README.md)
describes locally generated periodic ABCI state snapshots, but a downloadable
historical baseline for the requested period has not been located.

The official [event schema](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/nodes/l1-data-schemas)
documents miscellaneous events as staking and ledger/funding events, not a
complete historical contract catalogue. Metadata LIST confirms an August 1
miscellaneous-event object, but it was not downloaded for this investigation.

Transaction logs are a possible reconstruction route, not yet a qualified
adapter or proven complete baseline. A bounded LIST of
`replica_cmds/2026-07-25T07:35:28Z/20260801/` returned these first two objects:

| File | Compressed bytes |
| --- | ---: |
| 1093460000.lz4 | 729,417,176 |
| 1093470000.lz4 | 711,003,901 |

Neither was downloaded. Each exceeds the probe's 64 MiB ceiling; the cost and
coverage of a complete reconstruction have not been estimated. Do not extrapolate
a cheap fills-only pull into authorization for these substantially larger logs.

**Stop condition:** a trustworthy historical instrument catalogue (contract
identity, collateral/settlement, classification and lifetime/configuration
history) is not yet sourced. The user explicitly requested stopping when a
required data source is unknown. No importer implementation, bulk download or
real-data strategy run proceeds on fabricated metadata. A separately approved
investigation of historical state providers or transaction-log reconstruction
is the next decision; the overall real-data objective remains incomplete.

The user subsequently approved the reconstruction investigation without large
log downloads. Its measured costs, small explorer sample and remaining baseline
requirements are recorded in
[the reconstruction feasibility report](hyperliquid-metadata-reconstruction-feasibility.md).
